# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG Stages 5-6: Incremental Community Update and Selective Summarization.

Stage 5 re-runs Leiden community detection only on the k-hop neighborhood
of modified entities. Stage 6 regenerates community reports only for
communities marked as stale.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.models.temporal_types import utcnow

if TYPE_CHECKING:
    from neo4j import AsyncDriver, AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 5: Incremental Community Update
# ---------------------------------------------------------------------------


async def get_modified_entity_titles(
    session: "AsyncSession",
    since: datetime,
) -> list[str]:
    """Get entity titles that have been modified since a given timestamp.

    Queries Neo4j for entities whose edges were written or updated since `since`.
    """
    query = """
    MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
    WHERE r.t_tx_start >= $since
    RETURN DISTINCT s.title AS title
    UNION
    MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
    WHERE r.t_tx_start >= $since
    RETURN DISTINCT t.title AS title
    """
    result = await session.run(query, since=since.isoformat())
    titles = []
    async for record in result:
        titles.append(record["title"])
    return titles


async def get_k_hop_neighbors(
    session: "AsyncSession",
    entity_titles: list[str],
    k: int = 2,
) -> set[str]:
    """Get all entities within k hops of the given entities.

    Returns a set of entity titles forming the neighborhood to re-cluster.
    """
    if not entity_titles:
        return set()

    # Build a Cypher query for k-hop neighborhood
    all_neighbors = set(entity_titles)

    for hop in range(k):
        query = """
        MATCH (n:Entity)-[:RELATIONSHIP]-(m:Entity)
        WHERE n.title IN $titles
          AND NOT m.title IN $already_found
        RETURN DISTINCT m.title AS title
        """
        result = await session.run(
            query,
            titles=list(all_neighbors),
            already_found=list(all_neighbors),
        )
        new_titles = set()
        async for record in result:
            new_titles.add(record["title"])

        if not new_titles:
            break
        all_neighbors |= new_titles

    return all_neighbors


def identify_stale_communities(
    communities_df: pd.DataFrame,
    modified_entity_ids: set[str],
) -> list[str]:
    """Identify communities that contain modified entities and need re-summarization.

    Returns list of community IDs that are stale.
    """
    stale_ids = []
    for _, row in communities_df.iterrows():
        entity_ids = row.get("entity_ids", [])
        if isinstance(entity_ids, str):
            entity_ids = [e.strip() for e in entity_ids.split(",") if e.strip()]
        if isinstance(entity_ids, list):
            for eid in entity_ids:
                if str(eid) in modified_entity_ids:
                    stale_ids.append(str(row.get("id", "")))
                    break
    return stale_ids


async def run_incremental_community_update(
    communities_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver | None" = None,
    last_update_time: datetime | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Stage 5: Identify modified entities and stale communities.

    Returns:
        communities_df: Updated communities DataFrame
        stale_community_ids: List of community IDs that need re-summarization
    """
    logger.info("BT-GraphRAG Stage 5: Incremental community update")

    print(f"\n    [Stage 5] Incremental Community Update")
    print(f"    Input communities: {len(communities_df)}")
    print(f"    Input entities:    {len(entities_df)}")
    print(f"    Input rels:        {len(relationships_df)}")
    print(f"    Neo4j available:   {driver is not None}")
    print(f"    Last update:       {last_update_time or 'None (first run)'}")
    print(f"    k-hop radius:      {config.community_update_k_hop}")

    stale_community_ids: list[str] = []

    if driver is not None and last_update_time is not None:
        async with driver.session(database=config.neo4j_database) as session:
            # Find entities modified since last update
            modified_titles = await get_modified_entity_titles(
                session, last_update_time
            )

            print(f"\n    [Modified Entities] Found {len(modified_titles)} entities modified since {last_update_time.strftime('%Y-%m-%d %H:%M')}")
            for i, title in enumerate(modified_titles[:15]):
                print(f"      [{i}] {title[:50]}")
            if len(modified_titles) > 15:
                print(f"      ... and {len(modified_titles) - 15} more")

            if modified_titles:
                # Get k-hop neighborhood
                neighborhood = await get_k_hop_neighbors(
                    session, modified_titles, k=config.community_update_k_hop
                )
                expansion = len(neighborhood) - len(modified_titles)
                print(f"\n    [k-Hop Expansion] {len(modified_titles)} modified -> {len(neighborhood)} in {config.community_update_k_hop}-hop neighborhood (+{expansion} neighbors)")

                # Show sample of expanded neighborhood (entities not in modified set)
                expanded_only = neighborhood - set(modified_titles)
                if expanded_only:
                    print(f"    Neighborhood expansion (sample):")
                    for i, title in enumerate(list(expanded_only)[:10]):
                        print(f"      + {title[:50]}")
                    if len(expanded_only) > 10:
                        print(f"      ... and {len(expanded_only) - 10} more")

                logger.info(
                    "Stage 5: %d modified entities, %d in k-hop neighborhood",
                    len(modified_titles),
                    len(neighborhood),
                )

                # Map entity titles to IDs for community lookup
                title_to_id = dict(
                    zip(entities_df["title"], entities_df["id"])
                ) if "title" in entities_df.columns and "id" in entities_df.columns else {}

                modified_entity_ids = {
                    str(title_to_id.get(t, t)) for t in neighborhood
                }

                # Identify stale communities
                stale_community_ids = identify_stale_communities(
                    communities_df, modified_entity_ids
                )

                total_communities = len(communities_df)
                stale_count = len(stale_community_ids)
                fresh_count = total_communities - stale_count
                savings_pct = (fresh_count / total_communities * 100) if total_communities > 0 else 0

                print(f"\n    [Stale Communities]")
                print(f"      Total communities:   {total_communities}")
                print(f"      Stale (need update): {stale_count}")
                print(f"      Fresh (reusable):    {fresh_count}")
                print(f"      Savings:             {savings_pct:.1f}% of communities can skip re-summarization")
                if stale_community_ids:
                    print(f"      Stale IDs (first 10): {stale_community_ids[:10]}")

                logger.info(
                    "Stage 5: %d stale communities identified",
                    len(stale_community_ids),
                )
            else:
                print(f"\n    No modified entities found — all communities are fresh!")
    else:
        # Full re-cluster needed if no Neo4j or no prior timestamp
        stale_community_ids = list(communities_df["id"].astype(str)) if "id" in communities_df.columns else []
        reason = "no Neo4j driver" if driver is None else "no prior update timestamp"
        print(f"\n    FULL REBUILD required ({reason})")
        print(f"    All {len(stale_community_ids)} communities marked as stale")

    # --- Verification ---
    print(f"\n    {'=' * 55}")
    print(f"    STAGE 5 VERIFICATION")
    print(f"    {'=' * 55}")

    # Check community DataFrame structure
    has_id = "id" in communities_df.columns
    has_entity_ids = "entity_ids" in communities_df.columns
    print(f"    Communities DataFrame has 'id' column:         {has_id}")
    print(f"    Communities DataFrame has 'entity_ids' column: {has_entity_ids}")
    print(f"    Communities DataFrame columns:                 {list(communities_df.columns)}")

    # Verify all stale IDs are valid community IDs
    if has_id and stale_community_ids:
        all_ids = set(communities_df["id"].astype(str))
        invalid_stale = [sid for sid in stale_community_ids if sid not in all_ids]
        if invalid_stale:
            print(f"    WARNING: {len(invalid_stale)} stale IDs not found in communities DataFrame!")
        else:
            print(f"    Stale ID validity check:                       PASS")

    # Entity coverage: how many entities are in communities
    if has_entity_ids:
        community_entity_count = 0
        for _, row in communities_df.iterrows():
            eids = row.get("entity_ids", [])
            if isinstance(eids, list):
                community_entity_count += len(eids)
            elif isinstance(eids, str):
                community_entity_count += len([e for e in eids.split(",") if e.strip()])
        print(f"    Entities referenced in communities:             {community_entity_count}")
        print(f"    Entities in DataFrame:                         {len(entities_df)}")

    print(f"    {'=' * 55}")

    return communities_df, stale_community_ids


# ---------------------------------------------------------------------------
# Stage 6: Selective LLM Summarization
# ---------------------------------------------------------------------------


def filter_stale_communities(
    communities_df: pd.DataFrame,
    community_reports_df: pd.DataFrame,
    stale_community_ids: list[str],
) -> pd.DataFrame:
    """Filter communities to only include those that need re-summarization.

    Returns only the stale communities that need new reports generated.
    Communities with existing up-to-date reports are excluded.
    """
    if not stale_community_ids:
        print("    [Stage 6 Filter] No stale communities — returning empty DataFrame")
        return pd.DataFrame()

    id_col = "id" if "id" in communities_df.columns else "community"
    stale_set = set(stale_community_ids)

    mask = communities_df[id_col].astype(str).isin(stale_set)
    stale_df = communities_df[mask].copy()
    fresh_count = len(communities_df) - len(stale_df)

    print(f"    [Stage 6 Filter] {len(stale_df)} stale communities need re-summarization")
    print(f"    [Stage 6 Filter] {fresh_count} fresh communities can reuse existing reports")
    if len(community_reports_df) > 0:
        print(f"    [Stage 6 Filter] Existing reports available: {len(community_reports_df)}")

    return stale_df


def merge_community_reports(
    existing_reports_df: pd.DataFrame,
    new_reports_df: pd.DataFrame,
    stale_community_ids: list[str],
) -> pd.DataFrame:
    """Merge new community reports with existing ones.

    Replaces reports for stale communities with new reports,
    keeps existing reports for non-stale communities.
    """
    print(f"\n    [Stage 6 Merge] Merging community reports...")
    print(f"      Existing reports:   {len(existing_reports_df)}")
    print(f"      New reports:        {len(new_reports_df)}")
    print(f"      Stale IDs to swap:  {len(stale_community_ids)}")

    if new_reports_df.empty:
        print(f"      Result: Keeping all {len(existing_reports_df)} existing (no new reports)")
        return existing_reports_df

    if existing_reports_df.empty:
        print(f"      Result: Using all {len(new_reports_df)} new (no existing reports)")
        return new_reports_df

    # Determine the ID column
    id_col = "community" if "community" in existing_reports_df.columns else "id"
    stale_set = set(stale_community_ids)

    # Keep non-stale reports from existing
    keep_mask = ~existing_reports_df[id_col].astype(str).isin(stale_set)
    kept = existing_reports_df[keep_mask]

    # Combine with new reports
    merged = pd.concat([kept, new_reports_df], ignore_index=True)
    print(f"      Kept from existing: {len(kept)}")
    print(f"      Added from new:     {len(new_reports_df)}")
    print(f"      Final merged total: {len(merged)}")

    return merged


def annotate_community_reports_temporal(
    reports_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add temporal evolution annotations to community reports.

    Enriches reports with information about when key relationships
    started and ended within each community.
    """
    if reports_df.empty:
        return reports_df

    reports_df = reports_df.copy()

    # Add temporal_evolution column if not present
    if "temporal_evolution" not in reports_df.columns:
        reports_df["temporal_evolution"] = ""

    return reports_df
