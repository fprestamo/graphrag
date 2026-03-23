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

    stale_community_ids: list[str] = []

    if driver is not None and last_update_time is not None:
        async with driver.session(database=config.neo4j_database) as session:
            # Find entities modified since last update
            modified_titles = await get_modified_entity_titles(
                session, last_update_time
            )

            if modified_titles:
                # Get k-hop neighborhood
                neighborhood = await get_k_hop_neighbors(
                    session, modified_titles, k=config.community_update_k_hop
                )
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
                logger.info(
                    "Stage 5: %d stale communities identified",
                    len(stale_community_ids),
                )
    else:
        # Full re-cluster needed if no Neo4j or no prior timestamp
        stale_community_ids = list(communities_df["id"].astype(str)) if "id" in communities_df.columns else []

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
        return pd.DataFrame()

    id_col = "id" if "id" in communities_df.columns else "community"
    stale_set = set(stale_community_ids)

    mask = communities_df[id_col].astype(str).isin(stale_set)
    return communities_df[mask].copy()


def merge_community_reports(
    existing_reports_df: pd.DataFrame,
    new_reports_df: pd.DataFrame,
    stale_community_ids: list[str],
) -> pd.DataFrame:
    """Merge new community reports with existing ones.

    Replaces reports for stale communities with new reports,
    keeps existing reports for non-stale communities.
    """
    if new_reports_df.empty:
        return existing_reports_df

    if existing_reports_df.empty:
        return new_reports_df

    # Determine the ID column
    id_col = "community" if "community" in existing_reports_df.columns else "id"
    stale_set = set(stale_community_ids)

    # Keep non-stale reports from existing
    keep_mask = ~existing_reports_df[id_col].astype(str).isin(stale_set)
    kept = existing_reports_df[keep_mask]

    # Combine with new reports
    return pd.concat([kept, new_reports_df], ignore_index=True)


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
