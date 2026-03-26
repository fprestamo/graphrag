# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Cross-Graph Relationship Resolution (CGRR).

Resolves newly extracted relationship types against the canonical relation
types already present in the graph, preventing Relationship Aliasing — the
counterpart to Entity Aliasing that CGER addresses.

Without CGRR, ETCDR's conflict queries filter on `relation_type` and will
miss conflicts across alias boundaries (e.g., "IS_CEO_OF" vs "LEADS").

Section 2.5.1 of the BT-GraphRAG design document.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import pandas as pd

from graphrag.bt_graphrag.entity_resolution.scorers import (
    RelationshipScorer,
    compute_relationship_composite_score as compute_relationship_score,
)
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from neo4j import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Verification for Hard Cases
# ---------------------------------------------------------------------------

CGRR_VERIFICATION_PROMPT = """You are a knowledge graph expert. Determine whether the following two relationship types refer to the same real-world predicate (the same kind of relationship between entities).

Relationship A:
- Relation Type: {type_a}
- Description: {desc_a}
- Example: ({source_a}) -> ({target_a})

Relationship B:
- Relation Type: {type_b}
- Description: {desc_b}
- Example: ({source_b}) -> ({target_b})

Do these two relation types represent the same predicate? Answer ONLY with one of:
- SAME: They describe the same kind of relationship (e.g., "leads" and "is CEO of")
- DIFFERENT: They describe fundamentally different relationships (e.g., "works for" and "works with")
- UNCERTAIN: Cannot determine with available information

Answer:"""


async def llm_verify_relationship_match(
    candidate: dict[str, str],
    existing: dict[str, str],
    model: "LLMCompletion",
) -> str:
    """Use LLM to verify whether two relation types are the same predicate.

    Returns "SAME", "DIFFERENT", or "UNCERTAIN".
    """
    from graphrag_llm.utils import CompletionMessagesBuilder

    prompt = CGRR_VERIFICATION_PROMPT.format(
        type_a=candidate.get("relation_type", ""),
        desc_a=candidate.get("description", "")[:500],
        source_a=candidate.get("source", ""),
        target_a=candidate.get("target", ""),
        type_b=existing.get("relation_type", ""),
        desc_b=existing.get("description", "")[:500],
        source_b=existing.get("source", ""),
        target_b=existing.get("target", ""),
    )

    messages = CompletionMessagesBuilder().add_user_message(prompt).build()
    response = await model.completion_async(messages=messages)
    answer = response.content.strip().upper()

    if "SAME" in answer:
        return "SAME"
    if "DIFFERENT" in answer:
        return "DIFFERENT"
    return "UNCERTAIN"


# ---------------------------------------------------------------------------
# Neo4j: Fetch existing relation types
# ---------------------------------------------------------------------------


async def get_existing_relation_types(
    session: "AsyncSession",
) -> list[dict[str, str]]:
    """Query Neo4j for all distinct relation types with a sample edge for each.

    Returns a list of dicts with keys: relation_type, description, source, target.
    """
    query = """
    MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
    WHERE r.t_tx_end = $infinity AND r.t_valid_end = $infinity
    WITH r.relation_type AS rel_type,
         collect({desc: r.description, source: s.title, target: t.title})[0] AS sample,
         count(*) AS edge_count
    RETURN rel_type, sample.desc AS description,
           sample.source AS source, sample.target AS target,
           edge_count
    ORDER BY edge_count DESC
    """
    from graphrag.bt_graphrag.models.temporal_types import INFINITY_ISO

    result = await session.run(query, infinity=INFINITY_ISO)
    records = []
    async for record in result:
        records.append({
            "relation_type": record["rel_type"] or "",
            "description": record["description"] or "",
            "source": record["source"] or "",
            "target": record["target"] or "",
            "edge_count": record["edge_count"],
        })
    return records


async def get_existing_relations_for_entities(
    session: "AsyncSession",
    entity_titles: list[str],
) -> list[dict[str, Any]]:
    """Query Neo4j for relation types that share at least one entity with the batch.

    Only returns relation types where at least one endpoint (source or target)
    is in entity_titles.  Each result includes the entity sets so that
    per-candidate-type overlap can be checked in Python.

    Returns a list of dicts with keys:
        relation_type, description, source, target (sample edge),
        all_sources, all_targets (for overlap checking), edge_count.
    """
    if not entity_titles:
        return []

    query = """
    MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
    WHERE r.t_tx_end = $infinity AND r.t_valid_end = $infinity
      AND (s.title IN $entity_titles OR t.title IN $entity_titles)
    WITH r.relation_type AS rel_type,
         collect(DISTINCT s.title) AS all_sources,
         collect(DISTINCT t.title) AS all_targets,
         collect({desc: r.description, source: s.title, target: t.title})[0] AS sample,
         count(*) AS edge_count
    RETURN rel_type, sample.desc AS description,
           sample.source AS source, sample.target AS target,
           all_sources, all_targets, edge_count
    ORDER BY edge_count DESC
    """
    from graphrag.bt_graphrag.models.temporal_types import INFINITY_ISO

    result = await session.run(
        query, entity_titles=entity_titles, infinity=INFINITY_ISO,
    )
    records = []
    async for record in result:
        records.append({
            "relation_type": record["rel_type"] or "",
            "description": record["description"] or "",
            "source": record["source"] or "",
            "target": record["target"] or "",
            "all_sources": record["all_sources"] or [],
            "all_targets": record["all_targets"] or [],
            "edge_count": record["edge_count"],
        })
    return records


# ---------------------------------------------------------------------------
# Main CGRR Pipeline
# ---------------------------------------------------------------------------


async def resolve_relationships(
    relationships_df: pd.DataFrame,
    config: BTGraphRAGConfig,
    session: "AsyncSession",
    model: "LLMCompletion | None" = None,
    relationship_scorer: RelationshipScorer = compute_relationship_score,
) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    """Resolve candidate relationship types against existing graph predicates.

    For each unique relation_type in the incoming batch, scores it against
    all existing relation types in Neo4j. Types scoring above the merge
    threshold are normalized to the canonical form. Hard cases are sent
    to LLM verification.

    Returns:
        relationships_df: Updated DataFrame with normalized relation_types
        normalize_map: Dict mapping original relation types to canonical forms
        phase_b_log: List of Phase B (intra-batch) resolution log entries
    """
    normalize_map: dict[str, str] = {}
    phase_b_log: list[dict[str, Any]] = []

    if relationships_df.empty:
        print("    [CGRR] Skipping — no relationships to resolve")
        return relationships_df, normalize_map, phase_b_log

    # Ensure relation_type column exists
    if "relation_type" not in relationships_df.columns:
        print("    [CGRR] Skipping — no 'relation_type' column present")
        return relationships_df, normalize_map

    auto_merges = 0
    llm_merges = 0
    llm_rejections = 0
    below_threshold = 0
    exact_match_skip = 0

    # --- Phase A: Resolve against existing Neo4j relation types ---
    # Collect all entity titles from the incoming batch so we only
    # compare against existing relationships that share at least one entity.
    batch_entities = list(set(
        relationships_df["source"].dropna().unique().tolist()
        + relationships_df["target"].dropna().unique().tolist()
    ))

    existing_types = await get_existing_relations_for_entities(session, batch_entities)

    # Get unique candidate relation types from the batch
    candidate_types = relationships_df["relation_type"].unique().tolist()
    cand_type_counts = relationships_df["relation_type"].value_counts().to_dict()

    if not existing_types:
        print(f"    [CGRR] Phase A: Skipping — graph is empty (first run); "
              f"proceeding to intra-batch Phase B for {len(candidate_types)} types")
    else:
        print(f"\n    [CGRR] Phase A: Resolving {len(candidate_types)} candidate relation types "
              f"against {len(existing_types)} existing types")
        print(f"    [CGRR] Thresholds: auto_merge >= {config.cgrr_merge_threshold}, "
              f"LLM_zone = [{config.cgrr_llm_threshold_low}, {config.cgrr_merge_threshold})")
        print(f"    [CGRR] Weights: bm25={config.cgrr_bm25_weight}, "
              f"semantic={config.cgrr_semantic_weight}, "
              f"endpoint={config.cgrr_endpoint_weight}")

        # Show existing canonical types in graph
        print(f"\n    [CGRR] Existing canonical relation types in graph:")
        for i, et in enumerate(existing_types[:20]):
            print(f"      [{i:2d}] {et['relation_type'][:35]:35s}  ({et['edge_count']:3d} edges)  "
                  f"e.g. ({et['source'][:18]}) -> ({et['target'][:18]})")
        if len(existing_types) > 20:
            print(f"      ... and {len(existing_types) - 20} more")

        # Show incoming candidate types
        print(f"\n    [CGRR] Incoming candidate relation types ({len(candidate_types)}):")
        for i, ct in enumerate(candidate_types):
            edge_count = cand_type_counts.get(ct, 0)
            sample_row = relationships_df[relationships_df["relation_type"] == ct].iloc[0]
            s_src = str(sample_row.get("source", "?"))[:18]
            s_tgt = str(sample_row.get("target", "?"))[:18]
            s_desc = str(sample_row.get("description", ""))[:35]
            print(f"      [{i:2d}] {ct[:35]:35s}  ({edge_count:3d} edges)  "
                  f"e.g. ({s_src}) -> ({s_tgt}): '{s_desc}'")

        print(f"\n    [CGRR] Processing each candidate type:")
        print(f"    {'-' * 65}")

        for cand_idx, cand_type in enumerate(candidate_types):
            # Check if this type already exists exactly in the graph
            exact_exists = any(et["relation_type"] == cand_type for et in existing_types)
            if exact_exists:
                exact_match_skip += 1
                print(f"\n    [{cand_idx}] '{cand_type[:35]}' — EXACT MATCH in graph (skip)")
                continue

            cand_sample = relationships_df[relationships_df["relation_type"] == cand_type].iloc[0]
            cand_desc = str(cand_sample.get("description", ""))
            cand_source = str(cand_sample.get("source", ""))
            cand_target = str(cand_sample.get("target", ""))
            cand_edge_count = cand_type_counts.get(cand_type, 0)

            cand_rows = relationships_df[relationships_df["relation_type"] == cand_type]
            cand_entities = set(
                cand_rows["source"].dropna().tolist()
                + cand_rows["target"].dropna().tolist()
            )

            print(f"\n    [{cand_idx}] Candidate: '{cand_type[:40]}' ({cand_edge_count} edges)")
            print(f"         Sample: ({cand_source[:20]}) -> ({cand_target[:20]})")
            print(f"         Desc: '{cand_desc[:60]}'")

            scored_matches: list[tuple[float, dict[str, float], dict[str, str]]] = []
            skipped_no_overlap = 0

            for existing in existing_types:
                if cand_type == existing["relation_type"]:
                    continue
                existing_entities_set = set(
                    existing.get("all_sources", [existing.get("source", "")])
                    + existing.get("all_targets", [existing.get("target", "")])
                )
                if not cand_entities.intersection(existing_entities_set):
                    skipped_no_overlap += 1
                    continue
                score, breakdown = relationship_scorer(
                    cand_type, cand_desc, cand_source, cand_target,
                    existing["relation_type"], existing["description"],
                    existing["source"], existing["target"],
                    config,
                )
                scored_matches.append((score, breakdown, existing))

            scored_matches.sort(key=lambda x: x[0], reverse=True)

            if skipped_no_overlap > 0:
                print(f"         Skipped {skipped_no_overlap} existing types (no shared entities)")
            print(f"         Top comparisons ({len(scored_matches)} with entity overlap):")
            for rank, (score, breakdown, match) in enumerate(scored_matches[:5]):
                zone = ""
                if score >= config.cgrr_merge_threshold:
                    zone = " << AUTO-MERGE"
                elif score >= config.cgrr_llm_threshold_low:
                    zone = " << LLM-ZONE"
                print(f"           #{rank+1} score={score:.4f}  '{match['relation_type'][:30]:30s}' "
                      f"bm25={breakdown['bm25_type']:.3f} sem={breakdown['semantic_desc']:.3f} "
                      f"endpt={breakdown['endpoint_match']:.3f}{zone}")

            if not scored_matches:
                print(f"           (no comparisons available)")
                continue

            best_score, best_breakdown, best_match_record = scored_matches[0]
            match_type = best_match_record["relation_type"]

            if best_score >= config.cgrr_merge_threshold:
                normalize_map[cand_type] = match_type
                auto_merges += 1
                print(f"         DECISION: AUTO-NORMALIZE -> '{match_type[:35]}'")
                print(f"           Weighted: w_bm25={best_breakdown['w_bm25']:.3f} "
                      f"w_sem={best_breakdown['w_semantic']:.3f} "
                      f"w_endpt={best_breakdown['w_endpoint']:.3f}")
                logger.info(
                    "CGRR: Auto-normalizing '%s' -> '%s' (score=%.3f)",
                    cand_type, match_type, best_score,
                )
            elif best_score >= config.cgrr_llm_threshold_low and model is not None:
                print(f"         DECISION: LLM VERIFICATION needed (score={best_score:.4f})")
                print(f"           Comparing: '{cand_type[:30]}' vs '{match_type[:30]}'")
                verdict = await llm_verify_relationship_match(
                    candidate={
                        "relation_type": cand_type,
                        "description": cand_desc,
                        "source": cand_source,
                        "target": cand_target,
                    },
                    existing={
                        "relation_type": match_type,
                        "description": best_match_record.get("description", ""),
                        "source": best_match_record.get("source", ""),
                        "target": best_match_record.get("target", ""),
                    },
                    model=model,
                )
                if verdict == "SAME":
                    normalize_map[cand_type] = match_type
                    llm_merges += 1
                    print(f"           LLM verdict: SAME -> NORMALIZED to '{match_type[:30]}'")
                    logger.info(
                        "CGRR: LLM-confirmed '%s' -> '%s' (score=%.3f)",
                        cand_type, match_type, best_score,
                    )
                else:
                    llm_rejections += 1
                    print(f"           LLM verdict: {verdict} -> KEPT SEPARATE")
                    logger.info(
                        "CGRR: LLM rejected '%s' vs '%s' (score=%.3f, verdict=%s)",
                        cand_type, match_type, best_score, verdict,
                    )
            else:
                below_threshold += 1
                print(f"         DECISION: BELOW THRESHOLD (best score={best_score:.4f} < "
                      f"{config.cgrr_llm_threshold_low})")
                print(f"           Keeping '{cand_type[:35]}' as separate type")

    # Apply Phase A normalization map
    original_types = relationships_df["relation_type"].copy()
    if normalize_map:
        relationships_df = relationships_df.copy()
        relationships_df["relation_type"] = relationships_df["relation_type"].map(
            lambda rt: normalize_map.get(rt, rt)
        )

    # --- Phase B: Intra-batch relation type resolution (new vs new) ---
    # Only process types not already normalized to an existing canonical type.
    remaining_types = [
        rt for rt in relationships_df["relation_type"].unique()
        if rt not in normalize_map.values()
    ]

    intra_auto_normalizes = 0
    intra_llm_normalizes = 0
    intra_rejections_b = 0

    if len(remaining_types) > 1:
        print(f"\n    [CGRR] Phase B: Intra-batch resolution ({len(remaining_types)} remaining types)")
        intra_normalize: dict[str, str] = {}
        canonical_types: list[dict[str, str]] = []

        for rt in remaining_types:
            sample = relationships_df[relationships_df["relation_type"] == rt].iloc[0]
            cand_desc = str(sample.get("description", ""))
            cand_src = str(sample.get("source", ""))
            cand_tgt = str(sample.get("target", ""))

            if not canonical_types:
                canonical_types.append({
                    "relation_type": rt, "description": cand_desc,
                    "source": cand_src, "target": cand_tgt,
                })
                continue

            best_score = 0.0
            best_breakdown: dict[str, float] = {}
            best_canon: dict[str, str] | None = None

            for canon in canonical_types:
                score, breakdown = relationship_scorer(
                    rt, cand_desc, cand_src, cand_tgt,
                    canon["relation_type"], canon["description"],
                    canon["source"], canon["target"],
                    config,
                )
                if score > best_score:
                    best_score = score
                    best_breakdown = breakdown
                    best_canon = canon

            match_type = best_canon["relation_type"] if best_canon else ""

            if best_canon is not None and best_score >= config.cgrr_merge_threshold:
                intra_normalize[rt] = match_type
                intra_auto_normalizes += 1
                phase_b_log.append({
                    "phase": "B_intra_batch",
                    "relation_type": rt,
                    "edge_count": cand_type_counts.get(rt, 0),
                    "best_match": match_type,
                    "best_score": round(best_score, 4),
                    "decision": "AUTO_NORMALIZE",
                    "top_comparisons": [],
                })
                print(f"    [INTRA] AUTO-NORMALIZE: '{rt[:30]}' -> '{match_type[:30]}'  "
                      f"score={best_score:.4f}  "
                      f"bm25={best_breakdown.get('bm25_type', 0):.3f}  "
                      f"sem={best_breakdown.get('semantic_desc', 0):.3f}  "
                      f"endpt={best_breakdown.get('endpoint_match', 0):.3f}")
                logger.info(
                    "CGRR: Intra-batch normalize '%s' -> '%s' (score=%.3f)",
                    rt, match_type, best_score,
                )
            elif (best_canon is not None
                  and best_score >= config.cgrr_llm_threshold_low
                  and model is not None):
                print(f"    [INTRA] LLM-ZONE: '{rt[:30]}' vs '{match_type[:30]}'  "
                      f"score={best_score:.4f}")
                verdict = await llm_verify_relationship_match(
                    candidate={"relation_type": rt, "description": cand_desc,
                               "source": cand_src, "target": cand_tgt},
                    existing=best_canon,
                    model=model,
                )
                if verdict == "SAME":
                    intra_normalize[rt] = match_type
                    intra_llm_normalizes += 1
                    phase_b_log.append({
                        "phase": "B_intra_batch",
                        "relation_type": rt,
                        "edge_count": cand_type_counts.get(rt, 0),
                        "best_match": match_type,
                        "best_score": round(best_score, 4),
                        "decision": "LLM_NORMALIZE",
                        "top_comparisons": [],
                    })
                    print(f"           LLM verdict: SAME -> NORMALIZED")
                    logger.info(
                        "CGRR: Intra-batch LLM normalize '%s' -> '%s' (score=%.3f)",
                        rt, match_type, best_score,
                    )
                else:
                    intra_rejections_b += 1
                    phase_b_log.append({
                        "phase": "B_intra_batch",
                        "relation_type": rt,
                        "edge_count": cand_type_counts.get(rt, 0),
                        "best_match": match_type,
                        "best_score": round(best_score, 4),
                        "decision": "LLM_ZONE_REJECTED",
                        "top_comparisons": [],
                    })
                    print(f"           LLM verdict: {verdict} -> KEPT SEPARATE")
                    canonical_types.append({
                        "relation_type": rt, "description": cand_desc,
                        "source": cand_src, "target": cand_tgt,
                    })
            else:
                if best_score > 0.3:
                    phase_b_log.append({
                        "phase": "B_intra_batch",
                        "relation_type": rt,
                        "edge_count": cand_type_counts.get(rt, 0),
                        "best_match": match_type,
                        "best_score": round(best_score, 4),
                        "decision": "BELOW_THRESHOLD",
                        "top_comparisons": [],
                    })
                    print(f"    [INTRA] BELOW: '{rt[:30]}' best='{match_type[:30]}'  "
                          f"score={best_score:.4f}")
                canonical_types.append({
                    "relation_type": rt, "description": cand_desc,
                    "source": cand_src, "target": cand_tgt,
                })

        if intra_normalize:
            relationships_df = relationships_df.copy()
            relationships_df["relation_type"] = relationships_df["relation_type"].map(
                lambda rt: intra_normalize.get(rt, rt)
            )
            normalize_map.update(intra_normalize)
        else:
            print(f"    [CGRR] Intra-batch: no additional normalizations found")
    else:
        print(f"\n    [CGRR] Phase B: Intra-batch skipped (0-1 remaining types)")

    # --- Verification Summary ---
    edges_affected = sum(
        int((original_types == orig_type).sum())
        for orig_type in normalize_map
    )

    print(f"\n    {'=' * 65}")
    print(f"    CGRR VERIFICATION SUMMARY")
    print(f"    {'=' * 65}")
    print(f"    Candidate relation types:        {len(candidate_types)}")
    print(f"    Existing types in graph:         {len(existing_types)}")
    print(f"    Exact match (already canonical): {exact_match_skip}")
    print(f"    Phase A auto-normalized:         {auto_merges}")
    print(f"    Phase A LLM-confirmed:           {llm_merges}")
    print(f"    Phase A LLM-rejected:            {llm_rejections}")
    print(f"    Phase A below threshold:         {below_threshold}")
    print(f"    Phase B intra auto-normalized:   {intra_auto_normalizes}")
    print(f"    Phase B intra LLM-confirmed:     {intra_llm_normalizes}")
    print(f"    Phase B intra LLM-rejected:      {intra_rejections_b}")
    print(f"    Total normalizations (A+B):      {len(normalize_map)}")
    print(f"    Total edges affected:            {edges_affected}")

    if normalize_map:
        print(f"\n    Normalization Map:")
        for orig, canon in normalize_map.items():
            n_edges = int((original_types == orig).sum())
            print(f"      '{orig}' -> '{canon}'  ({n_edges} edges)")

        # Verify cardinality inheritance
        print(f"\n    Cardinality Inheritance Check:")
        for orig, canon in normalize_map.items():
            orig_card = config.get_cardinality(orig)
            canon_card = config.get_cardinality(canon)
            if orig_card != canon_card:
                print(f"      CHANGED: '{orig[:25]}' {orig_card} -> '{canon[:25]}' {canon_card}")
            else:
                print(f"      SAME:    '{orig[:25]}' {orig_card} == '{canon[:25]}' {canon_card}")

    # Before/After relation type table
    final_types_list = sorted(relationships_df["relation_type"].unique())
    original_types_list = sorted(original_types.unique())
    print(f"\n    Before/After Relation Types:")
    print(f"      BEFORE ({len(original_types_list)} unique):")
    for rt in original_types_list:
        count = int((original_types == rt).sum())
        marker = " -> NORMALIZED" if rt in normalize_map else ""
        print(f"        {rt[:40]:40s}  ({count:3d} edges){marker}")
    print(f"      AFTER ({len(final_types_list)} unique):")
    for rt in final_types_list:
        count = int((relationships_df["relation_type"] == rt).sum())
        print(f"        {rt[:40]:40s}  ({count:3d} edges)")

    print(f"    {'=' * 65}")

    return relationships_df, normalize_map, phase_b_log


def apply_normalize_map_to_cardinality(
    relationships_df: pd.DataFrame,
    normalize_map: dict[str, str],
    config: BTGraphRAGConfig,
) -> pd.DataFrame:
    """Update cardinality column based on normalized relation types.

    When a relation type is normalized, it should inherit the canonical
    type's cardinality classification.
    """
    if not normalize_map or relationships_df.empty:
        return relationships_df

    if "cardinality" not in relationships_df.columns:
        return relationships_df

    relationships_df = relationships_df.copy()
    # Re-derive cardinality from the (now-normalized) relation_type
    relationships_df["cardinality"] = relationships_df["relation_type"].apply(
        lambda rt: config.get_cardinality(rt)
    )

    return relationships_df
