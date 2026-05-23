# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Cross-Graph Relationship Resolution (CGRR).

Resolves newly extracted relationship types against the canonical
relation types already present in the graph, preventing Relationship
Aliasing — the counterpart to Entity Aliasing that CGER addresses.

The procedure mirrors the simplified CGER design:

1. **Cosine pre-filter.** For each unique relation type in the incoming
   batch, the top-K existing relation types by description-embedding
   cosine similarity are retrieved (Neo4j vector index or in-memory
   fallback). The best-scoring candidate whose cosine exceeds
   ``cgrr_cosine_threshold`` triggers the next step.
2. **LLM verification.** The LLM receives both relation types' names,
   descriptions and a sample ``(source) -> (target)`` endpoint pair,
   and must answer one of:

   * ``SAME`` — the two strings denote the same predicate; normalise
     the candidate to the existing canonical form.
   * ``DIFFERENT`` — they denote different predicates; keep them apart.

   Relation types have no active period of their own, so the temporal
   verdict that CGER uses does not apply here: the LLM only decides
   whether the two labels refer to the same predicate.

Without CGRR, ETCDR's conflict queries filter on ``relation_type`` and
will miss conflicts across alias boundaries (e.g. ``IS_CEO_OF`` vs
``LEADS``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from graphrag.bt_graphrag.entity_resolution.scorers import (
    RelationshipScorer,
    description_cosine_relationship_scorer,
)
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from neo4j import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosine similarity helper for top-K pre-filtering
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _top_k_by_embedding(
    query_embedding: list[float] | None,
    existing_types: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Return the top-K existing relation types by description-embedding cosine.

    Falls back to all existing types if the query embedding is missing.
    """
    if query_embedding is None or not existing_types:
        return existing_types

    scored = []
    for et in existing_types:
        emb = et.get("description_embedding")
        if emb is not None:
            sim = _cosine_similarity(query_embedding, emb)
        else:
            sim = 0.0
        scored.append((sim, et))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [et for _, et in scored[:top_k]]


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------

CGRR_VERIFICATION_PROMPT = """You are a knowledge-graph expert. Two candidate relationship types are presented below, each with a short description and a sample endpoint pair. Decide whether they denote the **same predicate**.

Your default answer is DIFFERENT. Only answer SAME when both strings clearly describe the same kind of relationship between the same kinds of entities (e.g. "IS_CEO_OF" and "LEADS" when used between a person and a company; "WORKS_FOR" and "EMPLOYED_BY"; "BORN_IN" and "PLACE_OF_BIRTH"). When the predicates differ in scope, direction, or the kind of entities they connect, answer DIFFERENT.

Relationship A:
- Relation Type: {type_a}
- Description: {desc_a}
- Example: ({source_a}) -> ({target_a})

Relationship B:
- Relation Type: {type_b}
- Description: {desc_b}
- Example: ({source_b}) -> ({target_b})

Answer with EXACTLY ONE of the following tokens and nothing else:
- SAME
- DIFFERENT

Answer:"""


async def llm_verify_relationship_match(
    candidate: dict[str, str],
    existing: dict[str, str],
    model: "LLMCompletion",
) -> str:
    """Use the LLM to decide whether two relation types denote the same predicate.

    Returns ``"SAME"`` or ``"DIFFERENT"``. Any unrecognised response is
    treated as ``"DIFFERENT"`` (conservative default — never normalise
    on ambiguous evidence).
    """
    from graphrag_llm.utils import CompletionMessagesBuilder

    prompt = CGRR_VERIFICATION_PROMPT.format(
        type_a=candidate.get("relation_type", ""),
        desc_a=(candidate.get("description", "") or "")[:500],
        source_a=candidate.get("source", ""),
        target_a=candidate.get("target", ""),
        type_b=existing.get("relation_type", ""),
        desc_b=(existing.get("description", "") or "")[:500],
        source_b=existing.get("source", ""),
        target_b=existing.get("target", ""),
    )

    messages = CompletionMessagesBuilder().add_user_message(prompt).build()
    response = await model.completion_async(messages=messages)
    answer = response.content.strip().upper()

    valid = {"SAME", "DIFFERENT"}
    first_verdict = next(
        (tok for tok in answer.replace(":", " ").replace(",", " ").split()
         if tok in valid),
        "",
    )
    return first_verdict if first_verdict in valid else "DIFFERENT"


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
    relationship_scorer: RelationshipScorer = description_cosine_relationship_scorer,
) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    """Resolve candidate relationship types against existing graph predicates.

    For each unique relation_type in the incoming batch, ranks existing
    relation types by description-embedding cosine similarity, picks
    the best same-endpoint-sharing candidate, and — if its cosine
    exceeds ``config.cgrr_cosine_threshold`` — defers the merge
    decision to the LLM. The LLM answers ``SAME`` (normalise to the
    canonical type) or ``DIFFERENT`` (keep separate).

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

    if "relation_type" not in relationships_df.columns:
        print("    [CGRR] Skipping — no 'relation_type' column present")
        return relationships_df, normalize_map, phase_b_log

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

    from graphrag.bt_graphrag.neo4j_store import get_existing_relation_types_with_embeddings

    existing_types = await get_existing_relation_types_with_embeddings(
        session, entity_titles=batch_entities,
    )

    candidate_types = relationships_df["relation_type"].unique().tolist()
    cand_type_counts = relationships_df["relation_type"].value_counts().to_dict()

    cgrr_top_k = getattr(config, "cgrr_candidate_top_k", 10)

    if not existing_types:
        print(f"    [CGRR] Phase A: Skipping — graph is empty (first run); "
              f"proceeding to intra-batch Phase B for {len(candidate_types)} types")
    else:
        print(f"\n    [CGRR] Phase A: Resolving {len(candidate_types)} candidate relation types "
              f"against {len(existing_types)} existing types (top-K={cgrr_top_k})")
        print(f"    [CGRR] LLM trigger: cosine >= {config.cgrr_cosine_threshold}")

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

            cand_emb = (
                cand_sample.get("description_embedding")
                if "description_embedding" in cand_sample.index else None
            )

            cand_rows = relationships_df[relationships_df["relation_type"] == cand_type]
            cand_entities = set(
                cand_rows["source"].dropna().tolist()
                + cand_rows["target"].dropna().tolist()
            )

            print(f"\n    [{cand_idx}] Candidate: '{cand_type[:40]}' ({cand_edge_count} edges)")
            print(f"         Sample: ({cand_source[:20]}) -> ({cand_target[:20]})")
            print(f"         Desc: '{cand_desc[:60]}'")

            # Top-K pre-filter: try Neo4j vector index first, fall back to in-memory cosine
            narrowed: list[dict[str, Any]] = []
            if cand_emb is not None:
                from graphrag.bt_graphrag.neo4j_store import vector_search_relationships
                try:
                    vec_results = await vector_search_relationships(
                        session, cand_emb, top_k=cgrr_top_k,
                    )
                except Exception:
                    vec_results = []
                if vec_results:
                    seen_types: set[str] = set()
                    for vr in vec_results:
                        rt = vr.get("relation_type", "")
                        if rt and rt not in seen_types:
                            seen_types.add(rt)
                            match_rec = next(
                                (et for et in existing_types if et["relation_type"] == rt),
                                None,
                            )
                            if match_rec:
                                narrowed.append(match_rec)
                            else:
                                narrowed.append({
                                    "relation_type": rt,
                                    "description": vr.get("description", ""),
                                    "source": vr.get("source", ""),
                                    "target": vr.get("target", ""),
                                    "all_sources": [vr.get("source", "")],
                                    "all_targets": [vr.get("target", "")],
                                    "edge_count": 1,
                                    "description_embedding": vr.get("description_embedding"),
                                })
                    print(f"         Neo4j vector search: {len(narrowed)} types from top-{cgrr_top_k} edges")

            if not narrowed:
                narrowed = _top_k_by_embedding(cand_emb, existing_types, cgrr_top_k)
                if len(narrowed) < len(existing_types):
                    print(f"         In-memory top-K: {len(existing_types)} -> {len(narrowed)} candidates (k={cgrr_top_k})")

            cand_record = {
                "relation_type": cand_type,
                "description": cand_desc,
                "source": cand_source,
                "target": cand_target,
                "description_embedding": cand_emb,
            }

            scored_matches: list[tuple[float, dict[str, float], dict[str, Any]]] = []
            skipped_no_overlap = 0

            for existing in narrowed:
                if cand_type == existing["relation_type"]:
                    continue
                existing_entities_set = set(
                    existing.get("all_sources", [existing.get("source", "")])
                    + existing.get("all_targets", [existing.get("target", "")])
                )
                if not cand_entities.intersection(existing_entities_set):
                    skipped_no_overlap += 1
                    continue
                score, breakdown = relationship_scorer(cand_record, existing, config)
                scored_matches.append((score, breakdown, existing))

            scored_matches.sort(key=lambda x: x[0], reverse=True)

            if skipped_no_overlap > 0:
                print(f"         Skipped {skipped_no_overlap} existing types (no shared entities)")
            print(f"         Top candidates ({len(scored_matches)} with entity overlap):")
            for rank, (score, _bd, match) in enumerate(scored_matches[:5]):
                zone = " << LLM" if score >= config.cgrr_cosine_threshold else ""
                print(f"           #{rank+1} cosine={score:.4f}  "
                      f"'{match['relation_type'][:30]:30s}'{zone}")

            if not scored_matches:
                print(f"           (no comparisons available)")
                continue

            best_score, _best_bd, best_match_record = scored_matches[0]
            match_type = best_match_record["relation_type"]

            if best_score >= config.cgrr_cosine_threshold and model is not None:
                print(f"         LLM: '{cand_type[:30]}' vs '{match_type[:30]}'  cosine={best_score:.4f}")
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
                        "CGRR: LLM-confirmed '%s' -> '%s' (cosine=%.3f)",
                        cand_type, match_type, best_score,
                    )
                else:
                    llm_rejections += 1
                    print(f"           LLM verdict: DIFFERENT -> KEPT SEPARATE")
                    logger.info(
                        "CGRR: LLM rejected '%s' vs '%s' (cosine=%.3f)",
                        cand_type, match_type, best_score,
                    )
            else:
                below_threshold += 1
                print(f"         BELOW THRESHOLD: best cosine={best_score:.4f} < "
                      f"{config.cgrr_cosine_threshold} — keeping '{cand_type[:35]}' separate")

    # Apply Phase A normalization map
    original_types = relationships_df["relation_type"].copy()
    if normalize_map:
        relationships_df = relationships_df.copy()
        relationships_df["relation_type"] = relationships_df["relation_type"].map(
            lambda rt: normalize_map.get(rt, rt)
        )

    # --- Phase B: Intra-batch relation type resolution (new vs new) ---
    remaining_types = [
        rt for rt in relationships_df["relation_type"].unique()
        if rt not in normalize_map.values()
    ]

    intra_llm_normalizes = 0
    intra_rejections_b = 0

    if len(remaining_types) > 1:
        print(f"\n    [CGRR] Phase B: Intra-batch resolution ({len(remaining_types)} remaining types)")
        intra_normalize: dict[str, str] = {}
        canonical_types: list[dict[str, Any]] = []

        # Pre-compute entity sets per relation type for overlap checking
        _type_entities: dict[str, set[str]] = {}
        for rt in remaining_types:
            rows_rt = relationships_df[relationships_df["relation_type"] == rt]
            _type_entities[rt] = set(
                rows_rt["source"].dropna().tolist()
                + rows_rt["target"].dropna().tolist()
            )

        for rt in remaining_types:
            sample = relationships_df[relationships_df["relation_type"] == rt].iloc[0]
            cand_desc = str(sample.get("description", ""))
            cand_src = str(sample.get("source", ""))
            cand_tgt = str(sample.get("target", ""))
            cand_entities = _type_entities[rt]
            cand_emb = (
                sample.get("description_embedding")
                if "description_embedding" in sample.index else None
            )

            if not canonical_types:
                canonical_types.append({
                    "relation_type": rt,
                    "description": cand_desc,
                    "source": cand_src,
                    "target": cand_tgt,
                    "description_embedding": cand_emb,
                })
                continue

            cand_record = {
                "relation_type": rt,
                "description": cand_desc,
                "source": cand_src,
                "target": cand_tgt,
                "description_embedding": cand_emb,
            }

            best_score = 0.0
            best_canon: dict[str, Any] | None = None

            for canon in canonical_types:
                canon_entities = _type_entities.get(canon["relation_type"], set())
                if not cand_entities.intersection(canon_entities):
                    continue
                score, _ = relationship_scorer(cand_record, canon, config)
                if score > best_score:
                    best_score = score
                    best_canon = canon

            match_type = best_canon["relation_type"] if best_canon else ""

            if (best_canon is not None
                    and best_score >= config.cgrr_cosine_threshold
                    and model is not None):
                print(f"    [INTRA] LLM: '{rt[:30]}' vs '{match_type[:30]}'  "
                      f"cosine={best_score:.4f}")
                verdict = await llm_verify_relationship_match(
                    candidate={"relation_type": rt, "description": cand_desc,
                               "source": cand_src, "target": cand_tgt},
                    existing={"relation_type": match_type,
                              "description": best_canon.get("description", ""),
                              "source": best_canon.get("source", ""),
                              "target": best_canon.get("target", "")},
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
                        "CGRR: Intra-batch LLM normalize '%s' -> '%s' (cosine=%.3f)",
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
                        "decision": "LLM_DIFFERENT",
                        "top_comparisons": [],
                    })
                    print(f"           LLM verdict: DIFFERENT -> KEPT SEPARATE")
                    canonical_types.append({
                        "relation_type": rt,
                        "description": cand_desc,
                        "source": cand_src,
                        "target": cand_tgt,
                        "description_embedding": cand_emb,
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
                          f"cosine={best_score:.4f}")
                canonical_types.append({
                    "relation_type": rt,
                    "description": cand_desc,
                    "source": cand_src,
                    "target": cand_tgt,
                    "description_embedding": cand_emb,
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
    print(f"    Cosine LLM trigger:              {config.cgrr_cosine_threshold}")
    print(f"    Phase A LLM merges (SAME):       {llm_merges}")
    print(f"    Phase A LLM rejects (DIFFERENT): {llm_rejections}")
    print(f"    Phase A below threshold:         {below_threshold}")
    print(f"    Phase B intra LLM merges:        {intra_llm_normalizes}")
    print(f"    Phase B intra LLM rejects:       {intra_rejections_b}")
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
    relationships_df["cardinality"] = relationships_df["relation_type"].apply(
        lambda rt: config.get_cardinality(rt)
    )

    return relationships_df
