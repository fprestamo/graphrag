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
    from neo4j import AsyncDriver, AsyncSession

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
    """Return the top-K existing relation types by relation-type-embedding cosine.

    Falls back to all existing types if the query embedding is missing.
    """
    if query_embedding is None or not existing_types:
        return existing_types

    scored = []
    for et in existing_types:
        emb = et.get("relation_type_embedding")
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

CGRR_VERIFICATION_PROMPT = """You are a knowledge-graph expert. Two candidate relationship types are presented below, each with a short description and one sample endpoint pair. Decide whether they denote the **same predicate**.

Focus on the *predicate* — what kind of relationship the label expresses — NOT on the specific entities mentioned in the description or example. Each description and example is just one instance of the relation; different instances naturally mention different people, companies, places or dates. Two predicates are SAME only when, applied to any pair of entities, they would assert the exact same kind of fact.

**Default to DIFFERENT.** Only answer SAME when the two labels are interchangeable synonyms (tense/voice/word-order variants, or one is a strict paraphrase of the other). When in doubt — when the predicates are merely related, overlapping, or in the same general domain — answer DIFFERENT. Merging distinct predicates loses information; keeping near-synonyms separate is a small, recoverable cost.

Answer SAME only when both labels denote the exact same predicate. Examples of SAME:
- "WORKS_FOR" / "EMPLOYED_AT" / "EMPLOYED_BY" — tense/voice variants of the employment relation.
- "BORN_IN" / "PLACE_OF_BIRTH" — direct paraphrase of the birthplace relation.
- "MARRIED_TO" / "IS_MARRIED_TO" / "SPOUSE_OF" — interchangeable expressions of the spousal relation.
- "FOUNDED" / "ESTABLISHED" / "CO_FOUNDED" — variants of the founding/creation relation between a person and an organisation.

Answer DIFFERENT when the predicates differ in meaning, scope, direction, specificity, or the kind of fact they assert. Examples of DIFFERENT:
- "IS_CEO_OF" vs "FOUNDED" — both involve a person and a company, but one asserts leadership and the other asserts creation.
- "ACQUIRED" vs "MERGED_WITH" — both involve two companies, but acquisition is directional whereas merger is symmetric.
- "SUCCEEDED_BY" vs "IS_PRESIDENT_OF" — succession links two presidents whereas IS_PRESIDENT_OF links a president to an organisation.
- "OWNS" vs "SUBSIDIARY_OF" — opposite directions of ownership.
- "LOCATED_IN" vs "PART_OF" — geographic containment is not the same as structural/organisational membership.
- "BORN_IN" vs "LIVED_IN" — birthplace is a one-time fact; residence is an ongoing/repeatable fact.
- "WORKS_FOR" vs "FOUNDED" — both link a person to an organisation, but employment ≠ creation.
- "SERVES" vs "PROVIDES_ACCESS_TO" — overlapping but distinct: serving an area is broader than granting access.
- "PART_OF" vs "MEMBER_OF" — structural inclusion is not the same as membership/affiliation.
- "INFLUENCED" vs "TAUGHT" — teaching is one mechanism of influence, not a synonym for it.
- "DIED_IN" vs "BURIED_IN" — death location and burial location are distinct facts.

Do NOT answer DIFFERENT just because the two examples mention unrelated entities or domains. The examples are illustrative; the predicate is what matters.

Do NOT answer SAME just because the two predicates touch the same general topic (employment, family, geography, leadership). Same topic ≠ same predicate.

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
    driver: "AsyncDriver | None" = None,
    phase_b_top_k: int | None = None,
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

    # Manual short-circuit: skip CGRR entirely (no scoring, no temp DB writes).
    print("    [CGRR] Skipping — disabled by manual short-circuit at top of resolve_relationships")
    return relationships_df, normalize_map, phase_b_log

    if relationships_df.empty:
        print("    [CGRR] Skipping — no relationships to resolve")
        return relationships_df, normalize_map, phase_b_log

    if "relation_type" not in relationships_df.columns:
        print("    [CGRR] Skipping — no 'relation_type' column present")
        return relationships_df, normalize_map, phase_b_log

    # Drop rows with missing/empty relation_type so they don't poison
    # unique()/iloc[0] lookups downstream. Such rows can't be resolved
    # against canonical predicates anyway — they have no label to match.
    null_mask = relationships_df["relation_type"].isna() | (
        relationships_df["relation_type"].astype(str).str.strip() == ""
    )
    null_count = int(null_mask.sum())
    if null_count:
        print(f"    [CGRR] Skipping {null_count} row(s) with missing/empty relation_type")
        relationships_df = relationships_df.loc[~null_mask].copy()
        if relationships_df.empty:
            print("    [CGRR] Skipping — no relationships left after dropping null types")
            return relationships_df, normalize_map, phase_b_log

    llm_merges = 0
    llm_rejections = 0
    below_threshold = 0
    exact_match_skip = 0

    # --- Phase A: Resolve against existing Neo4j relation types ---
    # Synonym detection compares the predicate labels themselves, not the
    # specific edges, so we fetch every active relation type from the graph
    # rather than restricting to types that share an entity with this batch.
    from graphrag.bt_graphrag.neo4j_store import get_existing_relation_types_with_embeddings

    existing_types = await get_existing_relation_types_with_embeddings(session)

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

        total_cands = len(candidate_types)
        for cand_idx, cand_type in enumerate(candidate_types):
            if (cand_idx + 1) % 100 == 0:
                print(f"    [CGRR] Phase A progress: {cand_idx + 1}/{total_cands}  "
                      f"exact={exact_match_skip}  merges={llm_merges}  "
                      f"rejects={llm_rejections}  below={below_threshold}")
            exact_exists = any(et["relation_type"] == cand_type for et in existing_types)
            if exact_exists:
                exact_match_skip += 1
                continue

            cand_sample = relationships_df[relationships_df["relation_type"] == cand_type].iloc[0]
            cand_desc = str(cand_sample.get("description", ""))
            cand_source = str(cand_sample.get("source", ""))
            cand_target = str(cand_sample.get("target", ""))

            cand_emb = (
                cand_sample.get("relation_type_embedding")
                if "relation_type_embedding" in cand_sample.index else None
            )

            # Top-K pre-filter on relation_type_embedding. The Neo4j vector
            # index is keyed on description_embedding (the wrong signal for
            # predicate-synonym detection), so we use in-memory cosine.
            narrowed = _top_k_by_embedding(cand_emb, existing_types, cgrr_top_k)

            cand_record = {
                "relation_type": cand_type,
                "description": cand_desc,
                "source": cand_source,
                "target": cand_target,
                "relation_type_embedding": cand_emb,
            }

            scored_matches: list[tuple[float, dict[str, float], dict[str, Any]]] = []

            for existing in narrowed:
                if cand_type == existing["relation_type"]:
                    continue
                score, breakdown = relationship_scorer(cand_record, existing, config)
                scored_matches.append((score, breakdown, existing))

            scored_matches.sort(key=lambda x: x[0], reverse=True)

            if not scored_matches:
                continue

            best_score, _best_bd, best_match_record = scored_matches[0]
            match_type = best_match_record["relation_type"]

            if best_score >= config.cgrr_cosine_threshold and model is not None:
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
                    logger.info(
                        "CGRR: LLM-confirmed '%s' -> '%s' (cosine=%.3f)",
                        cand_type, match_type, best_score,
                    )
                else:
                    llm_rejections += 1
                    logger.info(
                        "CGRR: LLM rejected '%s' vs '%s' (cosine=%.3f)",
                        cand_type, match_type, best_score,
                    )
            else:
                below_threshold += 1

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
        phase_b_k = (
            phase_b_top_k
            if phase_b_top_k is not None
            else getattr(config, "cgrr_candidate_top_k", 10)
        )
        use_temp_db = driver is not None and bool(getattr(config, "cgrr_phase_b_temp_db", ""))
        batch_db: Any = None
        if use_temp_db:
            from graphrag.bt_graphrag.neo4j_store import CGRRBatchDB
            batch_db = CGRRBatchDB(
                driver=driver,  # type: ignore[arg-type]
                db_name=config.cgrr_phase_b_temp_db,
                vector_dimensions=config.neo4j_vector_dimensions,
            )

        print(f"\n    [CGRR] Phase B: Intra-batch resolution "
              f"({len(remaining_types)} remaining types, top_k={phase_b_k}, "
              f"backend={'neo4j_temp_db' if use_temp_db else 'in_memory'})")

        intra_normalize: dict[str, str] = {}
        canonical_types: list[dict[str, Any]] = []
        canonical_by_rt: dict[str, dict[str, Any]] = {}
        kept_rts: set[str] = set()

        # Pre-collect one sample row per remaining relation type so the
        # temp DB can bulk-load every candidate's embedding up front.
        rt_records: list[dict[str, Any]] = []
        rt_record_by_rt: dict[str, dict[str, Any]] = {}
        for rt in remaining_types:
            sample = relationships_df[relationships_df["relation_type"] == rt].iloc[0]
            rec = {
                "relation_type": rt,
                "description": str(sample.get("description", "")),
                "source": str(sample.get("source", "")),
                "target": str(sample.get("target", "")),
                "relation_type_embedding": (
                    sample.get("relation_type_embedding")
                    if "relation_type_embedding" in sample.index else None
                ),
            }
            rt_records.append(rec)
            rt_record_by_rt[rt] = rec

        async def _pick_phase_b_candidates(
            rec: dict[str, Any],
        ) -> list[dict[str, Any]]:
            """Return up to top-K canonical candidates for one rel type."""
            emb = rec.get("relation_type_embedding") or []
            rt_label = rec["relation_type"]

            if batch_db is not None and emb:
                cand_refs = await batch_db.find_candidates(
                    embedding=list(emb),
                    top_k=phase_b_k,
                    exclude_relation_type=rt_label,
                    allowed_relation_types=kept_rts,
                )
                return [
                    canonical_by_rt[c["relation_type"]]
                    for c in cand_refs
                    if c["relation_type"] in canonical_by_rt
                ]

            # In-memory fallback: cosine over canonical_types
            if not canonical_types:
                return []
            cand_record = {"relation_type_embedding": emb}
            scored: list[tuple[float, dict[str, Any]]] = []
            for canon in canonical_types:
                score, _ = relationship_scorer(cand_record, canon, config)
                scored.append((score, canon))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:phase_b_k]]

        async def _phase_b_loop() -> None:
            nonlocal intra_llm_normalizes, intra_rejections_b
            total_b = len(rt_records)
            step_b = 0
            for rec in rt_records:
                step_b += 1
                if step_b % 100 == 0:
                    print(f"    [CGRR] Phase B progress: {step_b}/{total_b}  "
                          f"merges={intra_llm_normalizes}  "
                          f"rejects={intra_rejections_b}")
                rt = rec["relation_type"]
                cand_desc = rec["description"]
                cand_src = rec["source"]
                cand_tgt = rec["target"]
                cand_emb = rec["relation_type_embedding"]

                if not canonical_types:
                    canonical_types.append(rec)
                    canonical_by_rt[rt] = rec
                    kept_rts.add(rt)
                    continue

                candidate_pool = await _pick_phase_b_candidates(rec)

                best_score = 0.0
                best_canon: dict[str, Any] | None = None
                for canon in candidate_pool:
                    score, _ = relationship_scorer(rec, canon, config)
                    if score > best_score:
                        best_score = score
                        best_canon = canon

                match_type = best_canon["relation_type"] if best_canon else ""

                if (best_canon is not None
                        and best_score >= config.cgrr_cosine_threshold
                        and model is not None):
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
                            "candidate_pool_size": len(candidate_pool),
                            "top_comparisons": [],
                        })
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
                            "candidate_pool_size": len(candidate_pool),
                            "top_comparisons": [],
                        })
                        canonical_types.append(rec)
                        canonical_by_rt[rt] = rec
                        kept_rts.add(rt)
                else:
                    if best_score > 0.3:
                        phase_b_log.append({
                            "phase": "B_intra_batch",
                            "relation_type": rt,
                            "edge_count": cand_type_counts.get(rt, 0),
                            "best_match": match_type,
                            "best_score": round(best_score, 4),
                            "decision": "BELOW_THRESHOLD",
                            "candidate_pool_size": len(candidate_pool),
                            "top_comparisons": [],
                        })
                    canonical_types.append(rec)
                    canonical_by_rt[rt] = rec
                    kept_rts.add(rt)

        if batch_db is not None:
            try:
                async with batch_db:
                    await batch_db.bulk_load(rt_records)
                    print(f"    [CGRR] Phase B: temp Neo4j DB '{batch_db.db_name}' "
                          f"loaded with {len(rt_records)} relation types")
                    await _phase_b_loop()
            except Exception as temp_err:
                logger.warning(
                    "CGRR Phase B: temp Neo4j DB failed (%s); "
                    "falling back to in-memory candidate selection",
                    temp_err,
                )
                print(f"    [CGRR] Phase B: temp DB unavailable ({temp_err}); "
                      f"falling back to in-memory")
                batch_db = None
                intra_normalize.clear()
                canonical_types.clear()
                canonical_by_rt.clear()
                kept_rts.clear()
                intra_llm_normalizes = 0
                intra_rejections_b = 0
                await _phase_b_loop()
        else:
            await _phase_b_loop()

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
