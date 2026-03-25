# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Cross-Graph Entity Resolution (CGER).

Resolves newly extracted entities against the full existing graph using
a five-signal composite scoring function. Prevents entity aliasing by
matching across ingestion batches.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from graphrag.bt_graphrag.models.config import BTGraphRAGConfig

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Similarity Signals
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_similarity(s1: str, s2: str) -> float:
    """Character n-gram Jaccard similarity between two strings."""
    n = 3  # trigram
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1.lower() == s2.lower() else 0.0
    set1 = {s1[i:i + n].lower() for i in range(len(s1) - n + 1)}
    set2 = {s2[i:i + n].lower() for i in range(len(s2) - n + 1)}
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0


def bm25_name_score(query: str, candidate: str, k1: float = 1.5, b: float = 0.75) -> float:
    """Simplified BM25 score for name matching.

    Treats each name as a short document and scores term overlap.
    """
    q_terms = set(query.lower().split())
    c_terms = candidate.lower().split()
    if not q_terms or not c_terms:
        return 0.0

    avg_len = max(len(c_terms), 1)
    score = 0.0
    for term in q_terms:
        tf = c_terms.count(term)
        idf = 1.0  # single-document IDF approximation
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * len(c_terms) / avg_len)
        score += idf * numerator / denominator
    return min(score / max(len(q_terms), 1), 1.0)


def temporal_overlap_score(
    e1_start: datetime | None,
    e1_end: datetime | float | None,
    e2_start: datetime | None,
    e2_end: datetime | float | None,
) -> float:
    """Measure temporal overlap between two entities' active periods.

    Returns 0-1 score. High overlap suggests same entity; low overlap
    suggests different entities (e.g. company vs its spin-off).
    """
    if e1_start is None or e2_start is None:
        return 0.5  # Unknown: neutral score

    inf = math.inf

    def _to_ts(v: datetime | float | str | None) -> float:
        if v is None or v == inf:
            return 1e18  # far future
        if isinstance(v, datetime):
            return v.timestamp()
        if isinstance(v, str):
            from dateutil.parser import parse as _parse
            return _parse(v).timestamp()
        return float(v)

    s1, e1 = _to_ts(e1_start), _to_ts(e1_end)
    s2, e2 = _to_ts(e2_start), _to_ts(e2_end)

    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)
    overlap = max(0.0, overlap_end - overlap_start)

    union_start = min(s1, s2)
    union_end = max(e1, e2)
    union = max(union_end - union_start, 1.0)

    return overlap / union


def relation_context_similarity(
    e1_relations: list[str],
    e2_relations: list[str],
) -> float:
    """Simple Jaccard-based relation context similarity.

    Compares the sets of relation types/neighbors between two entities.
    """
    if not e1_relations or not e2_relations:
        return 0.0
    s1 = set(r.lower() for r in e1_relations)
    s2 = set(r.lower() for r in e2_relations)
    intersection = s1 & s2
    union = s1 | s2
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Composite Scoring
# ---------------------------------------------------------------------------


def compute_composite_score(
    new_entity: dict[str, Any],
    existing_entity: dict[str, Any],
    config: BTGraphRAGConfig,
    verbose: bool = False,
) -> tuple[float, dict[str, float]]:
    """Compute the five-signal composite entity resolution score.

    score = w1*cosine(desc) + w2*BM25(name) + w3*Jaccard(name)
            + w4*TemporalOverlap + w5*RelationContext

    Returns (composite_score, signal_breakdown_dict).
    """
    # Signal 1: Description embedding cosine similarity
    emb_new = new_entity.get("description_embedding", [])
    emb_existing = existing_entity.get("description_embedding", [])
    s1 = cosine_similarity(emb_new, emb_existing)

    # Signal 2: BM25 name score
    name_new = new_entity.get("title", "")
    name_existing = existing_entity.get("title", "")
    s2 = bm25_name_score(name_new, name_existing)

    # Signal 3: Jaccard name similarity
    s3 = jaccard_similarity(name_new, name_existing)

    # Signal 4: Temporal overlap
    s4 = temporal_overlap_score(
        new_entity.get("active_start"),
        new_entity.get("active_end"),
        existing_entity.get("active_start"),
        existing_entity.get("active_end"),
    )

    # Signal 5: Relation context similarity
    s5 = relation_context_similarity(
        new_entity.get("relation_types", []),
        existing_entity.get("relation_types", []),
    )

    composite = (
        config.cger_embedding_weight * s1
        + config.cger_bm25_weight * s2
        + config.cger_jaccard_weight * s3
        + config.cger_temporal_overlap_weight * s4
        + config.cger_relation_context_weight * s5
    )

    # When description embeddings are unavailable on either side, signal S1 is
    # always 0 and the embedding weight acts as a dead penalty on every comparison.
    # Normalize the composite to the range [0, 1] relative to the signals that
    # are actually available, so thresholds retain their intended meaning.
    embedding_available = bool(emb_new) and bool(emb_existing)
    if not embedding_available:
        available_weight = (
            config.cger_bm25_weight
            + config.cger_jaccard_weight
            + config.cger_temporal_overlap_weight
            + config.cger_relation_context_weight
        )
        if available_weight > 0:
            composite = composite / available_weight

    breakdown = {
        "cosine_emb": s1,
        "bm25_name": s2,
        "jaccard_name": s3,
        "temporal_overlap": s4,
        "relation_ctx": s5,
        "normalized": not embedding_available,
        "w_cosine": config.cger_embedding_weight * s1,
        "w_bm25": config.cger_bm25_weight * s2,
        "w_jaccard": config.cger_jaccard_weight * s3,
        "w_temporal": config.cger_temporal_overlap_weight * s4,
        "w_relation": config.cger_relation_context_weight * s5,
    }

    if verbose:
        print(f"      Score breakdown: '{name_new}' vs '{name_existing}'"
              + (" [NORMALIZED — no embeddings]" if not embedding_available else ""))
        print(f"        S1 Cosine(desc):    {s1:.4f} × {config.cger_embedding_weight} = {config.cger_embedding_weight * s1:.4f}")
        print(f"        S2 BM25(name):      {s2:.4f} × {config.cger_bm25_weight} = {config.cger_bm25_weight * s2:.4f}")
        print(f"        S3 Jaccard(name):   {s3:.4f} × {config.cger_jaccard_weight} = {config.cger_jaccard_weight * s3:.4f}")
        print(f"        S4 TemporalOverlap: {s4:.4f} × {config.cger_temporal_overlap_weight} = {config.cger_temporal_overlap_weight * s4:.4f}")
        print(f"        S5 RelationCtx:     {s5:.4f} × {config.cger_relation_context_weight} = {config.cger_relation_context_weight * s5:.4f}")
        print(f"        COMPOSITE:          {composite:.4f}")

    return composite, breakdown


# ---------------------------------------------------------------------------
# LLM Verification for Hard Cases
# ---------------------------------------------------------------------------

CGER_VERIFICATION_PROMPT = """You are an entity resolution expert. Determine whether the following two entities refer to the same real-world entity.

Entity A:
- Name: {name_a}
- Type: {type_a}
- Description: {desc_a}
- Active Period: {period_a}
- Known Relations: {relations_a}

Entity B:
- Name: {name_b}
- Type: {type_b}
- Description: {desc_b}
- Active Period: {period_b}
- Known Relations: {relations_b}

Are these the same entity? Answer ONLY with one of:
- SAME: They are the same real-world entity
- DIFFERENT: They are different entities
- UNCERTAIN: Cannot determine with available information

Answer:"""


async def llm_verify_entity_match(
    entity_a: dict[str, Any],
    entity_b: dict[str, Any],
    model: "LLMCompletion",
) -> str:
    """Use LLM to verify whether two entities are the same.

    Returns "SAME", "DIFFERENT", or "UNCERTAIN".
    """
    from graphrag_llm.utils import CompletionMessagesBuilder

    def _period(e: dict) -> str:
        start = e.get("active_start", "unknown")
        end = e.get("active_end", "present")
        if end == math.inf:
            end = "present"
        return f"{start} to {end}"

    prompt = CGER_VERIFICATION_PROMPT.format(
        name_a=entity_a.get("title", ""),
        type_a=entity_a.get("type", ""),
        desc_a=entity_a.get("description", "")[:500],
        period_a=_period(entity_a),
        relations_a=", ".join(entity_a.get("relation_types", [])[:10]),
        name_b=entity_b.get("title", ""),
        type_b=entity_b.get("type", ""),
        desc_b=entity_b.get("description", "")[:500],
        period_b=_period(entity_b),
        relations_b=", ".join(entity_b.get("relation_types", [])[:10]),
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
# Main CGER Pipeline
# ---------------------------------------------------------------------------


async def resolve_entities(
    new_entities: pd.DataFrame,
    existing_entities: pd.DataFrame,
    config: BTGraphRAGConfig,
    model: "LLMCompletion | None" = None,
) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    """Resolve new entities against the existing graph.

    For each new entity, computes composite scores against all existing
    entities. Entities scoring above the merge threshold are merged.
    Hard cases (between low and high thresholds) are sent to LLM.

    Returns:
        resolved_entities: The new entities DataFrame with merged IDs
        merge_map: Dict mapping merged entity titles to their canonical titles
        phase_b_log: Debug log entries from Phase B (intra-batch) resolution
    """
    merge_map: dict[str, str] = {}
    phase_b_log: list[dict[str, Any]] = []

    if new_entities.empty:
        print("    [CGER] Skipping — no new entities to process")
        return new_entities, merge_map, phase_b_log

    auto_merges = 0
    llm_merges = 0
    llm_rejections = 0
    no_match_count = 0
    below_threshold_count = 0

    # --- Phase A: Resolve new entities against existing graph ---
    existing_records = existing_entities.to_dict("records") if not existing_entities.empty else []

    if not existing_records:
        print(f"    [CGER] Phase A: Skipping — graph is empty (first run); "
              f"proceeding to intra-batch Phase B for {len(new_entities)} entities")
    else:
        print(f"\n    [CGER] Phase A: Resolving {len(new_entities)} new entities "
              f"against {len(existing_records)} existing")
        print(f"    [CGER] Thresholds: auto_merge >= {config.cger_merge_threshold}, "
              f"LLM_zone = [{config.cger_llm_threshold_low}, {config.cger_merge_threshold})")
        print(f"    [CGER] Weights: emb={config.cger_embedding_weight}, bm25={config.cger_bm25_weight}, "
              f"jaccard={config.cger_jaccard_weight}, temporal={config.cger_temporal_overlap_weight}, "
              f"relation={config.cger_relation_context_weight}")
        print()

        for idx, new_row in new_entities.iterrows():
            new_entity = dict(new_row)
            best_score = 0.0
            best_breakdown: dict[str, float] = {}
            best_match: dict[str, Any] | None = None

            for existing in existing_records:
                score, breakdown = compute_composite_score(new_entity, existing, config)
                if score > best_score:
                    best_score = score
                    best_breakdown = breakdown
                    best_match = existing

            entity_title = str(new_entity.get("title", "?"))

            if best_match is None:
                no_match_count += 1
                print(f"    [{idx}] '{entity_title[:35]}' — no candidates found")
                continue

            match_title = str(best_match.get("title", "?"))

            if best_score >= config.cger_merge_threshold:
                merge_map[new_entity["title"]] = best_match["title"]
                auto_merges += 1
                print(f"    [{idx}] AUTO-MERGE: '{entity_title[:30]}' -> '{match_title[:30]}'  score={best_score:.4f}")
                print(f"         Signals: cosine={best_breakdown.get('cosine_emb', 0):.3f}  "
                      f"bm25={best_breakdown.get('bm25_name', 0):.3f}  "
                      f"jaccard={best_breakdown.get('jaccard_name', 0):.3f}  "
                      f"temporal={best_breakdown.get('temporal_overlap', 0):.3f}  "
                      f"relation={best_breakdown.get('relation_ctx', 0):.3f}")
                logger.info(
                    "CGER: Auto-merging '%s' -> '%s' (score=%.3f)",
                    new_entity["title"], best_match["title"], best_score,
                )
            elif best_score >= config.cger_llm_threshold_low and model is not None:
                print(f"    [{idx}] LLM-ZONE: '{entity_title[:30]}' vs '{match_title[:30]}'  score={best_score:.4f}")
                print(f"         Signals: cosine={best_breakdown.get('cosine_emb', 0):.3f}  "
                      f"bm25={best_breakdown.get('bm25_name', 0):.3f}  "
                      f"jaccard={best_breakdown.get('jaccard_name', 0):.3f}  "
                      f"temporal={best_breakdown.get('temporal_overlap', 0):.3f}  "
                      f"relation={best_breakdown.get('relation_ctx', 0):.3f}")
                verdict = await llm_verify_entity_match(new_entity, best_match, model)
                if verdict == "SAME":
                    merge_map[new_entity["title"]] = best_match["title"]
                    llm_merges += 1
                    print(f"         LLM verdict: SAME -> MERGED")
                    logger.info(
                        "CGER: LLM-confirmed merge '%s' -> '%s' (score=%.3f)",
                        new_entity["title"], best_match["title"], best_score,
                    )
                else:
                    llm_rejections += 1
                    print(f"         LLM verdict: {verdict} -> KEPT SEPARATE")
                    logger.info(
                        "CGER: LLM rejected merge '%s' vs '%s' (score=%.3f, verdict=%s)",
                        new_entity["title"], best_match["title"], best_score, verdict,
                    )
            else:
                below_threshold_count += 1
                if best_score > 0.3:
                    print(f"    [{idx}] BELOW THRESHOLD: '{entity_title[:30]}' best='{match_title[:30]}'  score={best_score:.4f}")

    # Apply Phase A merge map to new entities
    if merge_map:
        new_entities = new_entities.copy()
        new_entities["title"] = new_entities["title"].map(
            lambda t: merge_map.get(t, t)
        )

    # --- Phase B: Intra-batch resolution (new vs new) ---
    # Only process entities not already remapped to an existing graph entity.
    unmerged = new_entities[~new_entities["title"].isin(merge_map.values())]

    intra_auto_merges = 0
    intra_llm_merges = 0
    intra_rejections = 0

    if len(unmerged) > 1:
        print(f"\n    [CGER] Phase B: Intra-batch resolution ({len(unmerged)} unmerged entities)")
        intra_merge_map: dict[str, str] = {}
        seen_entities: list[dict[str, Any]] = []

        for _, row in unmerged.iterrows():
            entity = dict(row)
            entity_title = str(entity.get("title", ""))

            # Skip if already resolved in Phase A (guards against chain-merge edge cases)
            if entity_title in merge_map:
                continue

            if not seen_entities:
                seen_entities.append(entity)
                continue

            best_score = 0.0
            best_breakdown: dict[str, float] = {}
            best_match: dict[str, Any] | None = None

            for seen in seen_entities:
                score, breakdown = compute_composite_score(entity, seen, config)
                if score > best_score:
                    best_score = score
                    best_breakdown = breakdown
                    best_match = seen

            match_title = str(best_match.get("title", "?")) if best_match else ""

            if best_match is not None and best_score >= config.cger_merge_threshold:
                intra_merge_map[entity_title] = match_title
                intra_auto_merges += 1
                phase_b_log.append({
                    "phase": "B_intra_batch",
                    "entity": entity_title,
                    "type": str(entity.get("type", "?")),
                    "best_match": match_title,
                    "best_score": round(best_score, 4),
                    "decision": "AUTO_MERGE",
                    "top_comparisons": [],
                })
                print(f"    [INTRA] AUTO-MERGE: '{entity_title[:30]}' -> '{match_title[:30]}'  "
                      f"score={best_score:.4f}")
                print(f"           cosine={best_breakdown.get('cosine_emb', 0):.3f}  "
                      f"bm25={best_breakdown.get('bm25_name', 0):.3f}  "
                      f"jaccard={best_breakdown.get('jaccard_name', 0):.3f}  "
                      f"temporal={best_breakdown.get('temporal_overlap', 0):.3f}  "
                      f"relation={best_breakdown.get('relation_ctx', 0):.3f}")
                logger.info(
                    "CGER: Intra-batch merge '%s' -> '%s' (score=%.3f)",
                    entity_title, match_title, best_score,
                )
            elif (best_match is not None
                  and best_score >= config.cger_llm_threshold_low
                  and model is not None):
                print(f"    [INTRA] LLM-ZONE: '{entity_title[:30]}' vs '{match_title[:30]}'  "
                      f"score={best_score:.4f}")
                verdict = await llm_verify_entity_match(entity, best_match, model)
                if verdict == "SAME":
                    intra_merge_map[entity_title] = match_title
                    intra_llm_merges += 1
                    phase_b_log.append({
                        "phase": "B_intra_batch",
                        "entity": entity_title,
                        "type": str(entity.get("type", "?")),
                        "best_match": match_title,
                        "best_score": round(best_score, 4),
                        "decision": "LLM_MERGE",
                        "top_comparisons": [],
                    })
                    print(f"           LLM verdict: SAME -> MERGED")
                    logger.info(
                        "CGER: Intra-batch LLM merge '%s' -> '%s' (score=%.3f)",
                        entity_title, match_title, best_score,
                    )
                else:
                    intra_rejections += 1
                    phase_b_log.append({
                        "phase": "B_intra_batch",
                        "entity": entity_title,
                        "type": str(entity.get("type", "?")),
                        "best_match": match_title,
                        "best_score": round(best_score, 4),
                        "decision": "LLM_ZONE_REJECTED",
                        "top_comparisons": [],
                    })
                    print(f"           LLM verdict: {verdict} -> KEPT SEPARATE")
                    seen_entities.append(entity)
            else:
                if best_score > 0.3:
                    phase_b_log.append({
                        "phase": "B_intra_batch",
                        "entity": entity_title,
                        "type": str(entity.get("type", "?")),
                        "best_match": match_title,
                        "best_score": round(best_score, 4),
                        "decision": "BELOW_THRESHOLD",
                        "top_comparisons": [],
                    })
                    print(f"    [INTRA] BELOW: '{entity_title[:30]}' best='{match_title[:30]}'  "
                          f"score={best_score:.4f}")
                seen_entities.append(entity)

        if intra_merge_map:
            new_entities = new_entities.copy()
            new_entities["title"] = new_entities["title"].map(
                lambda t: intra_merge_map.get(t, t)
            )
            merge_map.update(intra_merge_map)
        else:
            print(f"    [CGER] Intra-batch: no additional merges found")
    else:
        print(f"\n    [CGER] Phase B: Intra-batch skipped (0-1 unmerged entities)")

    # Deduplicate entity rows created by Phase A/B merging
    # (e.g. two new entities both resolved to the same canonical title)
    _before_dedup = len(new_entities)
    new_entities = new_entities.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    _dedup_dropped = _before_dedup - len(new_entities)
    if _dedup_dropped:
        print(f"\n    [CGER] Deduplicated {_dedup_dropped} entity row(s) after merging")

    # --- Verification Summary ---
    print(f"\n    {'=' * 55}")
    print(f"    CGER VERIFICATION SUMMARY")
    print(f"    {'=' * 55}")
    print(f"    Total new entities processed:   {len(new_entities)}")
    print(f"    Existing entities compared:     {len(existing_records)}")
    print(f"    Phase A auto-merges:            {auto_merges}")
    print(f"    Phase A LLM-confirmed:          {llm_merges}")
    print(f"    Phase A LLM-rejected:           {llm_rejections}")
    print(f"    Phase A below threshold:        {below_threshold_count}")
    print(f"    Phase A no candidates:          {no_match_count}")
    print(f"    Phase B intra auto-merges:      {intra_auto_merges}")
    print(f"    Phase B intra LLM-confirmed:    {intra_llm_merges}")
    print(f"    Phase B intra LLM-rejected:     {intra_rejections}")
    print(f"    Total merges (A+B):             {len(merge_map)}")

    # Verify no circular merges
    circular = False
    for src, dst in merge_map.items():
        if dst in merge_map and merge_map[dst] != dst:
            circular = True
            print(f"    WARNING: Potential chain merge: '{src}' -> '{dst}' -> '{merge_map[dst]}'")
    if not circular:
        print(f"    Circular merge check:           PASS")

    print(f"    {'=' * 55}")

    return new_entities, merge_map, phase_b_log


def apply_merge_map_to_relationships(
    relationships: pd.DataFrame,
    merge_map: dict[str, str],
) -> pd.DataFrame:
    """Update source/target in relationships based on entity merge map."""
    if not merge_map or relationships.empty:
        return relationships

    relationships = relationships.copy()
    relationships["source"] = relationships["source"].map(
        lambda s: merge_map.get(s, s)
    )
    relationships["target"] = relationships["target"].map(
        lambda t: merge_map.get(t, t)
    )
    return relationships
