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

    def _to_ts(v: datetime | float | None) -> float:
        if v is None or v == inf:
            return 1e18  # far future
        if isinstance(v, datetime):
            return v.timestamp()
        return v

    s1, e1 = e1_start.timestamp(), _to_ts(e1_end)
    s2, e2 = e2_start.timestamp(), _to_ts(e2_end)

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
) -> float:
    """Compute the five-signal composite entity resolution score.

    score = w1*cosine(desc) + w2*BM25(name) + w3*Jaccard(name)
            + w4*TemporalOverlap + w5*RelationContext
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

    return (
        config.cger_embedding_weight * s1
        + config.cger_bm25_weight * s2
        + config.cger_jaccard_weight * s3
        + config.cger_temporal_overlap_weight * s4
        + config.cger_relation_context_weight * s5
    )


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
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Resolve new entities against the existing graph.

    For each new entity, computes composite scores against all existing
    entities. Entities scoring above the merge threshold are merged.
    Hard cases (between low and high thresholds) are sent to LLM.

    Returns:
        resolved_entities: The new entities DataFrame with merged IDs
        merge_map: Dict mapping merged entity titles to their canonical titles
    """
    merge_map: dict[str, str] = {}

    if existing_entities.empty or new_entities.empty:
        return new_entities, merge_map

    existing_records = existing_entities.to_dict("records")

    for idx, new_row in new_entities.iterrows():
        new_entity = dict(new_row)
        best_score = 0.0
        best_match: dict[str, Any] | None = None

        for existing in existing_records:
            score = compute_composite_score(new_entity, existing, config)
            if score > best_score:
                best_score = score
                best_match = existing

        if best_match is None:
            continue

        if best_score >= config.cger_merge_threshold:
            # Automatic merge
            merge_map[new_entity["title"]] = best_match["title"]
            logger.info(
                "CGER: Auto-merging '%s' -> '%s' (score=%.3f)",
                new_entity["title"], best_match["title"], best_score,
            )
        elif best_score >= config.cger_llm_threshold_low and model is not None:
            # LLM verification for hard cases
            verdict = await llm_verify_entity_match(new_entity, best_match, model)
            if verdict == "SAME":
                merge_map[new_entity["title"]] = best_match["title"]
                logger.info(
                    "CGER: LLM-confirmed merge '%s' -> '%s' (score=%.3f)",
                    new_entity["title"], best_match["title"], best_score,
                )
            else:
                logger.info(
                    "CGER: LLM rejected merge '%s' vs '%s' (score=%.3f, verdict=%s)",
                    new_entity["title"], best_match["title"], best_score, verdict,
                )

    # Apply merge map to new entities
    if merge_map:
        new_entities = new_entities.copy()
        new_entities["title"] = new_entities["title"].map(
            lambda t: merge_map.get(t, t)
        )

    return new_entities, merge_map


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
