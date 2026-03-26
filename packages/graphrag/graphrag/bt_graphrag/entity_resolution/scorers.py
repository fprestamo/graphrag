# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Modular scoring functions for CGER and CGRR resolution.

Contains all similarity signals and composite scoring functions used by
Cross-Graph Entity Resolution (CGER) and Cross-Graph Relationship
Resolution (CGRR).  Extracting them into a single module makes the
evaluators reusable, testable, and easier to extend.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from graphrag.bt_graphrag.models.config import BTGraphRAGConfig


# ---------------------------------------------------------------------------
# Scorer callable type aliases
# ---------------------------------------------------------------------------

#: Callable that scores two entity dicts and returns (composite, breakdown).
EntityScorer = Callable[
    [dict[str, Any], dict[str, Any], "BTGraphRAGConfig"],
    tuple[float, dict[str, float]],
]

#: Callable that scores two relationships by their fields and returns (composite, breakdown).
#: Signature: (cand_type, cand_desc, cand_src, cand_tgt,
#:             exist_type, exist_desc, exist_src, exist_tgt, config)
RelationshipScorer = Callable[
    [str, str, str, str, str, str, str, str, "BTGraphRAGConfig"],
    tuple[float, dict[str, float]],
]


# ---------------------------------------------------------------------------
# Entity Similarity Signals (used by CGER)
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
# Relationship Similarity Signals (used by CGRR)
# ---------------------------------------------------------------------------


def bm25_relation_score(
    query: str, candidate: str, k1: float = 1.5, b: float = 0.75
) -> float:
    """BM25 lexical similarity between two relation type strings.

    Normalizes both to tokens (splits on underscores and spaces) before
    computing term overlap.
    """
    def _tokenize(s: str) -> list[str]:
        return [w.lower() for w in s.replace("_", " ").split() if w]

    q_terms = set(_tokenize(query))
    c_terms = _tokenize(candidate)
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


def semantic_description_similarity(
    desc_a: str, desc_b: str
) -> float:
    """Word-overlap proxy for semantic similarity between descriptions.

    Uses Jaccard similarity on word sets as a lightweight approximation.
    When embedding vectors are available, cosine similarity should be
    used instead.
    """
    if not desc_a or not desc_b:
        return 0.0
    words_a = set(desc_a.lower().split())
    words_b = set(desc_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def endpoint_match_score(
    candidate_source: str,
    candidate_target: str,
    existing_source: str,
    existing_target: str,
) -> float:
    """Binary signal boosted when candidate shares same endpoints as existing.

    Returns 1.0 if both source AND target match, 0.5 if one matches, 0.0 otherwise.
    Comparison is case-insensitive.
    """
    source_match = candidate_source.strip().lower() == existing_source.strip().lower()
    target_match = candidate_target.strip().lower() == existing_target.strip().lower()
    if source_match and target_match:
        return 1.0
    if source_match or target_match:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Composite Scoring: Entities (CGER)
# ---------------------------------------------------------------------------


def compute_entity_composite_score(
    new_entity: dict[str, Any],
    existing_entity: dict[str, Any],
    config: "BTGraphRAGConfig",
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
        print(f"        S1 Cosine(desc):    {s1:.4f} x {config.cger_embedding_weight} = {config.cger_embedding_weight * s1:.4f}")
        print(f"        S2 BM25(name):      {s2:.4f} x {config.cger_bm25_weight} = {config.cger_bm25_weight * s2:.4f}")
        print(f"        S3 Jaccard(name):   {s3:.4f} x {config.cger_jaccard_weight} = {config.cger_jaccard_weight * s3:.4f}")
        print(f"        S4 TemporalOverlap: {s4:.4f} x {config.cger_temporal_overlap_weight} = {config.cger_temporal_overlap_weight * s4:.4f}")
        print(f"        S5 RelationCtx:     {s5:.4f} x {config.cger_relation_context_weight} = {config.cger_relation_context_weight * s5:.4f}")
        print(f"        COMPOSITE:          {composite:.4f}")

    return composite, breakdown


# ---------------------------------------------------------------------------
# Alternative Entity Scorers
# ---------------------------------------------------------------------------


def embedding_only_entity_scorer(
    new_entity: dict[str, Any],
    existing_entity: dict[str, Any],
    config: "BTGraphRAGConfig",
) -> tuple[float, dict[str, float]]:
    """Score two entities using only the cosine similarity of their description embeddings.

    This is the simplest possible semantic scorer — a single signal with no
    lexical or temporal fallbacks.  It returns 0.0 whenever either entity
    lacks a ``description_embedding``.

    score = cosine(new.description_embedding, existing.description_embedding)
    """
    emb_new = new_entity.get("description_embedding") or []
    emb_existing = existing_entity.get("description_embedding") or []
    score = cosine_similarity(emb_new, emb_existing)

    breakdown = {
        "cosine_emb": score,
        "embedding_available": bool(emb_new) and bool(emb_existing),
    }
    return score, breakdown


def citation_and_description_entity_scorer(
    new_entity: dict[str, Any],
    existing_entity: dict[str, Any],
    config: "BTGraphRAGConfig",
    w_desc: float = 0.5,
    w_cite: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Score two entities by combining description and citation context embeddings.

    Two signals are fused:

    * **S1 — description cosine**: cosine similarity of ``description_embedding``
      (the summarized description of the entity, embedded at extraction time).
    * **S2 — citation cosine**: cosine similarity of ``text_unit_embedding``
      (the mean embedding of all text units that cite the entity, computed by
      :func:`~graphrag.bt_graphrag.temporal_extraction.embedding_enrichment\
.enrich_entities_with_text_unit_embeddings`).

    Score = w_desc * S1 + w_cite * S2

    Weights are renormalized to the available signals so that a missing
    embedding does not silently penalise every comparison:

    * Both available  → full weighted sum.
    * Only S1         → score = S1  (description only).
    * Only S2         → score = S2  (citation context only).
    * Neither         → score = 0.0.

    Parameters
    ----------
    w_desc:
        Weight for description embedding cosine similarity (default 0.5).
    w_cite:
        Weight for citation/text-unit embedding cosine similarity (default 0.5).
    """
    emb_desc_new = new_entity.get("description_embedding") or []
    emb_desc_ex = existing_entity.get("description_embedding") or []
    emb_cite_new = new_entity.get("text_unit_embedding") or []
    emb_cite_ex = existing_entity.get("text_unit_embedding") or []

    has_desc = bool(emb_desc_new) and bool(emb_desc_ex)
    has_cite = bool(emb_cite_new) and bool(emb_cite_ex)

    s1 = cosine_similarity(emb_desc_new, emb_desc_ex) if has_desc else 0.0
    s2 = cosine_similarity(emb_cite_new, emb_cite_ex) if has_cite else 0.0

    if has_desc and has_cite:
        total_w = w_desc + w_cite
        score = (w_desc * s1 + w_cite * s2) / total_w
    elif has_desc:
        score = s1
    elif has_cite:
        score = s2
    else:
        score = 0.0

    breakdown = {
        "cosine_desc": s1,
        "cosine_cite": s2,
        "has_desc_emb": has_desc,
        "has_cite_emb": has_cite,
        "w_desc": w_desc * s1 if has_desc else 0.0,
        "w_cite": w_cite * s2 if has_cite else 0.0,
    }
    return score, breakdown


# ---------------------------------------------------------------------------
# Composite Scoring: Relationships (CGRR)
# ---------------------------------------------------------------------------


def compute_relationship_composite_score(
    candidate_rel_type: str,
    candidate_description: str,
    candidate_source: str,
    candidate_target: str,
    existing_rel_type: str,
    existing_description: str,
    existing_source: str,
    existing_target: str,
    config: "BTGraphRAGConfig",
) -> tuple[float, dict[str, float]]:
    """Compute the three-signal composite relationship resolution score.

    score = w1*BM25(type) + w2*SemanticSim(desc) + w3*EndpointMatch

    Returns (composite_score, signal_breakdown_dict).
    """
    # Signal 1: BM25 lexical matching on relation type
    s1 = bm25_relation_score(candidate_rel_type, existing_rel_type)

    # Signal 2: Semantic similarity of descriptions
    s2 = semantic_description_similarity(candidate_description, existing_description)

    # Signal 3: Endpoint match
    s3 = endpoint_match_score(
        candidate_source, candidate_target,
        existing_source, existing_target,
    )

    composite = (
        config.cgrr_bm25_weight * s1
        + config.cgrr_semantic_weight * s2
        + config.cgrr_endpoint_weight * s3
    )

    breakdown = {
        "bm25_type": s1,
        "semantic_desc": s2,
        "endpoint_match": s3,
        "w_bm25": config.cgrr_bm25_weight * s1,
        "w_semantic": config.cgrr_semantic_weight * s2,
        "w_endpoint": config.cgrr_endpoint_weight * s3,
    }

    return composite, breakdown
