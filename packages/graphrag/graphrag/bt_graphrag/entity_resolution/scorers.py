# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Scoring functions for CGER and CGRR resolution.

CGER (entity resolution) now uses a single signal: cosine similarity
between the two entities' description embeddings. Pairs that exceed the
cosine threshold are deferred to an LLM, which decides whether to merge
using both the descriptions and the temporal context of each entity.

CGRR (relationship-type resolution) retains its three-signal composite
(BM25 on the type string, semantic similarity on the description, and an
endpoint-match indicator) because relation types lack a meaningful
"active period" of their own.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from graphrag.bt_graphrag.models.config import BTGraphRAGConfig


# ---------------------------------------------------------------------------
# Scorer callable type aliases
# ---------------------------------------------------------------------------

#: Callable that scores two entity dicts and returns (cosine, breakdown).
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
# Cosine similarity
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


# ---------------------------------------------------------------------------
# Entity scorer (CGER): description-embedding cosine only
# ---------------------------------------------------------------------------


def description_cosine_entity_scorer(
    new_entity: dict[str, Any],
    existing_entity: dict[str, Any],
    config: "BTGraphRAGConfig",
) -> tuple[float, dict[str, float]]:
    """Score two entities by the cosine similarity of their description embeddings.

    This is the only entity scorer CGER uses. Temporal reasoning is not
    folded into the score itself; instead, the LLM verification step
    receives each entity's active period as part of its context and is
    instructed to keep entities separate when the temporal gap between
    them is significant.

    Returns 0.0 whenever either entity lacks a ``description_embedding``.
    Entity-type filtering is the caller's responsibility (see CGER's
    ``_types_match``).
    """
    emb_new = new_entity.get("description_embedding") or []
    emb_existing = existing_entity.get("description_embedding") or []
    score = cosine_similarity(emb_new, emb_existing)
    breakdown = {
        "cosine_emb": score,
        "embedding_available": bool(emb_new) and bool(emb_existing),
    }
    return score, breakdown


# ---------------------------------------------------------------------------
# Relationship signals (CGRR)
# ---------------------------------------------------------------------------


def jaccard_similarity(s1: str, s2: str) -> float:
    """Character n-gram Jaccard similarity between two strings.

    Used by the semantic-only relationship scorer as a fallback when the
    relation descriptions are empty.
    """
    n = 3  # trigram
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1.lower() == s2.lower() else 0.0
    set1 = {s1[i:i + n].lower() for i in range(len(s1) - n + 1)}
    set2 = {s2[i:i + n].lower() for i in range(len(s2) - n + 1)}
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0


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

    Returns:
        1.0 if both source AND target match (same direction),
        0.75 if both match but swapped (direction inversion: A->B vs B->A),
        0.5 if one endpoint matches,
        0.0 otherwise.
    Comparison is case-insensitive.
    """
    cs = candidate_source.strip().lower()
    ct = candidate_target.strip().lower()
    es = existing_source.strip().lower()
    et = existing_target.strip().lower()

    source_match = cs == es
    target_match = ct == et
    if source_match and target_match:
        return 1.0
    if cs == et and ct == es:
        return 0.75
    if source_match or target_match or cs == et or ct == es:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Composite scoring: Relationships (CGRR)
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
    """
    s1 = bm25_relation_score(candidate_rel_type, existing_rel_type)
    s2 = semantic_description_similarity(candidate_description, existing_description)
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


# ---------------------------------------------------------------------------
# Alternative relationship scorers (CGRR)
# ---------------------------------------------------------------------------


def bm25_only_relationship_scorer(
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
    """Score two relation types using only BM25 lexical similarity on the type string."""
    s1 = bm25_relation_score(candidate_rel_type, existing_rel_type)
    return s1, {"bm25_type": s1, "semantic_desc": 0.0, "endpoint_match": 0.0}


def semantic_only_relationship_scorer(
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
    """Score two relations using only word-overlap semantic similarity of descriptions.

    Falls back to Jaccard on the type string if either description is empty.
    """
    if candidate_description and existing_description:
        s2 = semantic_description_similarity(candidate_description, existing_description)
    else:
        s2 = jaccard_similarity(candidate_rel_type, existing_rel_type)
    return s2, {"bm25_type": 0.0, "semantic_desc": s2, "endpoint_match": 0.0}


def type_and_endpoint_relationship_scorer(
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
    """Score two relations using BM25 on type string plus endpoint match."""
    s1 = bm25_relation_score(candidate_rel_type, existing_rel_type)
    s3 = endpoint_match_score(
        candidate_source, candidate_target,
        existing_source, existing_target,
    )
    total_w = config.cgrr_bm25_weight + config.cgrr_endpoint_weight
    score = (
        (config.cgrr_bm25_weight * s1 + config.cgrr_endpoint_weight * s3) / total_w
        if total_w > 0 else (s1 + s3) / 2.0
    )
    return score, {
        "bm25_type": s1,
        "semantic_desc": 0.0,
        "endpoint_match": s3,
        "w_bm25": config.cgrr_bm25_weight * s1,
        "w_endpoint": config.cgrr_endpoint_weight * s3,
    }
