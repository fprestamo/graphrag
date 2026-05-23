# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Scoring functions for CGER and CGRR resolution.

Both CGER (entity resolution) and CGRR (relationship-type resolution)
now use a single signal: cosine similarity between the description
embeddings of the two items being compared. Pairs whose cosine exceeds
the configured threshold are deferred to an LLM, which is the only
component that ever triggers a merge.

The two scorers differ in what the LLM receives:

* CGER feeds the LLM the descriptions **and** the active periods of
  each entity, and the LLM may answer ``SAME``, ``DIFFERENT_ENTITY`` or
  ``DIFFERENT_TEMPORAL`` (entities can be the same referent in two
  temporally-separated states).
* CGRR feeds the LLM only the descriptions and a sample endpoint pair
  for each relation type, and the LLM answers ``SAME`` or
  ``DIFFERENT``. Relation types have no active period of their own, so
  the temporal verdict does not apply.
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

#: Callable that scores two relationship dicts and returns (cosine, breakdown).
#: Each dict must expose at least ``description_embedding``; CGRR also reads
#: ``relation_type``, ``description``, ``source`` and ``target`` for logging
#: and for the LLM prompt.
RelationshipScorer = Callable[
    [dict[str, Any], dict[str, Any], "BTGraphRAGConfig"],
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
# Relationship scorer (CGRR): description-embedding cosine only
# ---------------------------------------------------------------------------


def description_cosine_relationship_scorer(
    candidate: dict[str, Any],
    existing: dict[str, Any],
    config: "BTGraphRAGConfig",
) -> tuple[float, dict[str, float]]:
    """Score two relation types by the cosine similarity of their description embeddings.

    Mirrors :func:`description_cosine_entity_scorer` but operates on
    relation-type records. The endpoint pair, relation-type string and
    description text are reserved for the LLM prompt; the score itself
    is the single cosine signal.

    Returns 0.0 whenever either relation lacks a
    ``description_embedding``. Same-endpoint and same-type filtering is
    the caller's responsibility.
    """
    emb_cand = candidate.get("description_embedding") or []
    emb_exist = existing.get("description_embedding") or []
    score = cosine_similarity(emb_cand, emb_exist)
    breakdown = {
        "cosine_emb": score,
        "embedding_available": bool(emb_cand) and bool(emb_exist),
    }
    return score, breakdown
