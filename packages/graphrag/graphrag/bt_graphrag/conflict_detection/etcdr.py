# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Edge-Level Temporal Conflict Detection and Resolution (ETCDR).

Stage 3 of the BT-GraphRAG pipeline. Before any relationship is written to
the graph, ETCDR executes bidirectional conflict queries against Neo4j,
classifies the conflict type, and routes to one of four resolution strategies:
Evolution, Correction, Corroboration, or Disagreement.

Section 2.6 of the BT-GraphRAG design document.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.models.temporal_types import (
    INFINITY,
    INFINITY_ISO,
    MINUS_INFINITY,
    ConflictResult,
    RelationCardinality,
    ResolutionStrategy,
    TemporalRelationship,
    TemporalStateQuad,
    utcnow,
)

if TYPE_CHECKING:
    from graphrag.bt_graphrag.neo4j_store import ETCDRBatchDB
    from graphrag_llm.completion import LLMCompletion
    from neo4j import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON Resolution Log
# ---------------------------------------------------------------------------

_resolution_log: list[dict[str, Any]] = []
"""In-memory buffer for resolution log entries, flushed to disk at the end."""


def _log_resolution(
    candidate: TemporalRelationship,
    existing: TemporalRelationship | None,
    strategy: ResolutionStrategy,
    confidence: float,
    phase: str,
    conflict_type: str,
) -> None:
    """Append an entry to the in-memory resolution log.

    Args:
        candidate: The incoming edge that triggered the conflict.
        existing: The conflicting edge (Neo4j or intra-batch).  None when
            there is no conflict.
        strategy: The chosen resolution strategy.
        confidence: Router confidence score.
        phase: ``"intra_batch"`` or ``"neo4j"`` indicating where the conflict
            was detected.
        conflict_type: ``"SUBJECT_SIDE"``, ``"OBJECT_SIDE"``, ``"BOTH_SIDES"``,
            or ``"NONE"``.
    """
    def _ts(v: float | datetime | None) -> str:
        if v == INFINITY:
            return "present"
        if v is None or v == MINUS_INFINITY:
            return "unknown"
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%dT%H:%M:%S")
        return "present"

    entry: dict[str, Any] = {
        "timestamp": utcnow().isoformat(),
        "phase": phase,
        "problem": strategy.value,
        "conflict_type": conflict_type,
        "confidence": round(confidence, 4),
        "candidate": {
            "source": candidate.source,
            "relation_type": candidate.relation_type,
            "target": candidate.target,
            "description": (candidate.description or "")[:120],
            "valid_start": _ts(candidate.temporal_quad.t_valid_start if candidate.temporal_quad else None),
            "valid_end": _ts(candidate.temporal_quad.t_valid_end if candidate.temporal_quad else None),
        },
        "conflicting_edge": None,
        "resolution": _RESOLUTION_DESCRIPTIONS.get(strategy, strategy.value),
    }
    if existing is not None:
        entry["conflicting_edge"] = {
            "id": existing.id or "",
            "source": existing.source,
            "relation_type": existing.relation_type,
            "target": existing.target,
            "description": (existing.description or "")[:120],
            "valid_start": _ts(existing.temporal_quad.t_valid_start if existing.temporal_quad else None),
            "valid_end": _ts(existing.temporal_quad.t_valid_end if existing.temporal_quad else None),
        }
    _resolution_log.append(entry)


_RESOLUTION_DESCRIPTIONS: dict[ResolutionStrategy, str] = {
    ResolutionStrategy.EVOLUTION: "World changed: closed old edge valid_end, inserted new edge",
    ResolutionStrategy.CORRECTION: "Old edge was wrong: retracted via tx_end, inserted corrected edge",
    ResolutionStrategy.CORROBORATION: "Same fact confirmed: incremented support_count on existing edge",
    ResolutionStrategy.DISAGREEMENT: "Ambiguous conflict: inserted candidate as disputed, existing unchanged",
    ResolutionStrategy.NEW_EDGE: "No conflict: inserted candidate as a new independent edge",
}


def flush_resolution_log(output_dir: str | Path | None = None) -> Path | None:
    """Write the accumulated resolution log to ``etcdr_resolution_log.json``.

    Called once after all edges in a batch have been processed.

    Args:
        output_dir: Directory where the JSON file will be written.
            Falls back to current working directory if not provided.

    Returns:
        Path to the written file, or None if there were no log entries.
    """
    if not _resolution_log:
        return None

    if output_dir is None:
        output_dir = Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "etcdr_resolution_log.json"

    # Append to existing log if present
    existing_entries: list[dict[str, Any]] = []
    if log_path.exists():
        try:
            existing_entries = json.loads(log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_entries = []

    all_entries = existing_entries + list(_resolution_log)
    log_path.write_text(
        json.dumps(all_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("ETCDR: Wrote %d resolution log entries to %s", len(_resolution_log), log_path)
    _resolution_log.clear()
    return log_path


# ---------------------------------------------------------------------------
# Decision Router LLM Prompts
# ---------------------------------------------------------------------------

DECISION_ROUTER_PROMPT = """You are a temporal knowledge graph expert tasked with resolving a conflict between an existing edge and a new candidate edge.

## Existing Edge (conflicting)
- Subject: {existing_subject}
- Relation: {existing_relation_type}
- Object: {existing_object}
- Valid Time: [{existing_t_valid_start} → {existing_t_valid_end}]
- Description: {existing_description}
- Source: {existing_source}
- Trust Score: {existing_trust}

## Candidate Edge (incoming)
- Subject: {candidate_subject}
- Relation: {candidate_relation_type}
- Object: {candidate_object}
- Valid Time: [{candidate_t_valid_start} → {candidate_t_valid_end}]
- Description: {candidate_description}
- Source: {candidate_source}
- Trust Score: {candidate_trust}

## Relation Cardinality
{cardinality}: {cardinality_explanation}

## Conflict Type
{conflict_type}

## Reading the Valid Time bounds
- A real date (e.g. ``2020-03-15``) means the extractor anchored that endpoint to a specific point in time.
- ``present`` in the right bound means the fact is still being asserted (open-ended).
- ``unknown`` in either bound means the extractor could NOT anchor that endpoint. Reason about temporal ordering from the descriptions and the source authority; if neither breaks the tie, prefer DISAGREEMENT over guessing EVOLUTION/CORRECTION.

## Task
Based on the evidence above, select the most appropriate resolution strategy:

1. **EVOLUTION** - The world genuinely changed. The existing edge was true for its period, and the candidate edge represents a new true state that strictly succeeds it (e.g. role transition from Chairman to Honorary Chairman; constituency change from Hertford to Finsbury). Close the existing edge's valid-time end at the candidate's start time and insert the candidate.

2. **CORRECTION** - The existing edge was wrong (factual error, hallucination, or bad source). Retroactively invalidate it (close transaction time) and insert the corrected candidate, inheriting the old edge's valid-time period. Reserve this for cases where one edge clearly contradicts the other; do NOT use it for trivial differences (≤ a few days in event dates, name-spelling variants, or interval refinement).

3. **CORROBORATION** - The candidate says the same thing as the existing edge. No new edge needed; just update the support count and provenance.
   Choose CORROBORATION whenever any of these hold:
   - Same fact paraphrased: descriptions paraphrase the same event/role even if one is more verbose or uses synonyms.
   - Trivial date drift: dates differ by ≤ a few days for a brief event, or by ≤ a few months for a multi-year role with otherwise matching boundaries.
   - Interval refinement: one side has ``unknown``/``present`` where the other has a concrete date that does not contradict it (e.g. existing ``unknown → present`` and candidate ``2015 → 2018`` for the same role).
   - Spelling variants of the same name after canonicalisation (e.g. "Peter" vs "Pierre" for the same merged entity).

4. **DISAGREEMENT** - There is genuine ambiguity or conflicting evidence with no clear resolution. Insert the candidate as "disputed" without modifying the existing edge.

5. **NEW_EDGE** - The two edges describe **independent facts that do not share an exclusivity slot**, even though they were retrieved by the cardinality query. The candidate should be inserted as a fresh, independent edge with no change to the existing one.
   Choose NEW_EDGE whenever any of these hold:
   - The two relation types are different and describe non-overlapping roles for the same subject (e.g. ``LEADS conference`` in 1911 vs ``SUCCEEDED_BY person`` in 1913 — leading an event and being succeeded as principal are different exclusivity slots).
   - The objects differ and the two facts can coexist for the subject without violating cardinality (e.g. ``MEMBER_OF House of Commons`` as a continuous role vs ``PARTICIPATED_IN House of Commons`` for a specific event in 2012).
   - The candidate adds a new dimension of the subject's biography that does not contradict, replace, or restate the existing edge.

Note: if the two relations are different types but describe the same pair (NON_EXCLUSIVE), first judge whether the descriptions are semantically compatible (both can be true simultaneously) or conflicting (one negates the other). If compatible and they describe distinct sub-facts, prefer NEW_EDGE. If they describe the same underlying fact via synonymous relation types (e.g. ``CHALLENGED`` vs ``OPPOSED`` for the same leadership contest in the same week), prefer CORROBORATION. If one clearly negates the other, choose EVOLUTION or CORRECTION based on temporal ordering.

Respond in this exact format:
STRATEGY: <one of EVOLUTION, CORRECTION, CORROBORATION, DISAGREEMENT, NEW_EDGE>
CONFIDENCE: <0.0 to 1.0>
REASONING: <one sentence explaining the choice>
"""

CARDINALITY_EXPLANATIONS = {
    RelationCardinality.BOTH_EXCLUSIVE: "One-to-one: at most one subject can hold this relation with this object at a time, AND each subject can hold it with at most one object at a time.",
    RelationCardinality.SUBJECT_EXCLUSIVE: "One-to-many: a single subject can hold at most one active instance of this relation (e.g. a person has one nationality at a time).",
    RelationCardinality.OBJECT_EXCLUSIVE: "Many-to-one: at most one subject can hold this relation with a given object (e.g. a country has one capital at a time).",
    RelationCardinality.NON_EXCLUSIVE: "Many-to-many: no cardinality constraint; multiple different relations between the same pair can coexist. However, a candidate may still semantically conflict with an existing edge between the same source and target (e.g. WORKS_AT vs LEFT_COMPANY). The router must judge whether the two descriptions are compatible or contradictory.",
}


# ---------------------------------------------------------------------------
# Conflict Detection Queries
# ---------------------------------------------------------------------------


async def run_subject_side_query(
    session: "AsyncSession",
    subject: str,
    relation_type: str,
    t_event: datetime | None = None,
) -> list[dict[str, Any]]:
    """Sub-query S: find active edges where this subject already holds this relation.

    For late arrivals, queries the graph state at t_event.
    Returns list of conflicting edge property dicts.
    """
    if t_event is not None:
        # Late-arrival path: query historical state at t_event
        query = """
        MATCH (s:Entity {title: $subject})-[e:RELATIONSHIP]->(o:Entity)
        WHERE e.relation_type = $relation_type
          AND e.t_valid_start <= $t_event
          AND e.t_valid_end > $t_event
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, o.title AS object_title
        """
        result = await session.run(
            query,
            subject=subject,
            relation_type=relation_type,
            infinity=INFINITY_ISO,
            t_event=t_event.isoformat(),
        )
    else:
        # Normal path: every non-retracted edge (t_tx_end = INFINITY) is a
        # candidate for conflict. Valid-time overlap with the candidate is
        # judged downstream by the resolver so closed historical intervals
        # remain eligible for CORROBORATION / EVOLUTION / CORRECTION.
        query = """
        MATCH (s:Entity {title: $subject})-[e:RELATIONSHIP]->(o:Entity)
        WHERE e.relation_type = $relation_type
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, o.title AS object_title
        """
        result = await session.run(
            query, subject=subject, relation_type=relation_type,
            infinity=INFINITY_ISO,
        )

    records = []
    async for record in result:
        edge_data = dict(record["e"])
        edge_data["_object_title"] = record["object_title"]
        records.append(edge_data)
    return records


async def run_object_side_query(
    session: "AsyncSession",
    subject: str,
    relation_type: str,
    obj: str,
    t_event: datetime | None = None,
) -> list[dict[str, Any]]:
    """Sub-query O: find active edges where a *different* subject holds this exclusive relation to this object.

    Activated only for OBJECT_EXCLUSIVE and BOTH_EXCLUSIVE relations.
    This is the key addition that ZEP/Graphiti misses.
    """
    if t_event is not None:
        query = """
        MATCH (s:Entity)-[e:RELATIONSHIP]->(o:Entity {title: $obj})
        WHERE e.relation_type = $relation_type
          AND s.title <> $subject
          AND e.t_valid_start <= $t_event
          AND e.t_valid_end > $t_event
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, s.title AS subject_title
        """
        result = await session.run(
            query,
            obj=obj,
            relation_type=relation_type,
            subject=subject,
            infinity=INFINITY_ISO,
            t_event=t_event.isoformat(),
        )
    else:
        # Normal path: see run_subject_side_query — valid-time overlap is
        # delegated to the resolver so historical edges stay visible.
        query = """
        MATCH (s:Entity)-[e:RELATIONSHIP]->(o:Entity {title: $obj})
        WHERE e.relation_type = $relation_type
          AND s.title <> $subject
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, s.title AS subject_title
        """
        result = await session.run(
            query, obj=obj, relation_type=relation_type, subject=subject,
            infinity=INFINITY_ISO,
        )

    records = []
    async for record in result:
        edge_data = dict(record["e"])
        edge_data["_subject_title"] = record["subject_title"]
        records.append(edge_data)
    return records


async def run_same_pair_query(
    session: "AsyncSession",
    subject: str,
    relation_type: str,
    obj: str,
    t_event: datetime | None = None,
) -> list[dict[str, Any]]:
    """Find active edges between the exact same (subject, relation_type, object) triple.

    Used for NON_EXCLUSIVE and OBJECT_EXCLUSIVE cardinalities where the
    subject-side query would be too broad.  Detects duplicates/corroborations
    only when both endpoints match.
    """
    if t_event is not None:
        query = """
        MATCH (s:Entity {title: $subject})-[e:RELATIONSHIP]->(o:Entity {title: $obj})
        WHERE e.relation_type = $relation_type
          AND e.t_valid_start <= $t_event
          AND e.t_valid_end > $t_event
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, o.title AS object_title
        """
        result = await session.run(
            query,
            subject=subject,
            relation_type=relation_type,
            obj=obj,
            infinity=INFINITY_ISO,
            t_event=t_event.isoformat(),
        )
    else:
        # Normal path: see run_subject_side_query — valid-time overlap is
        # delegated to the resolver so historical edges stay visible.
        query = """
        MATCH (s:Entity {title: $subject})-[e:RELATIONSHIP]->(o:Entity {title: $obj})
        WHERE e.relation_type = $relation_type
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, o.title AS object_title
        """
        result = await session.run(
            query, subject=subject, relation_type=relation_type, obj=obj,
            infinity=INFINITY_ISO,
        )

    records = []
    async for record in result:
        edge_data = dict(record["e"])
        edge_data["_object_title"] = record["object_title"]
        records.append(edge_data)
    return records


async def run_subject_any_type_query(
    session: "AsyncSession",
    subject: str,
    candidate_relation_type: str,
    t_event: datetime | None = None,
) -> list[dict[str, Any]]:
    """Cross-type subject-side query: subject's active edges of any *subject-exclusive* type.

    Used for SUBJECT_EXCLUSIVE and BOTH_EXCLUSIVE to surface edges whose
    relation type differs from the candidate's but may still semantically
    conflict (e.g. existing IS_CEO_OF vs candidate LEADS — same subject,
    different type names, potentially contradictory meaning).

    Only edges whose own cardinality claims subject-side exclusivity
    (SUBJECT_EXCLUSIVE or BOTH_EXCLUSIVE) are returned: a NON_EXCLUSIVE
    existing edge by definition coexists and cannot contest the candidate's
    exclusive slot, so including it would only feed false positives to the
    router.

    The candidate's own relation type is excluded to avoid self-comparison.
    """
    exclusive_cardinalities = [
        RelationCardinality.SUBJECT_EXCLUSIVE.value,
        RelationCardinality.BOTH_EXCLUSIVE.value,
    ]
    if t_event is not None:
        query = """
        MATCH (s:Entity {title: $subject})-[e:RELATIONSHIP]->(o:Entity)
        WHERE e.relation_type <> $candidate_relation_type
          AND e.cardinality IN $exclusive_cardinalities
          AND e.t_valid_start <= $t_event
          AND e.t_valid_end > $t_event
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, o.title AS object_title
        """
        result = await session.run(
            query,
            subject=subject,
            candidate_relation_type=candidate_relation_type,
            exclusive_cardinalities=exclusive_cardinalities,
            infinity=INFINITY_ISO,
            t_event=t_event.isoformat(),
        )
    else:
        # Normal path: see run_subject_side_query — valid-time overlap is
        # delegated to the resolver so historical edges stay visible.
        query = """
        MATCH (s:Entity {title: $subject})-[e:RELATIONSHIP]->(o:Entity)
        WHERE e.relation_type <> $candidate_relation_type
          AND e.cardinality IN $exclusive_cardinalities
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, o.title AS object_title
        """
        result = await session.run(
            query,
            subject=subject,
            candidate_relation_type=candidate_relation_type,
            exclusive_cardinalities=exclusive_cardinalities,
            infinity=INFINITY_ISO,
        )

    records = []
    async for record in result:
        edge_data = dict(record["e"])
        edge_data["_object_title"] = record["object_title"]
        records.append(edge_data)
    return records


async def run_object_any_type_query(
    session: "AsyncSession",
    subject: str,
    obj: str,
    candidate_relation_type: str,
    t_event: datetime | None = None,
) -> list[dict[str, Any]]:
    """Cross-type object-side query: object's incoming active edges of any *object-exclusive* type.

    Used for OBJECT_EXCLUSIVE and BOTH_EXCLUSIVE to surface edges whose
    relation type differs from the candidate's but may still semantically
    conflict (e.g. existing IS_PRESIDENT_OF vs candidate GOVERNS — different
    names, potentially same exclusive role).

    Only edges whose own cardinality claims object-side exclusivity
    (OBJECT_EXCLUSIVE or BOTH_EXCLUSIVE) are returned: a NON_EXCLUSIVE
    existing edge by definition coexists and cannot contest the candidate's
    exclusive slot on the object, so including it would only feed false
    positives to the router.

    The candidate's own relation type and the candidate's subject are excluded.
    """
    exclusive_cardinalities = [
        RelationCardinality.OBJECT_EXCLUSIVE.value,
        RelationCardinality.BOTH_EXCLUSIVE.value,
    ]
    if t_event is not None:
        query = """
        MATCH (s:Entity)-[e:RELATIONSHIP]->(o:Entity {title: $obj})
        WHERE e.relation_type <> $candidate_relation_type
          AND e.cardinality IN $exclusive_cardinalities
          AND s.title <> $subject
          AND e.t_valid_start <= $t_event
          AND e.t_valid_end > $t_event
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, s.title AS subject_title
        """
        result = await session.run(
            query,
            obj=obj,
            subject=subject,
            candidate_relation_type=candidate_relation_type,
            exclusive_cardinalities=exclusive_cardinalities,
            infinity=INFINITY_ISO,
            t_event=t_event.isoformat(),
        )
    else:
        # Normal path: see run_subject_side_query — valid-time overlap is
        # delegated to the resolver so historical edges stay visible.
        query = """
        MATCH (s:Entity)-[e:RELATIONSHIP]->(o:Entity {title: $obj})
        WHERE e.relation_type <> $candidate_relation_type
          AND e.cardinality IN $exclusive_cardinalities
          AND s.title <> $subject
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, s.title AS subject_title
        """
        result = await session.run(
            query,
            obj=obj,
            subject=subject,
            candidate_relation_type=candidate_relation_type,
            exclusive_cardinalities=exclusive_cardinalities,
            infinity=INFINITY_ISO,
        )

    records = []
    async for record in result:
        edge_data = dict(record["e"])
        edge_data["_subject_title"] = record["subject_title"]
        records.append(edge_data)
    return records


async def run_source_target_query(
    session: "AsyncSession",
    subject: str,
    obj: str,
    candidate_relation_type: str,
    t_event: datetime | None = None,
) -> list[dict[str, Any]]:
    """NON_EXCLUSIVE conflict query: find all active edges between (subject, obj).

    Unlike run_same_pair_query, this does NOT filter by relation_type.  Used
    for NON_EXCLUSIVE cardinality where the same pair can hold multiple relation
    types, but a candidate may still semantically conflict with an existing edge
    of a different type (e.g. WORKS_AT vs LEFT_COMPANY for the same pair).

    The candidate's own relation type is excluded so an edge is never compared
    against itself when it is already persisted.
    """
    if t_event is not None:
        query = """
        MATCH (s:Entity {title: $subject})-[e:RELATIONSHIP]->(o:Entity {title: $obj})
        WHERE e.relation_type <> $candidate_relation_type
          AND e.t_valid_start <= $t_event
          AND e.t_valid_end > $t_event
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, o.title AS object_title
        """
        result = await session.run(
            query,
            subject=subject,
            obj=obj,
            candidate_relation_type=candidate_relation_type,
            infinity=INFINITY_ISO,
            t_event=t_event.isoformat(),
        )
    else:
        # Normal path: see run_subject_side_query — valid-time overlap is
        # delegated to the resolver so historical edges stay visible.
        query = """
        MATCH (s:Entity {title: $subject})-[e:RELATIONSHIP]->(o:Entity {title: $obj})
        WHERE e.relation_type <> $candidate_relation_type
          AND e.t_tx_end = $infinity
        RETURN properties(e) AS e, o.title AS object_title
        """
        result = await session.run(
            query,
            subject=subject,
            obj=obj,
            candidate_relation_type=candidate_relation_type,
            infinity=INFINITY_ISO,
        )

    records = []
    async for record in result:
        edge_data = dict(record["e"])
        edge_data["_object_title"] = record["object_title"]
        records.append(edge_data)
    return records


# ---------------------------------------------------------------------------
# Intra-Batch Conflict Detection
# ---------------------------------------------------------------------------


def _is_temporally_active(
    rel: TemporalRelationship,
    t_event: datetime | None = None,
) -> bool:
    """Check if a relationship is temporally active (mirrors Neo4j query conditions)."""
    quad = rel.temporal_quad
    if quad is None:
        return True  # no temporal info, assume active

    if t_event is not None:
        # Late-arrival: check if active at t_event
        return (
            quad.t_valid_start <= t_event
            and quad.t_valid_end > t_event
            and quad.t_tx_start <= t_event
            and quad.t_tx_end > t_event
        )
    # Normal: edge is transactionally current (not retracted). Valid-time
    # overlap with the candidate is judged downstream by the resolver, so
    # historical edges (t_valid_end < INFINITY) remain visible as candidates
    # for CORROBORATION / EVOLUTION / CORRECTION over closed intervals.
    return quad.t_tx_end >= INFINITY


def find_intra_batch_subject_conflicts(
    candidate: TemporalRelationship,
    accepted_batch: list[TemporalRelationship],
    t_event: datetime | None = None,
) -> list[TemporalRelationship]:
    """Subject-side conflict detection within the current batch.

    Mirrors run_subject_side_query but checks in-memory batch relationships
    instead of Neo4j. Finds batch members where the same subject already holds
    the same relation type.
    """
    conflicts = []
    for rel in accepted_batch:
        if (
            rel.source == candidate.source
            and rel.relation_type == candidate.relation_type
            and rel.id != candidate.id
            and _is_temporally_active(rel, t_event)
        ):
            conflicts.append(rel)
    return conflicts


def find_intra_batch_object_conflicts(
    candidate: TemporalRelationship,
    accepted_batch: list[TemporalRelationship],
    t_event: datetime | None = None,
) -> list[TemporalRelationship]:
    """Object-side conflict detection within the current batch.

    Mirrors run_object_side_query but checks in-memory batch relationships.
    Finds batch members where a different subject holds this exclusive relation
    to the same object.
    """
    conflicts = []
    for rel in accepted_batch:
        if (
            rel.target == candidate.target
            and rel.relation_type == candidate.relation_type
            and rel.source != candidate.source
            and rel.id != candidate.id
            and _is_temporally_active(rel, t_event)
        ):
            conflicts.append(rel)
    return conflicts


def find_intra_batch_same_pair_conflicts(
    candidate: TemporalRelationship,
    accepted_batch: list[TemporalRelationship],
    t_event: datetime | None = None,
) -> list[TemporalRelationship]:
    """Same-pair conflict detection within the current batch.

    Mirrors run_same_pair_query but checks in-memory batch relationships.
    Finds batch members with the exact same (source, relation_type, target)
    triple.  Used for NON_EXCLUSIVE and OBJECT_EXCLUSIVE cardinalities where
    the full subject-side check would produce false positives.
    """
    conflicts = []
    for rel in accepted_batch:
        if (
            rel.source == candidate.source
            and rel.target == candidate.target
            and rel.relation_type == candidate.relation_type
            and rel.id != candidate.id
            and _is_temporally_active(rel, t_event)
        ):
            conflicts.append(rel)
    return conflicts


def find_intra_batch_source_target_conflicts(
    candidate: TemporalRelationship,
    accepted_batch: list[TemporalRelationship],
    t_event: datetime | None = None,
) -> list[TemporalRelationship]:
    """NON_EXCLUSIVE intra-batch conflict detection across all relation types.

    Mirrors run_source_target_query but checks in-memory batch relationships.
    Finds batch members that share the same (source, target) pair but have a
    *different* relation type, so the router can judge semantic compatibility.
    """
    conflicts = []
    for rel in accepted_batch:
        if (
            rel.source == candidate.source
            and rel.target == candidate.target
            and rel.relation_type != candidate.relation_type
            and rel.id != candidate.id
            and _is_temporally_active(rel, t_event)
        ):
            conflicts.append(rel)
    return conflicts


def find_intra_batch_subject_any_type_conflicts(
    candidate: TemporalRelationship,
    accepted_batch: list[TemporalRelationship],
    t_event: datetime | None = None,
) -> list[TemporalRelationship]:
    """Cross-type subject-side intra-batch detection.

    Mirrors run_subject_any_type_query. Finds batch members where the same
    subject holds any relation type other than the candidate's, restricted
    to edges that themselves carry subject-side exclusivity
    (SUBJECT_EXCLUSIVE or BOTH_EXCLUSIVE). NON_EXCLUSIVE batch edges cannot
    contest the candidate's exclusive slot and are skipped.
    """
    subject_exclusive_cardinalities = {
        RelationCardinality.SUBJECT_EXCLUSIVE,
        RelationCardinality.BOTH_EXCLUSIVE,
    }
    conflicts = []
    for rel in accepted_batch:
        if (
            rel.source == candidate.source
            and rel.relation_type != candidate.relation_type
            and rel.cardinality in subject_exclusive_cardinalities
            and rel.id != candidate.id
            and _is_temporally_active(rel, t_event)
        ):
            conflicts.append(rel)
    return conflicts


def find_intra_batch_object_any_type_conflicts(
    candidate: TemporalRelationship,
    accepted_batch: list[TemporalRelationship],
    t_event: datetime | None = None,
) -> list[TemporalRelationship]:
    """Cross-type object-side intra-batch detection.

    Mirrors run_object_any_type_query. Finds batch members where a different
    subject points to the same target with any relation type other than the
    candidate's, restricted to edges that themselves carry object-side
    exclusivity (OBJECT_EXCLUSIVE or BOTH_EXCLUSIVE). NON_EXCLUSIVE batch
    edges cannot contest the candidate's exclusive slot on the object and
    are skipped.
    """
    object_exclusive_cardinalities = {
        RelationCardinality.OBJECT_EXCLUSIVE,
        RelationCardinality.BOTH_EXCLUSIVE,
    }
    conflicts = []
    for rel in accepted_batch:
        if (
            rel.target == candidate.target
            and rel.source != candidate.source
            and rel.relation_type != candidate.relation_type
            and rel.cardinality in object_exclusive_cardinalities
            and rel.id != candidate.id
            and _is_temporally_active(rel, t_event)
        ):
            conflicts.append(rel)
    return conflicts


# ---------------------------------------------------------------------------
# Intra-Batch Resolution Actions
# ---------------------------------------------------------------------------


def _evolution_resolve_endpoints(
    a_valid_start: datetime,
    a_valid_end: datetime,
    b_valid_start: datetime,
) -> tuple[datetime | None, datetime]:
    """Resolve the EVOLUTION endpoints when either side may be unknown.

    Encodes the fallback table:

    | B.start                | A.end                  | A.start  | Action                                  |
    |------------------------|------------------------|----------|-----------------------------------------|
    | known                  | known                  | (any)    | A.end <- B.start                        |
    | known                  | unknown (INFINITY)     | (any)    | A.end <- B.start                        |
    | unknown (MINUS_INF)    | known                  | (any)    | B.start <- A.end (A unchanged)          |
    | unknown (MINUS_INF)    | unknown (INFINITY)     | known    | A.end <- A.start, B.start <- A.start    |
    | unknown                | unknown                | MINUS_INF| both collapse to MINUS_INFINITY         |

    Returns ``(new_a_end, new_b_start)``:
    - ``new_a_end is None`` means A should NOT be modified (case 3).
    - ``new_b_start`` is the value the caller must assign to B's
      ``t_valid_start`` (mutating the candidate in place).
    """
    b_start_known = b_valid_start != MINUS_INFINITY
    a_end_known = a_valid_end != INFINITY

    if b_start_known:
        return b_valid_start, b_valid_start
    if a_end_known:
        return None, a_valid_end
    # Case 4: both endpoints unknown -> collapse to A.start (which itself
    # may be MINUS_INFINITY when A is fully temporally unknown).
    return a_valid_start, a_valid_start


def apply_intra_batch_evolution(
    existing: TemporalRelationship,
    candidate: TemporalRelationship,
) -> None:
    """Evolution on a batch relationship: close A's valid-time end and
    align B's valid-time start, in-memory, following the fallback table
    in ``_evolution_resolve_endpoints``."""
    if not (existing.temporal_quad and candidate.temporal_quad):
        return
    quad_e = existing.temporal_quad
    quad_c = candidate.temporal_quad

    new_a_end, new_b_start = _evolution_resolve_endpoints(
        a_valid_start=quad_e.t_valid_start,
        a_valid_end=quad_e.t_valid_end,
        b_valid_start=quad_c.t_valid_start,
    )
    if new_a_end is not None:
        quad_e.close_valid_time(new_a_end)
    quad_c.t_valid_start = new_b_start

    logger.info(
        "ETCDR [INTRA-BATCH EVOLUTION]: batch edge %s -> A.t_valid_end=%s, "
        "candidate.t_valid_start=%s",
        existing.id,
        quad_e.t_valid_end,
        quad_c.t_valid_start,
    )


def apply_intra_batch_correction(
    existing: TemporalRelationship,
    t_now: datetime,
) -> None:
    """Correction on a batch relationship: retract it in-memory."""
    if existing.temporal_quad:
        existing.temporal_quad.retract(t_now)
    existing.status = "retracted"
    logger.info(
        "ETCDR [INTRA-BATCH CORRECTION]: Retracted batch edge %s",
        existing.id,
    )


def apply_intra_batch_corroboration(
    existing: TemporalRelationship,
    candidate: TemporalRelationship,
) -> None:
    """Corroboration on a batch relationship: increment support in-memory."""
    existing.support_count += 1
    existing.confidence = min(existing.confidence + 0.05, 1.0)
    logger.info(
        "ETCDR [INTRA-BATCH CORROBORATION]: Incremented support on batch edge %s",
        existing.id,
    )


# ---------------------------------------------------------------------------
# Top-K candidate ranking
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    """Plain cosine similarity. Returns 0.0 when either vector is missing/empty."""
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _is_same_pair(
    existing: TemporalRelationship,
    candidate: TemporalRelationship,
) -> bool:
    """True iff ``existing`` shares the exact (source, relation_type, target)
    triple of ``candidate``. Same-pair entries are duplicate-check candidates
    and bypass the cosine threshold in :func:`_topk_by_description`."""
    return (
        existing.source == candidate.source
        and existing.target == candidate.target
        and existing.relation_type == candidate.relation_type
    )


def _topk_by_description(
    candidate: TemporalRelationship,
    pool: list[tuple[TemporalRelationship, str]],
    top_k: int,
    cosine_threshold: float,
) -> list[tuple[TemporalRelationship, str, float]]:
    """Filter, sort and truncate the candidate-conflict pool.

    Parameters
    ----------
    candidate:
        The incoming edge whose ``description_embedding`` anchors the
        cosine ranking.
    pool:
        ``(existing, origin)`` tuples where ``origin`` is either ``"neo4j"``
        (edge persisted in the main DB) or ``"intra_batch"`` (edge living
        in the scratch ``etcdrbatch`` DB / in-memory accepted batch).
    top_k:
        Maximum number of entries to keep.
    cosine_threshold:
        Cosine floor below which non-same-pair entries are dropped.

    Returns
    -------
    list of ``(existing, origin, cosine)`` ordered by:

    1. Same-relation-type entries before cross-type (so an exact
       CORROBORATION candidate fires before any EVOLUTION on a
       cross-type neighbour).
    2. Cosine descending.
    3. Edge id ascending — stable tie-breaker for reproducible runs.

    Same-pair entries (source/target/relation_type all match the
    candidate's triple) bypass ``cosine_threshold``; they are duplicate
    checks and must always reach the LLM router.
    """
    cand_emb = candidate.description_embedding or []
    scored: list[tuple[float, bool, str, TemporalRelationship, str]] = []
    for existing, origin in pool:
        is_same_pair = _is_same_pair(existing, candidate)
        cosine = _cosine_similarity(cand_emb, existing.description_embedding)
        if not is_same_pair and cosine < cosine_threshold:
            continue
        same_rt = existing.relation_type == candidate.relation_type
        scored.append((cosine, same_rt, existing.id or "", existing, origin))
    # Sort key: same_rt first (True > False so -int), cosine desc (-cosine),
    # id asc (string compare).
    scored.sort(key=lambda x: (-int(x[1]), -x[0], x[2]))
    return [(rel, origin, cos) for cos, _, _, rel, origin in scored[:top_k]]


def _reduce_strategies(
    strategies: list[ResolutionStrategy],
) -> ResolutionStrategy:
    """Collapse the per-top-K strategies to a single candidate-level outcome.

    Priority (highest wins):

    1. CORROBORATION — candidate was absorbed by a duplicate.
    2. DISAGREEMENT — candidate is kept but disputed.
    3. CORRECTION — at least one existing edge was retracted.
    4. EVOLUTION — at least one existing edge's valid_end was closed.
    5. NEW_EDGE — no impact (default).
    """
    if not strategies:
        return ResolutionStrategy.NEW_EDGE
    for prio in (
        ResolutionStrategy.CORROBORATION,
        ResolutionStrategy.DISAGREEMENT,
        ResolutionStrategy.CORRECTION,
        ResolutionStrategy.EVOLUTION,
    ):
        if prio in strategies:
            return prio
    return ResolutionStrategy.NEW_EDGE


# ---------------------------------------------------------------------------
# Cardinality-aware retrieval (Phase A + Phase B share the same Cypher)
# ---------------------------------------------------------------------------


async def _run_cardinality_queries(
    session: "AsyncSession",
    candidate: TemporalRelationship,
    cardinality: RelationCardinality,
    query_time: datetime | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Run the cardinality-aware Cypher passes against ``session``.

    Returns ``(subject_records, object_records, ran_object_query)``. The
    same helper is used for Phase A (main DB session) and Phase B
    (scratch ``etcdrbatch`` DB session) since both share the
    ``(:Entity)-[:RELATIONSHIP]->(:Entity)`` schema.
    """
    subject_records: list[dict[str, Any]] = []
    object_records: list[dict[str, Any]] = []
    ran_object_query = False

    if cardinality == RelationCardinality.NON_EXCLUSIVE:
        subject_records = await run_same_pair_query(
            session=session,
            subject=candidate.source,
            relation_type=candidate.relation_type,
            obj=candidate.target,
            t_event=query_time,
        )
        subject_records += await run_source_target_query(
            session=session,
            subject=candidate.source,
            obj=candidate.target,
            candidate_relation_type=candidate.relation_type,
            t_event=query_time,
        )
    elif cardinality == RelationCardinality.SUBJECT_EXCLUSIVE:
        subject_records = await run_subject_side_query(
            session=session,
            subject=candidate.source,
            relation_type=candidate.relation_type,
            t_event=query_time,
        )
        subject_records += await run_subject_any_type_query(
            session=session,
            subject=candidate.source,
            candidate_relation_type=candidate.relation_type,
            t_event=query_time,
        )
    elif cardinality == RelationCardinality.OBJECT_EXCLUSIVE:
        subject_records = await run_same_pair_query(
            session=session,
            subject=candidate.source,
            relation_type=candidate.relation_type,
            obj=candidate.target,
            t_event=query_time,
        )
        ran_object_query = True
        object_records = await run_object_side_query(
            session=session,
            subject=candidate.source,
            relation_type=candidate.relation_type,
            obj=candidate.target,
            t_event=query_time,
        )
        object_records += await run_object_any_type_query(
            session=session,
            subject=candidate.source,
            obj=candidate.target,
            candidate_relation_type=candidate.relation_type,
            t_event=query_time,
        )
    else:  # BOTH_EXCLUSIVE
        subject_records = await run_subject_side_query(
            session=session,
            subject=candidate.source,
            relation_type=candidate.relation_type,
            t_event=query_time,
        )
        subject_records += await run_subject_any_type_query(
            session=session,
            subject=candidate.source,
            candidate_relation_type=candidate.relation_type,
            t_event=query_time,
        )
        ran_object_query = True
        object_records = await run_object_side_query(
            session=session,
            subject=candidate.source,
            relation_type=candidate.relation_type,
            obj=candidate.target,
            t_event=query_time,
        )
        object_records += await run_object_any_type_query(
            session=session,
            subject=candidate.source,
            obj=candidate.target,
            candidate_relation_type=candidate.relation_type,
            t_event=query_time,
        )

    return subject_records, object_records, ran_object_query


# ---------------------------------------------------------------------------
# Decision Router
# ---------------------------------------------------------------------------


async def _route_one(
    candidate: TemporalRelationship,
    existing: TemporalRelationship,
    conflict_type: str,
    config: BTGraphRAGConfig,
    model: "LLMCompletion | None",
) -> tuple[ResolutionStrategy, float]:
    """Route a single (candidate, existing) pair through the Decision Router.

    Falls back to :func:`_heuristic_route` when no LLM is configured, and
    downgrades to DISAGREEMENT when the LLM confidence drops below
    :attr:`BTGraphRAGConfig.etcdr_confidence_threshold`.
    """
    if model is None:
        return _heuristic_route(candidate, existing)
    strategy, confidence = await _llm_route(
        candidate, existing, conflict_type, model,
    )
    if confidence < config.etcdr_confidence_threshold:
        logger.info(
            "ETCDR: Low confidence (%.2f) → defaulting to DISAGREEMENT",
            confidence,
        )
        return ResolutionStrategy.DISAGREEMENT, confidence
    return strategy, confidence


async def route_conflict(
    candidate: TemporalRelationship,
    conflict_result: ConflictResult,
    config: BTGraphRAGConfig,
    model: "LLMCompletion | None" = None,
) -> tuple[ResolutionStrategy, float]:
    """Classify the conflict and return (strategy, confidence).

    Uses LLM when available; otherwise applies deterministic heuristics.
    Falls back to DISAGREEMENT when confidence < threshold.
    """
    if not conflict_result.has_conflicts:
        return ResolutionStrategy.CORROBORATION, 1.0  # no real conflict

    # Merge Neo4j and intra-batch conflicts for routing
    all_subject = (
        conflict_result.subject_conflicts
        + conflict_result.intra_batch_subject_conflicts
    )
    all_object = (
        conflict_result.object_conflicts
        + conflict_result.intra_batch_object_conflicts
    )
    all_conflicts = all_subject + all_object
    conflict_type = "SUBJECT_SIDE" if all_subject else "OBJECT_SIDE"
    if all_subject and all_object:
        conflict_type = "BOTH_SIDES"

    if model is not None:
        strategy, confidence = await _llm_route(
            candidate, all_conflicts[0], conflict_type, model
        )
        if confidence < config.etcdr_confidence_threshold:
            logger.info(
                "ETCDR: Low confidence (%.2f) → defaulting to DISAGREEMENT",
                confidence,
            )
            return ResolutionStrategy.DISAGREEMENT, confidence
        return strategy, confidence

    # Heuristic fallback: use temporal ordering
    return _heuristic_route(candidate, all_conflicts[0])


async def _llm_route(
    candidate: TemporalRelationship,
    existing: TemporalRelationship,
    conflict_type: str,
    model: "LLMCompletion",
) -> tuple[ResolutionStrategy, float]:
    """Use LLM to classify the conflict resolution strategy."""
    from graphrag_llm.utils import CompletionMessagesBuilder

    def _ts(v: float | datetime | None) -> str:
        if v == INFINITY:
            return "present"
        if v is None or v == MINUS_INFINITY:
            return "unknown"
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%d")
        return "present"

    quad_c = candidate.temporal_quad
    quad_e = existing.temporal_quad

    cardinality = candidate.cardinality
    prompt = DECISION_ROUTER_PROMPT.format(
        existing_subject=existing.source,
        existing_object=existing.target,
        existing_relation_type=existing.relation_type,
        existing_t_valid_start=_ts(quad_e.t_valid_start if quad_e else None),
        existing_t_valid_end=_ts(quad_e.t_valid_end if quad_e else None),
        existing_description=existing.description or "",
        existing_source=existing.provenance[0].source_url if existing.provenance else "unknown",
        existing_trust=existing.provenance[0].trust_score if existing.provenance else 1.0,
        candidate_subject=candidate.source,
        candidate_object=candidate.target,
        candidate_relation_type=candidate.relation_type,
        candidate_t_valid_start=_ts(quad_c.t_valid_start if quad_c else None),
        candidate_t_valid_end=_ts(quad_c.t_valid_end if quad_c else None),
        candidate_description=candidate.description or "",
        candidate_source=candidate.provenance[0].source_url if candidate.provenance else "unknown",
        candidate_trust=candidate.provenance[0].trust_score if candidate.provenance else 1.0,
        cardinality=cardinality.value,
        cardinality_explanation=CARDINALITY_EXPLANATIONS.get(cardinality, ""),
        conflict_type=conflict_type,
    )

    messages = CompletionMessagesBuilder().add_user_message(prompt).build()
    response = await model.completion_async(messages=messages)
    text = response.content.strip()

    strategy = ResolutionStrategy.DISAGREEMENT
    confidence = 0.5

    for line in text.splitlines():
        if line.startswith("STRATEGY:"):
            strat_str = line.split(":", 1)[1].strip().upper()
            try:
                strategy = ResolutionStrategy(strat_str)
            except ValueError:
                pass
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    return strategy, confidence


def _heuristic_route(
    candidate: TemporalRelationship,
    existing: TemporalRelationship,
) -> tuple[ResolutionStrategy, float]:
    """Deterministic heuristic routing when no LLM is available."""
    quad_c = candidate.temporal_quad
    quad_e = existing.temporal_quad

    if quad_c is None or quad_e is None:
        return ResolutionStrategy.DISAGREEMENT, 0.5

    c_start = quad_c.t_valid_start
    e_start = quad_e.t_valid_start

    # If candidate starts after existing → likely Evolution
    if c_start > e_start:
        return ResolutionStrategy.EVOLUTION, 0.75

    # If candidate starts before existing and existing trust is low → Correction
    if c_start < e_start and candidate.confidence > existing.confidence:
        return ResolutionStrategy.CORRECTION, 0.65

    # Description similarity check for Corroboration
    if candidate.description and existing.description:
        desc_c = set(candidate.description.lower().split())
        desc_e = set(existing.description.lower().split())
        if desc_c and desc_e:
            overlap = len(desc_c & desc_e) / len(desc_c | desc_e)
            if overlap > 0.6:
                return ResolutionStrategy.CORROBORATION, 0.8

    return ResolutionStrategy.DISAGREEMENT, 0.5


# ---------------------------------------------------------------------------
# Resolution Actions in Neo4j
# ---------------------------------------------------------------------------


async def apply_evolution(
    session: "AsyncSession",
    existing_edge_id: str,
    candidate: TemporalRelationship,
) -> None:
    """Evolution: close existing edge's valid-time, align candidate's start.

    The world genuinely changed: old edge was true, new edge is now true.
    When either A.t_valid_end (existing) or B.t_valid_start (candidate) is
    unknown, the fallback table in ``_evolution_resolve_endpoints`` decides
    what to write. The candidate's ``t_valid_start`` is mutated in-place so
    the caller persists it with the resolved value.
    """
    if candidate.temporal_quad is None:
        return

    # Read A's current temporal endpoints so we can apply the fallback table.
    read_result = await session.run(
        """
        MATCH ()-[e]->()
        WHERE e.id = $edge_id
        RETURN e.t_valid_start AS start, e.t_valid_end AS end
        """,
        edge_id=existing_edge_id,
    )
    record = await read_result.single()
    if record is None:
        logger.warning(
            "ETCDR [EVOLUTION]: existing edge %s not found; skipping",
            existing_edge_id,
        )
        return

    a_valid_start = (
        datetime.fromisoformat(record["start"]) if record["start"] else MINUS_INFINITY
    )
    a_valid_end = (
        datetime.fromisoformat(record["end"]) if record["end"] else INFINITY
    )

    new_a_end, new_b_start = _evolution_resolve_endpoints(
        a_valid_start=a_valid_start,
        a_valid_end=a_valid_end,
        b_valid_start=candidate.temporal_quad.t_valid_start,
    )

    # Always mutate the candidate so downstream persistence uses the
    # resolved t_valid_start (cases 3 and 4 may shift it).
    candidate.temporal_quad.t_valid_start = new_b_start

    if new_a_end is not None:
        await session.run(
            """
            MATCH ()-[e]->()
            WHERE e.id = $edge_id
            SET e.t_valid_end = $t_valid_end
            """,
            edge_id=existing_edge_id,
            t_valid_end=new_a_end.isoformat(),
        )

    logger.info(
        "ETCDR [EVOLUTION]: edge %s -> A.t_valid_end=%s, candidate.t_valid_start=%s",
        existing_edge_id,
        new_a_end.isoformat() if new_a_end is not None else "(unchanged)",
        candidate.temporal_quad.t_valid_start,
    )


async def apply_correction(
    session: "AsyncSession",
    existing_edge_id: str,
    t_now: datetime,
) -> None:
    """Correction: retroactively invalidate the existing edge (set t_tx_end).

    The existing edge was wrong; preserve it for audit but stop believing it.
    """
    await session.run(
        """
        MATCH ()-[e]->()
        WHERE e.id = $edge_id
        SET e.t_tx_end = $t_tx_end
        """,
        edge_id=existing_edge_id,
        t_tx_end=t_now.isoformat(),
    )
    logger.info(
        "ETCDR [CORRECTION]: Retroactively invalidated edge %s at tx_time %s",
        existing_edge_id,
        t_now,
    )


async def apply_corroboration(
    session: "AsyncSession",
    existing_edge_id: str,
    candidate: TemporalRelationship,
) -> None:
    """Corroboration: increment support count and append provenance.

    No new edge is created; the existing edge is strengthened.
    """
    new_provenance = (
        [p.to_dict() for p in candidate.provenance] if candidate.provenance else []
    )
    await session.run(
        """
        MATCH ()-[e]->()
        WHERE e.id = $edge_id
        SET e.support_count = e.support_count + 1,
            e.confidence = CASE
                WHEN e.confidence + 0.05 > 1.0 THEN 1.0
                ELSE e.confidence + 0.05
            END
        """,
        edge_id=existing_edge_id,
        provenance=str(new_provenance),
    )
    logger.info(
        "ETCDR [CORROBORATION]: Incremented support_count on edge %s",
        existing_edge_id,
    )


# ---------------------------------------------------------------------------
# Main ETCDR Entry Point
# ---------------------------------------------------------------------------


async def detect_and_resolve(
    candidate: TemporalRelationship,
    session: "AsyncSession",
    config: BTGraphRAGConfig,
    model: "LLMCompletion | None" = None,
    is_late_arrival: bool = False,
    t_event: datetime | None = None,
    edge_index: int = -1,
    accepted_batch: list[TemporalRelationship] | None = None,
    batch_db: "ETCDRBatchDB | None" = None,
) -> ConflictResult:
    """Run cardinality-aware conflict detection and resolve sequentially.

    Performs two-phase conflict detection — Phase A against the main
    BT-GraphRAG DB (``session``) and Phase B against the scratch ETCDR
    batch DB (``batch_db``) when provided, or the in-memory
    ``accepted_batch`` list as a fallback. Both phases share the exact
    cardinality-aware Cypher of :func:`_run_cardinality_queries` since
    the scratch DB mirrors the main schema.

    Each phase's structurally-filtered pool is ranked **independently**
    by cosine similarity between the candidate's ``description_embedding``
    and each pool member's, truncated to ``config.etcdr_topk`` and
    floored by ``config.etcdr_cosine_threshold``. Phase A and Phase B
    each get their own top-K budget so a dense main graph cannot crowd
    intra-batch contradictions out of the router. Same-pair entries
    (those sharing the candidate's exact (source, relation_type, target)
    triple) bypass the cosine floor — duplicate checks must always reach
    the router.

    The combined top-K (Phase A first, then Phase B — persisted state is
    "older" than the current batch) is routed *sequentially* through the
    Decision Router, with each strategy applied against the right
    backing store immediately:

    - **CORROBORATION**: support++ on the existing edge; loop **breaks**
      (candidate absorbed into the duplicate).
    - **EVOLUTION**: close the existing edge's valid_end; continue.
    - **CORRECTION**: retract the existing edge's tx_end and remember its
      valid-time for the candidate to inherit; continue.
    - **DISAGREEMENT**: mark the candidate as disputed; continue.
    - **NEW_EDGE**: no mutation; continue.

    The single ``conflict_result.strategy`` reported back is the
    highest-priority outcome across the loop (see :func:`_reduce_strategies`).

    Query strategy depends on the relation's cardinality:
    - **NON_EXCLUSIVE**: same-pair + cross-type same-(source,target).
    - **SUBJECT_EXCLUSIVE**: subject-side (same-type + cross-type).
    - **OBJECT_EXCLUSIVE**: same-pair + object-side (same-type + cross-type).
    - **BOTH_EXCLUSIVE**: full subject-side + full object-side passes.

    Args:
        candidate: The new relationship. Must carry
            ``description_embedding`` so the top-K cosine ranking works.
        session: Neo4j async session bound to the main BT-GraphRAG DB.
        config: BT-GraphRAG configuration (top-K, cosine threshold,
            confidence threshold all read from here).
        model: Optional LLM for the Decision Router. When ``None`` the
            heuristic fallback is used.
        is_late_arrival: Whether this document is a late arrival.
        t_event: Event timestamp for late-arrival historical queries.
        edge_index: Index of the edge being processed (for display).
        accepted_batch: Running list of previously accepted relationships.
            Used as the in-memory Phase B source when ``batch_db`` is
            ``None``.
        batch_db: Optional scratch DB holding the running accepted batch
            with a relationship vector index. When provided, Phase B
            retrieval and mutation flow through it instead of
            ``accepted_batch``.

    Returns:
        ConflictResult with the reduced strategy and the per-side
        conflict lists for downstream reporting.
    """
    t_now = utcnow()
    query_time = t_event if is_late_arrival and t_event else None

    cardinality = RelationCardinality(
        config.get_cardinality(candidate.relation_type)
    )
    candidate.cardinality = cardinality

    # --- Diagnostic header ---
    quad_c = candidate.temporal_quad
    tv_start = (
        quad_c.t_valid_start.strftime("%Y-%m-%d")
        if quad_c and quad_c.t_valid_start else "?"
    )
    tv_end = (
        "INF"
        if (quad_c and quad_c.t_valid_end >= INFINITY)
        else (quad_c.t_valid_end.strftime("%Y-%m-%d") if quad_c else "?")
    )
    prefix = f"      [{edge_index}]" if edge_index >= 0 else "      "
    print(
        f"{prefix} Candidate: ({candidate.source[:25]}) "
        f"-[{candidate.relation_type[:25]}]-> ({candidate.target[:25]})"
    )
    print(
        f"{prefix}   Valid=[{tv_start} -> {tv_end}]  "
        f"conf={candidate.confidence:.2f}  cardinality={cardinality.value}"
    )
    if is_late_arrival:
        print(
            f"{prefix}   LATE ARRIVAL — querying historical state at "
            f"t_event={t_event}"
        )

    # --- Phase A: cardinality-aware retrieval against the main DB ---
    (
        subject_records_a,
        object_records_a,
        ran_object_query,
    ) = await _run_cardinality_queries(
        session=session,
        candidate=candidate,
        cardinality=cardinality,
        query_time=query_time,
    )

    # Phase A diagnostic
    if cardinality == RelationCardinality.NON_EXCLUSIVE:
        query_label = "source-target (same-pair + cross-type)"
    elif cardinality == RelationCardinality.OBJECT_EXCLUSIVE:
        query_label = "same-pair + object-side"
    else:
        query_label = "subject-side + cross-type"
    print(
        f"{prefix}   Sub-query S ({query_label}): "
        f"{len(subject_records_a)} conflict(s)"
    )
    for i, rec in enumerate(subject_records_a[:3]):
        obj_title = rec.get("_object_title", "?")
        rel_type = rec.get("relation_type", "?")
        desc = str(rec.get("description", ""))[:40]
        print(
            f"{prefix}     S[{i}]: -[{rel_type[:20]}]-> "
            f"({obj_title[:25]})  desc='{desc}'"
        )
    if len(subject_records_a) > 3:
        print(f"{prefix}     ... and {len(subject_records_a) - 3} more")

    if ran_object_query:
        print(
            f"{prefix}   Sub-query O (object-side):  "
            f"{len(object_records_a)} conflict(s)"
        )
        for i, rec in enumerate(object_records_a[:3]):
            subj_title = rec.get("_subject_title", "?")
            desc = str(rec.get("description", ""))[:40]
            print(
                f"{prefix}     O[{i}]: ({subj_title[:25]}) -> "
                f" desc='{desc}'"
            )
        if len(object_records_a) > 3:
            print(f"{prefix}     ... and {len(object_records_a) - 3} more")
    else:
        print(
            f"{prefix}   Sub-query O: SKIPPED "
            f"(cardinality={cardinality.value} does not require object-side check)"
        )

    # --- Materialise Phase A records as TemporalRelationship objects ---
    def _to_temporal_rel(
        record: dict[str, Any], is_object_side: bool = False,
    ) -> TemporalRelationship:
        quad = TemporalStateQuad(
            t_valid_start=datetime.fromisoformat(record["t_valid_start"])
            if record.get("t_valid_start")
            else t_now,
            t_valid_end=datetime.fromisoformat(record["t_valid_end"])
            if record.get("t_valid_end")
            else INFINITY,
            t_tx_start=datetime.fromisoformat(record["t_tx_start"])
            if record.get("t_tx_start")
            else t_now,
            t_tx_end=datetime.fromisoformat(record["t_tx_end"])
            if record.get("t_tx_end")
            else INFINITY,
        )
        return TemporalRelationship(
            id=record.get("id", ""),
            source=(
                record.get("_subject_title", candidate.source)
                if is_object_side else candidate.source
            ),
            target=record.get("_object_title", candidate.target),
            # Preserve the existing edge's own relation_type (may differ
            # from the candidate's for cross-type conflicts).
            relation_type=record.get("relation_type") or candidate.relation_type,
            description=record.get("description"),
            weight=record.get("weight", 1.0),
            confidence=record.get("confidence", 1.0),
            temporal_quad=quad,
            cardinality=cardinality,
            description_embedding=record.get("description_embedding"),
        )

    subject_conflicts = [_to_temporal_rel(r, False) for r in subject_records_a]
    object_conflicts = [_to_temporal_rel(r, True) for r in object_records_a]

    # --- Phase B: retrieve intra-batch conflicts ---
    intra_subj: list[TemporalRelationship] = []
    intra_obj: list[TemporalRelationship] = []
    phase_b_source: str = "none"

    if batch_db is not None:
        async with batch_db.session() as b_session:
            (
                subject_records_b,
                object_records_b,
                _,
            ) = await _run_cardinality_queries(
                session=b_session,
                candidate=candidate,
                cardinality=cardinality,
                query_time=query_time,
            )
        intra_subj = [_to_temporal_rel(r, False) for r in subject_records_b]
        intra_obj = [_to_temporal_rel(r, True) for r in object_records_b]
        phase_b_source = f"scratch DB '{batch_db.db_name}'"
    elif accepted_batch:
        intra_subj, intra_obj = _intra_batch_inmemory_conflicts(
            candidate, accepted_batch, cardinality, query_time,
        )
        phase_b_source = f"in-memory batch ({len(accepted_batch)} edges)"

    # Phase B diagnostic
    if cardinality == RelationCardinality.NON_EXCLUSIVE:
        ib_label = "source-target (same-pair + cross-type)"
    elif cardinality == RelationCardinality.SUBJECT_EXCLUSIVE:
        ib_label = "subject-side (same-type + cross-type)"
    elif cardinality == RelationCardinality.OBJECT_EXCLUSIVE:
        ib_label = "same-pair + object-side (same-type + cross-type)"
    else:  # BOTH_EXCLUSIVE
        ib_label = "subject+object (same-type + cross-type)"
    if intra_subj or intra_obj:
        print(
            f"{prefix}   Intra-batch S ({ib_label}, src={phase_b_source}): "
            f"{len(intra_subj)} conflict(s)"
        )
        for i, rel in enumerate(intra_subj[:3]):
            print(
                f"{prefix}     IB-S[{i}]: -[{rel.relation_type[:20]}]-> "
                f"({rel.target[:25]})  desc='{(rel.description or '')[:40]}'"
            )
        if intra_obj:
            print(
                f"{prefix}   Intra-batch O (object-side, src={phase_b_source}): "
                f"{len(intra_obj)} conflict(s)"
            )
            for i, rel in enumerate(intra_obj[:3]):
                print(
                    f"{prefix}     IB-O[{i}]: ({rel.source[:25]}) "
                    f"-[{rel.relation_type[:20]}]->  "
                    f"desc='{(rel.description or '')[:40]}'"
                )
    elif phase_b_source != "none":
        print(f"{prefix}   Intra-batch: no conflicts in {phase_b_source}")

    # --- Build conflict_result ---
    conflict_result = ConflictResult(
        candidate=candidate,
        subject_conflicts=subject_conflicts,
        object_conflicts=object_conflicts,
        intra_batch_subject_conflicts=intra_subj,
        intra_batch_object_conflicts=intra_obj,
    )

    if not conflict_result.has_conflicts:
        conflict_result.strategy = ResolutionStrategy.NEW_EDGE
        conflict_result.confidence = 1.0
        print(f"{prefix}   Result: NO CONFLICT — insert as new edge")
        _log_resolution(
            candidate=candidate,
            existing=None,
            strategy=ResolutionStrategy.NEW_EDGE,
            confidence=1.0,
            phase="none",
            conflict_type="NONE",
        )
        return conflict_result

    # --- Build per-phase pools for top-K ranking ---
    # Phase A (Neo4j-persisted) and Phase B (intra-batch) each get an
    # independent ``etcdr_topk`` budget so a dense main graph cannot
    # crowd intra-batch contradictions out of the router.
    neo4j_pool: list[tuple[TemporalRelationship, str]] = [
        (rel, "neo4j") for rel in subject_conflicts + object_conflicts
    ]
    intra_pool: list[tuple[TemporalRelationship, str]] = [
        (rel, "intra_batch") for rel in intra_subj + intra_obj
    ]

    # Conflict-type label used inside the DECISION_ROUTER prompt
    _all_subj = subject_conflicts + intra_subj
    _all_obj = object_conflicts + intra_obj
    conflict_type_label = "SUBJECT_SIDE" if _all_subj else "OBJECT_SIDE"
    if _all_subj and _all_obj:
        conflict_type_label = "BOTH_SIDES"

    # --- Top-K ranking per phase (cosine on description_embedding) ---
    topk_a = _topk_by_description(
        candidate=candidate,
        pool=neo4j_pool,
        top_k=config.etcdr_topk,
        cosine_threshold=config.etcdr_cosine_threshold,
    )
    topk_b = _topk_by_description(
        candidate=candidate,
        pool=intra_pool,
        top_k=config.etcdr_topk,
        cosine_threshold=config.etcdr_cosine_threshold,
    )
    # Phase A first, then Phase B: persisted state is "older" than the
    # current batch, so mutations on it happen before intra-batch ones.
    # A CORROBORATION in either phase breaks the loop; sequential mutations
    # within each phase preserve the post-resolution view for the next pair.
    topk = topk_a + topk_b

    print(
        f"{prefix}   CONFLICT DETECTED — "
        f"top-K[Phase A]={len(topk_a)}/{len(neo4j_pool)}, "
        f"top-K[Phase B]={len(topk_b)}/{len(intra_pool)} "
        f"(cosine ≥ {config.etcdr_cosine_threshold:.2f}, "
        f"budget top_k={config.etcdr_topk} per phase)"
    )

    if not topk:
        # Structural pool was non-empty but cosine threshold filtered
        # everything out — treat the candidate as a genuinely new edge.
        conflict_result.strategy = ResolutionStrategy.NEW_EDGE
        conflict_result.confidence = 1.0
        print(f"{prefix}   Result: NO TOP-K MATCH — insert as new edge")
        _log_resolution(
            candidate=candidate,
            existing=None,
            strategy=ResolutionStrategy.NEW_EDGE,
            confidence=1.0,
            phase="none",
            conflict_type=conflict_type_label,
        )
        return conflict_result

    # --- Per-origin apply helpers (close over session / batch_db) ---
    async def _apply_evolution_for(
        existing: TemporalRelationship, origin: str,
    ) -> None:
        if not existing.id:
            return
        if origin == "neo4j":
            await apply_evolution(session, existing.id, candidate)
        elif origin == "intra_batch":
            if batch_db is not None:
                async with batch_db.session() as b_session:
                    await apply_evolution(b_session, existing.id, candidate)
            else:
                apply_intra_batch_evolution(existing, candidate)

    async def _apply_correction_for(
        existing: TemporalRelationship, origin: str,
    ) -> None:
        if not existing.id:
            return
        if origin == "neo4j":
            await apply_correction(session, existing.id, t_now)
        elif origin == "intra_batch":
            if batch_db is not None:
                async with batch_db.session() as b_session:
                    await apply_correction(b_session, existing.id, t_now)
            else:
                apply_intra_batch_correction(existing, t_now)

    async def _apply_corroboration_for(
        existing: TemporalRelationship, origin: str,
    ) -> None:
        if not existing.id:
            return
        if origin == "neo4j":
            await apply_corroboration(session, existing.id, candidate)
        elif origin == "intra_batch":
            if batch_db is not None:
                async with batch_db.session() as b_session:
                    await apply_corroboration(b_session, existing.id, candidate)
            else:
                apply_intra_batch_corroboration(existing, candidate)

    # --- Sequential routing + per-existing apply ---
    per_existing_strategies: list[ResolutionStrategy] = []
    per_existing_confidences: list[float] = []
    candidate_inherits_from: TemporalRelationship | None = None

    for idx, (existing, origin, cosine) in enumerate(topk):
        strategy, confidence = await _route_one(
            candidate=candidate,
            existing=existing,
            conflict_type=conflict_type_label,
            config=config,
            model=model,
        )
        per_existing_strategies.append(strategy)
        per_existing_confidences.append(confidence)

        print(
            f"{prefix}   Top-K[{idx}] ({origin}, cosine={cosine:.3f}, "
            f"rt='{existing.relation_type[:20]}') → {strategy.value} "
            f"conf={confidence:.2f}"
        )

        if strategy == ResolutionStrategy.CORROBORATION:
            await _apply_corroboration_for(existing, origin)
            _log_resolution(
                candidate, existing, strategy, confidence,
                origin, conflict_type_label,
            )
            print(
                f"{prefix}     Action: support++ on "
                f"'{(existing.id or '')[:12]}...' (origin={origin}) — "
                f"candidate absorbed, stopping loop"
            )
            break

        elif strategy == ResolutionStrategy.EVOLUTION:
            await _apply_evolution_for(existing, origin)
            _log_resolution(
                candidate, existing, strategy, confidence,
                origin, conflict_type_label,
            )
            print(
                f"{prefix}     Action: closed valid_end on "
                f"'{(existing.id or '')[:12]}...' (origin={origin})"
            )

        elif strategy == ResolutionStrategy.CORRECTION:
            await _apply_correction_for(existing, origin)
            if candidate_inherits_from is None:
                candidate_inherits_from = existing
            _log_resolution(
                candidate, existing, strategy, confidence,
                origin, conflict_type_label,
            )
            print(
                f"{prefix}     Action: retracted tx_end on "
                f"'{(existing.id or '')[:12]}...' (origin={origin})"
            )

        elif strategy == ResolutionStrategy.DISAGREEMENT:
            candidate.status = "disputed"
            _log_resolution(
                candidate, existing, strategy, confidence,
                origin, conflict_type_label,
            )
            print(f"{prefix}     Action: candidate marked DISPUTED")

        elif strategy == ResolutionStrategy.NEW_EDGE:
            # Router decided the retrieved edge is an independent fact
            # (different exclusivity slot, non-overlapping role, …). No
            # mutation to either side; record it so the log shows the
            # Router actively dismissed the conflict.
            _log_resolution(
                candidate, existing, strategy, confidence,
                origin, conflict_type_label,
            )
            print(
                f"{prefix}     Action: Router judged independent — "
                f"no mutation to '{(existing.id or '')[:12]}...'"
            )

    # Candidate inherits valid-time from the first CORRECTED existing edge
    # (skipped if a CORROBORATION absorbed the candidate earlier).
    if (
        candidate_inherits_from is not None
        and candidate.temporal_quad is not None
        and candidate_inherits_from.temporal_quad is not None
    ):
        candidate.temporal_quad.t_valid_start = (
            candidate_inherits_from.temporal_quad.t_valid_start
        )
        candidate.temporal_quad.t_valid_end = (
            candidate_inherits_from.temporal_quad.t_valid_end
        )
        print(
            f"{prefix}   Candidate inherited valid-time from corrected edge "
            f"'{(candidate_inherits_from.id or '')[:12]}...'"
        )

    # --- Reduce per-existing strategies to one candidate-level outcome ---
    final_strategy = _reduce_strategies(per_existing_strategies)
    final_confidence = (
        max(per_existing_confidences) if per_existing_confidences else 0.0
    )
    conflict_result.strategy = final_strategy
    conflict_result.confidence = final_confidence

    print(
        f"{prefix}   Final strategy: {final_strategy.value} "
        f"confidence={final_confidence:.2f}"
    )

    return conflict_result


def _intra_batch_inmemory_conflicts(
    candidate: TemporalRelationship,
    accepted_batch: list[TemporalRelationship],
    cardinality: RelationCardinality,
    query_time: datetime | None,
) -> tuple[list[TemporalRelationship], list[TemporalRelationship]]:
    """In-memory Phase B retrieval used when no ``ETCDRBatchDB`` is configured.

    Mirrors the cardinality dispatch of :func:`_run_cardinality_queries`
    but operates on the running ``accepted_batch`` list via the
    ``find_intra_batch_*`` Python helpers.
    """
    intra_subj: list[TemporalRelationship] = []
    intra_obj: list[TemporalRelationship] = []
    if cardinality == RelationCardinality.NON_EXCLUSIVE:
        intra_subj = find_intra_batch_same_pair_conflicts(
            candidate, accepted_batch, t_event=query_time,
        )
        intra_subj += find_intra_batch_source_target_conflicts(
            candidate, accepted_batch, t_event=query_time,
        )
    elif cardinality == RelationCardinality.SUBJECT_EXCLUSIVE:
        intra_subj = find_intra_batch_subject_conflicts(
            candidate, accepted_batch, t_event=query_time,
        )
        intra_subj += find_intra_batch_subject_any_type_conflicts(
            candidate, accepted_batch, t_event=query_time,
        )
    elif cardinality == RelationCardinality.OBJECT_EXCLUSIVE:
        intra_subj = find_intra_batch_same_pair_conflicts(
            candidate, accepted_batch, t_event=query_time,
        )
        intra_obj = find_intra_batch_object_conflicts(
            candidate, accepted_batch, t_event=query_time,
        )
        intra_obj += find_intra_batch_object_any_type_conflicts(
            candidate, accepted_batch, t_event=query_time,
        )
    else:  # BOTH_EXCLUSIVE
        intra_subj = find_intra_batch_subject_conflicts(
            candidate, accepted_batch, t_event=query_time,
        )
        intra_subj += find_intra_batch_subject_any_type_conflicts(
            candidate, accepted_batch, t_event=query_time,
        )
        intra_obj = find_intra_batch_object_conflicts(
            candidate, accepted_batch, t_event=query_time,
        )
        intra_obj += find_intra_batch_object_any_type_conflicts(
            candidate, accepted_batch, t_event=query_time,
        )
    return intra_subj, intra_obj
