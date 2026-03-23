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

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.models.temporal_types import (
    INFINITY,
    INFINITY_ISO,
    ConflictResult,
    RelationCardinality,
    ResolutionStrategy,
    TemporalRelationship,
    TemporalStateQuad,
    utcnow,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from neo4j import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision Router LLM Prompts
# ---------------------------------------------------------------------------

DECISION_ROUTER_PROMPT = """You are a temporal knowledge graph expert tasked with resolving a conflict between an existing edge and a new candidate edge.

## Existing Edge (conflicting)
- Subject: {existing_subject}
- Relation: {relation_type}
- Object: {existing_object}
- Valid Time: [{existing_t_valid_start} → {existing_t_valid_end}]
- Description: {existing_description}
- Source: {existing_source}
- Trust Score: {existing_trust}

## Candidate Edge (incoming)
- Subject: {candidate_subject}
- Relation: {relation_type}
- Object: {candidate_object}
- Valid Time: [{candidate_t_valid_start} → {candidate_t_valid_end}]
- Description: {candidate_description}
- Source: {candidate_source}
- Trust Score: {candidate_trust}

## Relation Cardinality
{cardinality}: {cardinality_explanation}

## Conflict Type
{conflict_type}

## Task
Based on the evidence above, select the most appropriate resolution strategy:

1. **EVOLUTION** - The world genuinely changed. The existing edge was true for its period, and the candidate edge represents a new true state. Close the existing edge's valid-time end at the candidate's start time and insert the candidate.

2. **CORRECTION** - The existing edge was wrong (factual error, hallucination, or bad source). Retroactively invalidate it (close transaction time) and insert the corrected candidate, inheriting the old edge's valid-time period.

3. **CORROBORATION** - The candidate says the same thing as the existing edge. No new edge needed; just update the support count and provenance.

4. **DISAGREEMENT** - There is genuine ambiguity or conflicting evidence with no clear resolution. Insert the candidate as "disputed" without modifying the existing edge.

Respond in this exact format:
STRATEGY: <one of EVOLUTION, CORRECTION, CORROBORATION, DISAGREEMENT>
CONFIDENCE: <0.0 to 1.0>
REASONING: <one sentence explaining the choice>
"""

CARDINALITY_EXPLANATIONS = {
    RelationCardinality.BOTH_EXCLUSIVE: "One-to-one: at most one subject can hold this relation with this object at a time, AND each subject can hold it with at most one object at a time.",
    RelationCardinality.SUBJECT_EXCLUSIVE: "One-to-many: a single subject can hold at most one active instance of this relation (e.g. a person has one nationality at a time).",
    RelationCardinality.OBJECT_EXCLUSIVE: "Many-to-one: at most one subject can hold this relation with a given object (e.g. a country has one capital at a time).",
    RelationCardinality.NON_EXCLUSIVE: "Many-to-many: no exclusivity constraint; multiple subjects/objects can hold this relation simultaneously.",
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
          AND e.t_tx_start <= $t_event
          AND e.t_tx_end > $t_event
        RETURN properties(e) AS e, o.title AS object_title
        """
        result = await session.run(
            query,
            subject=subject,
            relation_type=relation_type,
            t_event=t_event.isoformat(),
        )
    else:
        # Normal path: query current state (end times = INFINITY sentinel)
        query = """
        MATCH (s:Entity {title: $subject})-[e:RELATIONSHIP]->(o:Entity)
        WHERE e.relation_type = $relation_type
          AND e.t_tx_end = $infinity
          AND e.t_valid_end = $infinity
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
          AND e.t_tx_start <= $t_event
          AND e.t_tx_end > $t_event
        RETURN properties(e) AS e, s.title AS subject_title
        """
        result = await session.run(
            query,
            obj=obj,
            relation_type=relation_type,
            subject=subject,
            t_event=t_event.isoformat(),
        )
    else:
        query = """
        MATCH (s:Entity)-[e:RELATIONSHIP]->(o:Entity {title: $obj})
        WHERE e.relation_type = $relation_type
          AND s.title <> $subject
          AND e.t_tx_end = $infinity
          AND e.t_valid_end = $infinity
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


# ---------------------------------------------------------------------------
# Decision Router
# ---------------------------------------------------------------------------


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

    all_conflicts = conflict_result.subject_conflicts + conflict_result.object_conflicts
    conflict_type = (
        "SUBJECT_SIDE" if conflict_result.subject_conflicts else "OBJECT_SIDE"
    )
    if conflict_result.subject_conflicts and conflict_result.object_conflicts:
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
        if v is None or v == INFINITY:
            return "present"
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%d")
        return "present"

    quad_c = candidate.temporal_quad
    quad_e = existing.temporal_quad

    cardinality = candidate.cardinality
    prompt = DECISION_ROUTER_PROMPT.format(
        existing_subject=existing.source,
        existing_object=existing.target,
        relation_type=existing.relation_type,
        existing_t_valid_start=_ts(quad_e.t_valid_start if quad_e else None),
        existing_t_valid_end=_ts(quad_e.t_valid_end if quad_e else None),
        existing_description=existing.description or "",
        existing_source=existing.provenance[0].source_url if existing.provenance else "unknown",
        existing_trust=existing.provenance[0].trust_score if existing.provenance else 1.0,
        candidate_subject=candidate.source,
        candidate_object=candidate.target,
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
    t_now: datetime,
) -> None:
    """Evolution: close existing edge's valid-time, insert new edge.

    The world genuinely changed: old edge was true, new edge is now true.
    """
    candidate_t_valid_start = (
        candidate.temporal_quad.t_valid_start if candidate.temporal_quad else t_now
    )

    # Close the existing edge's valid-time end
    await session.run(
        """
        MATCH ()-[e]->()
        WHERE e.id = $edge_id
        SET e.t_valid_end = $t_valid_end
        """,
        edge_id=existing_edge_id,
        t_valid_end=candidate_t_valid_start.isoformat(),
    )
    logger.info(
        "ETCDR [EVOLUTION]: Closed edge %s valid_end at %s",
        existing_edge_id,
        candidate_t_valid_start,
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
) -> ConflictResult:
    """Run bidirectional conflict detection and resolve conflicts.

    For OBJECT_EXCLUSIVE and BOTH_EXCLUSIVE relations, runs both
    subject-side and object-side queries before committing any edge.

    Args:
        candidate: The new relationship to check.
        session: Neo4j async session.
        config: BT-GraphRAG configuration.
        model: Optional LLM for the Decision Router.
        is_late_arrival: Whether this document is a late arrival.
        t_event: Event timestamp for late-arrival historical queries.

    Returns:
        ConflictResult with strategy and resolved conflicts.
    """
    t_now = utcnow()
    query_time = t_event if is_late_arrival and t_event else None

    cardinality = RelationCardinality(
        config.get_cardinality(candidate.relation_type)
    )
    candidate.cardinality = cardinality

    # --- Sub-query S: subject-side ---
    subject_conflict_records = await run_subject_side_query(
        session=session,
        subject=candidate.source,
        relation_type=candidate.relation_type,
        t_event=query_time,
    )

    # --- Sub-query O: object-side (only for exclusive relations) ---
    object_conflict_records: list[dict[str, Any]] = []
    if cardinality in (RelationCardinality.OBJECT_EXCLUSIVE, RelationCardinality.BOTH_EXCLUSIVE):
        object_conflict_records = await run_object_side_query(
            session=session,
            subject=candidate.source,
            relation_type=candidate.relation_type,
            obj=candidate.target,
            t_event=query_time,
        )

    # Convert raw records to TemporalRelationship objects
    def _to_temporal_rel(record: dict[str, Any], is_object_side: bool = False) -> TemporalRelationship:
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
            source=record.get("_subject_title", candidate.source) if is_object_side else candidate.source,
            target=record.get("_object_title", candidate.target),
            relation_type=candidate.relation_type,
            description=record.get("description"),
            weight=record.get("weight", 1.0),
            confidence=record.get("confidence", 1.0),
            temporal_quad=quad,
            cardinality=cardinality,
        )

    subject_conflicts = [_to_temporal_rel(r, False) for r in subject_conflict_records]
    object_conflicts = [_to_temporal_rel(r, True) for r in object_conflict_records]

    conflict_result = ConflictResult(
        candidate=candidate,
        subject_conflicts=subject_conflicts,
        object_conflicts=object_conflicts,
    )

    if not conflict_result.has_conflicts:
        # No conflicts: candidate can be written as-is
        conflict_result.strategy = ResolutionStrategy.CORROBORATION
        conflict_result.confidence = 1.0
        return conflict_result

    # Run Decision Router
    strategy, confidence = await route_conflict(
        candidate=candidate,
        conflict_result=conflict_result,
        config=config,
        model=model,
    )
    conflict_result.strategy = strategy
    conflict_result.confidence = confidence

    # --- Apply resolution actions ---
    all_conflicts = subject_conflicts + object_conflicts

    if strategy == ResolutionStrategy.EVOLUTION:
        for existing in all_conflicts:
            if existing.id:
                await apply_evolution(session, existing.id, candidate, t_now)

    elif strategy == ResolutionStrategy.CORRECTION:
        for existing in all_conflicts:
            if existing.id:
                await apply_correction(session, existing.id, t_now)

    elif strategy == ResolutionStrategy.CORROBORATION:
        for existing in all_conflicts:
            if existing.id:
                await apply_corroboration(session, existing.id, candidate)

    elif strategy == ResolutionStrategy.DISAGREEMENT:
        candidate.status = "disputed"
        logger.info(
            "ETCDR [DISAGREEMENT]: Marking candidate (%s, %s, %s) as disputed",
            candidate.source,
            candidate.relation_type,
            candidate.target,
        )

    return conflict_result
