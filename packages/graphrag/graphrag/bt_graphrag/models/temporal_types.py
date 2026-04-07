# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Core temporal data types for BT-GraphRAG.

Defines the bitemporal model: Temporal State Quad, relation cardinality
ontology, epistemic states, resolution strategies, and provenance records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INFINITY = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
"""Far-future sentinel for open-ended timestamps (edge is still active / still believed).

Using an explicit datetime instead of NULL ensures Neo4j properties always
exist, eliminating "property does not exist" warnings and enabling clean
comparisons without IS NULL fallbacks.
"""

INFINITY_ISO = INFINITY.isoformat()
"""Pre-computed ISO-8601 string of INFINITY for serialization."""


def utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Relation Cardinality Ontology (Section 2.6.1)
# ---------------------------------------------------------------------------


class RelationCardinality(str, Enum):
    """Four-way relation cardinality ontology.

    Classifies every relation type by its structural exclusivity constraints.
    """

    SUBJECT_EXCLUSIVE = "SUBJECT_EXCLUSIVE"
    """One subject can hold at most one active instance (e.g. is_nationality_of)."""

    OBJECT_EXCLUSIVE = "OBJECT_EXCLUSIVE"
    """One object can have at most one active subject (e.g. has_capital_city)."""

    BOTH_EXCLUSIVE = "BOTH_EXCLUSIVE"
    """One-to-one: exclusive on both sides (e.g. is_CEO_of)."""

    NON_EXCLUSIVE = "NON_EXCLUSIVE"
    """No exclusivity constraint; history accumulated (e.g. worked_at)."""


# ---------------------------------------------------------------------------
# Epistemic States (Section 2.7)
# ---------------------------------------------------------------------------


class EpistemicState(str, Enum):
    """The four epistemic states derivable from the Temporal State Quad."""

    CURRENT_TRUTH = "CURRENT_TRUTH"
    """t_valid_end = inf AND t_tx_end = inf: True now, believed now."""

    HISTORICAL_TRUTH = "HISTORICAL_TRUTH"
    """t_valid_end < inf AND t_tx_end = inf: Was true, still believed."""

    RETRACTED_ERROR = "RETRACTED_ERROR"
    """t_valid_end = inf AND t_tx_end < inf: Believed true but retracted."""

    RETROACTIVE_CORRECTION = "RETROACTIVE_CORRECTION"
    """t_valid_end < inf AND t_tx_end < inf: Historical fact, now superseded."""


# ---------------------------------------------------------------------------
# Resolution Strategies (Section 2.6.3)
# ---------------------------------------------------------------------------


class ResolutionStrategy(str, Enum):
    """Decision Router output strategies for conflict resolution."""

    EVOLUTION = "EVOLUTION"
    """World genuinely changed: close old edge's t_valid_end, insert new edge."""

    CORRECTION = "CORRECTION"
    """Prior edge was wrong: set t_tx_end on old edge, insert corrected edge."""

    CORROBORATION = "CORROBORATION"
    """Candidate matches existing: increment support_count, append provenance."""

    DISAGREEMENT = "DISAGREEMENT"
    """No clear resolution: insert candidate with status=disputed."""

    NEW_EDGE = "NEW_EDGE"
    """No conflict detected: insert candidate as a new, independent edge."""


# ---------------------------------------------------------------------------
# Temporal State Quad (Section 2.7)
# ---------------------------------------------------------------------------


@dataclass
class TemporalStateQuad:
    """Every edge carries this four-timestamp structure.

    Implements the SQL:2011 bitemporal standard at the knowledge graph level.
    """

    t_valid_start: datetime
    """When the fact started being true in the world."""

    t_valid_end: datetime = INFINITY
    """When the fact stopped being true (INFINITY = still true)."""

    t_tx_start: datetime = field(default_factory=utcnow)
    """When the system came to believe this fact."""

    t_tx_end: datetime = INFINITY
    """When the system stopped believing this fact (INFINITY = still believed)."""

    @property
    def epistemic_state(self) -> EpistemicState:
        """Derive the epistemic state from the quad timestamps."""
        valid_open = self.t_valid_end >= INFINITY
        tx_open = self.t_tx_end >= INFINITY
        if valid_open and tx_open:
            return EpistemicState.CURRENT_TRUTH
        if not valid_open and tx_open:
            return EpistemicState.HISTORICAL_TRUTH
        if valid_open and not tx_open:
            return EpistemicState.RETRACTED_ERROR
        return EpistemicState.RETROACTIVE_CORRECTION

    def is_active(self) -> bool:
        """Return True if this edge is currently believed and currently valid."""
        return self.epistemic_state == EpistemicState.CURRENT_TRUTH

    def close_valid_time(self, end: datetime) -> None:
        """Close the valid-time interval (Evolution strategy)."""
        self.t_valid_end = end

    def retract(self, tx_end: datetime | None = None) -> None:
        """Retract this edge (Correction strategy) by closing transaction time."""
        self.t_tx_end = tx_end or utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for Neo4j property storage."""
        return {
            "t_valid_start": self.t_valid_start.isoformat(),
            "t_valid_end": self.t_valid_end.isoformat() if isinstance(self.t_valid_end, datetime) else INFINITY_ISO,
            "t_tx_start": self.t_tx_start.isoformat(),
            "t_tx_end": self.t_tx_end.isoformat() if isinstance(self.t_tx_end, datetime) else INFINITY_ISO,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TemporalStateQuad:
        """Deserialize from a Neo4j property dict."""
        def _parse(v: str | None) -> datetime:
            if v is None or v == INFINITY_ISO:
                return INFINITY
            return datetime.fromisoformat(v)

        return cls(
            t_valid_start=datetime.fromisoformat(d["t_valid_start"]),
            t_valid_end=_parse(d.get("t_valid_end")),
            t_tx_start=datetime.fromisoformat(d["t_tx_start"]),
            t_tx_end=_parse(d.get("t_tx_end")),
        )


# ---------------------------------------------------------------------------
# Provenance Record (Section 2.4)
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceRecord:
    """Full audit traceability for every edge."""

    source_document_id: str
    """ID of the source document."""

    text_unit_id: str
    """ID of the source text unit."""

    source_url: str | None = None
    """URL or path to the original source."""

    t_valid: datetime | None = None
    """Document-level valid time."""

    t_tx: datetime | None = None
    """System ingestion time."""

    trust_score: float = 1.0
    """Trust score of the source (0.0 - 1.0)."""

    text_hash: str | None = None
    """SHA-256 hash of the source text."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_document_id": self.source_document_id,
            "text_unit_id": self.text_unit_id,
            "source_url": self.source_url,
            "t_valid": self.t_valid.isoformat() if self.t_valid else None,
            "t_tx": self.t_tx.isoformat() if self.t_tx else None,
            "trust_score": self.trust_score,
            "text_hash": self.text_hash,
        }


# ---------------------------------------------------------------------------
# Temporal Relationship (extended edge model)
# ---------------------------------------------------------------------------


@dataclass
class TemporalRelationship:
    """A relationship enriched with bitemporal metadata.

    Extends GraphRAG's Relationship with the Temporal State Quad,
    cardinality classification, provenance, and conflict metadata.
    """

    id: str
    source: str
    target: str
    relation_type: str
    description: str | None = None
    weight: float = 1.0
    confidence: float = 1.0
    temporal_quad: TemporalStateQuad | None = None
    cardinality: RelationCardinality = RelationCardinality.NON_EXCLUSIVE
    provenance: list[ProvenanceRecord] = field(default_factory=list)
    support_count: int = 1
    status: str = "active"  # active, disputed, retracted
    text_unit_ids: list[str] = field(default_factory=list)
    description_embedding: list[float] | None = None
    relation_type_embedding: list[float] | None = None

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to a flat dict suitable for Neo4j edge properties."""
        props: dict[str, Any] = {
            "id": self.id,
            "relation_type": self.relation_type,
            "description": self.description or "",
            "weight": self.weight,
            "confidence": self.confidence,
            "cardinality": self.cardinality.value,
            "support_count": self.support_count,
            "status": self.status,
        }
        if self.temporal_quad:
            props.update(self.temporal_quad.to_dict())
        if self.description_embedding:
            props["description_embedding"] = self.description_embedding
        if self.relation_type_embedding:
            props["relation_type_embedding"] = self.relation_type_embedding
        return props


# ---------------------------------------------------------------------------
# Temporal Entity (extended node model)
# ---------------------------------------------------------------------------


@dataclass
class TemporalEntity:
    """An entity enriched with temporal metadata.

    Extends GraphRAG's Entity with active-period tracking and
    cross-batch resolution metadata.
    """

    id: str
    title: str
    type: str | None = None
    description: str | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    active_start: datetime | None = None
    active_end: datetime = INFINITY
    text_unit_ids: list[str] = field(default_factory=list)
    community_ids: list[str] = field(default_factory=list)
    description_embedding: list[float] | None = None
    name_embedding: list[float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to a flat dict suitable for Neo4j node properties."""
        def _ts(v: datetime | None) -> str:
            if v is None or v == INFINITY:
                return INFINITY_ISO
            return v.isoformat()

        props: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "type": self.type or "",
            "description": self.description or "",
            "first_seen": _ts(self.first_seen),
            "last_seen": _ts(self.last_seen),
            "active_start": _ts(self.active_start),
            "active_end": _ts(self.active_end),
        }
        if self.description_embedding:
            props["description_embedding"] = self.description_embedding
        return props


# ---------------------------------------------------------------------------
# Conflict Result
# ---------------------------------------------------------------------------


@dataclass
class ConflictResult:
    """Result of conflict detection for a single candidate edge."""

    candidate: TemporalRelationship
    subject_conflicts: list[TemporalRelationship] = field(default_factory=list)
    object_conflicts: list[TemporalRelationship] = field(default_factory=list)
    # Intra-batch conflicts (within the same extraction run)
    intra_batch_subject_conflicts: list[TemporalRelationship] = field(
        default_factory=list
    )
    intra_batch_object_conflicts: list[TemporalRelationship] = field(
        default_factory=list
    )
    strategy: ResolutionStrategy | None = None
    confidence: float = 0.0

    @property
    def has_conflicts(self) -> bool:
        return (
            len(self.subject_conflicts) > 0
            or len(self.object_conflicts) > 0
            or len(self.intra_batch_subject_conflicts) > 0
            or len(self.intra_batch_object_conflicts) > 0
        )

    @property
    def has_neo4j_conflicts(self) -> bool:
        """Conflicts against already-persisted edges."""
        return len(self.subject_conflicts) > 0 or len(self.object_conflicts) > 0

    @property
    def has_intra_batch_conflicts(self) -> bool:
        """Conflicts against other edges in the same extraction batch."""
        return (
            len(self.intra_batch_subject_conflicts) > 0
            or len(self.intra_batch_object_conflicts) > 0
        )
