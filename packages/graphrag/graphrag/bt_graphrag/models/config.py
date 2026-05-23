# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG configuration model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BTGraphRAGConfig:
    """Configuration for the BT-GraphRAG bitemporal extension.

    This is added to GraphRagConfig to control all temporal pipeline stages.
    """

    # --- Global ---
    enabled: bool = True
    """Whether to enable the BT-GraphRAG temporal pipeline."""

    # --- Neo4j Connection ---
    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    """Neo4j connection URI."""

    neo4j_user: str = "neo4j"
    """Neo4j username."""

    neo4j_password: str = "12345678"
    """Neo4j password."""

    neo4j_database: str = "btgraphrag"
    """Neo4j database name."""

    # --- Prompt paths (optional, override hardcoded prompts) ---
    extraction_prompt: str | None = None
    """Path to temporal graph extraction prompt .txt file.
    If set, overrides the built-in TEMPORAL_GRAPH_EXTRACTION_PROMPT."""

    community_report_prompt: str | None = None
    """Path to temporal community report prompt .txt file.
    If set, overrides the built-in TEMPORAL_COMMUNITY_REPORT_PROMPT."""

    cardinality_prompt: str | None = None
    """Path to cardinality classification prompt .txt file.
    If set, overrides the built-in CARDINALITY_CLASSIFICATION_PROMPT."""

    # --- Stage 1: Temporal Extraction ---
    late_arrival_threshold_days: int = 30
    """Documents with t_tx - t_valid > this are flagged as late arrivals."""

    default_valid_time_field: str = "creation_date"
    """Document field to use as fallback for valid time."""

    temporal_extraction_model_id: str | None = None
    """Optional separate model for temporal expression normalization.
    Falls back to the extract_graph model if not set."""

    # --- Stage 2: CGER ---
    cger_enabled: bool = True
    """Enable Cross-Graph Entity Resolution."""

    cger_cosine_threshold: float = 0.85
    """Description-embedding cosine similarity at or above which CGER
    defers the merge decision to the LLM. Pairs with cosine below this
    value are never merged; the LLM is the only component that ever
    triggers a merge, and it does so using both the descriptions and the
    active periods of each entity."""

    cger_candidate_top_k: int = 10
    """Number of top-K candidates to retrieve via Neo4j vector search
    (or in-memory cosine fallback) per new entity."""

    cger_phase_b_temp_db: str = "cgerbatch"
    """Name of a pre-existing Neo4j database used by CGER Phase B as a
    scratch area for intra-batch candidate retrieval.  The database must
    be created manually (CREATE DATABASE <name>); the pipeline only wipes
    its contents before and after each Phase B run."""

    neo4j_vector_dimensions: int = 3072
    """Dimensionality of description_embedding vectors stored in Neo4j.
    Must match the embedding model output (e.g. 1536 for text-embedding-3-small,
    3072 for text-embedding-3-large).  Used to CREATE VECTOR INDEX."""

    # --- Stage 2b: CGRR (Cross-Graph Relationship Resolution) ---
    cgrr_enabled: bool = True
    """Enable Cross-Graph Relationship Resolution."""

    cgrr_cosine_threshold: float = 0.85
    """Description-embedding cosine similarity at or above which CGRR
    defers the merge decision to the LLM. Pairs with cosine below this
    value are never normalised; the LLM is the only component that ever
    triggers a normalisation, and it answers SAME or DIFFERENT using
    the two relation types' names, descriptions and a sample endpoint
    pair (no temporal context — relation types have no active period
    of their own)."""

    cgrr_candidate_top_k: int = 10
    """Number of top-K existing relation types to compare per candidate,
    pre-filtered by description embedding cosine similarity."""

    # --- Stage 3: ETCDR ---
    etcdr_enabled: bool = True
    """Enable Edge-Level Temporal Conflict Detection and Resolution."""

    etcdr_confidence_threshold: float = 0.7
    """Below this confidence, Decision Router defaults to DISAGREEMENT."""

    # Relation cardinality overrides: maps relation_type -> cardinality
    relation_cardinality_overrides: dict[str, str] = field(default_factory=dict)
    """Manual overrides for relation cardinality classification.
    Keys are relation types (uppercased), values are one of:
    SUBJECT_EXCLUSIVE, OBJECT_EXCLUSIVE, BOTH_EXCLUSIVE, NON_EXCLUSIVE."""

    # --- Debug / Diagnostics ---
    debug_output_dir: str | None = None
    """Directory to write debug files (CGER entities, CGRR relationships, ETCDR conflicts).
    If None, debug files are not written."""

    # --- Stage 5-6: Incremental Community Update ---
    community_update_k_hop: int = 2
    """k-hop neighborhood radius for incremental Leiden re-runs."""

    # --- Stage 7: Query ---
    temporal_decay_alpha: float = 0.1
    """Exponential decay parameter for temporal proximity scoring."""

    # --- Cardinality Ontology (default seed) ---
    default_cardinality_map: dict[str, str] = field(default_factory=lambda: {
        
    })
    """Seed cardinality map. LLM classification extends this at runtime."""

    def resolved_extraction_prompt(self) -> str:
        """Return the temporal extraction prompt, reading from file if configured."""
        from pathlib import Path

        from graphrag.bt_graphrag.prompts import TEMPORAL_GRAPH_EXTRACTION_PROMPT

        if self.extraction_prompt:
            p = Path(self.extraction_prompt)
            if p.exists():
                return p.read_text(encoding="utf-8")
        return TEMPORAL_GRAPH_EXTRACTION_PROMPT

    def resolved_community_report_prompt(self) -> str:
        """Return the temporal community report prompt, reading from file if configured."""
        from pathlib import Path

        from graphrag.bt_graphrag.prompts import TEMPORAL_COMMUNITY_REPORT_PROMPT

        if self.community_report_prompt:
            p = Path(self.community_report_prompt)
            if p.exists():
                return p.read_text(encoding="utf-8")
        return TEMPORAL_COMMUNITY_REPORT_PROMPT

    def resolved_cardinality_prompt(self) -> str:
        """Return the cardinality classification prompt, reading from file if configured."""
        from pathlib import Path

        from graphrag.bt_graphrag.prompts import CARDINALITY_CLASSIFICATION_PROMPT

        if self.cardinality_prompt:
            p = Path(self.cardinality_prompt)
            if p.exists():
                return p.read_text(encoding="utf-8")
        return CARDINALITY_CLASSIFICATION_PROMPT

    def get_cardinality(self, relation_type: str) -> str:
        """Return the cardinality for a relation type, checking overrides first."""
        key = relation_type.upper().replace(" ", "_")
        if key in self.relation_cardinality_overrides:
            return self.relation_cardinality_overrides[key]
        return self.default_cardinality_map.get(key, "NON_EXCLUSIVE")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for YAML/JSON config."""
        return {
            "enabled": self.enabled,
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "neo4j_database": self.neo4j_database,
            "late_arrival_threshold_days": self.late_arrival_threshold_days,
            "cger_enabled": self.cger_enabled,
            "etcdr_enabled": self.etcdr_enabled,
            "community_update_k_hop": self.community_update_k_hop,
            "temporal_decay_alpha": self.temporal_decay_alpha,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BTGraphRAGConfig:
        """Create from a dict (YAML/JSON config)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
