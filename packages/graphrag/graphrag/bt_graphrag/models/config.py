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
    enabled: bool = False
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

    cger_embedding_weight: float = 0.3
    """Weight w1 for cosine embedding similarity."""

    cger_bm25_weight: float = 0.25
    """Weight w2 for BM25 name matching."""

    cger_jaccard_weight: float = 0.15
    """Weight w3 for Jaccard character-level name similarity."""

    cger_temporal_overlap_weight: float = 0.15
    """Weight w4 for temporal overlap score."""

    cger_relation_context_weight: float = 0.15
    """Weight w5 for relation-context embedding similarity."""

    cger_merge_threshold: float = 0.85
    """Score above which entities are automatically merged."""

    cger_llm_threshold_low: float = 0.55
    """Score below which entities are kept separate."""

    cger_llm_threshold_high: float = 0.85
    """Score above which entities are automatically merged (before LLM check)."""

    cger_candidate_top_k: int = 20
    """Number of candidates to retrieve from each signal."""

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

    # --- Stage 5-6: Incremental Community Update ---
    community_update_k_hop: int = 2
    """k-hop neighborhood radius for incremental Leiden re-runs."""

    # --- Stage 7: Query ---
    temporal_decay_alpha: float = 0.1
    """Exponential decay parameter for temporal proximity scoring."""

    # --- Cardinality Ontology (default seed) ---
    default_cardinality_map: dict[str, str] = field(default_factory=lambda: {
        "IS_CEO_OF": "BOTH_EXCLUSIVE",
        "IS_PRESIDENT_OF": "BOTH_EXCLUSIVE",
        "IS_CHAIRMAN_OF": "BOTH_EXCLUSIVE",
        "IS_CAPITAL_OF": "OBJECT_EXCLUSIVE",
        "HAS_CAPITAL": "SUBJECT_EXCLUSIVE",
        "IS_NATIONALITY_OF": "SUBJECT_EXCLUSIVE",
        "IS_MARRIED_TO": "BOTH_EXCLUSIVE",
        "IS_SPOUSE_OF": "BOTH_EXCLUSIVE",
        "IS_HEADQUARTERED_IN": "SUBJECT_EXCLUSIVE",
        "HAS_POPULATION": "SUBJECT_EXCLUSIVE",
        "WORKED_AT": "NON_EXCLUSIVE",
        "COLLABORATED_WITH": "NON_EXCLUSIVE",
        "APPEARED_IN": "NON_EXCLUSIVE",
        "CO_AUTHORED": "NON_EXCLUSIVE",
        "PARTICIPATED_IN": "NON_EXCLUSIVE",
        "INVESTED_IN": "NON_EXCLUSIVE",
    })
    """Seed cardinality map. LLM classification extends this at runtime."""

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
