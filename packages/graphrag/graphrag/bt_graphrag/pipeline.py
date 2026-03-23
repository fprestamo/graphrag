# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG 7-Stage Pipeline Orchestrator.

Connects all seven stages of the bitemporal pipeline:
1. Temporal Extraction
2. Cross-Graph Entity Resolution (CGER)
3. Edge-Level Temporal Conflict Detection (ETCDR)
4. Bitemporal Graph Store (Neo4j)
5. Incremental Community Update
6. Selective LLM Summarization
7. Query Pipeline

This module provides the main entry point that the BT workflow calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pandas as pd

from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.models.temporal_types import (
    INFINITY,
    INFINITY_ISO,
    ProvenanceRecord,
    TemporalEntity,
    TemporalRelationship,
    TemporalStateQuad,
    utcnow,
)
from graphrag.bt_graphrag.temporal_extraction.temporal_normalization import (
    assign_document_timestamps,
    is_late_arrival,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from neo4j import AsyncDriver, AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 → 4 Pipeline: Extract, Resolve, Detect Conflicts, Store
# ---------------------------------------------------------------------------


async def run_bt_pipeline(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    text_units_df: pd.DataFrame,
    documents_df: pd.DataFrame,
    config: BTGraphRAGConfig,
    extraction_model: "LLMCompletion | None" = None,
    neo4j_driver: "AsyncDriver | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Stages 2-4 of the BT-GraphRAG pipeline on already-extracted data.

    Stage 1 (temporal extraction) is handled by the modified extract_graph
    workflow which produces entities/relationships with temporal columns.

    This function performs:
    - Stage 2: CGER (Cross-Graph Entity Resolution)
    - Stage 3: ETCDR (Conflict Detection and Resolution)
    - Stage 4: Neo4j dual write

    Returns the (possibly modified) entities and relationships DataFrames
    after temporal processing. The original parquet write still happens
    in the calling workflow.
    """
    t_now = utcnow()

    print("\n" + "=" * 70)
    print("  BT-GraphRAG Pipeline — Stages 2-4")
    print("=" * 70)
    print(f"  Entities received:      {len(entities_df)}")
    print(f"  Relationships received:  {len(relationships_df)}")
    print(f"  Text units:              {len(text_units_df)}")
    print(f"  Documents:               {len(documents_df)}")
    print(f"  Neo4j connected:         {neo4j_driver is not None}")
    print(f"  CGER enabled:            {config.cger_enabled}")
    print(f"  ETCDR enabled:           {config.etcdr_enabled}")
    print("=" * 70)
    # Determine document-level temporal metadata
    doc_timestamps = _extract_document_timestamps(documents_df, config)

    print(f"\n  [Temporal Metadata] Document timestamps extracted: {len(doc_timestamps)}")
    for doc_id, (tv, ttx) in list(doc_timestamps.items())[:5]:
        print(f"    doc={doc_id[:20]:20s}  t_valid={tv.strftime('%Y-%m-%d')}  t_tx={ttx.strftime('%Y-%m-%d')}")
    if len(doc_timestamps) > 5:
        print(f"    ... and {len(doc_timestamps) - 5} more")

    # Enrich entities with temporal metadata
    entities_df = _enrich_entities_temporal(entities_df, doc_timestamps, t_now)

    # Enrich relationships with temporal metadata (if not already present)
    relationships_df = _enrich_relationships_temporal(
        relationships_df, doc_timestamps, t_now
    )

    # Show sample enriched data
    print(f"\n  [Temporal Enrichment] Entities now have columns: {list(entities_df.columns)}")
    print(f"  [Temporal Enrichment] Sample entities:")
    for _, row in entities_df.head(5).iterrows():
        print(f"    • {str(row.get('title', '?'))[:40]:40s}  type={str(row.get('type', '?'))[:15]}  active_start={str(row.get('active_start', '?'))[:10]}")

    print(f"\n  [Temporal Enrichment] Relationships now have columns: {list(relationships_df.columns)}")
    print(f"  [Temporal Enrichment] Sample relationships:")
    for _, row in relationships_df.head(5).iterrows():
        src = str(row.get('source', '?'))[:20]
        tgt = str(row.get('target', '?'))[:20]
        desc = str(row.get('description', '?'))[:30]
        tv = str(row.get('t_valid_start', '?'))[:10]
        conf = row.get('confidence', '?')
        print(f"    • ({src}) —[{desc}]→ ({tgt})  valid={tv}  conf={conf}")

    # -----------------------------------------------------------------------
    # Check if Neo4j DB has any existing data (skip queries on empty DB)
    # -----------------------------------------------------------------------
    neo4j_has_data = False
    if neo4j_driver is not None:
        from graphrag.bt_graphrag.neo4j_store import init_schema
        await init_schema(neo4j_driver, database=config.neo4j_database)
        async with neo4j_driver.session(database=config.neo4j_database) as _s:
            _r = await _s.run("MATCH (n:Entity) RETURN count(n) AS cnt LIMIT 1")
            _rec = await _r.single()
            neo4j_has_data = (_rec is not None and _rec["cnt"] > 0)
        if not neo4j_has_data:
            print("  [Neo4j] Database is empty — skipping CGER & ETCDR queries (first run)")

    # -----------------------------------------------------------------------
    # Stage 2: Cross-Graph Entity Resolution (CGER)
    # -----------------------------------------------------------------------
    if config.cger_enabled and neo4j_driver is not None and neo4j_has_data:
        entities_df, relationships_df = await _run_cger(
            entities_df=entities_df,
            relationships_df=relationships_df,
            config=config,
            model=extraction_model,
            driver=neo4j_driver,
        )

    # -----------------------------------------------------------------------
    # Stage 3: ETCDR (Conflict Detection)
    # -----------------------------------------------------------------------
    if config.etcdr_enabled and neo4j_driver is not None and neo4j_has_data:
        relationships_df = await _run_etcdr(
            relationships_df=relationships_df,
            documents_df=documents_df,
            config=config,
            model=extraction_model,
            driver=neo4j_driver,
            doc_timestamps=doc_timestamps,
        )

    # -----------------------------------------------------------------------
    # Stage 4: Dual write to Neo4j
    # -----------------------------------------------------------------------
    if neo4j_driver is not None:
        await _write_to_neo4j(
            entities_df=entities_df,
            relationships_df=relationships_df,
            config=config,
            driver=neo4j_driver,
        )

    return entities_df, relationships_df


# ---------------------------------------------------------------------------
# Document timestamp extraction
# ---------------------------------------------------------------------------


def _extract_document_timestamps(
    documents_df: pd.DataFrame,
    config: BTGraphRAGConfig,
) -> dict[str, tuple[datetime, datetime]]:
    """Extract t_valid and t_tx for each document.

    Returns a map: document_id -> (t_valid, t_tx).
    """
    timestamps: dict[str, tuple[datetime, datetime]] = {}
    t_now = utcnow()

    for _, row in documents_df.iterrows():
        doc_id = str(row.get("id", ""))
        doc_dict = dict(row)

        t_valid, t_tx = assign_document_timestamps(
            doc_dict, valid_time_field=config.default_valid_time_field
        )
        timestamps[doc_id] = (t_valid, t_tx)

    return timestamps


# ---------------------------------------------------------------------------
# Temporal enrichment helpers
# ---------------------------------------------------------------------------


def _enrich_entities_temporal(
    entities_df: pd.DataFrame,
    doc_timestamps: dict[str, tuple[datetime, datetime]],
    t_now: datetime,
) -> pd.DataFrame:
    """Add temporal columns to entities if not already present."""
    entities_df = entities_df.copy()

    if "first_seen" not in entities_df.columns:
        entities_df["first_seen"] = t_now.isoformat()
    if "last_seen" not in entities_df.columns:
        entities_df["last_seen"] = t_now.isoformat()
    if "active_start" not in entities_df.columns:
        # Derive from the earliest document that mentions this entity
        active_starts = []
        for _, row in entities_df.iterrows():
            text_unit_ids = row.get("text_unit_ids", [])
            if isinstance(text_unit_ids, list) and text_unit_ids:
                # Use earliest known document timestamp
                earliest = t_now
                for doc_id, (t_valid, _) in doc_timestamps.items():
                    if t_valid < earliest:
                        earliest = t_valid
                active_starts.append(earliest.isoformat())
            else:
                active_starts.append(t_now.isoformat())
        entities_df["active_start"] = active_starts
    if "active_end" not in entities_df.columns:
        entities_df["active_end"] = INFINITY_ISO  # Open-ended (still active)

    return entities_df


def _enrich_relationships_temporal(
    relationships_df: pd.DataFrame,
    doc_timestamps: dict[str, tuple[datetime, datetime]],
    t_now: datetime,
) -> pd.DataFrame:
    """Add temporal columns to relationships if not already present."""
    relationships_df = relationships_df.copy()

    if "t_valid_start" not in relationships_df.columns:
        # Default: earliest document timestamp
        if doc_timestamps:
            earliest_valid = min(t for t, _ in doc_timestamps.values())
        else:
            earliest_valid = t_now
        relationships_df["t_valid_start"] = earliest_valid.isoformat()

    if "t_valid_end" not in relationships_df.columns:
        relationships_df["t_valid_end"] = INFINITY_ISO  # Still true in the world

    if "t_tx_start" not in relationships_df.columns:
        relationships_df["t_tx_start"] = t_now.isoformat()

    if "t_tx_end" not in relationships_df.columns:
        relationships_df["t_tx_end"] = INFINITY_ISO  # Still believed by the system

    if "confidence" not in relationships_df.columns:
        relationships_df["confidence"] = 1.0

    if "status" not in relationships_df.columns:
        relationships_df["status"] = "active"

    if "support_count" not in relationships_df.columns:
        relationships_df["support_count"] = 1

    if "relation_type" not in relationships_df.columns:
        relationships_df["relation_type"] = relationships_df["description"].apply(
            _normalize_relation_type
        )

    if "cardinality" not in relationships_df.columns:
        relationships_df["cardinality"] = "NON_EXCLUSIVE"

    return relationships_df


# ---------------------------------------------------------------------------
# Stage 2: CGER
# ---------------------------------------------------------------------------


async def _run_cger(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    config: BTGraphRAGConfig,
    model: "LLMCompletion | None",
    driver: "AsyncDriver",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Cross-Graph Entity Resolution against existing Neo4j graph."""
    from graphrag.bt_graphrag.entity_resolution.cger import (
        apply_merge_map_to_relationships,
        resolve_entities,
    )
    from graphrag.bt_graphrag.neo4j_store import get_active_edges

    logger.info("BT-GraphRAG Stage 2: Running CGER entity resolution")
    print("\n" + "-" * 70)
    print("  Stage 2: Cross-Graph Entity Resolution (CGER)")
    print("-" * 70)

    # Load existing entities from Neo4j for comparison
    async with driver.session(database=config.neo4j_database) as session:
        existing_edges = await get_active_edges(session)

    # Build existing entities DataFrame from Neo4j edge data
    existing_entity_titles = set()
    for edge in existing_edges:
        existing_entity_titles.add(edge.get("source", ""))
        existing_entity_titles.add(edge.get("target", ""))

    if existing_entity_titles:
        existing_entities_df = pd.DataFrame(
            [{"title": t} for t in existing_entity_titles if t]
        )
    else:
        existing_entities_df = pd.DataFrame(columns=["title"])

    print(f"  Existing entities in Neo4j:  {len(existing_entity_titles)}")
    print(f"  New entities to resolve:     {len(entities_df)}")

    # Resolve new entities against existing graph
    resolved_entities, merge_map = await resolve_entities(
        new_entities=entities_df,
        existing_entities=existing_entities_df,
        config=config,
        model=model,
    )

    # Apply merge map to relationships
    if merge_map:
        relationships_df = apply_merge_map_to_relationships(
            relationships_df, merge_map
        )
        print(f"  CGER MERGES ({len(merge_map)}):")
        for old_name, new_name in list(merge_map.items())[:10]:
            print(f"    {old_name} → {new_name}")
        if len(merge_map) > 10:
            print(f"    ... and {len(merge_map) - 10} more")
        logger.info("CGER: Merged %d entities", len(merge_map))
    else:
        print("  No entity merges required.")

    print(f"  Resolved entity count:       {len(resolved_entities)}")

    return resolved_entities, relationships_df


# ---------------------------------------------------------------------------
# Stage 3: ETCDR
# ---------------------------------------------------------------------------


async def _run_etcdr(
    relationships_df: pd.DataFrame,
    documents_df: pd.DataFrame,
    config: BTGraphRAGConfig,
    model: "LLMCompletion | None",
    driver: "AsyncDriver",
    doc_timestamps: dict[str, tuple[datetime, datetime]],
) -> pd.DataFrame:
    """Run Edge-Level Temporal Conflict Detection and Resolution."""
    from graphrag.bt_graphrag.conflict_detection.etcdr import detect_and_resolve
    from graphrag.bt_graphrag.neo4j_store import insert_relationship

    logger.info("BT-GraphRAG Stage 3: Running ETCDR conflict detection")
    print("\n" + "-" * 70)
    print("  Stage 3: Edge-Level Temporal Conflict Detection (ETCDR)")
    print("-" * 70)
    print(f"  Relationships to check:  {len(relationships_df)}")

    # Determine if this is a late arrival batch
    t_now = utcnow()
    batch_is_late = False
    batch_t_event = None

    if doc_timestamps:
        avg_valid = min(t for t, _ in doc_timestamps.values())
        if is_late_arrival(avg_valid, t_now, config.late_arrival_threshold_days):
            batch_is_late = True
            batch_t_event = avg_valid
            print(f"  ⚠ Late arrival detected! event_time={batch_t_event.strftime('%Y-%m-%d')}")
            logger.info(
                "ETCDR: Late arrival detected (event time=%s, threshold=%d days)",
                batch_t_event,
                config.late_arrival_threshold_days,
            )

    statuses = []
    async with driver.session(database=config.neo4j_database) as session:
        for idx, row in relationships_df.iterrows():
            # Build TemporalRelationship from DataFrame row
            t_valid_start_str = row.get("t_valid_start")
            t_valid_end_str = row.get("t_valid_end")
            t_tx_start_str = row.get("t_tx_start")

            t_valid_start = (
                datetime.fromisoformat(t_valid_start_str)
                if t_valid_start_str
                else t_now
            )
            t_valid_end = (
                datetime.fromisoformat(t_valid_end_str)
                if t_valid_end_str and t_valid_end_str != "None"
                else INFINITY
            )
            t_tx_start = (
                datetime.fromisoformat(t_tx_start_str)
                if t_tx_start_str
                else t_now
            )

            rel_type = _normalize_relation_type(row.get("description", "RELATED_TO"))

            candidate = TemporalRelationship(
                id=str(uuid4()),
                source=str(row.get("source", "")),
                target=str(row.get("target", "")),
                relation_type=rel_type,
                description=str(row.get("description", "")),
                weight=float(row.get("weight", 1.0)),
                confidence=float(row.get("confidence", 1.0)),
                temporal_quad=TemporalStateQuad(
                    t_valid_start=t_valid_start,
                    t_valid_end=t_valid_end,
                    t_tx_start=t_tx_start,
                    t_tx_end=INFINITY,
                ),
                text_unit_ids=row.get("text_unit_ids", []),
            )

            # Run bidirectional conflict detection
            conflict_result = await detect_and_resolve(
                candidate=candidate,
                session=session,
                config=config,
                model=model,
                is_late_arrival=batch_is_late,
                t_event=batch_t_event,
            )

            statuses.append(conflict_result.candidate.status)

            # Log each edge's resolution
            strategy_name = conflict_result.strategy.value if conflict_result.strategy else "N/A"
            n_subj = len(conflict_result.subject_conflicts) if conflict_result.subject_conflicts else 0
            n_obj = len(conflict_result.object_conflicts) if conflict_result.object_conflicts else 0
            if n_subj > 0 or n_obj > 0:
                print(f"    [{idx}] ({candidate.source[:20]}) —[{rel_type[:20]}]→ ({candidate.target[:20]})  "
                      f"conflicts: S={n_subj} O={n_obj}  strategy={strategy_name}")

    # Update status column in relationships
    if statuses and len(statuses) == len(relationships_df):
        relationships_df = relationships_df.copy()
        relationships_df["status"] = statuses

    # Summary
    from collections import Counter
    status_counts = Counter(statuses)
    print(f"\n  ETCDR Summary:")
    for st, cnt in status_counts.most_common():
        print(f"    {st:15s}: {cnt}")

    return relationships_df


# ---------------------------------------------------------------------------
# Stage 4: Neo4j dual write
# ---------------------------------------------------------------------------


async def _write_to_neo4j(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
) -> None:
    """Write entities and relationships to Neo4j with temporal metadata."""
    from graphrag.bt_graphrag.neo4j_store import (
        insert_relationship,
        upsert_entity,
    )

    logger.info("BT-GraphRAG Stage 4: Writing to Neo4j bitemporal store")
    print("\n" + "-" * 70)
    print("  Stage 4: Neo4j Bitemporal Write")
    print("-" * 70)
    print(f"  Entities to write:       {len(entities_df)}")
    print(f"  Relationships to write:  {len(relationships_df)}")

    # init_schema already called before Stage 3 in run_bt_pipeline

    async with driver.session(database=config.neo4j_database) as session:
        # Write entities
        for _, row in entities_df.iterrows():
            entity = TemporalEntity(
                id=str(row.get("id", str(uuid4()))),
                title=str(row.get("title", "")),
                type=str(row.get("type", "")),
                description=str(row.get("description", "")),
                first_seen=(
                    datetime.fromisoformat(row["first_seen"])
                    if row.get("first_seen")
                    else utcnow()
                ),
                last_seen=(
                    datetime.fromisoformat(row["last_seen"])
                    if row.get("last_seen")
                    else utcnow()
                ),
                active_start=(
                    datetime.fromisoformat(row["active_start"])
                    if row.get("active_start")
                    else utcnow()
                ),
                active_end=INFINITY,
            )
            await upsert_entity(session, entity)

        # Write relationships
        for _, row in relationships_df.iterrows():
            t_valid_start_str = row.get("t_valid_start")
            t_valid_end_str = row.get("t_valid_end")
            t_tx_start_str = row.get("t_tx_start")
            t_tx_end_str = row.get("t_tx_end")

            t_now = utcnow()
            t_valid_start = (
                datetime.fromisoformat(t_valid_start_str)
                if t_valid_start_str
                else t_now
            )
            t_valid_end = (
                datetime.fromisoformat(t_valid_end_str)
                if t_valid_end_str and t_valid_end_str != "None"
                else INFINITY
            )
            t_tx_start = (
                datetime.fromisoformat(t_tx_start_str)
                if t_tx_start_str
                else t_now
            )
            t_tx_end = (
                datetime.fromisoformat(t_tx_end_str)
                if t_tx_end_str and t_tx_end_str != "None"
                else INFINITY
            )

            rel_type = _normalize_relation_type(row.get("description", "RELATED_TO"))

            rel = TemporalRelationship(
                id=str(row.get("id", str(uuid4()))),
                source=str(row.get("source", "")),
                target=str(row.get("target", "")),
                relation_type=rel_type,
                description=str(row.get("description", "")),
                weight=float(row.get("weight", 1.0)),
                confidence=float(row.get("confidence", 1.0)),
                temporal_quad=TemporalStateQuad(
                    t_valid_start=t_valid_start,
                    t_valid_end=t_valid_end,
                    t_tx_start=t_tx_start,
                    t_tx_end=t_tx_end,
                ),
                status=str(row.get("status", "active")),
                support_count=int(row.get("support_count", 1)),
            )
            await insert_relationship(session, rel)

    entity_count = len(entities_df)
    rel_count = len(relationships_df)
    print(f"  ✓ Written {entity_count} entities and {rel_count} relationships to Neo4j")
    logger.info("BT-GraphRAG Stage 4: Neo4j write complete")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_relation_type(description: str) -> str:
    """Derive a normalized relation type from a description string.

    Converts natural language descriptions to UPPER_SNAKE_CASE
    relation types suitable for cardinality classification.
    """
    if not description:
        return "RELATED_TO"
    # Take first 5 words, uppercase, join with underscores
    words = description.upper().split()[:5]
    return "_".join(w for w in words if w.isalpha())[:50] or "RELATED_TO"
