# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG 7-Stage Pipeline Orchestrator.

Connects all stages of the bitemporal pipeline:
1.  Temporal Extraction
2.  Cross-Graph Entity Resolution (CGER)
2b. Cross-Graph Relationship Resolution (CGRR)
3.  Edge-Level Temporal Conflict Detection (ETCDR)
4.  Bitemporal Graph Store (Neo4j)
5.  Incremental Community Update
6.  Selective LLM Summarization
7.  Query Pipeline

This module provides the main entry point that the BT workflow calls.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pandas as pd

from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.models.temporal_types import (
    INFINITY,
    INFINITY_ISO,
    MINUS_INFINITY_ISO,
    ProvenanceRecord,
    RelationCardinality,
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
# Debug / Diagnostics Helpers
# ---------------------------------------------------------------------------


def _ensure_debug_dir(config: BTGraphRAGConfig) -> str | None:
    """Return the debug output directory path, creating it if needed. None if disabled."""
    if not config.debug_output_dir:
        return None
    os.makedirs(config.debug_output_dir, exist_ok=True)
    return config.debug_output_dir


def _save_debug_json(debug_dir: str | None, filename: str, data: Any) -> None:
    """Save data as JSON to the debug directory."""
    if not debug_dir:
        return
    path = os.path.join(debug_dir, filename)

    def _default(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "value"):
            return o.value
        if isinstance(o, float) and (o == float("inf") or o != o):
            return str(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default, ensure_ascii=False)
    print(f"    [DEBUG] Saved -> {path}")


def _save_debug_dataframe(debug_dir: str | None, filename: str, df: pd.DataFrame) -> None:
    """Save a DataFrame as CSV to the debug directory."""
    if not debug_dir:
        return
    path = os.path.join(debug_dir, filename)
    df.to_csv(path, index=False)
    print(f"    [DEBUG] Saved -> {path}")


def _print_entity_table(entities_df: pd.DataFrame, title: str, max_rows: int = 50) -> None:
    """Print a formatted table of entities."""
    print(f"\n    {'─' * 100}")
    print(f"    {title}")
    print(f"    {'─' * 100}")
    header = f"    {'#':>4}  {'TITLE':40s}  {'TYPE':15s}  {'ACTIVE_START':12s}  {'ACTIVE_END':12s}  {'DESCRIPTION':40s}"
    print(header)
    print(f"    {'─' * 100}")
    for i, (_, row) in enumerate(entities_df.iterrows()):
        if i >= max_rows:
            print(f"    ... ({len(entities_df) - max_rows} more rows)")
            break
        title_val = str(row.get("title", "?"))[:40]
        type_val = str(row.get("type", "?"))[:15]
        as_val = str(row.get("active_start", "?"))[:12]
        ae_val = str(row.get("active_end", "?"))[:12]
        desc_val = str(row.get("description", ""))[:40]
        print(f"    {i:4d}  {title_val:40s}  {type_val:15s}  {as_val:12s}  {ae_val:12s}  {desc_val:40s}")
    print(f"    {'─' * 100}")
    print(f"    Total: {len(entities_df)} entities")


def _print_relationship_table(rel_df: pd.DataFrame, title: str, max_rows: int = 50) -> None:
    """Print a formatted table of relationships."""
    print(f"\n    {'─' * 120}")
    print(f"    {title}")
    print(f"    {'─' * 120}")
    header = (f"    {'#':>4}  {'SOURCE':22s}  {'RELATION_TYPE':25s}  {'TARGET':22s}  "
              f"{'T_VALID_START':12s}  {'T_VALID_END':12s}  {'CONF':5s}  {'STATUS':10s}")
    print(header)
    print(f"    {'─' * 120}")
    for i, (_, row) in enumerate(rel_df.iterrows()):
        if i >= max_rows:
            print(f"    ... ({len(rel_df) - max_rows} more rows)")
            break
        src = str(row.get("source", "?"))[:22]
        rt = str(row.get("relation_type", row.get("description", "?")))[:25]
        tgt = str(row.get("target", "?"))[:22]
        vs = str(row.get("t_valid_start", "?"))[:12]
        ve = str(row.get("t_valid_end", "?"))[:12]
        conf = f"{float(row.get('confidence', 0)):.2f}" if row.get("confidence") is not None else "?"
        status = str(row.get("status", "?"))[:10]
        print(f"    {i:4d}  {src:22s}  {rt:25s}  {tgt:22s}  {vs:12s}  {ve:12s}  {conf:5s}  {status:10s}")
    print(f"    {'─' * 120}")
    print(f"    Total: {len(rel_df)} relationships")


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
    - Stage 2:  CGER (Cross-Graph Entity Resolution)
    - Stage 2b: CGRR (Cross-Graph Relationship Resolution)
    - Stage 3:  ETCDR (Conflict Detection and Resolution)
    - Stage 4:  Neo4j dual write

    Returns the (possibly modified) entities and relationships DataFrames
    after temporal processing. The original parquet write still happens
    in the calling workflow.
    """
    t_now = utcnow()

    print("\n" + "=" * 70)
    print("  BT-GraphRAG Pipeline — Stages 2, 2b, 3, 4")
    print("=" * 70)
    print(f"  Entities received:      {len(entities_df)}")
    print(f"  Relationships received:  {len(relationships_df)}")
    print(f"  Text units:              {len(text_units_df)}")
    print(f"  Documents:               {len(documents_df)}")
    print(f"  Neo4j connected:         {neo4j_driver is not None}")
    print(f"  CGER enabled:            {config.cger_enabled}")
    print(f"  CGRR enabled:            {config.cgrr_enabled}")
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
        relationships_df, doc_timestamps, t_now, config=config,
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
        await init_schema(
            neo4j_driver,
            database=config.neo4j_database,
            vector_dimensions=config.neo4j_vector_dimensions,
        )
        async with neo4j_driver.session(database=config.neo4j_database) as _s:
            _r = await _s.run("MATCH (n:Entity) RETURN count(n) AS cnt LIMIT 1")
            _rec = await _r.single()
            neo4j_has_data = (_rec is not None and _rec["cnt"] > 0)
        if not neo4j_has_data:
            print("  [Neo4j] Database is empty — skipping CGER & ETCDR queries (first run)")

    debug_dir = _ensure_debug_dir(config)

    # -----------------------------------------------------------------------
    # Stage 2: Cross-Graph Entity Resolution (CGER)
    # -----------------------------------------------------------------------
    if config.cger_enabled and neo4j_driver is not None:
        entities_df, relationships_df = await _run_cger(
            entities_df=entities_df,
            relationships_df=relationships_df,
            config=config,
            model=extraction_model,
            driver=neo4j_driver,
        )

    # --- CGER debug: save entities & print table ---
    _print_entity_table(entities_df, "ENTITIES AFTER STAGE 2 (CGER)")
    _save_debug_dataframe(debug_dir, "stage2_cger_entities.csv", entities_df)
    _save_debug_json(debug_dir, "stage2_cger_entities.json", entities_df.to_dict("records"))

    # -----------------------------------------------------------------------
    # Self-loop filter (post-CGER): CGER merges may collapse source==target
    # -----------------------------------------------------------------------
    if "source" in relationships_df.columns and "target" in relationships_df.columns:
        self_loop_mask = (
            relationships_df["source"].str.strip().str.lower()
            == relationships_df["target"].str.strip().str.lower()
        )
        n_self_loops = int(self_loop_mask.sum())
        if n_self_loops > 0:
            dropped = relationships_df[self_loop_mask]
            print(f"\n  [Self-Loop Filter] Dropping {n_self_loops} self-referencing edge(s):")
            for _, row in dropped.iterrows():
                src = str(row.get("source", "?"))[:30]
                rt = str(row.get("relation_type", row.get("description", "?")))[:30]
                print(f"    DROPPED: ({src}) -[{rt}]-> ({src})")
            relationships_df = relationships_df[~self_loop_mask].reset_index(drop=True)
        else:
            print(f"\n  [Self-Loop Filter] No self-loops found — OK")

    # -----------------------------------------------------------------------
    # Stage 2b: Cross-Graph Relationship Resolution (CGRR)
    # -----------------------------------------------------------------------
    if config.cgrr_enabled and neo4j_driver is not None:
        relationships_df = await _run_cgrr(
            relationships_df=relationships_df,
            config=config,
            model=extraction_model,
            driver=neo4j_driver,
        )
    elif not config.cgrr_enabled:
        print("\n  [CGRR] WARNING: CGRR is DISABLED. Enabling it is recommended to avoid relationship aliasing.")
        print("  [CGRR] Set cgrr_enabled=True in config to enable.")

    # --- CGRR debug: save relationships & print table ---
    _print_relationship_table(relationships_df, "RELATIONSHIPS AFTER STAGE 2b (CGRR)")
    _save_debug_dataframe(debug_dir, "stage2b_cgrr_relationships.csv", relationships_df)
    _save_debug_json(debug_dir, "stage2b_cgrr_relationships.json", relationships_df.to_dict("records"))

    # -----------------------------------------------------------------------
    # Stage 2c: LLM Cardinality Classification
    # -----------------------------------------------------------------------
    relationships_df = await _classify_cardinalities(
        relationships_df=relationships_df,
        config=config,
        model=extraction_model,
    )

    # -----------------------------------------------------------------------
    # Stage 3: ETCDR (Conflict Detection)
    # -----------------------------------------------------------------------
    # Run ETCDR even on first iteration: while Neo4j has no pre-existing
    # edges, intra-batch conflict detection still catches contradictions
    # between relationships extracted in the same run.
    if config.etcdr_enabled and neo4j_driver is not None:
        relationships_df = await _run_etcdr(
            relationships_df=relationships_df,
            documents_df=documents_df,
            config=config,
            model=extraction_model,
            driver=neo4j_driver,
            doc_timestamps=doc_timestamps,
        )

    # --- ETCDR debug: save relationships & print table ---
    _print_relationship_table(relationships_df, "RELATIONSHIPS AFTER STAGE 3 (ETCDR)")
    _save_debug_dataframe(debug_dir, "stage3_etcdr_relationships.csv", relationships_df)
    _save_debug_json(debug_dir, "stage3_etcdr_relationships.json", relationships_df.to_dict("records"))

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

    # -----------------------------------------------------------------------
    # Final Pipeline Verification
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  BT-GraphRAG Pipeline — FINAL VERIFICATION")
    print("=" * 70)
    print(f"  Entities output:         {len(entities_df)}")
    print(f"  Relationships output:    {len(relationships_df)}")

    # Check required columns are present
    required_entity_cols = ["title", "type", "first_seen", "last_seen", "active_start", "active_end"]
    missing_entity_cols = [c for c in required_entity_cols if c not in entities_df.columns]
    if missing_entity_cols:
        print(f"  WARNING: Entity DataFrame missing columns: {missing_entity_cols}")
    else:
        print(f"  Entity temporal columns:  PASS (all present)")

    required_rel_cols = ["source", "target", "t_valid_start", "t_valid_end", "t_tx_start", "t_tx_end",
                         "confidence", "status", "support_count"]
    missing_rel_cols = [c for c in required_rel_cols if c not in relationships_df.columns]
    if missing_rel_cols:
        print(f"  WARNING: Relationship DataFrame missing columns: {missing_rel_cols}")
    else:
        print(f"  Relationship temporal columns: PASS (all present)")

    # Check for null values in critical columns
    if "title" in entities_df.columns:
        null_titles = entities_df["title"].isna().sum()
        if null_titles > 0:
            print(f"  WARNING: {null_titles} entities have null titles")
        else:
            print(f"  Entity title null check: PASS")

    if "source" in relationships_df.columns and "target" in relationships_df.columns:
        null_sources = relationships_df["source"].isna().sum()
        null_targets = relationships_df["target"].isna().sum()
        if null_sources > 0 or null_targets > 0:
            print(f"  WARNING: {null_sources} null sources, {null_targets} null targets in relationships")
        else:
            print(f"  Relationship endpoints:  PASS (no nulls)")

    # Check referential integrity: all relationship sources/targets exist in entities
    if "title" in entities_df.columns and "source" in relationships_df.columns:
        entity_titles = set(entities_df["title"].tolist())
        rel_sources = set(relationships_df["source"].tolist())
        rel_targets = set(relationships_df["target"].tolist())
        missing_sources = rel_sources - entity_titles
        missing_targets = rel_targets - entity_titles
        if missing_sources:
            print(f"  WARNING: {len(missing_sources)} relationship sources not in entities: "
                  f"{list(missing_sources)[:5]}")
        if missing_targets:
            print(f"  WARNING: {len(missing_targets)} relationship targets not in entities: "
                  f"{list(missing_targets)[:5]}")
        if not missing_sources and not missing_targets:
            print(f"  Referential integrity:   PASS (all endpoints exist in entities)")

    # Temporal consistency check
    if "t_valid_start" in relationships_df.columns and "t_valid_end" in relationships_df.columns:
        from graphrag.bt_graphrag.models.temporal_types import INFINITY_ISO
        invalid_temporal = 0
        for _, row in relationships_df.iterrows():
            vs = str(row.get("t_valid_start", ""))
            ve = str(row.get("t_valid_end", ""))
            if vs and ve and ve != INFINITY_ISO and ve != "None":
                try:
                    if datetime.fromisoformat(vs) > datetime.fromisoformat(ve):
                        invalid_temporal += 1
                except (ValueError, TypeError):
                    pass
        if invalid_temporal > 0:
            print(f"  WARNING: {invalid_temporal} relationships have t_valid_start > t_valid_end")
        else:
            print(f"  Temporal ordering check: PASS (all valid_start <= valid_end)")

    print("=" * 70 + "\n")

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
    config: BTGraphRAGConfig | None = None,
) -> pd.DataFrame:
    """Add temporal columns to relationships if not already present."""
    relationships_df = relationships_df.copy()

    # Assign a canonical edge id up front so the same UUID flows through
    # ETCDR (candidate.id), Neo4j (insert_relationship), and parquet
    # (finalize_relationships). Otherwise each stage generates its own
    # uuid4 and the three stores diverge.
    if "id" not in relationships_df.columns:
        relationships_df["id"] = [str(uuid4()) for _ in range(len(relationships_df))]
    else:
        mask = relationships_df["id"].isna() | (relationships_df["id"].astype(str).str.len() == 0)
        if mask.any():
            relationships_df.loc[mask, "id"] = [
                str(uuid4()) for _ in range(int(mask.sum()))
            ]

    if "t_valid_start" not in relationships_df.columns:
        relationships_df["t_valid_start"] = MINUS_INFINITY_ISO

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
        # No relation_type column at all — fall back for every row
        relationships_df["relation_type"] = relationships_df["description"].apply(
            _normalize_relation_type
        )
    else:
        # Column exists but some rows may be empty (legacy LLM output).
        # Fill only the missing values with the fallback.
        mask = relationships_df["relation_type"].isna() | (relationships_df["relation_type"] == "")
        if mask.any():
            relationships_df.loc[mask, "relation_type"] = (
                relationships_df.loc[mask, "description"].apply(_normalize_relation_type)
            )

    if "cardinality" not in relationships_df.columns:
        if config is not None:
            relationships_df["cardinality"] = relationships_df["relation_type"].apply(
                lambda rt: config.get_cardinality(rt) if pd.notna(rt) else "NON_EXCLUSIVE"
            )
        else:
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
    """Run Cross-Graph Entity Resolution against existing Neo4j graph.

    Uses the Neo4j vector index on Entity.description_embedding to
    retrieve the top-K most similar existing entities per new entity,
    then runs composite scoring on those candidates only.
    Falls back to loading all entities if the vector index is unavailable.
    """
    from graphrag.bt_graphrag.entity_resolution.cger import (
        apply_merge_map_to_relationships,
        resolve_entities,
    )
    from graphrag.bt_graphrag.neo4j_store import (
        get_all_entities,
        vector_search_entities,
    )

    logger.info("BT-GraphRAG Stage 2: Running CGER entity resolution")
    print("\n" + "-" * 70)
    print("  Stage 2: Cross-Graph Entity Resolution (CGER)")
    print("-" * 70)

    top_k = config.cger_candidate_top_k

    # --- Retrieve existing entity candidates from Neo4j ---
    # Strategy: per new entity, query the Neo4j vector index for the
    # top-K nearest neighbours by description_embedding cosine similarity.
    # This gives us full entity properties (type, description, embeddings,
    # temporal fields) which the old approach was missing.
    #
    # If the vector index call fails (e.g. index not populated yet on
    # first run, or Neo4j < 5.11), fall back to loading all entities.
    use_vector_index = True
    all_existing_entities: list[dict[str, Any]] | None = None

    async with driver.session(database=config.neo4j_database) as session:
        # Quick probe: does the graph have entities at all?
        probe = await session.run("MATCH (n:Entity) RETURN count(n) AS cnt LIMIT 1")
        probe_rec = await probe.single()
        existing_count = probe_rec["cnt"] if probe_rec else 0

        if existing_count == 0:
            print(f"  Existing entities in Neo4j:  0 (first run)")
            print(f"  New entities to resolve:     {len(entities_df)}")
            # Build an empty DataFrame so resolve_entities skips Phase A
            existing_entities_df = pd.DataFrame(columns=["title"])
            candidate_map: dict[str, list[dict[str, Any]]] = {}
        else:
            print(f"  Existing entities in Neo4j:  {existing_count}")
            print(f"  New entities to resolve:     {len(entities_df)}")
            print(f"  Top-K per entity:            {top_k}")

            # Try vector-search per new entity
            candidate_map = {}
            for _, new_row in entities_df.iterrows():
                new_emb = new_row.get("description_embedding")
                title = str(new_row.get("title", ""))
                if new_emb is not None and isinstance(new_emb, list) and len(new_emb) > 0:
                    try:
                        candidates = await vector_search_entities(
                            session, new_emb, top_k=top_k,
                        )
                        candidate_map[title] = candidates
                    except Exception as vec_err:
                        if use_vector_index:
                            logger.warning(
                                "Neo4j vector index query failed (%s); "
                                "falling back to full entity load", vec_err,
                            )
                            print(f"  [CGER] Vector index unavailable — "
                                  f"falling back to full entity load")
                            use_vector_index = False
                            break

            # Fallback: load all entities with full properties
            if not use_vector_index or not candidate_map:
                all_existing_entities = await get_all_entities(session)
                print(f"  [CGER] Loaded {len(all_existing_entities)} entities "
                      f"with full properties (fallback)")

            # Build existing_entities_df from either the union of all
            # retrieved candidates or the full entity list
            if all_existing_entities is not None:
                existing_entities_df = pd.DataFrame(all_existing_entities)
                candidate_map = {}  # let cger.py do in-memory top-K
            else:
                # Union all unique candidates across new entities
                seen_titles: set[str] = set()
                all_candidates: list[dict[str, Any]] = []
                for cands in candidate_map.values():
                    for c in cands:
                        t = c.get("title", "")
                        if t and t not in seen_titles:
                            seen_titles.add(t)
                            all_candidates.append(c)
                existing_entities_df = (
                    pd.DataFrame(all_candidates) if all_candidates
                    else pd.DataFrame(columns=["title"])
                )
                print(f"  [CGER] Retrieved {len(all_candidates)} unique "
                      f"candidate entities via vector index")

    debug_dir = _ensure_debug_dir(config)

    # --- CGER vector search log ---
    _cger_search_log: list[dict[str, Any]] = []
    for _entity_title, _cands in candidate_map.items():
        _cger_search_log.append({
            "entity": _entity_title,
            "method": "neo4j_vector_index" if use_vector_index and candidate_map else "in_memory_fallback",
            "top_k": top_k,
            "candidates_returned": len(_cands),
            "candidates": [
                {
                    "title": c.get("title", ""),
                    "type": c.get("type", ""),
                    "vector_score": round(c.get("_vector_score", 0.0), 4),
                    "description": (c.get("description", "") or "")[:100],
                }
                for c in _cands[:10]
            ],
        })
    _save_debug_json(debug_dir, "cger_vector_search_log.json", _cger_search_log)

    cger_resolution_log: list[dict[str, Any]] = []

    # --- Phase A pre-logging: cosine similarity per new entity vs its candidates ---
    print(f"\n  Phase A: New vs Existing (Neo4j)")

    from graphrag.bt_graphrag.entity_resolution.scorers import (
        description_cosine_entity_scorer,
    )

    existing_records: list[dict[str, Any]] = (
        [{str(k): v for k, v in r.items()} for r in existing_entities_df.to_dict("records")]
        if not existing_entities_df.empty else []
    )

    for _, new_row in entities_df.iterrows():
        new_entity: dict[str, Any] = {str(k): v for k, v in dict(new_row).items()}
        entity_title = str(new_entity.get("title", "?"))

        if candidate_map and entity_title in candidate_map:
            records_for_entity = candidate_map[entity_title]
        else:
            records_for_entity = existing_records

        best_score = 0.0
        best_match_title = ""
        all_comparisons: list[dict[str, Any]] = []

        for existing in records_for_entity:
            score, breakdown = description_cosine_entity_scorer(
                new_entity, existing, config
            )
            ex_title = str(existing.get("title", "?"))
            comp = {
                "existing_entity": ex_title,
                "score": round(score, 4),
                "cosine_emb": round(breakdown.get("cosine_emb", 0), 4),
                "embedding_available": bool(breakdown.get("embedding_available", False)),
            }
            all_comparisons.append(comp)
            if score > best_score:
                best_score = score
                best_match_title = ex_title

        all_comparisons.sort(key=lambda c: c["score"], reverse=True)
        cger_resolution_log.append({
            "phase": "A_cross_graph",
            "entity": entity_title,
            "type": str(new_entity.get("type", "?")),
            "best_match": best_match_title,
            "best_score": round(best_score, 4),
            "decision": (
                "LLM_CHECK" if best_score >= config.cger_cosine_threshold
                else "BELOW_THRESHOLD"
            ),
            "top_comparisons": all_comparisons[:10],
        })

    entity_scorer = description_cosine_entity_scorer
    print(f"  CGER scorer: description_cosine (LLM trigger at cosine >= {config.cger_cosine_threshold})")

    # resolve_entities handles Phase A (new vs existing) and Phase B (intra-batch).
    # Pass the driver so Phase B can spin up a temporary Neo4j database for
    # same-name + top-10 description-embedding candidate retrieval.
    resolved_entities, merge_map, phase_b_log = await resolve_entities(
        new_entities=entities_df,
        existing_entities=existing_entities_df,
        config=config,
        model=model,
        entity_scorer=entity_scorer,
        candidate_map=candidate_map if candidate_map else None,
        driver=driver,
        phase_b_top_k=top_k,
    )

    # Apply full merge map to relationships
    if merge_map:
        relationships_df = apply_merge_map_to_relationships(
            relationships_df, merge_map
        )
        print(f"\n  CGER TOTAL MERGES ({len(merge_map)}):")
        for old_name, new_name in list(merge_map.items())[:10]:
            print(f"    {old_name} → {new_name}")
        if len(merge_map) > 10:
            print(f"    ... and {len(merge_map) - 10} more")
        logger.info("CGER: Merged %d entities (cross-graph + intra-batch)", len(merge_map))
    else:
        print("  No entity merges required.")

    print(f"  Resolved entity count:       {len(resolved_entities)}")

    # --- CGER merge map detail table ---
    if merge_map:
        print(f"\n  {'─' * 80}")
        print(f"  CGER MERGE MAP DETAIL")
        print(f"  {'─' * 80}")
        print(f"  {'#':>4}  {'ORIGINAL ENTITY':40s}  {'MERGED INTO':40s}")
        print(f"  {'─' * 80}")
        for i, (orig, canonical) in enumerate(merge_map.items()):
            print(f"  {i:4d}  {orig[:40]:40s}  {canonical[:40]:40s}")
        print(f"  {'─' * 80}")

    # Save CGER debug files (combine Phase A and Phase B logs)
    full_cger_log = cger_resolution_log + phase_b_log
    _save_debug_json(debug_dir, "stage2_cger_merge_map.json", merge_map)
    _save_debug_json(debug_dir, "stage2_cger_resolution_log.json", full_cger_log)

    return resolved_entities, relationships_df


# ---------------------------------------------------------------------------
# Stage 2b: CGRR (Cross-Graph Relationship Resolution)
# ---------------------------------------------------------------------------


async def _run_cgrr(
    relationships_df: pd.DataFrame,
    config: BTGraphRAGConfig,
    model: "LLMCompletion | None",
    driver: "AsyncDriver",
) -> pd.DataFrame:
    """Run Cross-Graph Relationship Resolution against existing Neo4j graph."""
    from graphrag.bt_graphrag.entity_resolution.cgrr import (
        apply_normalize_map_to_cardinality,
        resolve_relationships,
    )

    logger.info("BT-GraphRAG Stage 2b: Running CGRR relationship resolution")
    print("\n" + "-" * 70)
    print("  Stage 2b: Cross-Graph Relationship Resolution (CGRR)")
    print("-" * 70)
    print(f"  Relationships to resolve:  {len(relationships_df)}")
    print(f"  LLM available:             {model is not None}")
    print(f"  CGRR merge threshold:      {config.cgrr_merge_threshold}")
    print(f"  CGRR LLM threshold:        {config.cgrr_llm_threshold_low}")

    if "relation_type" not in relationships_df.columns:
        print("  Computing relation_type from descriptions (column not present)...")
        relationships_df = relationships_df.copy()
        relationships_df["relation_type"] = relationships_df["description"].apply(
            _normalize_relation_type
        )
    else:
        # Fill any missing values with the fallback
        mask = relationships_df["relation_type"].isna() | (relationships_df["relation_type"] == "")
        if mask.any():
            print(f"  Filling {mask.sum()} missing relation_type values from descriptions...")
            relationships_df = relationships_df.copy()
            relationships_df.loc[mask, "relation_type"] = (
                relationships_df.loc[mask, "description"].apply(_normalize_relation_type)
            )

    unique_before = relationships_df["relation_type"].nunique()
    type_counts_before = relationships_df["relation_type"].value_counts()
    print(f"\n  Relation type distribution BEFORE CGRR ({unique_before} unique):")
    for rt, count in type_counts_before.head(20).items():
        card = config.get_cardinality(str(rt))
        print(f"    {str(rt)[:40]:40s}  {count:4d} edges  [{card}]")
    if len(type_counts_before) > 20:
        print(f"    ... and {len(type_counts_before) - 20} more types")

    debug_dir = _ensure_debug_dir(config)
    cgrr_resolution_log: list[dict[str, Any]] = []

    # --- Phase A: Resolve against existing Neo4j relation types ---
    print(f"\n  Phase A: New vs Existing (Neo4j)")

    from graphrag.bt_graphrag.entity_resolution.cgrr import (
        compute_relationship_score,
        _top_k_by_embedding,
    )
    from graphrag.bt_graphrag.neo4j_store import get_existing_relation_types_with_embeddings

    # Pre-compute Phase A comparisons for the debug log
    # Only fetch existing types that share at least one entity with the batch
    batch_entities = list(set(
        relationships_df["source"].dropna().unique().tolist()
        + relationships_df["target"].dropna().unique().tolist()
    ))
    cgrr_top_k = getattr(config, "cgrr_candidate_top_k", 10)
    async with driver.session(database=config.neo4j_database) as session:
        existing_types_for_log = await get_existing_relation_types_with_embeddings(
            session, entity_titles=batch_entities,
        )

    candidate_types_before = relationships_df["relation_type"].unique().tolist()
    cand_type_counts = relationships_df["relation_type"].value_counts().to_dict()

    # --- CGRR vector search log ---
    _cgrr_search_log: list[dict[str, Any]] = []
    for cand_type in candidate_types_before:
        cand_sample = relationships_df[relationships_df["relation_type"] == cand_type].iloc[0]
        cand_emb_check = cand_sample.get("description_embedding") if "description_embedding" in cand_sample.index else None
        narrowed_check = _top_k_by_embedding(cand_emb_check, existing_types_for_log, cgrr_top_k)
        _cgrr_search_log.append({
            "relation_type": cand_type,
            "method": "in_memory_cosine_top_k",
            "top_k": cgrr_top_k,
            "existing_types_total": len(existing_types_for_log),
            "candidates_after_topk": len(narrowed_check),
            "has_embedding": cand_emb_check is not None,
            "top_matches": [
                {
                    "relation_type": et.get("relation_type", ""),
                    "edge_count": et.get("edge_count", 0),
                    "description": (et.get("description", "") or "")[:100],
                }
                for et in narrowed_check[:5]
            ],
        })
    _save_debug_json(debug_dir, "cgrr_vector_search_log.json", _cgrr_search_log)

    for cand_type in candidate_types_before:
        # Check exact match
        exact_exists = any(et["relation_type"] == cand_type for et in existing_types_for_log)
        if exact_exists:
            cgrr_resolution_log.append({
                "phase": "A_cross_graph",
                "relation_type": cand_type,
                "edge_count": cand_type_counts.get(cand_type, 0),
                "best_match": cand_type,
                "best_score": 1.0,
                "decision": "EXACT_MATCH",
                "top_comparisons": [],
            })
            continue

        cand_sample = relationships_df[relationships_df["relation_type"] == cand_type].iloc[0]
        cand_desc = str(cand_sample.get("description", ""))
        cand_src = str(cand_sample.get("source", ""))
        cand_tgt = str(cand_sample.get("target", ""))

        # Get candidate embedding for top-K pre-filter
        cand_emb = cand_sample.get("description_embedding") if "description_embedding" in cand_sample.index else None

        # Collect all entities for this candidate type
        cand_rows = relationships_df[relationships_df["relation_type"] == cand_type]
        cand_entities = set(
            cand_rows["source"].dropna().tolist()
            + cand_rows["target"].dropna().tolist()
        )

        # Top-K pre-filter by description embedding similarity
        narrowed = _top_k_by_embedding(cand_emb, existing_types_for_log, cgrr_top_k)

        all_comparisons: list[dict[str, Any]] = []
        for existing in narrowed:
            if cand_type == existing["relation_type"]:
                continue
            # Skip if no entity overlap
            existing_entities = set(
                existing.get("all_sources", [existing.get("source", "")])
                + existing.get("all_targets", [existing.get("target", "")])
            )
            if not cand_entities.intersection(existing_entities):
                continue
            score, breakdown = compute_relationship_score(
                candidate_rel_type=cand_type,
                candidate_description=cand_desc,
                candidate_source=cand_src,
                candidate_target=cand_tgt,
                existing_rel_type=existing["relation_type"],
                existing_description=existing["description"],
                existing_source=existing["source"],
                existing_target=existing["target"],
                config=config,
            )
            all_comparisons.append({
                "existing_type": existing["relation_type"],
                "score": round(score, 4),
                "bm25_type": round(breakdown.get("bm25_type", 0), 4),
                "semantic_desc": round(breakdown.get("semantic_desc", 0), 4),
                "endpoint_match": round(breakdown.get("endpoint_match", 0), 4),
            })

        all_comparisons.sort(key=lambda c: c["score"], reverse=True)
        best = all_comparisons[0] if all_comparisons else {}
        best_score_val = best.get("score", 0)
        cgrr_resolution_log.append({
            "phase": "A_cross_graph",
            "relation_type": cand_type,
            "edge_count": cand_type_counts.get(cand_type, 0),
            "best_match": best.get("existing_type", ""),
            "best_score": best_score_val,
            "decision": (
                "AUTO_NORMALIZE" if best_score_val >= config.cgrr_merge_threshold
                else "LLM_ZONE" if best_score_val >= config.cgrr_llm_threshold_low
                else "BELOW_THRESHOLD"
            ),
            "top_comparisons": all_comparisons[:10],
        })

    # Select relationship scorer based on config
    cgrr_scorer_name = getattr(config, "cgrr_scorer", "embedding_only")
    if cgrr_scorer_name == "bm25_only":
        from graphrag.bt_graphrag.entity_resolution.scorers import (
            bm25_only_relationship_scorer,
        )
        relationship_scorer = bm25_only_relationship_scorer
    elif cgrr_scorer_name == "type_and_endpoint":
        from graphrag.bt_graphrag.entity_resolution.scorers import (
            type_and_endpoint_relationship_scorer,
        )
        relationship_scorer = type_and_endpoint_relationship_scorer
    elif cgrr_scorer_name == "composite":
        from graphrag.bt_graphrag.entity_resolution.cgrr import (
            compute_relationship_score,
        )
        relationship_scorer = compute_relationship_score
    else:  # "embedding_only" (default)
        from graphrag.bt_graphrag.entity_resolution.scorers import (
            semantic_only_relationship_scorer,
        )
        relationship_scorer = semantic_only_relationship_scorer

    print(f"  CGRR scorer: {cgrr_scorer_name}")

    # resolve_relationships handles Phase A (new vs existing) and Phase B (intra-batch)
    async with driver.session(database=config.neo4j_database) as session:
        relationships_df, normalize_map, phase_b_log_cgrr = await resolve_relationships(
            relationships_df=relationships_df,
            config=config,
            session=session,
            model=model,
            relationship_scorer=relationship_scorer,
        )

    # Update cardinality column after normalization
    if normalize_map:
        relationships_df = apply_normalize_map_to_cardinality(
            relationships_df, normalize_map, config
        )
        print(f"\n  Cardinality column updated for {len(normalize_map)} normalized types")

    unique_after = relationships_df["relation_type"].nunique()
    type_counts_after = relationships_df["relation_type"].value_counts()
    print(f"\n  Relation type distribution AFTER CGRR ({unique_after} unique):")
    for rt, count in type_counts_after.head(20).items():
        card = config.get_cardinality(str(rt))
        change = ""
        if count != type_counts_before.get(rt, 0):
            old = type_counts_before.get(rt, 0)
            change = f"  (+{count - old})" if count > old else ""
        print(f"    {str(rt)[:40]:40s}  {count:4d} edges  [{card}]{change}")
    if len(type_counts_after) > 20:
        print(f"    ... and {len(type_counts_after) - 20} more types")

    reduction = unique_before - unique_after
    print(f"\n  CGRR complete: {unique_before} -> {unique_after} unique relation types "
          f"({reduction} types normalized)")
    print(f"  Normalizations applied: {len(normalize_map)}")
    if reduction > 0:
        print(f"  Impact: ETCDR will now correctly detect cross-alias conflicts")

    # --- CGRR normalize map detail table ---
    if normalize_map:
        print(f"\n  {'─' * 90}")
        print(f"  CGRR NORMALIZE MAP DETAIL")
        print(f"  {'─' * 90}")
        print(f"  {'#':>4}  {'ORIGINAL RELATION TYPE':40s}  {'NORMALIZED TO':40s}")
        print(f"  {'─' * 90}")
        for i, (orig, canonical) in enumerate(normalize_map.items()):
            print(f"  {i:4d}  {orig[:40]:40s}  {canonical[:40]:40s}")
        print(f"  {'─' * 90}")

    # Save CGRR debug files (combine Phase A and Phase B logs)
    full_cgrr_log = cgrr_resolution_log + phase_b_log_cgrr
    _save_debug_json(debug_dir, "stage2b_cgrr_normalize_map.json", normalize_map)
    _save_debug_json(debug_dir, "stage2b_cgrr_resolution_log.json", full_cgrr_log)

    return relationships_df


# ---------------------------------------------------------------------------
# Stage 2c: LLM Cardinality Classification
# ---------------------------------------------------------------------------


async def _classify_cardinalities(
    relationships_df: pd.DataFrame,
    config: BTGraphRAGConfig,
    model: "LLMCompletion | None",
) -> pd.DataFrame:
    """Classify cardinality for relation types not in the seed map using LLM.

    For each unique relation_type not already known in
    ``default_cardinality_map`` or ``relation_cardinality_overrides``,
    the LLM is asked to classify it.  Results are cached in
    ``config.relation_cardinality_overrides`` so subsequent batches
    benefit automatically.

    Returns the DataFrame with an updated ``cardinality`` column.
    """
    logger.info("BT-GraphRAG Stage 2c: Classifying relation cardinalities")
    print("\n" + "-" * 70)
    print("  Stage 2c: LLM Cardinality Classification")
    print("-" * 70)

    unique_types = relationships_df["relation_type"].dropna().unique().tolist()

    # Identify types that are NOT already classified
    unknown_types = []
    for rt in unique_types:
        key = rt.upper().replace(" ", "_")
        if key not in config.default_cardinality_map and key not in config.relation_cardinality_overrides:
            unknown_types.append(rt)

    already_known = len(unique_types) - len(unknown_types)
    print(f"  Relation types in batch:   {len(unique_types)}")
    print(f"  Already classified:        {already_known}")
    print(f"  Need classification:       {len(unknown_types)}")

    if not unknown_types:
        print("  All relation types already classified — skipping LLM calls")
        # Still refresh the cardinality column
        relationships_df = relationships_df.copy()
        relationships_df["cardinality"] = relationships_df["relation_type"].apply(
            lambda rt: config.get_cardinality(rt)
        )
        return relationships_df

    if model is None:
        print("  WARNING: No LLM available — falling back to NON_EXCLUSIVE for unknown types")
        relationships_df = relationships_df.copy()
        relationships_df["cardinality"] = relationships_df["relation_type"].apply(
            lambda rt: config.get_cardinality(rt)
        )
        return relationships_df

    print(f"\n  Classifying {len(unknown_types)} relation types via LLM...")
    from graphrag_llm.utils import CompletionMessagesBuilder

    prompt_template = config.resolved_cardinality_prompt()
    valid_values = {"SUBJECT_EXCLUSIVE", "OBJECT_EXCLUSIVE", "BOTH_EXCLUSIVE", "NON_EXCLUSIVE"}

    for rt in unknown_types:
        # Build context examples from the batch
        sample_rows = relationships_df[relationships_df["relation_type"] == rt].head(5)
        examples = []
        for _, row in sample_rows.iterrows():
            examples.append(
                f"  ({row.get('source', '?')}) -[{rt}]-> ({row.get('target', '?')}): "
                f"{str(row.get('description', ''))[:80]}"
            )
        context = "\n".join(examples) if examples else "No examples available"

        prompt = prompt_template.format(
            relation_type=rt,
            context_examples=context,
        )

        messages = CompletionMessagesBuilder().add_user_message(prompt).build()
        response = await model.completion_async(messages=messages)
        answer = response.content.strip().upper()

        # Parse — check in priority order (most restrictive first) so that
        # a verbose response containing "NON_EXCLUSIVE" in its explanation
        # does not shadow a more specific classification on the first line.
        classification = "NON_EXCLUSIVE"
        for v in ("BOTH_EXCLUSIVE", "SUBJECT_EXCLUSIVE", "OBJECT_EXCLUSIVE", "NON_EXCLUSIVE"):
            if v in answer:
                classification = v
                break

        key = rt.upper().replace(" ", "_")
        config.relation_cardinality_overrides[key] = classification

        edge_count = int((relationships_df["relation_type"] == rt).sum())
        print(f"    {rt[:40]:40s}  ({edge_count:3d} edges) -> {classification}")

    # Refresh the cardinality column with the new classifications
    relationships_df = relationships_df.copy()
    relationships_df["cardinality"] = relationships_df["relation_type"].apply(
        lambda rt: config.get_cardinality(rt)
    )

    # Summary
    card_dist = relationships_df["cardinality"].value_counts()
    print(f"\n  Cardinality distribution after classification:")
    for card, cnt in card_dist.items():
        print(f"    {card:20s}: {cnt}")
    print(f"  Total overrides cached:    {len(config.relation_cardinality_overrides)}")

    return relationships_df


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
    strategies_used = []
    confidence_values = []
    cardinality_dist: dict[str, int] = {}
    etcdr_debug_log: list[dict[str, Any]] = []  # per-edge conflict resolution log

    print(f"\n    Processing {len(relationships_df)} edges through ETCDR...")
    print(f"    Late arrival threshold: {config.late_arrival_threshold_days} days")
    print(f"    Confidence threshold:   {config.etcdr_confidence_threshold}")
    print(f"    LLM available:          {model is not None}")
    print(f"    Intra-batch detection:  ENABLED")
    print()

    # Accepted batch tracks relationships that passed ETCDR so far, enabling
    # intra-batch conflict detection for subsequent candidates.
    accepted_batch: list[TemporalRelationship] = []
    # All candidates in DataFrame order — needed to write back mutated temporal
    # quads (t_valid_end, t_tx_end) after intra-batch evolution/correction.
    all_candidates: list[TemporalRelationship] = []

    async with driver.session(database=config.neo4j_database) as session:
        for edge_num, (idx, row) in enumerate(relationships_df.iterrows()):
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

            rel_type = str(row.get("relation_type", "")).strip()
            if not rel_type:
                rel_type = _normalize_relation_type(row.get("description", "RELATED_TO"))

            row_id = row.get("id")
            edge_id = str(row_id) if row_id else str(uuid4())

            row_card = row.get("cardinality")
            card_enum = (
                RelationCardinality(str(row_card))
                if row_card and str(row_card) in RelationCardinality.__members__
                else RelationCardinality(config.get_cardinality(rel_type))
            )

            candidate = TemporalRelationship(
                id=edge_id,
                source=str(row.get("source", "")),
                target=str(row.get("target", "")),
                relation_type=rel_type,
                description=str(row.get("description", "")),
                weight=float(row.get("weight", 1.0)),
                confidence=float(row.get("confidence", 1.0)),
                cardinality=card_enum,
                temporal_quad=TemporalStateQuad(
                    t_valid_start=t_valid_start,
                    t_valid_end=t_valid_end,
                    t_tx_start=t_tx_start,
                    t_tx_end=INFINITY,
                ),
                text_unit_ids=row.get("text_unit_ids", []),
            )

            # Track cardinality
            card_val = config.get_cardinality(rel_type)
            cardinality_dist[card_val] = cardinality_dist.get(card_val, 0) + 1

            # Run bidirectional conflict detection (Neo4j + intra-batch)
            conflict_result = await detect_and_resolve(
                candidate=candidate,
                session=session,
                config=config,
                model=model,
                is_late_arrival=batch_is_late,
                t_event=batch_t_event,
                edge_index=edge_num,
                accepted_batch=accepted_batch,
            )

            statuses.append(conflict_result.candidate.status)
            strategy_val = conflict_result.strategy.value if conflict_result.strategy else "NONE"
            if conflict_result.strategy:
                strategies_used.append(strategy_val)
            confidence_values.append(conflict_result.confidence)

            # Track every candidate in order so we can write back mutated quads
            all_candidates.append(candidate)

            # Track accepted (non-retracted) candidates for intra-batch detection
            if candidate.status != "retracted":
                accepted_batch.append(candidate)

            # Build per-edge debug record
            etcdr_debug_log.append({
                "edge_index": edge_num,
                "source": candidate.source,
                "relation_type": candidate.relation_type,
                "target": candidate.target,
                "t_valid_start": candidate.temporal_quad.t_valid_start.isoformat() if candidate.temporal_quad else None,
                "t_valid_end": candidate.temporal_quad.t_valid_end.isoformat() if candidate.temporal_quad else None,
                "confidence": candidate.confidence,
                "cardinality": card_val,
                "is_late_arrival": batch_is_late,
                "subject_conflicts_count": len(conflict_result.subject_conflicts),
                "object_conflicts_count": len(conflict_result.object_conflicts),
                "intra_batch_subject_conflicts_count": len(conflict_result.intra_batch_subject_conflicts),
                "intra_batch_object_conflicts_count": len(conflict_result.intra_batch_object_conflicts),
                "has_conflicts": conflict_result.has_conflicts,
                "strategy": strategy_val,
                "decision_confidence": conflict_result.confidence,
                "status": conflict_result.candidate.status,
                "subject_conflicts": [
                    {
                        "id": sc.id,
                        "source": sc.source,
                        "target": sc.target,
                        "relation_type": sc.relation_type,
                        "description": sc.description,
                    }
                    for sc in conflict_result.subject_conflicts
                ],
                "object_conflicts": [
                    {
                        "id": oc.id,
                        "source": oc.source,
                        "target": oc.target,
                        "relation_type": oc.relation_type,
                        "description": oc.description,
                    }
                    for oc in conflict_result.object_conflicts
                ],
                "intra_batch_subject_conflicts": [
                    {
                        "id": isc.id,
                        "source": isc.source,
                        "target": isc.target,
                        "relation_type": isc.relation_type,
                        "description": isc.description,
                    }
                    for isc in conflict_result.intra_batch_subject_conflicts
                ],
                "intra_batch_object_conflicts": [
                    {
                        "id": ioc.id,
                        "source": ioc.source,
                        "target": ioc.target,
                        "relation_type": ioc.relation_type,
                        "description": ioc.description,
                    }
                    for ioc in conflict_result.intra_batch_object_conflicts
                ],
            })

            # Print separator between edges (for readability)
            if edge_num < len(relationships_df) - 1:
                print()

    # Write back status, t_valid_end, and t_tx_end to the DataFrame so that
    # Stage 4 (Neo4j write) sees all intra-batch mutations.
    #
    # Why all_candidates instead of the `statuses` list:
    #   - `statuses` is appended at candidate-processing time, so it captures
    #     the status *before* a later candidate may apply CORRECTION to it.
    #   - all_candidates holds live object references; any mutation applied by
    #     apply_intra_batch_evolution / apply_intra_batch_correction is already
    #     reflected here (Python reference semantics).
    if all_candidates and len(all_candidates) == len(relationships_df):
        relationships_df = relationships_df.copy()
        relationships_df["status"] = [c.status for c in all_candidates]
        relationships_df["t_valid_end"] = [
            c.temporal_quad.t_valid_end.isoformat() if c.temporal_quad else INFINITY_ISO
            for c in all_candidates
        ]
        relationships_df["t_tx_end"] = [
            c.temporal_quad.t_tx_end.isoformat() if c.temporal_quad else INFINITY_ISO
            for c in all_candidates
        ]

    # --- Verification Summary ---
    from collections import Counter
    status_counts = Counter(statuses)
    strategy_counts = Counter(strategies_used)

    print(f"\n    {'=' * 55}")
    print(f"    ETCDR VERIFICATION SUMMARY")
    print(f"    {'=' * 55}")
    print(f"    Total edges processed:  {len(relationships_df)}")
    print(f"    Late arrival batch:     {batch_is_late}")

    # Count intra-batch conflicts
    total_intra_batch = sum(
        1 for e in etcdr_debug_log
        if e.get("intra_batch_subject_conflicts_count", 0) > 0
        or e.get("intra_batch_object_conflicts_count", 0) > 0
    )

    print(f"\n    Edge Status Distribution:")
    for st, cnt in status_counts.most_common():
        bar = "#" * min(cnt, 40)
        print(f"      {st:15s}: {cnt:5d}  {bar}")

    print(f"\n    Resolution Strategy Distribution:")
    for st, cnt in strategy_counts.most_common():
        bar = "#" * min(cnt, 40)
        print(f"      {st:15s}: {cnt:5d}  {bar}")

    print(f"\n    Intra-batch conflicts:    {total_intra_batch} edge(s) had within-batch conflicts")
    print(f"    Accepted batch size:      {len(accepted_batch)} edge(s) tracked")

    print(f"\n    Cardinality Distribution:")
    for card, cnt in sorted(cardinality_dist.items()):
        print(f"      {card:20s}: {cnt}")

    if confidence_values:
        avg_conf = sum(confidence_values) / len(confidence_values)
        min_conf = min(confidence_values)
        max_conf = max(confidence_values)
        print(f"\n    Decision Confidence:")
        print(f"      min={min_conf:.3f}  max={max_conf:.3f}  avg={avg_conf:.3f}")

        # Flag low-confidence decisions
        low_conf = [c for c in confidence_values if c < config.etcdr_confidence_threshold]
        if low_conf:
            print(f"      WARNING: {len(low_conf)} decisions below threshold ({config.etcdr_confidence_threshold})")

    # Verify no active+disputed inconsistency
    active_and_disputed = sum(1 for s in statuses if s == "disputed")
    if active_and_disputed > 0:
        print(f"\n    Disputed edges:         {active_and_disputed} (will be inserted with status='disputed')")

    print(f"    {'=' * 55}")

    # --- ETCDR conflict resolution detail table ---
    if etcdr_debug_log:
        print(f"\n    {'─' * 130}")
        print(f"    ETCDR CONFLICT RESOLUTION DETAIL TABLE")
        print(f"    {'─' * 130}")
        print(f"    {'#':>4}  {'SOURCE':20s}  {'RELATION':22s}  {'TARGET':20s}  "
              f"{'STRATEGY':15s}  {'CONF':5s}  {'S_CNF':5s}  {'O_CNF':5s}  {'IB_S':5s}  {'IB_O':5s}  {'STATUS':10s}  {'CARDINALITY':18s}")
        print(f"    {'─' * 150}")
        for entry in etcdr_debug_log:
            print(f"    {entry['edge_index']:4d}  "
                  f"{str(entry['source'])[:20]:20s}  "
                  f"{str(entry['relation_type'])[:22]:22s}  "
                  f"{str(entry['target'])[:20]:20s}  "
                  f"{entry['strategy']:15s}  "
                  f"{entry['decision_confidence']:.2f}   "
                  f"{entry['subject_conflicts_count']:5d}  "
                  f"{entry['object_conflicts_count']:5d}  "
                  f"{entry.get('intra_batch_subject_conflicts_count', 0):5d}  "
                  f"{entry.get('intra_batch_object_conflicts_count', 0):5d}  "
                  f"{str(entry['status'])[:10]:10s}  "
                  f"{entry['cardinality']:18s}")
        print(f"    {'─' * 150}")

    # Save ETCDR debug log to file
    debug_dir = _ensure_debug_dir(config)
    _save_debug_json(debug_dir, "stage3_etcdr_conflict_log.json", etcdr_debug_log)

    # Save dedicated intra-batch conflict summary
    intra_batch_entries = [
        entry for entry in etcdr_debug_log
        if entry.get("intra_batch_subject_conflicts_count", 0) > 0
        or entry.get("intra_batch_object_conflicts_count", 0) > 0
    ]
    _save_debug_json(
        debug_dir,
        "stage3_etcdr_intra_batch_log.json",
        {
            "total_edges_processed": len(etcdr_debug_log),
            "edges_with_intra_batch_conflicts": len(intra_batch_entries),
            "accepted_batch_size": len(accepted_batch),
            "entries": intra_batch_entries,
        },
    )

    # Flush the ETCDR resolution JSON log
    from graphrag.bt_graphrag.conflict_detection.etcdr import flush_resolution_log

    log_path = flush_resolution_log(output_dir=debug_dir)
    if log_path:
        print(f"    ETCDR resolution log written to: {log_path}")

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
    from collections import Counter

    from graphrag.bt_graphrag.models.temporal_types import EpistemicState
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
    print(f"  Target database:         {config.neo4j_database}")

    # init_schema already called before Stage 3 in run_bt_pipeline

    entity_actions = Counter()  # CREATED / MATCHED
    rel_epistemic = Counter()   # epistemic state distribution
    rel_skipped = 0
    rel_written = 0

    async with driver.session(database=config.neo4j_database) as session:
        # Write entities
        print(f"\n    [Entities] Writing {len(entities_df)} entities...")
        for i, (_, row) in enumerate(entities_df.iterrows()):
            _emb = row.get("description_embedding")
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
                description_embedding=_emb if isinstance(_emb, list) else None,
            )
            action = await upsert_entity(session, entity)
            entity_actions[action] += 1

            # Print first 10 and then every 50th for large batches
            if i < 10 or (i > 0 and i % 50 == 0):
                print(f"      [{i}] {action:8s} '{entity.title[:40]:40s}'  type={str(entity.type)[:15]}  "
                      f"active={entity.active_start.strftime('%Y-%m-%d') if entity.active_start else '?'}")

        if len(entities_df) > 10:
            print(f"      ... ({len(entities_df)} total)")

        print(f"\n    Entity write summary:")
        print(f"      CREATED (new):   {entity_actions.get('CREATED', 0)}")
        print(f"      MATCHED (merge): {entity_actions.get('MATCHED', 0)}")

        # Write relationships
        print(f"\n    [Relationships] Writing {len(relationships_df)} relationships...")
        for i, (_, row) in enumerate(relationships_df.iterrows()):
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

            rel_type = str(row.get("relation_type", "")).strip()
            if not rel_type:
                rel_type = _normalize_relation_type(row.get("description", "RELATED_TO"))

            quad = TemporalStateQuad(
                t_valid_start=t_valid_start,
                t_valid_end=t_valid_end,
                t_tx_start=t_tx_start,
                t_tx_end=t_tx_end,
            )

            _desc_emb = row.get("description_embedding")
            _type_emb = row.get("relation_type_embedding")

            row_card = row.get("cardinality")
            card_enum = (
                RelationCardinality(str(row_card))
                if row_card and str(row_card) in RelationCardinality.__members__
                else RelationCardinality(config.get_cardinality(rel_type))
            )

            rel = TemporalRelationship(
                id=str(row.get("id") or uuid4()),
                source=str(row.get("source", "")),
                target=str(row.get("target", "")),
                relation_type=rel_type,
                description=str(row.get("description", "")),
                weight=float(row.get("weight", 1.0)),
                confidence=float(row.get("confidence", 1.0)),
                cardinality=card_enum,
                temporal_quad=quad,
                status=str(row.get("status", "active")),
                support_count=int(row.get("support_count", 1)),
                description_embedding=_desc_emb if isinstance(_desc_emb, list) else None,
                relation_type_embedding=_type_emb if isinstance(_type_emb, list) else None,
            )
            edge_id = await insert_relationship(session, rel)

            # Track epistemic state
            epistemic = quad.epistemic_state.value
            rel_epistemic[epistemic] += 1
            rel_written += 1

            # Print first 10 and then every 50th
            if i < 10 or (i > 0 and i % 50 == 0):
                tv_s = t_valid_start.strftime("%Y-%m-%d")
                tv_e = "INF" if t_valid_end >= INFINITY else t_valid_end.strftime("%Y-%m-%d")
                print(f"      [{i}] ({rel.source[:20]}) -[{rel_type[:20]}]-> ({rel.target[:20]})")
                print(f"           Quad=[{tv_s}->{tv_e}] tx=[{t_tx_start.strftime('%Y-%m-%d')}->"
                      f"{'INF' if t_tx_end >= INFINITY else t_tx_end.strftime('%Y-%m-%d')}]  "
                      f"state={epistemic}  status={rel.status}  conf={rel.confidence:.2f}")

        if len(relationships_df) > 10:
            print(f"      ... ({len(relationships_df)} total)")

    # -----------------------------------------------------------------------
    # Post-write Verification
    # -----------------------------------------------------------------------
    print(f"\n    {'=' * 55}")
    print(f"    STAGE 4 WRITE SUMMARY")
    print(f"    {'=' * 55}")
    print(f"    Entities:  {entity_actions.get('CREATED', 0)} created + "
          f"{entity_actions.get('MATCHED', 0)} merged = {sum(entity_actions.values())} total")
    print(f"    Edges:     {rel_written} written, {rel_skipped} skipped")
    print(f"\n    Epistemic State Distribution:")
    for state, count in sorted(rel_epistemic.items()):
        bar = "#" * min(count, 40)
        print(f"      {state:25s}: {count:5d}  {bar}")
    print(f"      {'TOTAL':25s}: {sum(rel_epistemic.values()):5d}")

    # Status distribution
    if "status" in relationships_df.columns:
        status_counts = relationships_df["status"].value_counts()
        print(f"\n    Edge Status Distribution:")
        for status, count in status_counts.items():
            print(f"      {str(status):15s}: {count}")

    # Confidence distribution
    if "confidence" in relationships_df.columns:
        confs = relationships_df["confidence"].astype(float)
        print(f"\n    Confidence Stats:")
        print(f"      min={confs.min():.3f}  max={confs.max():.3f}  "
              f"mean={confs.mean():.3f}  median={confs.median():.3f}")

    # Post-write Neo4j verification query
    print(f"\n    [Verification] Querying Neo4j for actual counts...")
    async with driver.session(database=config.neo4j_database) as verify_session:
        v_result = await verify_session.run(
            """
            MATCH (n:Entity) WITH count(n) AS entity_count
            OPTIONAL MATCH ()-[r:RELATIONSHIP]->() WITH entity_count, count(r) AS rel_count
            OPTIONAL MATCH ()-[r2:RELATIONSHIP]->() WHERE r2.t_tx_end = $infinity AND r2.t_valid_end = $infinity
            RETURN entity_count, rel_count, count(r2) AS active_rel_count
            """,
            infinity=INFINITY_ISO,
        )
        v_rec = await v_result.single()
        if v_rec:
            neo4j_entities = v_rec["entity_count"]
            neo4j_rels = v_rec["rel_count"]
            neo4j_active = v_rec["active_rel_count"]
            print(f"      Neo4j total entities:         {neo4j_entities}")
            print(f"      Neo4j total relationships:    {neo4j_rels}")
            print(f"      Neo4j active (CURRENT_TRUTH): {neo4j_active}")

            # Check for orphan entities (no relationships)
            orphan_result = await verify_session.run(
                """
                MATCH (n:Entity)
                WHERE NOT (n)-[:RELATIONSHIP]-() AND NOT (n)<-[:RELATIONSHIP]-()
                RETURN count(n) AS orphan_count
                """
            )
            orphan_rec = await orphan_result.single()
            orphan_count = orphan_rec["orphan_count"] if orphan_rec else 0
            if orphan_count > 0:
                print(f"      WARNING: {orphan_count} orphan entities (no relationships)")
            else:
                print(f"      Orphan entity check:          PASS (all connected)")

    print(f"    {'=' * 55}")
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
