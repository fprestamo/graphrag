# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG temporal graph extraction workflow.

Replaces the standard extract_graph workflow when BT-GraphRAG is enabled.
Uses temporal-aware prompts (Stage 1) and then runs the BT pipeline
(Stages 2-4) before writing results.
"""

import logging
from typing import TYPE_CHECKING

import pandas as pd
from graphrag_llm.completion import create_completion
from graphrag_llm.embedding import create_embedding

from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.pipeline import run_bt_pipeline
from graphrag.bt_graphrag.temporal_extraction.temporal_normalization import (
    assign_document_timestamps,
)
from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.extract_graph.extract_graph import (
    extract_graph as standard_extractor,
)
from graphrag.index.operations.summarize_descriptions.summarize_descriptions import (
    summarize_descriptions,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """BT-GraphRAG temporal graph extraction workflow.

    This workflow:
    1. Extracts entities/relationships using temporal-aware prompts (Stage 1)
    2. Runs CGER entity resolution (Stage 2)
    3. Runs ETCDR conflict detection (Stage 3)
    4. Writes to both parquet (standard) and Neo4j (Stage 4)
    """
    logger.info("Workflow started: bt_extract_graph")
    print("\n" + "#" * 70)
    print("  BT-GraphRAG Workflow: bt_extract_graph")
    print("#" * 70)

    reader = DataReader(context.output_table_provider)
    text_units = await reader.text_units()

    # Load documents for temporal metadata
    documents = await reader.documents()

    # Get BT config
    bt_config = _get_bt_config(config)

    print(f"\n  Input: {len(text_units)} text units, {len(documents)} documents")
    print(f"  BT-GraphRAG enabled: {bt_config.enabled}")
    print(f"  Model: {config.extract_graph.completion_model_id}")

    # -----------------------------------------------------------------------
    # Stage 1: Temporal Extraction
    # -----------------------------------------------------------------------
    # Create extraction model
    extraction_model_config = config.get_completion_model_config(
        config.extract_graph.completion_model_id
    )
    extraction_model = create_completion(
        extraction_model_config,
        cache=context.cache.child(config.extract_graph.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    # Build temporal-aware prompt
    temporal_prompt = _build_temporal_prompt(documents, bt_config)

    print("\n" + "-" * 70)
    print("  Stage 1: Temporal Graph Extraction (LLM)")
    print("-" * 70)
    print(f"  Entity types: {config.extract_graph.entity_types}")
    print(f"  Max gleanings: {config.extract_graph.max_gleanings}")
    print(f"  Concurrent requests: {config.concurrent_requests}")

    # Run extraction with temporal prompt
    extracted_entities, extracted_relationships = await standard_extractor(
        text_units=text_units,
        callbacks=context.callbacks,
        text_column="text",
        id_column="id",
        model=extraction_model,
        prompt=temporal_prompt,
        entity_types=config.extract_graph.entity_types,
        max_gleanings=config.extract_graph.max_gleanings,
        num_threads=config.concurrent_requests,
        async_type=config.async_mode,
    )

    if len(extracted_entities) == 0:
        error_msg = "BT Graph Extraction failed. No entities detected."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(extracted_relationships) == 0:
        error_msg = "BT Graph Extraction failed. No relationships detected."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Report extraction results
    print(f"\n  ✓ Stage 1 Complete:")
    print(f"    Entities extracted:      {len(extracted_entities)}")
    print(f"    Relationships extracted: {len(extracted_relationships)}")
    entity_types = extracted_entities['type'].value_counts() if 'type' in extracted_entities.columns else {}
    if len(entity_types) > 0:
        print(f"    Entity types breakdown:")
        for etype, count in entity_types.head(10).items():
            print(f"      {etype:20s}: {count}")
    print(f"    Sample entities:")
    for _, row in extracted_entities.head(8).iterrows():
        print(f"      • {str(row.get('title', '?'))[:45]:45s}  [{str(row.get('type', '?'))[:12]}]")
    print(f"    Sample relationships:")
    for _, row in extracted_relationships.head(8).iterrows():
        src = str(row.get('source', '?'))[:20]
        tgt = str(row.get('target', '?'))[:20]
        desc = str(row.get('description', '?'))[:35]
        print(f"      • ({src}) —> ({tgt}): {desc}")

    # Parse temporal fields from extraction results
    extracted_relationships = _parse_temporal_fields_from_extraction(
        extracted_relationships, documents, bt_config
    )
    print(f"\n  [Temporal Parse] Temporal columns added to relationships: "
          f"t_valid_start, t_valid_end, t_tx_start, t_tx_end, confidence, status, support_count")

    # Save raw copies
    raw_entities = extracted_entities.copy()
    raw_relationships = extracted_relationships.copy()

    # -----------------------------------------------------------------------
    # Summarize descriptions (same as standard)
    # -----------------------------------------------------------------------
    summarization_model_config = config.get_completion_model_config(
        config.summarize_descriptions.completion_model_id
    )
    summarization_prompts = config.summarize_descriptions.resolved_prompts()
    summarization_model = create_completion(
        summarization_model_config,
        cache=context.cache.child(config.summarize_descriptions.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    entity_summaries, relationship_summaries = await summarize_descriptions(
        entities_df=extracted_entities,
        relationships_df=extracted_relationships,
        callbacks=context.callbacks,
        model=summarization_model,
        max_summary_length=config.summarize_descriptions.max_length,
        max_input_tokens=config.summarize_descriptions.max_input_tokens,
        prompt=summarization_prompts.summarize_prompt,
        num_threads=config.concurrent_requests,
    )
    print(f"  ✓ Description summarization complete")
    print(f"    Entity summaries:       {len(entity_summaries)}")
    print(f"    Relationship summaries: {len(relationship_summaries)}")

    # Merge summaries back
    relationships = extracted_relationships.drop(columns=["description"], errors="ignore").merge(
        relationship_summaries, on=["source", "target"], how="left"
    )
    extracted_entities.drop(columns=["description"], inplace=True, errors="ignore")
    entities = extracted_entities.merge(entity_summaries, on="title", how="left")

    # -----------------------------------------------------------------------
    # Embedding enrichment — computed on summarized descriptions
    # -----------------------------------------------------------------------
    try:
        embedding_model_config = config.get_embedding_model_config(
            config.embed_text.embedding_model_id
        )
        embedding_model = create_embedding(
            embedding_model_config,
            cache=context.cache.child(config.embed_text.model_instance_name),
            cache_key_creator=cache_key_creator,
        )
        from graphrag.bt_graphrag.temporal_extraction.embedding_enrichment import (
            embed_dataframes,
            enrich_entities_with_text_unit_embeddings,
        )
        entities, relationships = await embed_dataframes(
            entities, relationships, embedding_model
        )
        entities = await enrich_entities_with_text_unit_embeddings(
            entities, text_units, embedding_model
        )
        ent_has = entities["description_embedding"].apply(bool).sum() if "description_embedding" in entities.columns else 0
        ent_cite_has = entities["text_unit_embedding"].apply(bool).sum() if "text_unit_embedding" in entities.columns else 0
        rel_has = relationships["description_embedding"].apply(bool).sum() if "description_embedding" in relationships.columns else 0
        rel_type_has = relationships["relation_type_embedding"].apply(bool).sum() if "relation_type_embedding" in relationships.columns else 0
        print(f"\n  ✓ Embeddings computed:")
        print(f"    entity description_embedding       : {ent_has} / {len(entities)}")
        print(f"    entity text_unit_embedding         : {ent_cite_has} / {len(entities)}")
        print(f"    relationship description_embedding : {rel_has} / {len(relationships)}")
        print(f"    relationship relation_type_embedding: {rel_type_has} / {len(relationships)}")
    except Exception as exc:
        logger.warning("Embedding enrichment failed — continuing without embeddings: %s", exc)
        print(f"\n  ⚠ Embedding enrichment skipped: {exc}")

    # -----------------------------------------------------------------------
    # Stages 2-4: BT Pipeline (CGER, ETCDR, Neo4j write)
    # -----------------------------------------------------------------------
    neo4j_driver = None
    if bt_config.enabled:
        try:
            neo4j_driver = await _get_neo4j_driver(bt_config)
            print(f"  ✓ Connected to Neo4j at {bt_config.neo4j_uri}")
        except Exception:
            print(f"  ✗ Could not connect to Neo4j at {bt_config.neo4j_uri} — parquet only")
            logger.warning(
                "Could not connect to Neo4j at %s. "
                "BT-GraphRAG will save to parquet only.",
                bt_config.neo4j_uri,
            )

    if bt_config.enabled:
        entities, relationships = await run_bt_pipeline(
            entities_df=entities,
            relationships_df=relationships,
            text_units_df=text_units,
            documents_df=documents,
            config=bt_config,
            extraction_model=extraction_model,
            neo4j_driver=neo4j_driver,
        )

    if neo4j_driver is not None:
        await neo4j_driver.close()

    # -----------------------------------------------------------------------
    # Write to parquet (standard GraphRAG output)
    # -----------------------------------------------------------------------
    await context.output_table_provider.write_dataframe("entities", entities)
    await context.output_table_provider.write_dataframe("relationships", relationships)
    print(f"\n  ✓ Written to parquet: {len(entities)} entities, {len(relationships)} relationships")

    if config.snapshots.raw_graph:
        await context.output_table_provider.write_dataframe(
            "raw_entities", raw_entities
        )
        await context.output_table_provider.write_dataframe(
            "raw_relationships", raw_relationships
        )

    logger.info("Workflow completed: bt_extract_graph")
    print("\n" + "#" * 70)
    print("  bt_extract_graph COMPLETE")
    print(f"  Final: {len(entities)} entities, {len(relationships)} relationships")
    print("#" * 70 + "\n")
    return WorkflowFunctionOutput(
        result={
            "entities": entities,
            "relationships": relationships,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_bt_config(config: GraphRagConfig) -> BTGraphRAGConfig:
    """Extract BTGraphRAGConfig from GraphRagConfig."""
    if hasattr(config, "bt_graphrag") and config.bt_graphrag is not None:
        bt_dict = config.bt_graphrag
        if isinstance(bt_dict, dict):
            return BTGraphRAGConfig.from_dict(bt_dict)
        if isinstance(bt_dict, BTGraphRAGConfig):
            return bt_dict
    return BTGraphRAGConfig(enabled=False)


def _build_temporal_prompt(
    documents: pd.DataFrame,
    bt_config: BTGraphRAGConfig,
) -> str:
    """Build the temporal-aware extraction prompt with document date context.

    Reads the prompt from the configured file path (bt_graphrag.extraction_prompt)
    if set, otherwise falls back to the built-in TEMPORAL_GRAPH_EXTRACTION_PROMPT.
    """
    # Determine the document date for the prompt
    doc_date = "unknown"
    if not documents.empty:
        for _, row in documents.iterrows():
            doc_dict = dict(row)
            t_valid, _ = assign_document_timestamps(
                doc_dict, bt_config.default_valid_time_field
            )
            doc_date = t_valid.strftime("%Y-%m-%d")
            break  # Use first document's date as reference

    prompt_template = bt_config.resolved_extraction_prompt()
    return prompt_template.replace("{document_date}", doc_date)


def _parse_temporal_fields_from_extraction(
    relationships_df: pd.DataFrame,
    documents_df: pd.DataFrame,
    bt_config: BTGraphRAGConfig,
) -> pd.DataFrame:
    """Parse temporal fields from the extraction results.

    The temporal-aware prompt asks the LLM to include valid_time_start
    and valid_time_end in the relationship output.  The standard extractor
    now captures these as ``valid_time_start`` / ``valid_time_end`` columns
    when the LLM returns 7-field relationship records.

    This function:
    1. Converts LLM-captured ``valid_time_start`` / ``valid_time_end`` into
       proper ISO timestamps (``t_valid_start`` / ``t_valid_end``).
    2. Falls back to parsing temporal anchors from descriptions when the
       LLM did not provide explicit temporal fields.
    3. Sets ``t_tx_start`` / ``t_tx_end`` (transaction time) and derived
       columns (``confidence``, ``status``, ``support_count``).
    """
    from datetime import datetime, timezone

    from graphrag.bt_graphrag.models.temporal_types import INFINITY_ISO, utcnow
    from graphrag.bt_graphrag.temporal_extraction.temporal_normalization import (
        assign_document_timestamps,
        extract_temporal_anchors,
        parse_date_from_string,
    )

    relationships_df = relationships_df.copy()
    t_now = utcnow()

    # Determine batch-level document timestamp
    batch_t_valid = t_now
    if not documents_df.empty:
        first_doc = dict(documents_df.iloc[0])
        batch_t_valid, _ = assign_document_timestamps(
            first_doc, bt_config.default_valid_time_field
        )

    # --- Convert LLM-captured valid_time_start/end → t_valid_start/end ------
    has_llm_start = "valid_time_start" in relationships_df.columns
    has_llm_end = "valid_time_end" in relationships_df.columns

    t_valid_starts: list[str] = []
    t_valid_ends: list[str] = []

    for _, row in relationships_df.iterrows():
        # -- t_valid_start --
        resolved_start: str | None = None

        # 1) Try LLM-captured field
        if has_llm_start:
            raw = str(row.get("valid_time_start", "") or "").strip()
            if raw and raw.upper() not in ("UNKNOWN", "N/A", ""):
                parsed = parse_date_from_string(raw, batch_t_valid)
                if parsed:
                    resolved_start = parsed.isoformat()

        # 2) Fallback: parse temporal anchors from description
        if resolved_start is None:
            desc = str(row.get("description", ""))
            anchors = extract_temporal_anchors(desc, batch_t_valid)
            if anchors:
                earliest = min(anchors, key=lambda a: a["start"])
                resolved_start = earliest["start"].isoformat()

        # 3) Final fallback: document date
        if resolved_start is None:
            resolved_start = batch_t_valid.isoformat()

        t_valid_starts.append(resolved_start)

        # -- t_valid_end --
        resolved_end: str | None = None

        if has_llm_end:
            raw = str(row.get("valid_time_end", "") or "").strip()
            if raw and raw.upper() not in ("UNKNOWN", "ONGOING", "N/A", "PRESENT", ""):
                parsed = parse_date_from_string(raw, batch_t_valid)
                if parsed:
                    resolved_end = parsed.isoformat()

        if resolved_end is None:
            # Check description for end-date anchors (ranges, "until" patterns)
            desc = str(row.get("description", ""))
            anchors = extract_temporal_anchors(desc, batch_t_valid)
            # Look for a range anchor with an end date
            for a in anchors:
                if a.get("end") and a["type"] == "range":
                    resolved_end = a["end"].isoformat()
                    break

        if resolved_end is None:
            resolved_end = INFINITY_ISO

        t_valid_ends.append(resolved_end)

    relationships_df["t_valid_start"] = t_valid_starts
    relationships_df["t_valid_end"] = t_valid_ends

    # Drop the raw LLM columns now that we've converted them
    relationships_df.drop(
        columns=["valid_time_start", "valid_time_end"],
        errors="ignore",
        inplace=True,
    )

    # --- Transaction time (always system time) --------------------------------
    if "t_tx_start" not in relationships_df.columns:
        relationships_df["t_tx_start"] = t_now.isoformat()

    if "t_tx_end" not in relationships_df.columns:
        relationships_df["t_tx_end"] = INFINITY_ISO

    # --- Derived columns ------------------------------------------------------
    if "confidence" not in relationships_df.columns:
        if "weight" in relationships_df.columns:
            relationships_df["confidence"] = relationships_df["weight"].apply(
                lambda w: min(float(w) / 10.0, 1.0) if pd.notna(w) else 1.0
            )
        else:
            relationships_df["confidence"] = 1.0

    if "status" not in relationships_df.columns:
        relationships_df["status"] = "active"

    if "support_count" not in relationships_df.columns:
        relationships_df["support_count"] = 1

    return relationships_df


async def _get_neo4j_driver(bt_config: BTGraphRAGConfig):
    """Create Neo4j async driver from BT config."""
    from graphrag.bt_graphrag.neo4j_store import create_neo4j_driver

    return await create_neo4j_driver(
        uri=bt_config.neo4j_uri,
        user=bt_config.neo4j_user,
        password=bt_config.neo4j_password,
    )
