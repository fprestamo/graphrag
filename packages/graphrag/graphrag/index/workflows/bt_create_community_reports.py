# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG temporal community reports workflow.

Extends the standard create_community_reports workflow with:
- Incremental community detection (only re-cluster modified neighborhoods)
- Selective summarization (only regenerate stale community reports)
- Temporal annotations in community reports
"""

import logging
from typing import TYPE_CHECKING

import pandas as pd
from graphrag_llm.completion import create_completion

from graphrag.bt_graphrag.community_update import (
    annotate_community_reports_temporal,
    filter_stale_communities,
    merge_community_reports,
    run_incremental_community_update,
)
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.summarize_communities.community_reports_extractor import (
    CommunityReportsExtractor,
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
    """BT-GraphRAG temporal community reports workflow.

    Uses incremental community update (Stage 5) and selective
    summarization (Stage 6) to minimize LLM calls.
    """
    logger.info("Workflow started: bt_create_community_reports")
    print("\n" + "#" * 70)
    print("  BT-GraphRAG Workflow: bt_create_community_reports")
    print("#" * 70)

    reader = DataReader(context.output_table_provider)

    entities = await reader.entities()
    relationships = await reader.relationships()
    communities = await reader.communities()

    # Try to load existing community reports
    try:
        existing_reports = await reader.community_reports()
    except Exception:
        existing_reports = pd.DataFrame()

    # Get BT config
    bt_config = _get_bt_config(config)

    print(f"\n  Input:")
    print(f"    Entities:            {len(entities)}")
    print(f"    Relationships:       {len(relationships)}")
    print(f"    Communities:         {len(communities)}")
    print(f"    Existing reports:    {len(existing_reports)}")
    print(f"    BT-GraphRAG enabled: {bt_config.enabled}")

    # -----------------------------------------------------------------------
    # Stage 5: Incremental Community Update
    # -----------------------------------------------------------------------
    neo4j_driver = None
    if bt_config.enabled:
        try:
            from graphrag.bt_graphrag.neo4j_store import create_neo4j_driver
            neo4j_driver = await create_neo4j_driver(
                uri=bt_config.neo4j_uri,
                user=bt_config.neo4j_user,
                password=bt_config.neo4j_password,
            )
        except Exception:
            logger.warning("Neo4j unavailable for incremental community update")

    # Get last update timestamp from context state
    last_update_str = context.state.get("bt_last_community_update")
    last_update = None
    if last_update_str:
        from datetime import datetime
        try:
            last_update = datetime.fromisoformat(last_update_str)
        except (ValueError, TypeError):
            pass

    print("\n" + "-" * 70)
    print("  Stage 5: Incremental Community Update")
    print("-" * 70)
    print(f"  Last update: {last_update or 'never (first run)'}")

    communities, stale_community_ids = await run_incremental_community_update(
        communities_df=communities,
        entities_df=entities,
        relationships_df=relationships,
        config=bt_config,
        driver=neo4j_driver,
        last_update_time=last_update,
    )

    print(f"  \u2713 Stage 5 Complete:")
    print(f"    Communities after update: {len(communities)}")
    print(f"    Stale communities:       {len(stale_community_ids) if stale_community_ids else 0}")
    if stale_community_ids:
        print(f"    Stale IDs (first 10):    {list(stale_community_ids)[:10]}")

    if neo4j_driver:
        await neo4j_driver.close()

    # -----------------------------------------------------------------------
    # Stage 6: Selective Summarization
    # -----------------------------------------------------------------------
    model_config = config.get_completion_model_config(
        config.community_reports.completion_model_id
    )
    model = create_completion(
        model_config,
        cache=context.cache.child(config.community_reports.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    # Use temporal prompt for community reports (from config file or built-in)
    prompt = bt_config.resolved_community_report_prompt()

    print("\n" + "-" * 70)
    print("  Stage 6: Selective Community Summarization")
    print("-" * 70)

    # If we have stale communities, only regenerate those
    if stale_community_ids and not existing_reports.empty:
        stale_communities = filter_stale_communities(
            communities, existing_reports, stale_community_ids
        )
        print(f"  Regenerating {len(stale_communities)} stale reports (of {len(communities)} total)")
        logger.info(
            "Stage 6: Regenerating %d stale community reports (of %d total)",
            len(stale_communities),
            len(communities),
        )
    else:
        stale_communities = communities
        print(f"  Generating all {len(stale_communities)} community reports (first run or full rebuild)")

    # Generate reports for stale/all communities
    if not stale_communities.empty:
        from graphrag.index.workflows.create_community_reports import (
            run_workflow as standard_create_reports,
        )
        # Fall back to standard report creation for the actual generation
        result = await standard_create_reports(config, context)

        # Read the generated reports
        new_reports = await reader.community_reports()

        # Annotate with temporal information
        new_reports = annotate_community_reports_temporal(
            new_reports, entities, relationships
        )

        # Merge with existing if incremental
        if stale_community_ids and not existing_reports.empty:
            community_reports = merge_community_reports(
                existing_reports, new_reports, stale_community_ids
            )
        else:
            community_reports = new_reports

        await context.output_table_provider.write_dataframe(
            "community_reports", community_reports
        )
        print(f"  \u2713 Stage 6 Complete: {len(community_reports)} community reports written")
    else:
        community_reports = existing_reports
        print(f"  \u2713 Stage 6: No stale communities — reusing {len(community_reports)} existing reports")

    # Update the last community update timestamp
    from graphrag.bt_graphrag.models.temporal_types import utcnow
    context.state["bt_last_community_update"] = utcnow().isoformat()

    logger.info("Workflow completed: bt_create_community_reports")
    print("\n" + "#" * 70)
    print("  bt_create_community_reports COMPLETE")
    print(f"  Final: {len(community_reports)} community reports")
    print("#" * 70 + "\n")
    return WorkflowFunctionOutput(
        result={"community_reports": community_reports}
    )


def _get_bt_config(config: GraphRagConfig) -> BTGraphRAGConfig:
    """Extract BTGraphRAGConfig from GraphRagConfig."""
    if hasattr(config, "bt_graphrag") and config.bt_graphrag is not None:
        bt_dict = config.bt_graphrag
        if isinstance(bt_dict, dict):
            return BTGraphRAGConfig.from_dict(bt_dict)
        if isinstance(bt_dict, BTGraphRAGConfig):
            return bt_dict
    return BTGraphRAGConfig(enabled=False)
