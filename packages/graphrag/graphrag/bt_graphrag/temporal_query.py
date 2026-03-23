# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG Stage 7: Temporal Query Pipeline.

Extends GraphRAG's search modes with temporal semantics:
- Local Temporal Search: filters edges valid at query time
- Global Temporal Search: MapReduce over temporally valid community reports
- Temporal Audit Search: reconstruct historical system states
- Dispute Resolution Search: surface disputed edges
- Temporal Query Decomposition: break complex temporal queries into sub-queries
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.models.temporal_types import utcnow
from graphrag.bt_graphrag.prompts import (
    DISPUTE_RESOLUTION_PROMPT,
    TEMPORAL_ANSWER_SYNTHESIS_PROMPT,
    TEMPORAL_QUERY_DECOMPOSITION_PROMPT,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal scoring
# ---------------------------------------------------------------------------


def temporal_decay_score(
    query_time: datetime,
    edge_time: datetime,
    alpha: float = 0.1,
) -> float:
    """Compute exponential decay score based on temporal proximity.

    Uses DyG-RAG's formulation: exp(-alpha * |t_q - t_e|)
    where the time difference is measured in days.
    """
    delta_days = abs((query_time - edge_time).total_seconds()) / 86400.0
    return math.exp(-alpha * delta_days)


# ---------------------------------------------------------------------------
# Local Temporal Search
# ---------------------------------------------------------------------------


async def local_temporal_search(
    query: str,
    query_time: datetime | None,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
    model: "LLMCompletion | None" = None,
) -> dict[str, Any]:
    """Local Temporal Search: filters to edges valid at query time T
    and currently believed (t_tx_end = inf).

    Uses temporal decay scoring to weight edges by proximity to query time.
    """
    from graphrag.bt_graphrag.neo4j_store import get_active_edges

    query_time = query_time or utcnow()

    async with driver.session(database=config.neo4j_database) as session:
        edges = await get_active_edges(session, at_time=query_time)

    # Score edges by temporal proximity
    scored_edges = []
    for edge in edges:
        t_valid_start_str = edge.get("t_valid_start")
        if t_valid_start_str:
            try:
                edge_time = datetime.fromisoformat(t_valid_start_str)
                score = temporal_decay_score(
                    query_time, edge_time, config.temporal_decay_alpha
                )
            except (ValueError, TypeError):
                score = 0.5
        else:
            score = 0.5

        scored_edges.append({**edge, "_temporal_score": score})

    # Sort by temporal score
    scored_edges.sort(key=lambda e: e.get("_temporal_score", 0), reverse=True)

    return {
        "edges": scored_edges[:50],  # Top 50 most temporally relevant
        "query_time": query_time.isoformat(),
        "total_matching_edges": len(scored_edges),
    }


# ---------------------------------------------------------------------------
# Temporal Audit Search
# ---------------------------------------------------------------------------


async def temporal_audit_search(
    query: str,
    audit_time: datetime,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
) -> dict[str, Any]:
    """Temporal Audit: 'What did the system believe at time T?'

    Queries by transaction-time to reconstruct historical system state.
    """
    from graphrag.bt_graphrag.neo4j_store import get_system_state_at

    async with driver.session(database=config.neo4j_database) as session:
        edges = await get_system_state_at(session, audit_time)

    return {
        "system_state_at": audit_time.isoformat(),
        "edges": edges,
        "total_edges": len(edges),
    }


# ---------------------------------------------------------------------------
# Dispute Resolution Search
# ---------------------------------------------------------------------------


async def dispute_resolution_search(
    query: str,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
    model: "LLMCompletion | None" = None,
    entity_title: str | None = None,
) -> dict[str, Any]:
    """Surface disputed edges for queries touching unresolved conflicts.

    Presents competing claims with provenance chains.
    """
    from graphrag.bt_graphrag.neo4j_store import get_disputed_edges

    async with driver.session(database=config.neo4j_database) as session:
        disputed = await get_disputed_edges(session, entity_title)

    result: dict[str, Any] = {
        "disputed_edges": disputed,
        "total_disputes": len(disputed),
    }

    # If LLM available, generate analysis
    if model is not None and disputed:
        from graphrag_llm.utils import CompletionMessagesBuilder

        edges_text = "\n".join(
            f"- ({e.get('source', '?')}) -[{e.get('relation_type', '?')}]-> ({e.get('target', '?')}): "
            f"{e.get('description', 'No description')}"
            for e in disputed[:10]
        )

        prompt = DISPUTE_RESOLUTION_PROMPT.format(
            disputed_edges=edges_text,
            query=query,
        )
        messages = CompletionMessagesBuilder().add_user_message(prompt).build()
        response = await model.completion_async(messages=messages)
        result["analysis"] = response.content

    return result


# ---------------------------------------------------------------------------
# Temporal Query Decomposition
# ---------------------------------------------------------------------------


async def decompose_temporal_query(
    query: str,
    model: "LLMCompletion",
    current_date: datetime | None = None,
) -> list[dict[str, Any]]:
    """Decompose a complex temporal query into simpler sub-queries.

    Each sub-query targets a specific time point or interval.
    """
    from graphrag_llm.utils import CompletionMessagesBuilder

    current_date = current_date or utcnow()

    prompt = TEMPORAL_QUERY_DECOMPOSITION_PROMPT.format(
        query=query,
        current_date=current_date.strftime("%Y-%m-%d"),
    )

    messages = CompletionMessagesBuilder().add_user_message(prompt).build()
    response = await model.completion_async(messages=messages)

    try:
        sub_queries = json.loads(response.content)
        if isinstance(sub_queries, list):
            return sub_queries
    except (json.JSONDecodeError, TypeError):
        pass

    return [{"sub_query": query, "temporal_constraint": "current", "query_type": "POINT_IN_TIME"}]


async def synthesize_temporal_answers(
    query: str,
    sub_results: list[dict[str, Any]],
    model: "LLMCompletion",
) -> str:
    """Synthesize sub-query results into a coherent temporal answer."""
    from graphrag_llm.utils import CompletionMessagesBuilder

    sub_results_text = "\n\n".join(
        f"Sub-query: {r.get('sub_query', '?')}\n"
        f"Time: {r.get('temporal_constraint', 'unknown')}\n"
        f"Result: {r.get('result', 'No result')}"
        for r in sub_results
    )

    prompt = TEMPORAL_ANSWER_SYNTHESIS_PROMPT.format(
        query=query,
        sub_results=sub_results_text,
    )

    messages = CompletionMessagesBuilder().add_user_message(prompt).build()
    response = await model.completion_async(messages=messages)
    return response.content


# ---------------------------------------------------------------------------
# Full Temporal Search (combines all modes)
# ---------------------------------------------------------------------------


async def temporal_search(
    query: str,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
    model: "LLMCompletion | None" = None,
    query_time: datetime | None = None,
    search_mode: str = "local",
) -> dict[str, Any]:
    """Unified temporal search entry point.

    Args:
        query: The search query.
        config: BT-GraphRAG configuration.
        driver: Neo4j async driver.
        model: Optional LLM for analysis and decomposition.
        query_time: Target time for temporal queries.
        search_mode: One of 'local', 'audit', 'dispute', 'decompose'.
    """
    if search_mode == "audit" and query_time:
        return await temporal_audit_search(query, query_time, config, driver)

    if search_mode == "dispute":
        return await dispute_resolution_search(
            query, config, driver, model
        )

    if search_mode == "decompose" and model is not None:
        sub_queries = await decompose_temporal_query(query, model, query_time)
        sub_results = []
        for sq in sub_queries:
            result = await local_temporal_search(
                sq.get("sub_query", query),
                query_time,
                config,
                driver,
                model,
            )
            sub_results.append({**sq, "result": result})

        answer = await synthesize_temporal_answers(query, sub_results, model)
        return {"answer": answer, "sub_queries": sub_results}

    # Default: local temporal search
    return await local_temporal_search(
        query, query_time, config, driver, model
    )
