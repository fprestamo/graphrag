# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG Stage 7: Temporal Query Pipeline.

Implements the query side of the proposal (see ``proposal.tex``,
section "Integración con el pipeline de GraphRAG y consulta temporal"):

1.  Analyse + decompose: extract entity mentions and split the question
    into one or more sub-queries, each with an explicit temporal type
    (POINT_IN_TIME, RANGE, EVOLUTION, COMPARISON) and an ISO date range.
2.  Resolve seed entities against the graph by title match (no embedding
    required at query time).
3.  Retrieve a k-hop temporal subgraph around the seeds, filtered as-of
    the sub-query's time window AND restricted to currently-believed
    edges (``t_tx_end = INFINITY``).
4.  Re-rank any ``status=disputed`` edges by support_count, confidence
    and recency of the latest transaction.
5.  Synthesise a final answer that respects the temporal scope of the
    original question.

The module keeps standalone helpers for ``temporal_audit_search`` and
``dispute_resolution_search`` for callers that want those individual
modes; the main entry point is ``temporal_query_pipeline``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.models.temporal_types import (
    INFINITY,
    INFINITY_ISO,
    MINUS_INFINITY,
    MINUS_INFINITY_ISO,
    utcnow,
)
from graphrag.bt_graphrag.prompts import (
    DISPUTE_RESOLUTION_PROMPT,
    TEMPORAL_ANSWER_SYNTHESIS_PROMPT,
    TEMPORAL_QUERY_ANALYSIS_PROMPT,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from neo4j import AsyncDriver, AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_iso_date(value: Any) -> datetime | None:
    """Parse a date string like 'YYYY-MM-DD' (and a few common variants)
    into a UTC datetime. Returns None for empty / sentinel inputs.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    if not s or s.upper() in {"UNKNOWN", "CURRENT", "ONGOING", "NONE", "NULL"}:
        return None
    # YYYY-MM-DD
    if _DATE_RE.match(s):
        try:
            return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    # ISO timestamp
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # YYYY-MM
    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return datetime(y, mo, 1, tzinfo=timezone.utc)
    # YYYY only
    m = re.match(r"^(\d{4})$", s)
    if m:
        return datetime(int(m.group(1)), 1, 1, tzinfo=timezone.utc)
    return None


def _format_iso(dt: datetime | None) -> str:
    if dt is None:
        return "UNKNOWN"
    if dt == INFINITY:
        return "ONGOING"
    if dt == MINUS_INFINITY:
        return "UNKNOWN"
    return dt.date().isoformat()


# ---------------------------------------------------------------------------
# Step 1: Query analysis + decomposition
# ---------------------------------------------------------------------------


def _heuristic_query_analysis(query: str, current_date: datetime) -> dict[str, Any]:
    """Lightweight fallback when no LLM is available.

    Returns a single POINT_IN_TIME / CURRENT sub-query with no entities
    pre-extracted. The pipeline will still attempt to retrieve a useful
    subgraph by other means (e.g. by treating the whole question as a
    title search).
    """
    return {
        "entities": [],
        "sub_queries": [
            {
                "sub_query": query,
                "entities": [],
                "query_type": "POINT_IN_TIME",
                "t_start": "CURRENT",
                "t_end": "CURRENT",
            }
        ],
    }


def _coerce_analysis(payload: Any, query: str, current_date: datetime) -> dict[str, Any]:
    """Validate / normalise the LLM analysis payload.

    Falls back to the heuristic structure on any malformation.
    """
    if not isinstance(payload, dict):
        return _heuristic_query_analysis(query, current_date)
    entities = payload.get("entities") or []
    if not isinstance(entities, list):
        entities = []
    entities = [str(e).strip() for e in entities if str(e).strip()]

    raw_subs = payload.get("sub_queries") or []
    if not isinstance(raw_subs, list) or not raw_subs:
        return {
            "entities": entities,
            "sub_queries": _heuristic_query_analysis(query, current_date)["sub_queries"],
        }

    subs: list[dict[str, Any]] = []
    for sq in raw_subs:
        if not isinstance(sq, dict):
            continue
        sub_entities = sq.get("entities") or []
        if not isinstance(sub_entities, list):
            sub_entities = []
        sub_entities = [str(e).strip() for e in sub_entities if str(e).strip()]
        subs.append({
            "sub_query": str(sq.get("sub_query") or query).strip() or query,
            "entities": sub_entities or list(entities),
            "query_type": str(sq.get("query_type") or "POINT_IN_TIME").strip().upper(),
            "t_start": str(sq.get("t_start") or "CURRENT").strip(),
            "t_end": str(sq.get("t_end") or sq.get("t_start") or "CURRENT").strip(),
        })

    if not subs:
        subs = _heuristic_query_analysis(query, current_date)["sub_queries"]

    return {"entities": entities, "sub_queries": subs}


async def analyze_and_decompose_query(
    query: str,
    model: "LLMCompletion",
    current_date: datetime | None = None,
) -> dict[str, Any]:
    """Run the analysis+decomposition LLM call.

    Returns a dict with keys ``entities`` (list[str]) and ``sub_queries``
    (list of {sub_query, entities, query_type, t_start, t_end}).
    """
    from graphrag_llm.utils import CompletionMessagesBuilder

    current_date = current_date or utcnow()
    prompt = TEMPORAL_QUERY_ANALYSIS_PROMPT.format(
        query=query,
        current_date=current_date.strftime("%Y-%m-%d"),
    )
    messages = CompletionMessagesBuilder().add_user_message(prompt).build()
    try:
        response = await model.completion_async(messages=messages)
        content = getattr(response, "content", "") or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("Query analysis LLM call failed: %s", exc)
        return _heuristic_query_analysis(query, current_date)

    # The model may wrap the JSON in code fences — strip them.
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try to isolate the outermost JSON object.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return _heuristic_query_analysis(query, current_date)
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return _heuristic_query_analysis(query, current_date)

    return _coerce_analysis(payload, query, current_date)


# ---------------------------------------------------------------------------
# Step 2: Resolve seed entities against the graph
# ---------------------------------------------------------------------------


async def resolve_seed_entities(
    session: "AsyncSession",
    entity_names: list[str],
) -> list[dict[str, Any]]:
    """Return graph entities matching the supplied names (title-based)."""
    from graphrag.bt_graphrag.neo4j_store import find_entities_by_titles

    if not entity_names:
        return []
    return await find_entities_by_titles(session, entity_names)


# ---------------------------------------------------------------------------
# Step 3: Retrieve as-of subgraph for a sub-query
# ---------------------------------------------------------------------------


def _sub_query_time_window(
    sub_query: dict[str, Any],
    query_time: datetime,
) -> tuple[datetime | None, tuple[datetime, datetime] | None]:
    """Compute (valid_at, valid_range) for the given sub-query.

    - POINT_IN_TIME: ``valid_at`` is set to t_start (or t_end if start is unknown).
    - RANGE / COMPARISON: ``valid_range`` is set to [t_start, t_end].
    - EVOLUTION: no temporal restriction (return None, None) so all
      historical states show up; the LLM can order them.
    - CURRENT / UNKNOWN sentinels resolve to ``query_time``.
    """
    qtype = sub_query.get("query_type", "POINT_IN_TIME").upper()
    raw_start = sub_query.get("t_start", "CURRENT")
    raw_end = sub_query.get("t_end", raw_start)

    def _resolve(raw: str) -> datetime | None:
        if not raw or str(raw).upper() in {"CURRENT", "ONGOING"}:
            return query_time
        if str(raw).upper() in {"UNKNOWN", "NONE", "NULL"}:
            return None
        return _parse_iso_date(raw)

    t_start = _resolve(raw_start)
    t_end = _resolve(raw_end)

    if qtype == "EVOLUTION":
        return None, None
    if qtype in {"RANGE", "COMPARISON"}:
        if t_start is None and t_end is None:
            return None, None
        rs = t_start or MINUS_INFINITY
        re_ = t_end or INFINITY
        if rs > re_:
            rs, re_ = re_, rs
        return None, (rs, re_)
    # Default: POINT_IN_TIME
    point = t_start or t_end or query_time
    return point, None


async def retrieve_temporal_subgraph(
    session: "AsyncSession",
    sub_query: dict[str, Any],
    seed_entities: list[dict[str, Any]],
    config: BTGraphRAGConfig,
    query_time: datetime,
    k_hop: int = 1,
    limit: int = 200,
) -> dict[str, Any]:
    """Run the as-of subgraph retrieval for a single sub-query.

    Returns ``{"edges": [...], "entities": [...]}``. All edges in the
    result satisfy the bitemporal filter; disputed edges are kept so
    they can be re-ranked by ``rank_disputed_edges`` downstream.
    """
    from graphrag.bt_graphrag.neo4j_store import (
        get_active_edges,
        get_subgraph_around_entities,
    )

    valid_at, valid_range = _sub_query_time_window(sub_query, query_time)
    seed_titles = [e["title"] for e in seed_entities if e.get("title")]

    if seed_titles:
        sub = await get_subgraph_around_entities(
            session,
            entity_titles=seed_titles,
            k_hop=k_hop,
            valid_at=valid_at,
            valid_range=valid_range,
            tx_at=None,  # currently-believed edges (t_tx_end = INFINITY)
            include_disputed=True,
            limit=limit,
        )
        if sub["edges"]:
            return sub
        # If the seeded subgraph is empty at the requested time, fall
        # back to ALL edges that touch the seeds (any valid period).
        # This is what lets the LLM still answer "the system has no
        # record of X at that time" instead of getting nothing.
        sub_unconstrained = await get_subgraph_around_entities(
            session,
            entity_titles=seed_titles,
            k_hop=k_hop,
            valid_at=None,
            valid_range=None,
            tx_at=None,
            include_disputed=True,
            limit=limit,
        )
        return sub_unconstrained

    # No seed entities resolved: fall back to a temporal scan of the
    # whole graph at the requested time. Capped by ``limit``.
    edges = await get_active_edges(
        session,
        at_time=valid_at if valid_at is not None else None,
    )
    return {
        "edges": edges[:limit],
        "entities": [],
        "seed_titles": [],
    }


# ---------------------------------------------------------------------------
# Step 4: Dispute weighting
# ---------------------------------------------------------------------------


def _dispute_score(edge: dict[str, Any], query_time: datetime) -> float:
    """Confidence × log(1+support_count) × recency-of-belief.

    Used to break ties between competing disputed edges.
    """
    try:
        conf = float(edge.get("confidence", 1.0) or 1.0)
    except (ValueError, TypeError):
        conf = 1.0
    try:
        support = int(edge.get("support_count", 1) or 1)
    except (ValueError, TypeError):
        support = 1
    tx_start_dt = _parse_iso_date(edge.get("t_tx_start"))
    if tx_start_dt is None:
        recency = 0.5
    else:
        # Map "recently believed" → higher recency.  Use exponential decay
        # over the gap between t_tx_start and the query time.
        delta_days = max(
            0.0, (query_time - tx_start_dt).total_seconds() / 86400.0
        )
        recency = math.exp(-0.005 * delta_days)
    return conf * math.log1p(max(0, support)) * recency


def rank_disputed_edges(
    edges: list[dict[str, Any]],
    query_time: datetime,
) -> list[dict[str, Any]]:
    """Annotate disputed edges with a ``_dispute_score`` and sort the
    list so that more reliable disputed claims appear first.

    Non-disputed edges are unaffected by the score but are returned in
    the same order.
    """
    for edge in edges:
        if str(edge.get("status", "")).lower() == "disputed":
            edge["_dispute_score"] = _dispute_score(edge, query_time)
        else:
            edge["_dispute_score"] = None
    edges.sort(
        key=lambda e: (
            str(e.get("status", "")).lower() == "disputed",  # disputed last
            -1 * (e.get("_dispute_score") or 0.0),
        )
    )
    return edges


# ---------------------------------------------------------------------------
# Step 5: Format edges + synthesise an answer
# ---------------------------------------------------------------------------


def _format_edge_line(edge: dict[str, Any]) -> str:
    """Format a single edge for inclusion in an LLM context block."""
    src = edge.get("source") or edge.get("source_title") or "?"
    tgt = edge.get("target") or edge.get("target_title") or "?"
    rel = edge.get("relation_type") or edge.get("type") or "RELATED_TO"
    tvs = edge.get("t_valid_start", "")
    tve = edge.get("t_valid_end", "")

    # Pretty-print INFINITY / MINUS_INFINITY sentinels.
    def _pretty(v: Any, side: str) -> str:
        if not v:
            return "UNKNOWN" if side == "start" else "ONGOING"
        s = str(v)
        if s == INFINITY_ISO:
            return "ONGOING"
        if s == MINUS_INFINITY_ISO:
            return "UNKNOWN"
        # Drop the time component for brevity
        return s.split("T", 1)[0]

    status = str(edge.get("status", "active") or "active")
    support = edge.get("support_count", 1)
    conf = edge.get("confidence", 1.0)
    try:
        conf_str = f"{float(conf):.2f}"
    except (ValueError, TypeError):
        conf_str = str(conf)

    desc = (edge.get("description") or "").strip()
    if len(desc) > 200:
        desc = desc[:200] + "…"

    return (
        f"- ({src}) -[{rel}]-> ({tgt}) "
        f"[{_pretty(tvs, 'start')} → {_pretty(tve, 'end')}] "
        f"{{status={status}, support={support}, confidence={conf_str}}}: "
        f"{desc}"
    )


def format_subgraph_for_llm(
    sub_query: dict[str, Any],
    edges: list[dict[str, Any]],
    max_edges: int,
) -> str:
    """Render the subgraph for one sub-query as a context block."""
    header_parts = [f"Sub-query: {sub_query.get('sub_query', '')}"]
    qtype = sub_query.get("query_type", "POINT_IN_TIME")
    t_start = sub_query.get("t_start", "?")
    t_end = sub_query.get("t_end", "?")
    header_parts.append(f"Type: {qtype}    Window: {t_start} → {t_end}")
    if not edges:
        header_parts.append("(no edges matched this sub-query)")
        return "\n".join(header_parts)

    edge_lines = [_format_edge_line(e) for e in edges[:max_edges]]
    header_parts.append("Edges:")
    header_parts.extend(edge_lines)
    if len(edges) > max_edges:
        header_parts.append(f"... ({len(edges) - max_edges} more edges truncated)")
    return "\n".join(header_parts)


async def synthesize_temporal_answer(
    query: str,
    sub_results: list[dict[str, Any]],
    model: "LLMCompletion",
    max_edges_per_sub: int = 30,
) -> str:
    """Combine the per-sub-query subgraphs into a final answer."""
    from graphrag_llm.utils import CompletionMessagesBuilder

    blocks = []
    for r in sub_results:
        blocks.append(
            format_subgraph_for_llm(
                sub_query=r.get("sub_query", {}),
                edges=r.get("edges", []),
                max_edges=max_edges_per_sub,
            )
        )
    sub_results_text = "\n\n".join(blocks) if blocks else "(no context retrieved)"

    prompt = TEMPORAL_ANSWER_SYNTHESIS_PROMPT.format(
        query=query, sub_results=sub_results_text
    )
    messages = CompletionMessagesBuilder().add_user_message(prompt).build()
    response = await model.completion_async(messages=messages)
    return (getattr(response, "content", "") or "").strip()


# Backwards-compatibility wrapper preserving the older name.
async def synthesize_temporal_answers(
    query: str,
    sub_results: list[dict[str, Any]],
    model: "LLMCompletion",
) -> str:
    return await synthesize_temporal_answer(query, sub_results, model)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def temporal_query_pipeline(
    query: str,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
    model: "LLMCompletion | None" = None,
    query_time: datetime | None = None,
    k_hop: int = 1,
    max_edges_per_sub: int = 50,
    max_total_edges: int = 200,
) -> dict[str, Any]:
    """Full BT-GraphRAG query pipeline.

    Steps:
        1. Analyse + decompose the question.
        2. For each sub-query, resolve seeds and retrieve the as-of
           subgraph.
        3. Re-rank any disputed edges.
        4. Synthesise the final answer (requires an LLM).

    Returns a dict with::

        {
            "answer": str,
            "sub_queries": [...],  # each entry has sub_query, edges, entities
            "edges_used": int,
            "query_time": str ISO,
        }
    """
    query_time = query_time or utcnow()

    # Step 1: analyse the question
    if model is not None:
        analysis = await analyze_and_decompose_query(query, model, query_time)
    else:
        analysis = _heuristic_query_analysis(query, query_time)

    sub_queries = analysis.get("sub_queries", [])
    if not sub_queries:
        sub_queries = _heuristic_query_analysis(query, query_time)["sub_queries"]

    # Step 2-4: retrieve subgraphs per sub-query
    sub_results: list[dict[str, Any]] = []
    total_edges = 0
    async with driver.session(database=config.neo4j_database) as session:
        for sq in sub_queries:
            entity_names = sq.get("entities") or analysis.get("entities") or []
            seeds = await resolve_seed_entities(session, entity_names)
            sub = await retrieve_temporal_subgraph(
                session=session,
                sub_query=sq,
                seed_entities=seeds,
                config=config,
                query_time=query_time,
                k_hop=k_hop,
                limit=max_edges_per_sub,
            )
            edges = rank_disputed_edges(sub["edges"], query_time)
            total_edges += len(edges)
            sub_results.append({
                "sub_query": sq,
                "edges": edges,
                "entities": sub.get("entities", []),
                "seeds": seeds,
            })
            if total_edges >= max_total_edges:
                break

    # Step 5: synthesise
    if model is None:
        answer = ""
    else:
        try:
            answer = await synthesize_temporal_answer(
                query=query,
                sub_results=sub_results,
                model=model,
                max_edges_per_sub=max_edges_per_sub,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Temporal answer synthesis failed: %s", exc)
            answer = ""

    return {
        "answer": answer,
        "sub_queries": sub_results,
        "edges_used": total_edges,
        "query_time": query_time.isoformat(),
        "analysis": analysis,
    }


# ---------------------------------------------------------------------------
# Legacy entry points (kept for direct callers / search_mode dispatch)
# ---------------------------------------------------------------------------


async def local_temporal_search(
    query: str,
    query_time: datetime | None,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
    model: "LLMCompletion | None" = None,
    k_hop: int = 1,
    max_edges: int = 50,
) -> dict[str, Any]:
    """Backwards-compatible local temporal search.

    Now delegates to ``temporal_query_pipeline`` and exposes the same
    ``{"edges", "query_time", "total_matching_edges", "answer"}`` shape
    callers (e.g. ``evaluation.core.runners.bt_answer``) already use.
    """
    result = await temporal_query_pipeline(
        query=query,
        config=config,
        driver=driver,
        model=model,
        query_time=query_time,
        k_hop=k_hop,
        max_edges_per_sub=max_edges,
    )
    # Flatten the per-sub-query edges into a single list for compatibility.
    flat_edges: list[dict[str, Any]] = []
    for sub in result.get("sub_queries", []):
        flat_edges.extend(sub.get("edges", []))
    return {
        "edges": flat_edges[:max_edges],
        "query_time": result.get("query_time"),
        "total_matching_edges": len(flat_edges),
        "answer": result.get("answer", ""),
        "sub_queries": result.get("sub_queries", []),
        "analysis": result.get("analysis", {}),
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
    """Surface disputed edges and (optionally) ask the LLM to weight them."""
    from graphrag.bt_graphrag.neo4j_store import get_disputed_edges

    async with driver.session(database=config.neo4j_database) as session:
        disputed = await get_disputed_edges(session, entity_title)

    disputed = rank_disputed_edges(disputed, utcnow())

    result: dict[str, Any] = {
        "disputed_edges": disputed,
        "total_disputes": len(disputed),
    }

    if model is not None and disputed:
        from graphrag_llm.utils import CompletionMessagesBuilder

        edges_text = "\n".join(_format_edge_line(e) for e in disputed[:20])
        prompt = DISPUTE_RESOLUTION_PROMPT.format(
            disputed_edges=edges_text,
            query=query,
        )
        messages = CompletionMessagesBuilder().add_user_message(prompt).build()
        try:
            response = await model.completion_async(messages=messages)
            result["analysis"] = response.content
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dispute LLM call failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Unified search entry point (used by older callers / external API)
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
        model: Optional LLM for analysis and synthesis.
        query_time: Target time for temporal queries.
        search_mode: One of 'local', 'audit', 'dispute'.
    """
    if search_mode == "audit":
        return await temporal_audit_search(
            query,
            query_time or utcnow(),
            config,
            driver,
        )

    if search_mode == "dispute":
        return await dispute_resolution_search(query, config, driver, model)

    return await temporal_query_pipeline(
        query=query,
        config=config,
        driver=driver,
        model=model,
        query_time=query_time,
    )


# ---------------------------------------------------------------------------
# Legacy helpers (kept for callers that imported them directly)
# ---------------------------------------------------------------------------


def temporal_decay_score(
    query_time: datetime,
    edge_time: datetime,
    alpha: float = 0.1,
) -> float:
    """Exponential decay score based on temporal proximity (days)."""
    delta_days = abs((query_time - edge_time).total_seconds()) / 86400.0
    return math.exp(-alpha * delta_days)


async def decompose_temporal_query(
    query: str,
    model: "LLMCompletion",
    current_date: datetime | None = None,
) -> list[dict[str, Any]]:
    """Backwards-compatible: returns just the sub-query list."""
    analysis = await analyze_and_decompose_query(query, model, current_date)
    return analysis.get("sub_queries", [])
