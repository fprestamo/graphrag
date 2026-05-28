# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG Stage 6(b): Temporal Query as a helper to ``local_search``.

This module no longer owns the answer synthesis. Instead, it plugs the
temporal knowledge it has (a bitemporal Neo4j graph + LLM-driven query
decomposition) into the *stages* of graphrag's vanilla ``local_search``:

  1. **Query → entities mapping** (stage 1 of ``LocalSearchMixedContext``):
     - Decompose the query into sub-queries with explicit time windows.
     - Resolve seed entities for each sub-query against the bitemporal
       graph (title + embedding NN). Hand the resolved titles to
       ``local_search`` as ``include_entity_names`` so its context
       builder is biased toward temporally-relevant candidates.
     - Reformulate the sub-query text with the window inline so the LLM
       at the synthesis step has the date anchor in the user prompt.

  2. **LLM synthesis** (stage 5):
     - When the decomposition produced a SINGLE sub-query, the answer
       from ``local_search`` *is* the final answer.
     - When it produced multiple, a small LLM call combines them.
     - When the final answer hedges / refuses, a refusal-recovery branch
       re-synthesizes using as-of evidence from the bitemporal graph
       (the legacy "PRIMARY / BACKGROUND" rendering, kept for this path).

The two graph-side stages of vanilla ``local_search`` (relationship
context, text-unit context) are not hookable from the outside without
either rewriting the parquet at index time or reaching into graphrag's
internal builder. That work is out of scope here; the decomposition +
seed-injection lift on stage 1 is what the bitemporal graph can
contribute today.

Public surface (kept stable for ``runners.bt_answer``):

    decompose_temporal_query, resolve_seed_entities, local_temporal_search
    rank_disputed_edges, temporal_decay_score, dispute_resolution_search
    temporal_query_pipeline, temporal_search
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

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
    TEMPORAL_QUERY_ANALYSIS_PROMPT,
    TEMPORAL_REFUSAL_RECOVERY_PROMPT,
    TEMPORAL_SUBANSWER_SYNTHESIS_PROMPT,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.embedding import LLMEmbedding
    from neo4j import AsyncDriver, AsyncSession

logger = logging.getLogger(__name__)


# Signature for the local_search adapter the runner injects. Returning
# ``None`` is treated as a "retriever failed / empty answer" — the
# pipeline routes those into the refusal-recovery branch.
LocalSearchProvider = Callable[..., Awaitable["str | None"]]


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_iso_date(value: Any) -> datetime | None:
    """Parse a date string into a UTC datetime; ``None`` on sentinels."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    if not s or s.upper() in {"UNKNOWN", "CURRENT", "ONGOING", "NONE", "NULL"}:
        return None
    if _DATE_RE.match(s):
        try:
            return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1, tzinfo=timezone.utc)
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


def _edge_bounds(edge: dict[str, Any]) -> tuple[datetime, datetime]:
    """Return ``(t_valid_start, t_valid_end)`` with sentinels for unknowns."""
    vs = _parse_iso_date(edge.get("t_valid_start"))
    ve = _parse_iso_date(edge.get("t_valid_end"))
    return (
        vs if vs is not None else MINUS_INFINITY,
        ve if ve is not None else INFINITY,
    )


# ---------------------------------------------------------------------------
# Step 1: Query analysis + decomposition
# ---------------------------------------------------------------------------


def _heuristic_query_analysis(query: str, current_date: datetime) -> dict[str, Any]:
    """Fallback when no LLM is available: one POINT_IN_TIME/CURRENT sub-query."""
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
    """Validate / normalise the LLM analysis payload."""
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
    """Run the analysis + decomposition LLM call.

    Returns ``{"entities": [str], "sub_queries": [{sub_query, entities,
    query_type, t_start, t_end}]}``.
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

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

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


async def decompose_temporal_query(
    query: str,
    model: "LLMCompletion",
    current_date: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return just the sub-query list — preserved for thesis traceability."""
    analysis = await analyze_and_decompose_query(query, model, current_date)
    return analysis.get("sub_queries", [])


# ---------------------------------------------------------------------------
# Step 2: Resolve seed entities against the bitemporal graph
# ---------------------------------------------------------------------------


async def resolve_seed_entities(
    session: "AsyncSession",
    entity_names: list[str],
    embedding_model: "LLMEmbedding | None" = None,
    top_k: int = 3,
    score_threshold: float = 0.55,
    prefer_title_match: bool = True,
) -> list[dict[str, Any]]:
    """Resolve entity names against the graph via title + embedding search.

    Two strategies are unioned and de-duped by title:

    1. **Title match** — exact / case-insensitive / substring matches.
    2. **Description-embedding NN** — vector search on
       ``Entity.description_embedding``, mirroring CGER's pre-filter from
       indexing time. Provides recall when the LLM-extracted name
       doesn't appear verbatim in the graph.

    Title matches always win the tiebreak; embedding hits fill up to
    ``top_k`` *additional* titles when ``prefer_title_match`` is set.
    """
    from graphrag.bt_graphrag.neo4j_store import (
        find_entities_by_titles,
        vector_search_entities,
    )

    if not entity_names:
        return []

    title_matches = await find_entities_by_titles(session, entity_names)

    embedding_matches: list[tuple[float, dict[str, Any]]] = []
    if embedding_model is not None and top_k > 0:
        nonempty = [n for n in entity_names if n and n.strip()]
        if nonempty:
            try:
                response = await embedding_model.embedding_async(input=nonempty)
                vectors = list(getattr(response, "embeddings", []) or [])
            except Exception as exc:  # noqa: BLE001
                logger.debug("Seed embedding call failed: %s", exc)
                vectors = []
            for name, vec in zip(nonempty, vectors):
                if not vec:
                    continue
                try:
                    hits = await vector_search_entities(
                        session, query_embedding=list(vec), top_k=top_k,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Vector seed search failed for %r: %s", name, exc)
                    continue
                for h in hits:
                    score = float(h.get("_vector_score") or 0.0)
                    if score < score_threshold:
                        continue
                    h["_query_term"] = name
                    embedding_matches.append((score, h))
    embedding_matches.sort(key=lambda x: x[0], reverse=True)

    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for ent in title_matches:
        title = ent.get("title")
        if not title or title in seen:
            continue
        seen.add(title)
        merged.append(ent)

    budget = top_k if prefer_title_match else len(embedding_matches)
    added = 0
    for _score, ent in embedding_matches:
        if added >= budget:
            break
        title = ent.get("title")
        if not title or title in seen:
            continue
        seen.add(title)
        merged.append(ent)
        added += 1
    return merged


# ---------------------------------------------------------------------------
# Step 3: Sub-query rendering (window-aware reformulation)
# ---------------------------------------------------------------------------


_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _pretty_date(s: str | None) -> str | None:
    """Render an ISO date as a human-friendly month/year string."""
    if not s:
        return None
    up = s.upper()
    if up in {"CURRENT", "ONGOING"}:
        return "today"
    if up in {"UNKNOWN", "NONE", "NULL"}:
        return None
    dt = _parse_iso_date(s)
    if dt is None:
        return s
    return f"{_MONTHS[dt.month - 1]} {dt.year}"


def _window_collapses_to_year(start_iso: str | None, end_iso: str | None) -> str | None:
    """If the window covers exactly a calendar year (YYYY-01-01 → YYYY-12-31),
    return ``"YYYY"`` so the caller can render "in 1992" instead of the
    verbose "between January 1992 and December 1992"."""
    if not start_iso or not end_iso:
        return None
    s = _parse_iso_date(start_iso)
    e = _parse_iso_date(end_iso)
    if s is None or e is None:
        return None
    if s.year != e.year:
        return None
    if (s.month, s.day) == (1, 1) and (e.month, e.day) == (12, 31):
        return str(s.year)
    return None


def _render_subquery_with_window(sub_query: dict[str, Any]) -> str:
    """Reformulate the sub-query so the time window is visible in the user
    prompt that ``local_search`` sees.

    The vanilla retriever ranks text chunks by entity match — it does not
    understand abstract temporal windows. Putting the window inline in
    the user query is the cheapest way to bias the synthesis LLM (and the
    embedding lookup of the question) toward time-relevant material.
    """
    base = (sub_query.get("sub_query") or "").strip()
    qtype = str(sub_query.get("query_type") or "POINT_IN_TIME").upper()
    raw_start = sub_query.get("t_start")
    raw_end = sub_query.get("t_end")
    start = _pretty_date(raw_start)
    end = _pretty_date(raw_end)
    year_collapse = _window_collapses_to_year(raw_start, raw_end)

    if qtype == "EVOLUTION":
        window: str | None = "across its full timeline"
    elif qtype == "COMPARISON" and start and end and start != end:
        window = f"comparing {start} and {end}"
    elif qtype == "RANGE":
        if year_collapse:
            window = f"during {year_collapse}"
        elif start and end == "today" and start != "today":
            window = f"from {start} onward"
        elif start == "today" and end and end != "today":
            window = f"up to {end}"
        elif start and end and start != end:
            window = f"between {start} and {end}"
        elif start and not end:
            window = f"from {start} onward"
        elif end and not start:
            window = f"up to {end}"
        else:
            window = None
    else:  # POINT_IN_TIME
        if year_collapse:
            window = f"in {year_collapse}"
        elif start and end and start != end:
            window = f"between {start} and {end}"
        elif start:
            window = f"in {start}"
        elif end:
            window = f"in {end}"
        else:
            window = None

    if not window:
        return base
    # If the base sub-query already restates the same window, skip the
    # suffix to avoid duplicates.
    if start and start.lower() in base.lower():
        return base
    if end and end.lower() in base.lower():
        return base
    if year_collapse and year_collapse in base:
        return base
    return f"{base} (time anchor: {window})"


# ---------------------------------------------------------------------------
# Step 4: As-of subgraph retrieval (refusal-recovery branch only)
# ---------------------------------------------------------------------------


def _sub_query_window(
    sub_query: dict[str, Any],
    query_time: datetime,
) -> tuple[datetime | None, tuple[datetime, datetime] | None, datetime | None]:
    """Return ``(valid_at, valid_range, anchor)`` for a sub-query."""
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
        return None, None, None
    if qtype in {"RANGE", "COMPARISON"}:
        if t_start is None and t_end is None:
            return None, None, None
        rs = t_start or MINUS_INFINITY
        re_ = t_end or INFINITY
        if rs > re_:
            rs, re_ = re_, rs
        anchor = t_start or t_end
        return None, (rs, re_), anchor
    point = t_start or t_end or query_time
    return point, None, point


def _edge_in_window(
    edge: dict[str, Any],
    valid_at: datetime | None,
    valid_range: tuple[datetime, datetime] | None,
) -> bool:
    es, ee = _edge_bounds(edge)
    if valid_at is not None:
        return es <= valid_at <= ee
    if valid_range is not None:
        rs, re_ = valid_range
        return es <= re_ and ee >= rs
    return True


def _edge_specificity(edge: dict[str, Any]) -> float:
    es, ee = _edge_bounds(edge)
    if es == MINUS_INFINITY and ee == INFINITY:
        return 1e-9
    if es == MINUS_INFINITY or ee == INFINITY:
        return 1e-6
    span_days = max(1.0, (ee - es).total_seconds() / 86400.0)
    return 1.0 / span_days


def _edge_contains(edge: dict[str, Any], anchor: datetime | None) -> bool:
    if anchor is None:
        return False
    es, ee = _edge_bounds(edge)
    return es <= anchor <= ee


def temporal_decay_score(
    query_time: datetime,
    edge_time: datetime,
    alpha: float = 0.1,
) -> float:
    delta_days = abs((query_time - edge_time).total_seconds()) / 86400.0
    return math.exp(-alpha * delta_days)


def _edge_decay(
    edge: dict[str, Any],
    anchor: datetime | None,
    alpha: float,
) -> float:
    if anchor is None:
        return 1.0
    es, ee = _edge_bounds(edge)
    if es == MINUS_INFINITY and ee == INFINITY:
        edge_time = anchor
    elif es == MINUS_INFINITY:
        edge_time = ee
    elif ee == INFINITY:
        edge_time = es
    else:
        edge_time = es + (ee - es) / 2
    return temporal_decay_score(anchor, edge_time, alpha)


def _edge_reliability(edge: dict[str, Any]) -> float:
    try:
        conf = float(edge.get("confidence", 1.0) or 1.0)
    except (ValueError, TypeError):
        conf = 1.0
    try:
        support = int(edge.get("support_count", 1) or 1)
    except (ValueError, TypeError):
        support = 1
    return conf * math.log1p(max(0, support))


def _edge_start_proximity(
    edge: dict[str, Any],
    anchor: datetime | None,
    alpha: float,
) -> float:
    if anchor is None:
        return 0.0
    es, _ = _edge_bounds(edge)
    if es == MINUS_INFINITY:
        return 0.0
    return temporal_decay_score(anchor, es, alpha)


def _rank_primary(
    edges: list[dict[str, Any]],
    anchor: datetime | None,
    valid_range: tuple[datetime, datetime] | None,  # noqa: ARG001 — kept for back-compat
    alpha: float,
    qtype: str = "POINT_IN_TIME",
) -> list[dict[str, Any]]:
    point_in_time = (qtype or "POINT_IN_TIME").upper() == "POINT_IN_TIME"

    def _key(edge: dict[str, Any]) -> tuple[float, ...]:
        contains = 1.0 if _edge_contains(edge, anchor) else 0.0
        spec = _edge_specificity(edge)
        decay = _edge_decay(edge, anchor, alpha)
        rel = _edge_reliability(edge)
        if point_in_time:
            start_prox = _edge_start_proximity(edge, anchor, alpha)
            return (contains, start_prox, spec, decay, rel)
        return (contains, spec, decay, rel)

    return sorted(edges, key=_key, reverse=True)


def _rank_background(
    edges: list[dict[str, Any]],
    anchor: datetime | None,
    alpha: float,
) -> list[dict[str, Any]]:
    def _key(edge: dict[str, Any]) -> tuple[float, float]:
        return (_edge_decay(edge, anchor, alpha), _edge_reliability(edge))
    return sorted(edges, key=_key, reverse=True)


def _strip_embeddings(edge: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in edge.items() if not k.endswith("_embedding")}


async def local_temporal_search(
    session: "AsyncSession",
    sub_query: dict[str, Any],
    seed_entities: list[dict[str, Any]],
    config: BTGraphRAGConfig,
    query_time: datetime,
    k_hop: int = 1,
    primary_limit: int = 30,
    background_limit: int = 10,
) -> dict[str, Any]:
    """As-of subgraph retrieval for a single sub-query.

    Only used in the refusal-recovery branch of the new pipeline. The
    PRIMARY / BACKGROUND distinction is kept because the recovery prompt
    consumes that exact rendering.
    """
    from graphrag.bt_graphrag.neo4j_store import (
        get_active_edges,
        get_subgraph_around_entities,
    )

    valid_at, valid_range, anchor = _sub_query_window(sub_query, query_time)
    alpha = float(getattr(config, "temporal_decay_alpha", 0.1) or 0.1)
    qtype = str(sub_query.get("query_type") or "POINT_IN_TIME").upper()
    seed_titles = [e["title"] for e in seed_entities if e.get("title")]

    result: dict[str, Any] = {
        "sub_query": sub_query,
        "seeds": seed_entities,
        "valid_at": valid_at,
        "valid_range": valid_range,
        "anchor": anchor,
        "edges": [],
        "background_edges": [],
    }

    if not seed_titles:
        edges = await get_active_edges(
            session, at_time=valid_at if valid_at is not None else None,
        )
        edges = [_strip_embeddings(e) for e in edges]
        result["edges"] = _rank_primary(edges, anchor, valid_range, alpha, qtype)[:primary_limit]
        return result

    over_fetch = max(primary_limit * 4, 80)
    primary_sub: dict[str, Any] = {"edges": []}
    if valid_at is not None or valid_range is not None:
        primary_sub = await get_subgraph_around_entities(
            session,
            entity_titles=seed_titles,
            k_hop=k_hop,
            valid_at=valid_at,
            valid_range=valid_range,
            tx_at=None,
            include_disputed=True,
            limit=over_fetch,
        )
    primary_edges = [_strip_embeddings(e) for e in primary_sub.get("edges", [])]

    wide_sub = await get_subgraph_around_entities(
        session,
        entity_titles=seed_titles,
        k_hop=k_hop,
        valid_at=None,
        valid_range=None,
        tx_at=None,
        include_disputed=True,
        limit=max(over_fetch, 200),
    )
    wide_edges = [_strip_embeddings(e) for e in wide_sub.get("edges", [])]

    if primary_edges:
        primary_edges = _rank_primary(primary_edges, anchor, valid_range, alpha, qtype)
        prim_ids = {e.get("id") for e in primary_edges if e.get("id")}
        prim_keys = {
            (e.get("source"), e.get("relation_type"), e.get("target"),
             e.get("t_valid_start"), e.get("t_valid_end"))
            for e in primary_edges
        }
        bg = [
            e for e in wide_edges
            if (e.get("id") and e.get("id") not in prim_ids)
            or (not e.get("id") and (
                e.get("source"), e.get("relation_type"), e.get("target"),
                e.get("t_valid_start"), e.get("t_valid_end"),
            ) not in prim_keys)
        ]
        bg = _rank_background(bg, anchor, alpha)
        result["edges"] = primary_edges[:primary_limit]
        result["background_edges"] = bg[:background_limit]
        return result

    ranked = _rank_background(wide_edges, anchor, alpha)
    promoted = ranked[:primary_limit]
    for e in promoted:
        e["_promoted_from_background"] = True
    result["edges"] = promoted
    result["background_edges"] = ranked[primary_limit:primary_limit + background_limit]
    return result


async def retrieve_temporal_subgraph(
    session: "AsyncSession",
    sub_query: dict[str, Any],
    seed_entities: list[dict[str, Any]],
    config: BTGraphRAGConfig,
    query_time: datetime,
    k_hop: int = 1,
    limit: int = 30,
) -> dict[str, Any]:
    """Backwards-compatible thin wrapper over ``local_temporal_search``."""
    return await local_temporal_search(
        session=session,
        sub_query=sub_query,
        seed_entities=seed_entities,
        config=config,
        query_time=query_time,
        k_hop=k_hop,
        primary_limit=limit,
    )


# ---------------------------------------------------------------------------
# Step 5: Dispute weighting (kept for the legacy dispute_resolution_search)
# ---------------------------------------------------------------------------


def _dispute_score(edge: dict[str, Any], query_time: datetime) -> float:
    rel = _edge_reliability(edge)
    tx_start_dt = _parse_iso_date(edge.get("t_tx_start"))
    if tx_start_dt is None:
        recency = 0.5
    else:
        delta_days = max(0.0, (query_time - tx_start_dt).total_seconds() / 86400.0)
        recency = math.exp(-0.005 * delta_days)
    return rel * recency


def rank_disputed_edges(
    edges: list[dict[str, Any]],
    query_time: datetime,
) -> list[dict[str, Any]]:
    for edge in edges:
        if str(edge.get("status", "")).lower() == "disputed":
            edge["_dispute_score"] = _dispute_score(edge, query_time)
        else:
            edge["_dispute_score"] = None
    disputed = [e for e in edges if e.get("_dispute_score") is not None]
    others = [e for e in edges if e.get("_dispute_score") is None]
    disputed.sort(key=lambda e: e.get("_dispute_score") or 0.0, reverse=True)
    return disputed + others


# ---------------------------------------------------------------------------
# Step 6: Refusal detection (shared with the recovery branch)
# ---------------------------------------------------------------------------


_REFUSAL_RE = re.compile(
    r"(no (record|information|data|specific(ally)?|specified|known|details|"
    r"direct (information|evidence)|mention|indication|evidence|sign)|"
    r"not (specified|provided|available|present|directly|in (the|this)|"
    r"explicitly|recorded|mentioned|listed|stated|found|clear|known|"
    r"indicated|documented|reported)|"
    r"did not (hold|join|belong|play|work|attend|live|specify|appear|have|"
    r"serve|study|teach|lead|run|manage|coach|sign|become)|"
    r"there is no|there's no|"
    r"information.*missing|cannot be (determined|confirmed|identified|"
    r"verified|established|ascertained)|"
    r"unable to (determine|confirm|identify|verify|find)|"
    r"data does not (provide|specify|contain|include|mention|indicate|"
    r"show|state|reveal)|"
    r"according to the (provided|available|given) (data|information|sources)|"
    r"based on the (provided|available|given) (data|information|sources)|"
    r"fails to (mention|specify|provide|include|indicate)|"
    r"is not (mentioned|specified|provided|recorded|listed|documented|"
    r"detailed|available))",
    re.IGNORECASE,
)


def _looks_like_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Step 7: Edge formatting (used in the refusal-recovery prompt)
# ---------------------------------------------------------------------------


def _format_edge_line(edge: dict[str, Any]) -> str:
    src = edge.get("source") or edge.get("source_title") or "?"
    tgt = edge.get("target") or edge.get("target_title") or "?"
    rel = edge.get("relation_type") or edge.get("type") or "RELATED_TO"
    vs = edge.get("t_valid_start", "")
    ve = edge.get("t_valid_end", "")

    def _pretty(v: Any, side: str) -> str:
        if not v:
            return "UNKNOWN" if side == "start" else "ONGOING"
        s = str(v)
        if s == INFINITY_ISO:
            return "ONGOING"
        if s == MINUS_INFINITY_ISO:
            return "UNKNOWN"
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
        f"[{_pretty(vs, 'start')} → {_pretty(ve, 'end')}] "
        f"{{status={status}, support={support}, confidence={conf_str}}}: "
        f"{desc}"
    )


def _format_sub_result_block(
    sub_query: dict[str, Any],
    primary: list[dict[str, Any]],
    background: list[dict[str, Any]],
    max_primary: int,
    max_background: int,
) -> str:
    lines = [f"Sub-query: {sub_query.get('sub_query', '')}"]
    qtype = sub_query.get("query_type", "POINT_IN_TIME")
    t_start = sub_query.get("t_start", "?")
    t_end = sub_query.get("t_end", "?")
    lines.append(f"Type: {qtype}    Window: {t_start} → {t_end}")

    promoted = any(e.get("_promoted_from_background") for e in primary)
    if promoted:
        lines.append(
            "PRIMARY edges (no edge in the requested window — these are the "
            "closest matches around the seeds; their validity does NOT "
            "intersect the question's window):"
        )
    else:
        lines.append(
            "PRIMARY edges (validity intersects the requested window — "
            "answer MUST come from these):"
        )
    if primary:
        lines.extend(_format_edge_line(e) for e in primary[:max_primary])
        if len(primary) > max_primary:
            lines.append(f"... ({len(primary) - max_primary} more primary edges truncated)")
    else:
        lines.append("(none)")

    if background and max_background > 0:
        lines.append("")
        lines.append(
            "BACKGROUND edges (out of window — disambiguation only):"
        )
        lines.extend(_format_edge_line(e) for e in background[:max_background])
        if len(background) > max_background:
            lines.append(
                f"... ({len(background) - max_background} more background edges truncated)"
            )

    return "\n".join(lines)


def format_subgraph_for_llm(
    sub_query: dict[str, Any],
    edges: list[dict[str, Any]],
    max_edges: int,
    valid_at: datetime | None = None,  # noqa: ARG001
    valid_range: tuple[datetime, datetime] | None = None,  # noqa: ARG001
    background_edges: list[dict[str, Any]] | None = None,
    max_background: int = 10,
) -> str:
    """Backwards-compatible wrapper used by some legacy callers."""
    return _format_sub_result_block(
        sub_query=sub_query,
        primary=edges,
        background=background_edges or [],
        max_primary=max_edges,
        max_background=max_background,
    )


# ---------------------------------------------------------------------------
# Step 8: Sub-answer synthesis (compound queries only)
# ---------------------------------------------------------------------------


def _format_subanswers_block(sub_results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, sr in enumerate(sub_results, start=1):
        sq = sr.get("sub_query") or {}
        qt = sq.get("query_type", "POINT_IN_TIME")
        ts, te = sq.get("t_start", "?"), sq.get("t_end", "?")
        text = (sq.get("sub_query") or "").strip()
        answer = (sr.get("answer") or "").strip() or "(no answer)"
        seeds = ", ".join(
            s.get("title", "?") for s in (sr.get("seeds") or [])[:5]
        ) or "(no seeds resolved)"
        lines.append(
            f"### Sub-question {i}\n"
            f"Window: {qt}  [{ts} → {te}]\n"
            f"Text: {text}\n"
            f"Seed entities biased into retrieval: {seeds}\n"
            f"Sub-answer: {answer}\n"
        )
    return "\n".join(lines)


async def _synthesize_subanswers(
    query: str,
    sub_results: list[dict[str, Any]],
    model: "LLMCompletion",
) -> str:
    from graphrag_llm.utils import CompletionMessagesBuilder

    prompt = TEMPORAL_SUBANSWER_SYNTHESIS_PROMPT.format(
        query=query,
        sub_answers_block=_format_subanswers_block(sub_results),
    )
    messages = CompletionMessagesBuilder().add_user_message(prompt).build()
    try:
        response = await model.completion_async(messages=messages)
        return (getattr(response, "content", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Sub-answer synthesis failed: %s", exc)
        # Fall back to the first non-empty sub-answer so we never return ""
        for sr in sub_results:
            ans = (sr.get("answer") or "").strip()
            if ans:
                return ans
        return ""


# ---------------------------------------------------------------------------
# Step 9: Refusal recovery via bitemporal graph evidence
# ---------------------------------------------------------------------------


async def _gather_graph_evidence(
    sub_results: list[dict[str, Any]],
    driver: "AsyncDriver",
    config: BTGraphRAGConfig,
    query_time: datetime,
    k_hop: int = 1,
    primary_limit: int = 12,
    background_limit: int = 8,
) -> list[dict[str, Any]]:
    """For each sub-query, run as-of retrieval on the graph using the
    seeds that the local_search call already resolved. Returns the same
    shape ``local_temporal_search`` does, one dict per sub-query.
    """
    out: list[dict[str, Any]] = []
    for sr in sub_results:
        sq = sr.get("sub_query") or {}
        seeds = sr.get("seeds") or []
        try:
            async with driver.session(database=config.neo4j_database) as session:
                graph_sub = await local_temporal_search(
                    session=session,
                    sub_query=sq,
                    seed_entities=seeds,
                    config=config,
                    query_time=query_time,
                    k_hop=k_hop,
                    primary_limit=primary_limit,
                    background_limit=background_limit,
                )
                graph_sub["edges"] = rank_disputed_edges(graph_sub["edges"], query_time)
        except Exception as exc:  # noqa: BLE001
            logger.debug("graph evidence retrieval failed: %s", exc)
            graph_sub = {
                "sub_query": sq,
                "seeds": seeds,
                "edges": [],
                "background_edges": [],
            }
        out.append(graph_sub)
    return out


async def _refusal_recovery_with_graph(
    query: str,
    prior_answer: str,
    graph_evidence: list[dict[str, Any]],
    model: "LLMCompletion",
    max_edges_per_sub: int = 12,
    max_background_per_sub: int = 8,
) -> str:
    from graphrag_llm.utils import CompletionMessagesBuilder

    blocks = [
        _format_sub_result_block(
            sub_query=g.get("sub_query", {}),
            primary=g.get("edges", []),
            background=g.get("background_edges", []),
            max_primary=max_edges_per_sub,
            max_background=max_background_per_sub,
        )
        for g in graph_evidence
    ]
    sub_evidence = "\n\n".join(blocks) if blocks else "(no graph evidence retrieved)"

    prompt = TEMPORAL_REFUSAL_RECOVERY_PROMPT.format(
        query=query,
        prior_answer=(prior_answer or "")[:600],
        sub_evidence_block=sub_evidence,
    )
    messages = CompletionMessagesBuilder().add_user_message(prompt).build()
    try:
        response = await model.completion_async(messages=messages)
        return (getattr(response, "content", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Refusal recovery LLM call failed: %s", exc)
        return prior_answer


# ---------------------------------------------------------------------------
# Step 10: Orchestrator — temporal helpers around local_search
# ---------------------------------------------------------------------------


def _empty_pipeline_result(
    query: str, query_time: datetime, reason: str = "no model"
) -> dict[str, Any]:
    return {
        "answer": "",
        "sub_queries": [],
        "edges_used": 0,
        "query_time": query_time.isoformat(),
        "analysis": {"entities": [], "sub_queries": [], "error": reason},
        "community_reports": [],
        "baseline_evidence": None,
    }


async def temporal_query_pipeline(
    query: str,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
    model: "LLMCompletion | None" = None,
    embedding_model: "LLMEmbedding | None" = None,
    query_time: datetime | None = None,
    local_search_provider: LocalSearchProvider | None = None,
    k_hop: int = 1,
    max_edges_per_sub: int = 12,  # noqa: ARG001 — kept for signature back-compat
    max_background_per_sub: int = 8,  # noqa: ARG001 — same
    semantic_top_k: int = 0,  # noqa: ARG001 — removed channel
    community_top_k: int = 0,  # noqa: ARG001 — removed channel
    max_total_edges: int = 200,  # noqa: ARG001 — kept for signature back-compat
    baseline_provider: Callable[[str], Awaitable[str | None]] | None = None,
    max_agent_iters: int = 5,  # noqa: ARG001 — kept for signature back-compat
) -> dict[str, Any]:
    """Temporal helper around graphrag's ``local_search``.

    Pipeline:

      1. Decompose the question into sub-queries with explicit windows
         (analyze_and_decompose_query). For most TimeQA-style questions
         this yields a single sub-query.
      2. Per sub-query, *in parallel*:
           a. Resolve seed entities against the bitemporal graph
              (resolve_seed_entities).
           b. Reformulate the sub-query text with the window inline.
           c. Call ``local_search_provider`` with the resolved titles as
              ``include_entity_names`` so graphrag's context builder
              biases entity-mapping toward time-relevant candidates.
      3. If a single sub-answer was produced, return it directly.
         If multiple were produced, run a small synthesis LLM call.
      4. If the final answer hedges / refuses, re-run synthesis using
         as-of evidence from the bitemporal graph
         (refusal-recovery branch).

    ``baseline_provider`` is accepted for signature back-compat with the
    previous pipeline; when ``local_search_provider`` is missing, it is
    used as a fallback (called once with the original query) so the
    pipeline still produces *something* during partial wiring.
    """
    query_time = query_time or utcnow()

    if model is None:
        return _empty_pipeline_result(query, query_time, reason="no model")

    # ── Stage 1: decompose ────────────────────────────────────────────
    analysis = await analyze_and_decompose_query(query, model, query_time)
    sub_queries = analysis.get("sub_queries") or []
    global_entities = analysis.get("entities") or []
    if not sub_queries:
        sub_queries = _heuristic_query_analysis(query, query_time)["sub_queries"]

    seed_top_k = int(getattr(config, "query_seed_top_k", 3) or 3)
    seed_threshold = float(
        getattr(config, "query_seed_embedding_threshold", 0.55) or 0.55
    )
    seed_prefer_title = bool(
        getattr(config, "query_seed_prefer_title_match", True)
    )

    # Resolve the provider once: if local_search_provider is missing,
    # fall back to the legacy baseline_provider wrapper.
    provider: LocalSearchProvider | None = local_search_provider
    if provider is None and baseline_provider is not None:
        async def _provider_compat(
            q: str, *, include_entity_names: list[str] | None = None,
        ) -> str | None:
            del include_entity_names  # legacy provider cannot accept seeds
            return await baseline_provider(q)
        provider = _provider_compat

    # ── Stage 2: per-sub-query retrieval (parallel) ──────────────────
    async def _process_one(sq: dict[str, Any]) -> dict[str, Any]:
        seed_names = sq.get("entities") or global_entities or []
        seeds: list[dict[str, Any]] = []
        try:
            async with driver.session(database=config.neo4j_database) as session:
                seeds = await resolve_seed_entities(
                    session, seed_names,
                    embedding_model=embedding_model,
                    top_k=seed_top_k,
                    score_threshold=seed_threshold,
                    prefer_title_match=seed_prefer_title,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("seed resolution failed for %r: %s", seed_names, exc)

        rendered = _render_subquery_with_window(sq)
        include_names = [s.get("title") for s in seeds if s.get("title")]

        answer: str | None = None
        if provider is not None:
            try:
                answer = await provider(
                    rendered, include_entity_names=include_names or None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("local_search_provider failed: %s", exc)
                answer = None

        return {
            "sub_query": sq,
            "seeds": seeds,
            "rendered_query": rendered,
            "include_entity_names": include_names,
            "answer": (answer or "").strip(),
            "edges": [],            # populated later if refusal recovery runs
            "background_edges": [],
        }

    sub_results: list[dict[str, Any]] = list(
        await asyncio.gather(*(_process_one(sq) for sq in sub_queries))
    )

    # ── Stage 3: synthesis ────────────────────────────────────────────
    if len(sub_results) == 1:
        final = sub_results[0].get("answer", "")
    else:
        final = await _synthesize_subanswers(query, sub_results, model)

    # ── Stage 4: refusal recovery ─────────────────────────────────────
    retried = False
    needs_recovery = (not final.strip()) or _looks_like_refusal(final)
    if needs_recovery:
        retried = True
        graph_evidence = await _gather_graph_evidence(
            sub_results=sub_results,
            driver=driver,
            config=config,
            query_time=query_time,
            k_hop=k_hop,
        )
        # Pipe the graph evidence back into the sub_results so the
        # caller can serialise it for telemetry.
        for sr, ge in zip(sub_results, graph_evidence):
            sr["edges"] = ge.get("edges", [])
            sr["background_edges"] = ge.get("background_edges", [])
        final = await _refusal_recovery_with_graph(
            query=query,
            prior_answer=final,
            graph_evidence=graph_evidence,
            model=model,
        )

    edges_used = sum(len(sr.get("edges", [])) for sr in sub_results)
    analysis_out = dict(analysis)
    analysis_out["refusal_retry"] = retried
    return {
        "answer": final,
        "sub_queries": sub_results,
        "edges_used": edges_used,
        "query_time": query_time.isoformat(),
        "analysis": analysis_out,
        "community_reports": [],
        "baseline_evidence": None,
    }


# ---------------------------------------------------------------------------
# Legacy entry points (kept stable; only the orchestrator was rewritten)
# ---------------------------------------------------------------------------


async def temporal_audit_search(
    query: str,
    audit_time: datetime,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
) -> dict[str, Any]:
    """Temporal Audit: 'What did the system believe at time T?'."""
    from graphrag.bt_graphrag.neo4j_store import get_system_state_at

    async with driver.session(database=config.neo4j_database) as session:
        edges = await get_system_state_at(session, audit_time)

    return {
        "system_state_at": audit_time.isoformat(),
        "edges": edges,
        "total_edges": len(edges),
    }


async def dispute_resolution_search(
    query: str,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
    model: "LLMCompletion | None" = None,
    entity_title: str | None = None,
) -> dict[str, Any]:
    """Surface ``disputed`` edges and (optionally) ask the LLM to weight them."""
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


async def temporal_search(
    query: str,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
    model: "LLMCompletion | None" = None,
    embedding_model: "LLMEmbedding | None" = None,
    query_time: datetime | None = None,
    search_mode: str = "local",
) -> dict[str, Any]:
    """Unified entry — kept as the thesis-named top-level dispatcher."""
    if search_mode == "audit":
        return await temporal_audit_search(
            query, query_time or utcnow(), config, driver,
        )
    if search_mode == "dispute":
        return await dispute_resolution_search(query, config, driver, model)
    return await temporal_query_pipeline(
        query=query,
        config=config,
        driver=driver,
        model=model,
        embedding_model=embedding_model,
        query_time=query_time,
    )
