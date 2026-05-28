# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG Stage 6(b): Temporal Query Pipeline.

This module implements the query side of the proposal
(see ``Proposal.tex``, sec. *Integración con el pipeline de GraphRAG y
consulta temporal*, and ``Implementation.tex``, sec. *Etapa 6 (b):
Consulta temporal*). The pipeline has the five canonical steps:

    1. ``decompose_temporal_query``        — analyse + decompose
    2. ``resolve_seed_entities``           — title + embedding lookup
    3. ``local_temporal_search``           — bitemporal as-of retrieval
    4. ``dispute_resolution_search``       — re-rank ``disputed`` edges
    5. ``synthesize_temporal_answer``      — final answer (LLM)

The orchestrator is ``temporal_query_pipeline`` (and ``temporal_search``
as the legacy entry that also dispatches the audit/dispute modes).

The key implementation choices that distinguish the rewritten pipeline
from the first iteration — driven by the diagnosis on the TimeQA
``hard`` split, where 77 % of incorrect answers had the supporting edge
present in Neo4j with the correct validity window:

  * **Two-bucket retrieval, not a single ranked list.** Each sub-query
    produces a PRIMARY bucket (edges whose ``t_valid_*`` intersects the
    sub-query window) and a BACKGROUND bucket (the rest). The buckets
    are shown to the LLM in *separate* blocks; PRIMARY is the only
    bucket the model is allowed to answer from when it is non-empty.
  * **Ranking inside PRIMARY** prefers, in order: edges whose interval
    contains the sub-query anchor; narrower intervals (specific facts
    beat generic 0001→9999 ones); ``temporal_decay_score`` against the
    anchor; ``confidence × log(1+support_count)``.
  * **Community reports** are only added when *all* sub-queries returned
    an empty PRIMARY bucket — they otherwise dilute atomic answers with
    the entity's "most prominent" associations.
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
    TEMPORAL_ANSWER_SYNTHESIS_PROMPT,
    TEMPORAL_QUERY_ANALYSIS_PROMPT,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.embedding import LLMEmbedding
    from neo4j import AsyncDriver, AsyncSession

logger = logging.getLogger(__name__)


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


# Thesis-traceability alias (Implementation.tex names the function this way).
async def decompose_temporal_query(
    query: str,
    model: "LLMCompletion",
    current_date: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return just the sub-query list — preserved for thesis traceability."""
    analysis = await analyze_and_decompose_query(query, model, current_date)
    return analysis.get("sub_queries", [])


# ---------------------------------------------------------------------------
# Step 2: Resolve seed entities against the graph
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

    1. **Title match** — exact / case-insensitive / substring matches
       (cheap, precise).
    2. **Description-embedding NN** — vector search on
       ``Entity.description_embedding``, mirroring CGER's cosine
       pre-filter from indexing time. Provides recall when the
       LLM-extracted name doesn't appear verbatim in the graph.

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
# Step 3: Local temporal search (per sub-query)
# ---------------------------------------------------------------------------


def _sub_query_window(
    sub_query: dict[str, Any],
    query_time: datetime,
) -> tuple[datetime | None, tuple[datetime, datetime] | None, datetime | None]:
    """Return ``(valid_at, valid_range, anchor)`` for a sub-query.

    - POINT_IN_TIME: ``valid_at`` is set to the explicit point;
      ``anchor`` = that same point.
    - RANGE / COMPARISON: ``valid_range`` covers ``[t_start, t_end]``;
      ``anchor`` is ``t_start`` (the typical TimeQA "after / between /
      from X" phrasing keys off the start), falling back to ``t_end``.
    - EVOLUTION: no temporal restriction (``None, None``); ``anchor`` is
      ``None`` and ranking ignores temporal proximity.
    - ``CURRENT`` / ``ONGOING`` resolve to ``query_time``; ``UNKNOWN``
      becomes ``None``.
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
    # POINT_IN_TIME
    point = t_start or t_end or query_time
    return point, None, point


def _edge_in_window(
    edge: dict[str, Any],
    valid_at: datetime | None,
    valid_range: tuple[datetime, datetime] | None,
) -> bool:
    """``True`` iff the edge's validity intersects the requested window.

    Unknown bounds are treated as open-ended on their side.
    """
    es, ee = _edge_bounds(edge)
    if valid_at is not None:
        return es <= valid_at <= ee
    if valid_range is not None:
        rs, re_ = valid_range
        return es <= re_ and ee >= rs
    return True


def _edge_specificity(edge: dict[str, Any]) -> float:
    """``1 / span_days`` — narrower intervals are more specific.

    Open intervals (``0001-01-01`` → ``9999-12-31``) collapse to a tiny
    score. Edges with at least one unknown bound get a *small* but
    non-zero score so they don't get dropped completely — they just lose
    to any narrower competitor.
    """
    es, ee = _edge_bounds(edge)
    if es == MINUS_INFINITY and ee == INFINITY:
        return 1e-9
    if es == MINUS_INFINITY or ee == INFINITY:
        # Half-open interval: treat as moderately generic.
        return 1e-6
    span_days = max(1.0, (ee - es).total_seconds() / 86400.0)
    return 1.0 / span_days


def _edge_contains(edge: dict[str, Any], anchor: datetime | None) -> bool:
    """Does the edge's validity contain the sub-query anchor?

    The anchor is the question's most representative timestamp
    (``t_start`` for "after / between", ``t_end`` for "before",
    the point itself for POINT_IN_TIME). Edges whose validity covers
    that instant — i.e. the edge was *true at the anchor* — get a
    boolean boost in the ranker.

    We deliberately do NOT require the edge to cover the whole
    requested range. A "after 2002-10" question whose gold edge is
    ``[2001, 2003]`` is a fact that was true AT the anchor; demanding
    coverage of ``[2002-10, NOW]`` would discard exactly that gold.
    """
    if anchor is None:
        return False
    es, ee = _edge_bounds(edge)
    return es <= anchor <= ee


def temporal_decay_score(
    query_time: datetime,
    edge_time: datetime,
    alpha: float = 0.1,
) -> float:
    """Exponential decay on temporal proximity (days).

    Preserved with the thesis-mentioned name (Implementation.tex,
    sec. 6b). The orchestrator passes ``alpha`` from
    ``BTGraphRAGConfig.temporal_decay_alpha``.
    """
    delta_days = abs((query_time - edge_time).total_seconds()) / 86400.0
    return math.exp(-alpha * delta_days)


def _edge_decay(
    edge: dict[str, Any],
    anchor: datetime | None,
    alpha: float,
) -> float:
    """Per-edge decay against the sub-query anchor.

    Uses the edge's midpoint when both bounds are known; falls back to
    whichever bound exists; returns ``1.0`` when no anchor is set.
    """
    if anchor is None:
        return 1.0
    es, ee = _edge_bounds(edge)
    if es == MINUS_INFINITY and ee == INFINITY:
        edge_time = anchor  # generic edge: don't penalise on decay
    elif es == MINUS_INFINITY:
        edge_time = ee
    elif ee == INFINITY:
        edge_time = es
    else:
        mid = es + (ee - es) / 2
        edge_time = mid
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
    """Decay on ``|t_valid_start − anchor|``.

    For POINT_IN_TIME "joined / became / appointed in X" questions, the
    intended answer is an edge whose validity *started* at the anchor —
    not one that merely contains it. This score lets the ranker prefer
    "joined in 1908" (start = 1908) over "was member 1900–1915" (start
    far from the anchor) even though both contain the anchor.

    Edges with an UNKNOWN start get ``0.0`` so they cannot outrank a
    well-anchored edge on this dimension.
    """
    if anchor is None:
        return 0.0
    es, _ = _edge_bounds(edge)
    if es == MINUS_INFINITY:
        return 0.0
    return temporal_decay_score(anchor, es, alpha)


def _rank_primary(
    edges: list[dict[str, Any]],
    anchor: datetime | None,
    valid_range: tuple[datetime, datetime] | None,  # kept for back-compat
    alpha: float,
    qtype: str = "POINT_IN_TIME",
) -> list[dict[str, Any]]:
    """Sort PRIMARY edges by the priority list documented at module top.

    Priority (all descending):
      1. edge interval contains the anchor (was the fact true *at* the anchor?)
      2. (POINT_IN_TIME only) start proximity — edges whose ``t_valid_start``
         is close to the anchor beat edges that merely contain it, so
         "joined in 1908" outranks "was member during 1900–1915".
      3. specificity (narrower interval beats a generic 0001 → 9999 one)
      4. temporal decay (edge midpoint closer to the anchor)
      5. reliability (confidence × log(1+support_count))
    """
    del valid_range  # unused, retained for API compatibility
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
    """Sort BACKGROUND edges by decay then reliability.

    Background edges are out-of-window, so specificity / containment
    don't apply. Closer (in time) and more reliable beats farther and
    weaker. Generic 0001→9999 edges retain a neutral decay so they can
    still surface when nothing else exists.
    """
    def _key(edge: dict[str, Any]) -> tuple[float, float]:
        return (_edge_decay(edge, anchor, alpha), _edge_reliability(edge))

    return sorted(edges, key=_key, reverse=True)


def _strip_embeddings(edge: dict[str, Any]) -> dict[str, Any]:
    """Drop heavy embedding fields before keeping the edge in memory."""
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
    """Run the as-of subgraph retrieval for a single sub-query.

    Returns::

        {
          "sub_query": <input sub_query>,
          "seeds": [...],
          "valid_at": datetime | None,
          "valid_range": (datetime, datetime) | None,
          "anchor": datetime | None,
          "edges": [...],            # PRIMARY (in-window), ranked
          "background_edges": [...], # out-of-window context
        }

    Stage A fetches in-window edges only (``valid_at`` /
    ``valid_range`` constraint + ``t_tx_end = INFINITY``). If Stage A
    returns nothing — or no seeds resolved — Stage B widens the search
    to all currently-believed edges around the seeds, and the rank pulls
    the most temporally-relevant ones into the primary bucket.
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
        # No seed entities: scan currently-active edges at the requested
        # time so the LLM can still respond "no record around <X>".
        edges = await get_active_edges(
            session, at_time=valid_at if valid_at is not None else None,
        )
        edges = [_strip_embeddings(e) for e in edges]
        result["edges"] = _rank_primary(edges, anchor, valid_range, alpha, qtype)[:primary_limit]
        return result

    # Stage A: strict in-window retrieval. We ask for a generous
    # over-fetch so the Python rerank has material to work with.
    over_fetch = max(primary_limit * 4, 80)
    primary_sub: dict[str, Any] = {"edges": []}
    if valid_at is not None or valid_range is not None:
        primary_sub = await get_subgraph_around_entities(
            session,
            entity_titles=seed_titles,
            k_hop=k_hop,
            valid_at=valid_at,
            valid_range=valid_range,
            tx_at=None,  # t_tx_end = INFINITY (currently-believed)
            include_disputed=True,
            limit=over_fetch,
        )
    primary_edges = [_strip_embeddings(e) for e in primary_sub.get("edges", [])]

    # Stage B: widen — all currently-believed edges around the seeds,
    # regardless of validity. Used both as background for the LLM and as
    # the fallback pool when Stage A is empty.
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
        # We have in-window edges. Rank them; the BACKGROUND bucket gets
        # whatever wide edges are NOT in the primary bucket (by id, or
        # by tuple shape as a fallback).
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

    # Stage A empty: promote the most temporally-relevant wide edges as
    # primary so the LLM has *something* to answer from. The rest stays
    # as background. Mark each promoted edge so the prompt can warn the
    # LLM that the window did not actually overlap.
    ranked = _rank_background(wide_edges, anchor, alpha)
    promoted = ranked[:primary_limit]
    for e in promoted:
        e["_promoted_from_background"] = True
    result["edges"] = promoted
    result["background_edges"] = ranked[primary_limit:primary_limit + background_limit]
    return result


# Thesis name kept as a thin wrapper; older callers used this signature.
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
# Step 4: Dispute weighting
# ---------------------------------------------------------------------------


def _dispute_score(edge: dict[str, Any], query_time: datetime) -> float:
    """``confidence × log(1+support_count) × recency-of-belief``."""
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
    """Annotate ``disputed`` edges with ``_dispute_score`` and re-sort.

    Non-disputed edges keep their relative order. Disputed edges are
    sorted by score descending. The annotated list is the same edges
    re-ordered: disputed ones first (highest-score first), then the
    non-disputed in their original order.
    """
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
# Step 5: Format edges + synthesise the answer
# ---------------------------------------------------------------------------


def _format_edge_line(edge: dict[str, Any]) -> str:
    """Render a single edge as one Markdown-ish bullet for the LLM."""
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
    """Render one sub-query block with PRIMARY and BACKGROUND sections."""
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
            "BACKGROUND edges (out of window — use only to disambiguate "
            "entities or as a last-resort tiebreak; never as the primary "
            "source of the answer):"
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
    valid_at: datetime | None = None,
    valid_range: tuple[datetime, datetime] | None = None,
    background_edges: list[dict[str, Any]] | None = None,
    max_background: int = 10,
) -> str:
    """Backwards-compatible wrapper for callers that pass a single list.

    New callers should use ``_format_sub_result_block`` directly with
    explicit PRIMARY / BACKGROUND lists.
    """
    return _format_sub_result_block(
        sub_query=sub_query,
        primary=edges,
        background=background_edges or [],
        max_primary=max_edges,
        max_background=max_background,
    )


def _format_baseline_block(baseline_evidence: str | None) -> str:
    """Render the optional BASELINE EVIDENCE block for the synthesis prompt.

    ``baseline_evidence`` is a short text-grounded answer produced by a
    non-temporal retriever (e.g. vanilla local_search) — see the
    ``baseline_provider`` plumbing in ``temporal_query_pipeline``. We
    render it as a clearly-delimited block so the LLM can apply the
    cross-check rules in the prompt's instructions section.
    """
    if not baseline_evidence:
        return ""
    text = baseline_evidence.strip()
    if not text:
        return ""
    return (
        "BASELINE EVIDENCE (text-grounded answer from non-temporal "
        "retrieval — use as a cross-check on PRIMARY; see instructions):\n"
        f"{text}\n\n"
    )


async def synthesize_temporal_answer(
    query: str,
    sub_results: list[dict[str, Any]],
    model: "LLMCompletion",
    max_edges_per_sub: int = 30,
    max_background_per_sub: int = 10,
    community_reports: list[dict[str, Any]] | None = None,
    community_max_chars_per_report: int = 1500,
    baseline_evidence: str | None = None,
) -> str:
    """Render each sub-query as a block and ask the LLM for one answer.

    Communities, when supplied, are prepended as background context
    (DRIFT-style primer). The orchestrator decides whether to include
    them at all.

    ``baseline_evidence`` is an optional text-grounded answer from a
    non-temporal retriever (e.g. vanilla local_search). When provided,
    it is rendered as a third bucket the prompt's instructions teach
    the LLM to cross-check against PRIMARY — the goal is to recover
    refusals and correct wrong-picks driven by noisy date extraction.
    """
    from graphrag_llm.utils import CompletionMessagesBuilder

    from graphrag.bt_graphrag.temporal_communities import (
        format_community_reports_for_llm,
    )

    blocks = [
        _format_sub_result_block(
            sub_query=r.get("sub_query", {}),
            primary=r.get("edges", []),
            background=r.get("background_edges", []),
            max_primary=max_edges_per_sub,
            max_background=max_background_per_sub,
        )
        for r in sub_results
    ]
    sub_results_text = "\n\n".join(blocks) if blocks else "(no context retrieved)"

    community_block = ""
    if community_reports:
        rendered = format_community_reports_for_llm(
            community_reports,
            max_chars_per_report=community_max_chars_per_report,
        )
        if rendered:
            community_block = (
                "Community reports (use only to disambiguate when PRIMARY "
                "edges are absent or ambiguous):\n"
                f"{rendered}\n\n"
            )

    baseline_block = _format_baseline_block(baseline_evidence)

    prompt = TEMPORAL_ANSWER_SYNTHESIS_PROMPT.format(
        query=query,
        sub_results=sub_results_text,
        community_context=community_block,
        baseline_block=baseline_block,
    )
    messages = CompletionMessagesBuilder().add_user_message(prompt).build()
    response = await model.completion_async(messages=messages)
    return (getattr(response, "content", "") or "").strip()


# Legacy alias preserved for older callers.
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
    embedding_model: "LLMEmbedding | None" = None,
    query_time: datetime | None = None,
    k_hop: int = 1,  # noqa: ARG001 — kept for signature back-compat
    max_edges_per_sub: int = 30,  # noqa: ARG001
    max_background_per_sub: int = 10,  # noqa: ARG001
    max_total_edges: int = 200,  # noqa: ARG001
    baseline_provider: Callable[[str], Awaitable[str | None]] | None = None,
    max_agent_iters: int = 8,
) -> dict[str, Any]:
    """Run the agent-based BT-GraphRAG temporal query pipeline.

    The model is given a toolbox (resolve_entities, time_window_search,
    wide_search, text_search, final_answer) and drives its own search
    loop until it decides it has enough to answer (or hits
    ``max_agent_iters``).

    Returns the same dict shape the legacy pipeline produced so callers
    (eg. ``runners.bt_answer``) keep working unchanged. The agent trace
    is surfaced under ``analysis['agent_trace']`` and the LLM-collected
    edges are exposed via a single synthetic sub_result entry — the
    old PRIMARY/BACKGROUND buckets are no longer meaningful in the
    agent model and we keep the field only for inspection.
    """
    from graphrag.bt_graphrag.temporal_agent import (
        AgentContext,
        run_temporal_agent,
    )

    query_time = query_time or utcnow()

    if model is None:
        # Agent loop needs an LLM; fall back to an empty answer so
        # callers still get a well-shaped dict.
        return {
            "answer": "",
            "sub_queries": [],
            "edges_used": 0,
            "query_time": query_time.isoformat(),
            "analysis": {"agent_trace": [], "iterations": 0},
            "community_reports": [],
            "baseline_evidence": None,
        }

    ctx = AgentContext(
        driver=driver,
        config=config,
        model=model,
        embedding_model=embedding_model,
        query_time=query_time,
        baseline_provider=baseline_provider,
    )

    try:
        agent_result = await run_temporal_agent(
            query=query,
            ctx=ctx,
            max_iters=max_agent_iters,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("temporal agent failed: %s", exc)
        return {
            "answer": "",
            "sub_queries": [],
            "edges_used": 0,
            "query_time": query_time.isoformat(),
            "analysis": {"agent_trace": [], "iterations": 0, "error": str(exc)},
            "community_reports": [],
            "baseline_evidence": None,
        }

    answer = agent_result.get("answer", "") or ""
    trace = agent_result.get("trace", []) or []
    edges = agent_result.get("edges_collected", []) or []
    baseline_evidence = agent_result.get("baseline_answer")
    iterations = agent_result.get("iterations", 0)
    ts_count = agent_result.get("text_search_count", 0)
    ts_empty = agent_result.get("text_search_empty", 0)

    sub_results = [
        {
            "sub_query": {"sub_query": query, "query_type": "AGENT"},
            "edges": edges,
            "background_edges": [],
            "seeds": [],
            "valid_at": None,
            "valid_range": None,
            "anchor": None,
            "entities": [],
        }
    ] if edges else []

    return {
        "answer": answer,
        "sub_queries": sub_results,
        "edges_used": len(edges),
        "query_time": query_time.isoformat(),
        "analysis": {
            "agent_trace": trace,
            "iterations": iterations,
            "text_search_count": ts_count,
            "text_search_empty": ts_empty,
        },
        "community_reports": [],
        "baseline_evidence": baseline_evidence,
    }


# ---------------------------------------------------------------------------
# Legacy entry points
# ---------------------------------------------------------------------------


async def temporal_audit_search(
    query: str,
    audit_time: datetime,
    config: BTGraphRAGConfig,
    driver: "AsyncDriver",
) -> dict[str, Any]:
    """Temporal Audit: 'What did the system believe at time T?'.

    Queries by transaction-time to reconstruct historical system state.
    Not part of the question-answering pipeline.
    """
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
    """Surface ``disputed`` edges and (optionally) ask the LLM to weight them.

    Implements the disambiguation-only mode that the thesis names as
    ``dispute_resolution_search``. The dispute weighting is exposed as
    ``rank_disputed_edges`` and is also applied inside the main pipeline
    over the PRIMARY bucket.
    """
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
    """Unified entry — the thesis-named top-level dispatcher.

    ``search_mode`` switches between the three modes described in
    Implementation.tex, sec. 6b:

      - ``local`` (default): full five-step pipeline
        (``temporal_query_pipeline``).
      - ``audit``: ``temporal_audit_search`` over transaction-time.
      - ``dispute``: ``dispute_resolution_search`` over disputed edges.
    """
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
