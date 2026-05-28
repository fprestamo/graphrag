# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG Stage 6(b) — Agent loop replacing the static pipeline.

The pipeline used to be a fixed waterfall: decompose → resolve → retrieve
→ background → synthesize. Every question went through the same five
steps with no opportunity to recover when one of them produced noisy
results.

Here the same primitives are exposed as **tools** and the LLM drives a
ReAct-style loop: at each turn it emits a JSON object choosing the next
tool to invoke (with arguments) or the final answer. The loop runs for
up to ``max_iters`` turns; a hard "give your best answer now" fallback
fires when the budget is spent.

No native function-calling — we keep the LLM contract minimal:

  assistant turn  →  {"tool": <name>, "args": {...}}   or   {"answer": "..."}

The orchestrator parses the JSON, runs the tool, appends the result as
the next user turn, and asks again.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.models.temporal_types import utcnow
from graphrag.bt_graphrag.prompts import (
    TEMPORAL_AGENT_FORCE_ANSWER_PROMPT,
    TEMPORAL_AGENT_SYSTEM_PROMPT,
    TEMPORAL_AGENT_USER_PROMPT,
)

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from graphrag_llm.types import LLMCompletion, LLMEmbedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool result formatting helpers
# ---------------------------------------------------------------------------

_DESC_TRUNCATE = 240
_TOOL_RESULT_TRUNCATE = 4500  # cap per-tool response size shown to the LLM


def _truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


def _format_edge_line(e: dict[str, Any]) -> str:
    src = e.get("source") or e.get("source_title") or "?"
    tgt = e.get("target") or e.get("target_title") or "?"
    rel = e.get("relation_type") or e.get("type") or "RELATED_TO"
    vs = e.get("t_valid_start") or "UNKNOWN"
    ve = e.get("t_valid_end") or "ONGOING"
    status = e.get("status") or "active"
    support = e.get("support_count") if e.get("support_count") is not None else 1
    conf = e.get("confidence")
    try:
        conf_str = f"{float(conf):.2f}" if conf is not None else "1.00"
    except (ValueError, TypeError):
        conf_str = "1.00"
    desc = _truncate(e.get("description") or "", _DESC_TRUNCATE)
    return (
        f"- ({src}) -[{rel}]-> ({tgt}) [{vs} → {ve}] "
        f"{{status={status}, support={support}, conf={conf_str}}}: {desc}"
    )


def _format_edges(edges: list[dict[str, Any]], limit: int) -> str:
    if not edges:
        return "(no edges)"
    lines = [_format_edge_line(e) for e in edges[:limit]]
    if len(edges) > limit:
        lines.append(f"... ({len(edges) - limit} more not shown)")
    return "\n".join(lines)


def _format_entities(ents: list[dict[str, Any]]) -> str:
    if not ents:
        return "(no entities resolved)"
    lines = []
    for e in ents:
        title = e.get("title") or "?"
        etype = e.get("type") or "?"
        desc = _truncate(e.get("description") or "", _DESC_TRUNCATE)
        lines.append(f"- {title} ({etype}): {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class AgentContext:
    """Mutable state threaded through every tool call.

    Holds the Neo4j driver, the config, the LLM/embedding handles, the
    baseline_provider (vanilla local_search) callable, and accumulates
    side-data we surface back in the pipeline result (collected edges,
    seeds resolved, baseline answer when text_search is invoked).
    """

    def __init__(
        self,
        *,
        driver: "AsyncDriver",
        config: BTGraphRAGConfig,
        model: "LLMCompletion",
        embedding_model: "LLMEmbedding | None",
        query_time: datetime,
        baseline_provider: Callable[[str], Awaitable[str | None]] | None,
    ) -> None:
        self.driver = driver
        self.config = config
        self.model = model
        self.embedding_model = embedding_model
        self.query_time = query_time
        self.baseline_provider = baseline_provider
        self.collected_edges: list[dict[str, Any]] = []
        self.resolved_titles: set[str] = set()
        self.baseline_answer: str | None = None
        self.text_search_count = 0
        self.text_search_empty = 0  # how many TS calls returned empty / null


# Each tool is `async def tool(ctx, args) -> str` returning the user-visible
# string the LLM will see as the next turn. The dispatcher catches exceptions
# and renders them as a tool error string.

async def _tool_resolve_entities(
    ctx: AgentContext, args: dict[str, Any]
) -> str:
    from graphrag.bt_graphrag.temporal_query import resolve_seed_entities

    names = args.get("names") or []
    if isinstance(names, str):
        names = [names]
    names = [str(n).strip() for n in names if str(n).strip()]
    if not names:
        return "Error: 'names' must be a non-empty list of strings."
    top_k = int(args.get("top_k", 3) or 3)
    top_k = max(1, min(top_k, 6))

    async with ctx.driver.session(database=ctx.config.neo4j_database) as session:
        ents = await resolve_seed_entities(
            session,
            names,
            embedding_model=ctx.embedding_model,
            top_k=top_k,
            score_threshold=getattr(
                ctx.config, "query_seed_embedding_threshold", 0.55,
            ),
            prefer_title_match=getattr(
                ctx.config, "query_seed_prefer_title_match", True,
            ),
        )
    for e in ents:
        t = e.get("title")
        if t:
            ctx.resolved_titles.add(t)
    return _format_entities(ents)


def _parse_date(s: str | None) -> datetime | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.upper() in {"NULL", "NONE", "UNKNOWN"}:
        return None
    if s.upper() in {"CURRENT", "TODAY", "ONGOING"}:
        return None  # caller will substitute query_time
    # Match YYYY-MM-DD or YYYY or YYYY-MM
    m = re.match(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$", s)
    if m:
        y, mo, d = m.group(1), m.group(2) or "01", m.group(3) or "01"
        try:
            from datetime import timezone
            return datetime(int(y), int(mo), int(d), tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


async def _tool_time_window_search(
    ctx: AgentContext, args: dict[str, Any]
) -> str:
    from graphrag.bt_graphrag.neo4j_store import get_subgraph_around_entities

    titles = args.get("entity_titles") or []
    if isinstance(titles, str):
        titles = [titles]
    titles = [str(t).strip() for t in titles if str(t).strip()]
    if not titles:
        return ("Error: 'entity_titles' must be a non-empty list. "
                "Call resolve_entities first.")
    t_start = _parse_date(args.get("t_start"))
    t_end = _parse_date(args.get("t_end"))
    if t_end is None:
        t_end = ctx.query_time
    if t_start is None:
        from graphrag.bt_graphrag.models.temporal_types import MINUS_INFINITY
        t_start = MINUS_INFINITY
    if t_start > t_end:
        t_start, t_end = t_end, t_start
    k_hop = max(1, min(int(args.get("k_hop", 1) or 1), 2))
    limit = max(1, min(int(args.get("limit", 12) or 12), 30))

    async with ctx.driver.session(database=ctx.config.neo4j_database) as session:
        sub = await get_subgraph_around_entities(
            session,
            entity_titles=titles,
            k_hop=k_hop,
            valid_at=None,
            valid_range=(t_start, t_end),
            tx_at=None,
            include_disputed=True,
            limit=limit,
        )
    edges = [
        {k: v for k, v in e.items() if not k.endswith("_embedding")}
        for e in sub.get("edges", [])
    ]
    ctx.collected_edges.extend(edges)
    header = (
        f"time_window_search returned {len(edges)} edge(s) for "
        f"{titles} in window [{t_start.date()} → {t_end.date()}]:"
    )
    return f"{header}\n{_format_edges(edges, limit)}"


async def _tool_wide_search(
    ctx: AgentContext, args: dict[str, Any]
) -> str:
    from graphrag.bt_graphrag.neo4j_store import get_subgraph_around_entities

    titles = args.get("entity_titles") or []
    if isinstance(titles, str):
        titles = [titles]
    titles = [str(t).strip() for t in titles if str(t).strip()]
    if not titles:
        return ("Error: 'entity_titles' must be a non-empty list. "
                "Call resolve_entities first.")
    k_hop = max(1, min(int(args.get("k_hop", 1) or 1), 2))
    limit = max(1, min(int(args.get("limit", 15) or 15), 30))

    async with ctx.driver.session(database=ctx.config.neo4j_database) as session:
        sub = await get_subgraph_around_entities(
            session,
            entity_titles=titles,
            k_hop=k_hop,
            valid_at=None,
            valid_range=None,
            tx_at=None,
            include_disputed=True,
            limit=limit,
        )
    edges = [
        {k: v for k, v in e.items() if not k.endswith("_embedding")}
        for e in sub.get("edges", [])
    ]
    ctx.collected_edges.extend(edges)
    header = (
        f"wide_search returned {len(edges)} edge(s) for {titles} "
        f"(no time filter):"
    )
    return f"{header}\n{_format_edges(edges, limit)}"


async def _tool_search_edges_by_description(
    ctx: AgentContext, args: dict[str, Any]
) -> str:
    """Semantic search over relationship descriptions.

    Embeds the caller's natural-language ``description`` and looks up the
    top-K edges in Neo4j's ``relationship_description_embedding`` vector
    index. This bypasses the seed-entity step entirely: the answer to
    "who was chair of Org X in 1898?" often lives in an edge whose
    description literally mentions the role + period, even when the
    title-match for "Org X" returned nothing useful.
    """
    from graphrag.bt_graphrag.neo4j_store import vector_search_relationships

    description = str(args.get("description") or "").strip()
    if not description:
        return ("Error: 'description' must be a non-empty natural-language "
                "string describing what the edge should contain.")
    if ctx.embedding_model is None:
        return ("Error: embedding model not available in this run; "
                "use wide_search or text_search instead.")
    top_k = max(1, min(int(args.get("top_k", 10) or 10), 20))

    try:
        emb_resp = await ctx.embedding_model.embedding_async(input=[description])
        vectors = list(getattr(emb_resp, "embeddings", []) or [])
    except Exception as exc:  # noqa: BLE001
        return f"Error: embedding call failed: {exc}"
    if not vectors or not vectors[0]:
        return "Error: embedding model returned no vector."

    async with ctx.driver.session(database=ctx.config.neo4j_database) as session:
        edges = await vector_search_relationships(
            session, vectors[0], top_k=top_k,
        )
    edges = [
        {k: v for k, v in e.items() if not k.endswith("_embedding")}
        for e in edges
    ]
    ctx.collected_edges.extend(edges)
    header = (
        f"search_edges_by_description returned {len(edges)} edge(s) "
        f"semantically matching: \"{description[:120]}\""
    )
    return f"{header}\n{_format_edges(edges, top_k)}"


async def _tool_text_search(
    ctx: AgentContext, args: dict[str, Any]
) -> str:
    question = str(args.get("question") or "").strip()
    if not question:
        return "Error: 'question' must be a non-empty string."
    if ctx.baseline_provider is None:
        return ("Error: text_search is not available in this run "
                "(no baseline_provider wired). Fall back to wide_search.")
    if ctx.text_search_count >= 3:
        return ("Error: text_search has been called 3 times already. "
                "Make your best guess from the evidence you already have.")
    ctx.text_search_count += 1
    answer = await ctx.baseline_provider(question)
    if not answer:
        ctx.text_search_empty += 1
        # Empty answer is a frequent failure mode of the underlying retriever.
        # Tell the agent exactly what to do next instead of letting it answer
        # blindly from the (possibly noisy) graph alone.
        remaining = 3 - ctx.text_search_count
        next_tool_hint = (
            "search_edges_by_description with a different phrasing"
            if remaining > 0 else "wide_search"
        )
        return (
            "(text_search returned no answer for this question.) "
            "The text retriever sometimes fails silently. "
            f"DO NOT answer yet — try {next_tool_hint}, or rephrase the question "
            "and call text_search again. Only fall back to the graph candidate "
            "if all alternatives are exhausted."
        )
    # Keep the first text_search answer as the "baseline_evidence" surface.
    if ctx.baseline_answer is None:
        ctx.baseline_answer = answer
    return f"text_search returned:\n{answer}"


async def _tool_final_answer(
    ctx: AgentContext, args: dict[str, Any]
) -> str:  # not actually used — handled in the loop
    return str(args.get("answer") or "").strip()


TOOL_DISPATCH: dict[str, Callable[[AgentContext, dict[str, Any]], Awaitable[str]]] = {
    "resolve_entities": _tool_resolve_entities,
    "time_window_search": _tool_time_window_search,
    "wide_search": _tool_wide_search,
    "search_edges_by_description": _tool_search_edges_by_description,
    "text_search": _tool_text_search,
    "final_answer": _tool_final_answer,
}


# ---------------------------------------------------------------------------
# JSON parsing for the LLM's per-turn decision
# ---------------------------------------------------------------------------


class Decision:
    __slots__ = ("kind", "tool", "args", "answer", "raw_error")

    def __init__(
        self,
        kind: str,
        *,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        answer: str | None = None,
        raw_error: str | None = None,
    ) -> None:
        self.kind = kind  # "tool" | "answer" | "error"
        self.tool = tool
        self.args = args or {}
        self.answer = answer
        self.raw_error = raw_error


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of an LLM response."""
    if not text:
        return None
    s = text.strip()
    # Strip code fences if any
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # Direct parse attempt
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Find first { ... } pair
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


# Patterns that indicate the model is refusing instead of answering. Matches
# vanilla-style hedging ("The data does not provide..."), agent denials
# ("did not hold any position", "no record of..."), and graph-empty
# capitulations. The check is conservative — only triggers on explicit
# refusal language, not on legitimately-negative facts ("X never married").
_REFUSAL_RE = re.compile(
    r"(no (record|information|data|specific(ally)?|specified|known|details|"
    r"direct (information|evidence)|mention)|"
    r"not (specified|provided|available|present|directly|in (the|this)|"
    r"explicitly|recorded|mentioned|listed|stated|found)|"
    r"did not (hold|join|belong|play|work|attend|live|specify|appear|have)|"
    r"there is no|information.*missing|cannot be (determined|confirmed|identified)|"
    r"data does not (provide|specify|contain|include|mention))",
    re.IGNORECASE,
)


def _looks_like_refusal(text: str) -> bool:
    """True when the model's answer is hedging instead of committing to an entity."""
    return bool(_REFUSAL_RE.search(text or ""))


def _parse_decision(text: str) -> Decision:
    obj = _extract_json_object(text)
    if obj is None:
        return Decision(
            "error",
            raw_error="Response was not valid JSON. Output exactly one JSON object.",
        )
    # final_answer can also come as a top-level "answer" key
    if "answer" in obj and "tool" not in obj:
        ans = str(obj.get("answer") or "").strip()
        if not ans:
            return Decision("error", raw_error="'answer' was empty.")
        return Decision("answer", answer=ans)
    tool = obj.get("tool")
    if not tool or not isinstance(tool, str):
        return Decision(
            "error",
            raw_error="JSON must include either 'answer' or a string 'tool'.",
        )
    args = obj.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return Decision(
            "error",
            raw_error="'args' must be a JSON object.",
        )
    # The tool name "final_answer" is recognised as terminating the loop.
    if tool == "final_answer":
        ans = str(args.get("answer") or "").strip()
        if not ans:
            return Decision("error", raw_error="final_answer requires non-empty 'answer'.")
        return Decision("answer", answer=ans)
    return Decision("tool", tool=tool, args=args)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def run_temporal_agent(
    query: str,
    ctx: AgentContext,
    max_iters: int = 8,
) -> dict[str, Any]:
    """Drive the LLM through tool-by-tool retrieval until it answers.

    Returns a dict with:
      - answer:           the final text the LLM emitted
      - iterations:       number of LLM decision turns used
      - trace:            list of {step, tool, args, result_preview}
      - edges_collected:  list of unique edges accumulated across tools
      - baseline_answer:  the first text_search result if invoked
    """
    from graphrag_llm.utils import CompletionMessagesBuilder

    system = TEMPORAL_AGENT_SYSTEM_PROMPT.format(
        current_date=ctx.query_time.date().isoformat(),
        max_iters=max_iters,
    )
    user_first = TEMPORAL_AGENT_USER_PROMPT.format(query=query)

    builder = (
        CompletionMessagesBuilder()
        .add_system_message(system)
        .add_user_message(user_first)
    )

    trace: list[dict[str, Any]] = []
    final_answer: str | None = None
    used_iters = 0
    pushback_done = False  # only allow ONE anti-refusal pushback per question

    for step in range(max_iters):
        used_iters = step + 1
        try:
            response = await ctx.model.completion_async(messages=builder.build())
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent LLM call failed at step %d: %s", step, exc)
            break
        content = (getattr(response, "content", "") or "").strip()
        decision = _parse_decision(content)

        if decision.kind == "answer":
            # Anti-refusal guard: if the model is hedging instead of committing
            # to an entity, push back ONCE with a strict re-prompt that demands
            # a concrete answer synthesised from all evidence gathered so far.
            # The check only fires when iterations remain and we haven't
            # pushed back yet — we don't want to loop indefinitely.
            if (
                not pushback_done
                and step < max_iters - 1
                and _looks_like_refusal(decision.answer)
            ):
                pushback_done = True
                builder.add_assistant_message(content)
                builder.add_user_message(
                    "REJECTED. You may NOT refuse or hedge ('no data', "
                    "'not specified', 'did not hold', 'cannot determine', etc.). "
                    "Re-read EVERY tool result above — the entities you "
                    "resolved, the edges from time_window_search and "
                    "search_edges_by_description, and the text_search "
                    "responses — and commit to the SINGLE most likely entity "
                    "for the question. If the time window matches multiple "
                    "candidates, pick the one with the strongest evidence. "
                    "If text_search hedged but graph edges named an entity, "
                    "use the graph entity. If the graph has no good match "
                    "but text_search mentioned any entity at all (even with "
                    "uncertain framing), use that. Output JSON now: "
                    '{"answer": "<single entity, one short sentence>"}'
                )
                trace.append({
                    "step": step,
                    "kind": "anti_refusal_pushback",
                    "refused_answer": decision.answer[:200],
                })
                continue

            final_answer = decision.answer
            trace.append({"step": step, "kind": "answer", "answer": decision.answer})
            break

        if decision.kind == "error":
            # Show the malformed output back and ask for a corrected JSON.
            builder.add_assistant_message(content)
            builder.add_user_message(
                f"Your previous response was invalid: {decision.raw_error}\n"
                "Reply with exactly one JSON object: "
                '{"tool": "...", "args": {...}} or {"answer": "..."}.'
            )
            trace.append({"step": step, "kind": "parse_error", "raw": content[:300]})
            continue

        # Tool execution path
        tool_name = decision.tool or ""
        tool_args = decision.args or {}
        tool_fn = TOOL_DISPATCH.get(tool_name)
        if tool_fn is None:
            builder.add_assistant_message(content)
            builder.add_user_message(
                f"Unknown tool '{tool_name}'. Available tools: "
                f"{', '.join(t for t in TOOL_DISPATCH if t != 'final_answer')}. "
                "Pick one of those or return a final_answer."
            )
            trace.append(
                {"step": step, "kind": "unknown_tool", "tool": tool_name}
            )
            continue

        try:
            result = await tool_fn(ctx, tool_args)
        except Exception as exc:  # noqa: BLE001
            result = f"Tool error: {exc}"
            logger.debug("tool %s failed: %s", tool_name, exc)
        result = _truncate(result, _TOOL_RESULT_TRUNCATE)
        builder.add_assistant_message(content)
        builder.add_user_message(f"{result}\n\nNext JSON action?")
        trace.append(
            {
                "step": step,
                "kind": "tool",
                "tool": tool_name,
                "args": tool_args,
                "result_preview": result[:400],
            }
        )

    if final_answer is None:
        # Out of iterations — force a final answer with whatever was gathered.
        builder.add_user_message(TEMPORAL_AGENT_FORCE_ANSWER_PROMPT)
        try:
            response = await ctx.model.completion_async(messages=builder.build())
            content = (getattr(response, "content", "") or "").strip()
            decision = _parse_decision(content)
            if decision.kind == "answer":
                final_answer = decision.answer
            else:
                final_answer = content  # take the raw text as best-effort
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent force-answer failed: %s", exc)
            final_answer = ""
        trace.append({"step": used_iters, "kind": "forced_answer", "answer": final_answer})

    # Dedupe collected edges by id (fall back to tuple shape)
    seen: set[str] = set()
    unique_edges: list[dict[str, Any]] = []
    for e in ctx.collected_edges:
        key = e.get("id") or (
            e.get("source"), e.get("relation_type"), e.get("target"),
            e.get("t_valid_start"), e.get("t_valid_end"),
        )
        k = str(key)
        if k in seen:
            continue
        seen.add(k)
        unique_edges.append(e)

    return {
        "answer": final_answer or "",
        "iterations": used_iters,
        "trace": trace,
        "edges_collected": unique_edges,
        "baseline_answer": ctx.baseline_answer,
        "text_search_count": ctx.text_search_count,
        "text_search_empty": ctx.text_search_empty,
    }
