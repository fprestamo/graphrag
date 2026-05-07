# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Thin async wrapper around BT-GraphRAG used by every benchmark.

The runner exposes a single coroutine, :func:`answer_question`, that:

1. Decomposes the (optionally temporal) query through BT-GraphRAG.
2. Runs Local Temporal Search to gather temporally-valid edges.
3. Synthesises a natural-language answer with the project's LLM client using
   :data:`graphrag.bt_graphrag.prompts.TEMPORAL_ANSWER_SYNTHESIS_PROMPT`.

The wrapper is *intentionally* loose: if Neo4j or the LLM clients are not
configured (e.g. when running ``--dry-run`` for CI smoke tests) it returns a
deterministic placeholder so that the dataset / metric plumbing can still be
exercised end-to-end without infrastructure.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Runtime configuration for the BT-GraphRAG evaluation runner."""

    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    )
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "12345678")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "btgraphrag")
    )

    model_id: str | None = field(default_factory=lambda: os.getenv("BTG_MODEL_ID"))
    """LLM model identifier (resolved through the project's LLM factory)."""

    max_edges: int = 50
    """Maximum number of edges fed into the answer-synthesis prompt."""

    dry_run: bool = False
    """If true, return a stub answer instead of contacting Neo4j / the LLM."""


# ---------------------------------------------------------------------------
# Lazy imports — keep the evaluation package import-light.
# ---------------------------------------------------------------------------


def _build_btgraphrag_config(cfg: RunnerConfig):
    from graphrag.bt_graphrag.models.config import BTGraphRAGConfig

    return BTGraphRAGConfig(
        enabled=True,
        neo4j_uri=cfg.neo4j_uri,
        neo4j_user=cfg.neo4j_user,
        neo4j_password=cfg.neo4j_password,
        neo4j_database=cfg.neo4j_database,
    )


async def _open_driver(cfg: RunnerConfig):
    from neo4j import AsyncGraphDatabase

    return AsyncGraphDatabase.driver(
        cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
    )


async def _build_llm(cfg: RunnerConfig):
    """Best-effort construction of an LLM completion client.

    Returns ``None`` if the project's LLM factory is not configured – the
    caller will then fall back to an extractive answer derived from the
    retrieved edges.
    """
    if cfg.dry_run:
        return None
    try:
        from graphrag_llm.factory import ModelFactory  # type: ignore

        if cfg.model_id and ModelFactory.is_supported_model(cfg.model_id):
            return ModelFactory.create_chat_model(cfg.model_id)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Could not initialise LLM (%s); falling back to extractive answer.", exc)
    return None


# ---------------------------------------------------------------------------
# Edge → context formatting
# ---------------------------------------------------------------------------


def _format_edges(edges: list[dict[str, Any]], limit: int) -> str:
    """Render retrieved edges as a compact textual context block."""
    lines: list[str] = []
    for e in edges[:limit]:
        src = e.get("source", "?")
        rel = e.get("relation_type") or e.get("type") or "?"
        tgt = e.get("target", "?")
        desc = e.get("description") or ""
        valid = e.get("t_valid_start")
        line = f"({src}) -[{rel}]-> ({tgt})"
        if valid:
            line += f"  [valid_from={valid}]"
        if desc:
            line += f"  :: {desc}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no temporal edges retrieved)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def answer_question(
    question: str,
    *,
    config: RunnerConfig | None = None,
    question_time: datetime | None = None,
) -> dict[str, Any]:
    """Answer a benchmark question with BT-GraphRAG.

    Returns a dict with at least ``answer`` (string) and ``raw`` (debug info).
    Never raises: connection / LLM problems are logged and surface as an
    ``error`` key in ``raw``.
    """
    cfg = config or RunnerConfig()

    if cfg.dry_run:
        return {
            "answer": "",
            "raw": {"dry_run": True, "question": question},
        }

    bt_cfg = _build_btgraphrag_config(cfg)
    driver = None
    try:
        driver = await _open_driver(cfg)
        from graphrag.bt_graphrag.temporal_query import local_temporal_search

        retrieval = await local_temporal_search(
            query=question,
            query_time=question_time,
            config=bt_cfg,
            driver=driver,
        )
        edges = retrieval.get("edges", []) or []
        context = _format_edges(edges, cfg.max_edges)

        llm = await _build_llm(cfg)
        if llm is None:
            # Fallback: emit the top edge description (good enough for smoke tests)
            answer = edges[0].get("description", "") if edges else ""
            return {
                "answer": answer,
                "raw": {
                    "retrieved_edges": len(edges),
                    "fallback": True,
                    "context": context,
                },
            }

        from graphrag.bt_graphrag.prompts import TEMPORAL_ANSWER_SYNTHESIS_PROMPT
        from graphrag_llm.utils import CompletionMessagesBuilder  # type: ignore

        prompt = TEMPORAL_ANSWER_SYNTHESIS_PROMPT.format(
            query=question,
            context=context,
            query_time=(question_time or datetime.now(timezone.utc)).isoformat(),
        )
        messages = CompletionMessagesBuilder().add_user_message(prompt).build()
        response = await llm.completion_async(messages=messages)
        return {
            "answer": (response.content or "").strip(),
            "raw": {
                "retrieved_edges": len(edges),
                "context": context,
            },
        }
    except Exception as exc:  # pragma: no cover - env dependent
        logger.exception("BT-GraphRAG runner failed for question %r", question)
        return {"answer": "", "raw": {"error": str(exc)}}
    finally:
        if driver is not None:
            try:
                await driver.close()
            except Exception:  # pragma: no cover
                pass
