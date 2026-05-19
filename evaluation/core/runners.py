"""Runners for the two systems under test (BT-GraphRAG and vanilla GraphRAG).

Both expose the same interface:

    async def <system>_answer(question: str, *, config=...) -> {"answer": str, "raw": dict}

They never raise: any failure is logged and returned in ``raw['error']``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BT-GraphRAG runner
# ---------------------------------------------------------------------------


@dataclass
class BTConfig:
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
    max_edges: int = 50
    dry_run: bool = False


def _format_edges(edges: list[dict[str, Any]], limit: int) -> str:
    lines: list[str] = []
    for e in edges[:limit]:
        src = e.get("source") or e.get("source_title") or "?"
        tgt = e.get("target") or e.get("target_title") or "?"
        rel = e.get("relation_type") or e.get("type") or "RELATED_TO"
        desc = e.get("description") or ""
        tstart = e.get("t_valid_start") or ""
        tend = e.get("t_valid_end") or ""
        period = f" [{tstart} → {tend}]" if (tstart or tend) else ""
        line = f"- ({src}) -[{rel}]-> ({tgt}){period}: {desc}".strip()
        lines.append(line)
    return "\n".join(lines) if lines else "(no relevant temporal edges found)"


def _build_bt_llm(model_id: str | None):
    if not model_id:
        return None
    try:
        from graphrag_llm.completion import create_completion
        from graphrag_llm.config import ModelConfig
        from graphrag_llm.config.types import LLMProviderType
    except ImportError as exc:
        logger.debug("graphrag_llm not importable: %s", exc)
        return None
    api_key = os.getenv("GRAPHRAG_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    try:
        cfg = ModelConfig(
            type=LLMProviderType.LiteLLM,
            model_provider=os.getenv("BTG_MODEL_PROVIDER", "openai"),
            model=model_id,
            api_key=api_key,
        )
        return create_completion(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not build BT LLM: %s", exc)
        return None


async def bt_answer(question: str, *, config: BTConfig | None = None) -> dict[str, Any]:
    cfg = config or BTConfig()
    if cfg.dry_run:
        return {"answer": "", "raw": {"dry_run": True}}

    try:
        from neo4j import AsyncGraphDatabase

        from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
        from graphrag.bt_graphrag.prompts import TEMPORAL_ANSWER_SYNTHESIS_PROMPT
        from graphrag.bt_graphrag.temporal_query import local_temporal_search
    except ImportError as exc:
        logger.warning("BT-GraphRAG dependencies missing: %s", exc)
        return {"answer": "", "raw": {"error": f"import: {exc}"}}

    bt_cfg = BTGraphRAGConfig(
        neo4j_uri=cfg.neo4j_uri,
        neo4j_user=cfg.neo4j_user,
        neo4j_password=cfg.neo4j_password,
        neo4j_database=cfg.neo4j_database,
    )

    llm = _build_bt_llm(cfg.model_id)
    driver = AsyncGraphDatabase.driver(
        cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
    )
    try:
        search = await local_temporal_search(
            query=question,
            query_time=None,
            config=bt_cfg,
            driver=driver,
            model=llm,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("local_temporal_search failed: %s", exc)
        try:
            await driver.close()
        except Exception:  # noqa: BLE001
            pass
        return {"answer": "", "raw": {"error": f"search: {exc}"}}

    edges = search.get("edges", []) or []
    context_text = _format_edges(edges, cfg.max_edges)

    if llm is None:
        await driver.close()
        return {
            "answer": "",
            "raw": {
                "error": "no LLM (set BTG_MODEL_ID)",
                "edges_found": len(edges),
                "context": context_text,
            },
        }

    try:
        from graphrag_llm.utils import CompletionMessagesBuilder

        prompt = TEMPORAL_ANSWER_SYNTHESIS_PROMPT.format(
            query=question,
            sub_results=context_text,
        )
        messages = CompletionMessagesBuilder().add_user_message(prompt).build()
        response = await llm.completion_async(messages=messages)
        answer = (getattr(response, "content", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Answer synthesis failed: %s", exc)
        answer = ""
        raw_extra = {"synthesis_error": str(exc)}
    else:
        raw_extra = {}
    finally:
        try:
            await driver.close()
        except Exception:  # noqa: BLE001
            pass

    return {
        "answer": answer,
        "raw": {
            "edges_found": len(edges),
            "edges_used": min(len(edges), cfg.max_edges),
            "query_time": search.get("query_time"),
            **raw_extra,
        },
    }


# ---------------------------------------------------------------------------
# Vanilla GraphRAG runner
# ---------------------------------------------------------------------------


@dataclass
class VanillaConfig:
    root_dir: Path = field(
        default_factory=lambda: Path(os.getenv("GRAPHRAG_ROOT", ".ragtest"))
    )
    data_dir: Path | None = None
    search_mode: str = field(
        default_factory=lambda: os.getenv("GRAPHRAG_SEARCH", "local")
    )
    community_level: int = 2
    response_type: str = "Single sentence"
    dry_run: bool = False


# Module-level cache: parquet reads are expensive; we want them once per process.
# Key: (str(root_dir), str(data_dir or "")).
_VANILLA_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


async def _load_vanilla_state(cfg: VanillaConfig) -> dict[str, Any]:
    key = (str(cfg.root_dir), str(cfg.data_dir or ""))
    if key in _VANILLA_CACHE:
        return _VANILLA_CACHE[key]

    from graphrag.config.load_config import load_config
    from graphrag.data_model.data_reader import DataReader
    from graphrag_storage import create_storage
    from graphrag_storage.tables.table_provider_factory import create_table_provider

    overrides: dict[str, Any] = {}
    if cfg.data_dir:
        overrides["output_storage"] = {"base_dir": str(cfg.data_dir)}
    graphrag_config = load_config(root_dir=cfg.root_dir, cli_overrides=overrides)

    storage_obj = create_storage(graphrag_config.output_storage)
    table_provider = create_table_provider(
        graphrag_config.table_provider, storage=storage_obj
    )
    reader = DataReader(table_provider)

    state: dict[str, Any] = {"config": graphrag_config}
    needed = ["entities", "communities", "community_reports"]
    if cfg.search_mode == "local":
        needed += ["text_units", "relationships"]
    for name in needed:
        state[name] = await getattr(reader, name)()
    state["covariates"] = (
        await reader.covariates() if await table_provider.has("covariates") else None
    )

    _VANILLA_CACHE[key] = state
    return state


async def vanilla_answer(
    question: str, *, config: VanillaConfig | None = None
) -> dict[str, Any]:
    cfg = config or VanillaConfig()
    if cfg.dry_run:
        return {"answer": "", "raw": {"dry_run": True}}

    try:
        import graphrag.api as api
    except ImportError as exc:
        logger.warning("graphrag.api not importable: %s", exc)
        return {"answer": "", "raw": {"error": f"import: {exc}"}}

    try:
        state = await _load_vanilla_state(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load GraphRAG index from %s: %s", cfg.root_dir, exc)
        return {"answer": "", "raw": {"error": f"load: {exc}"}}

    try:
        if cfg.search_mode == "global":
            response, context = await api.global_search(
                config=state["config"],
                entities=state["entities"],
                communities=state["communities"],
                community_reports=state["community_reports"],
                community_level=cfg.community_level,
                dynamic_community_selection=False,
                response_type=cfg.response_type,
                query=question,
            )
        else:
            response, context = await api.local_search(
                config=state["config"],
                entities=state["entities"],
                communities=state["communities"],
                community_reports=state["community_reports"],
                text_units=state["text_units"],
                relationships=state["relationships"],
                covariates=state.get("covariates"),
                community_level=cfg.community_level,
                response_type=cfg.response_type,
                query=question,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vanilla search (%s) failed: %s", cfg.search_mode, exc)
        return {"answer": "", "raw": {"error": f"search: {exc}"}}

    answer = response if isinstance(response, str) else str(response)
    return {
        "answer": answer.strip(),
        "raw": {
            "search_mode": cfg.search_mode,
            "has_context": bool(context),
        },
    }
