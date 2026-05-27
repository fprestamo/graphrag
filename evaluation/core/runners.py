"""Runners for the two systems under test (BT-GraphRAG and vanilla GraphRAG).

Both expose the same interface:

    async def <system>_answer(question: str, *, config=...) -> {"answer": str, "raw": dict}

They never raise: any failure is logged and returned in ``raw['error']``.
"""

from __future__ import annotations

import asyncio
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
    # Optional shared Neo4j driver. When provided, bt_answer reuses its
    # single connection pool instead of creating a new driver per call —
    # critical for high-concurrency evaluation (a per-call driver at
    # concurrency=100 opens 100 separate pools and the server starts
    # killing connections).
    driver: Any = None


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
        from graphrag.bt_graphrag.temporal_query import temporal_query_pipeline
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
    shared_driver = cfg.driver
    if shared_driver is not None:
        driver = shared_driver
        owns_driver = False
    else:
        driver = AsyncGraphDatabase.driver(
            cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
        )
        owns_driver = True

    result: dict[str, Any] | None = None
    error: str | None = None
    try:
        result = await temporal_query_pipeline(
            query=question,
            config=bt_cfg,
            driver=driver,
            model=llm,
            max_edges_per_sub=cfg.max_edges,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("temporal_query_pipeline failed: %s", exc)
        error = f"search: {exc}"
    finally:
        if owns_driver:
            try:
                await driver.close()
            except Exception:  # noqa: BLE001
                pass

    if error is not None or result is None:
        return {"answer": "", "raw": {"error": error or "search: no result"}}

    answer = result.get("answer", "")
    sub_results = result.get("sub_queries", []) or []
    edges_used = result.get("edges_used", 0)

    raw: dict[str, Any] = {
        "edges_used": edges_used,
        "query_time": result.get("query_time"),
        "num_sub_queries": len(sub_results),
    }
    analysis = result.get("analysis") or {}
    if analysis:
        raw["entities_extracted"] = analysis.get("entities", [])
        # Drop the heavy ``edges`` payload from each sub-query before
        # serialising — keep only the structured fields the LLM saw.
        raw["sub_queries"] = [
            {
                **(s.get("sub_query") or {}),
                "edges_matched": len(s.get("edges", [])),
                "seeds_resolved": [
                    {"title": e.get("title"), "type": e.get("type")}
                    for e in (s.get("seeds") or [])
                ],
            }
            for s in sub_results
        ]
    if llm is None:
        raw["error"] = "no LLM (set BTG_MODEL_ID)"
        # Surface the retrieved context for inspection.
        context_blocks = []
        for sub in sub_results:
            context_blocks.append(_format_edges(sub.get("edges", []), cfg.max_edges))
        raw["context"] = "\n\n".join(context_blocks) if context_blocks else ""

    return {"answer": answer, "raw": raw}


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
# Key: (str(abs root_dir), str(abs data_dir or "")).
_VANILLA_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
# Serialize concurrent loads: graphrag.config.load_config does os.chdir() and
# resolves paths relative to CWD, so two concurrent loads from a relative root
# race on the chdir and the second sees ".ragtest/.ragtest".
_VANILLA_LOAD_LOCK = asyncio.Lock()
# Snapshot the CWD at import time. Tasks that resolve a relative root_dir
# AFTER load_config's internal chdir would otherwise build paths like
# ".ragtest/.ragtest"; resolving against this base instead keeps every call
# in agreement on the absolute path, regardless of what load_config did to CWD.
_VANILLA_BASE_CWD = Path.cwd()


def _resolve_against_base(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = _VANILLA_BASE_CWD / path
    return path.resolve()


async def _load_vanilla_state(cfg: VanillaConfig) -> dict[str, Any]:
    # Resolve against the snapshotted base CWD, not the live one — see
    # _VANILLA_BASE_CWD above. Using Path(...).resolve() directly here would
    # double-prefix the root after load_config's chdir, even with the lock.
    root_abs = _resolve_against_base(cfg.root_dir)
    data_abs = _resolve_against_base(cfg.data_dir) if cfg.data_dir else None
    key = (str(root_abs), str(data_abs or ""))
    if key in _VANILLA_CACHE:
        return _VANILLA_CACHE[key]

    async with _VANILLA_LOAD_LOCK:
        if key in _VANILLA_CACHE:
            return _VANILLA_CACHE[key]

        from graphrag.config.load_config import load_config
        from graphrag.data_model.data_reader import DataReader
        from graphrag_storage import create_storage
        from graphrag_storage.tables.table_provider_factory import create_table_provider

        overrides: dict[str, Any] = {}
        if data_abs is not None:
            overrides["output_storage"] = {"base_dir": str(data_abs)}
        saved_cwd = Path.cwd()
        try:
            graphrag_config = load_config(root_dir=root_abs, cli_overrides=overrides)
        finally:
            os.chdir(saved_cwd)

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
