# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Community-aware retrieval for the BT-GraphRAG temporal query pipeline.

This module mirrors the *primer* phase of GraphRAG's DRIFT search: it
selects a small set of community reports that provide global context
for the user's question and injects them alongside the edge-level
evidence that the local subgraph retrieval already produces.

The reports themselves live on disk as the
``.../community_reports.parquet`` artefact written during indexing;
the entity → community mapping comes from ``communities.parquet`` and
the entity catalogue from ``entities.parquet``. None of these are in
Neo4j today, so we load them once into a process-wide
:class:`CommunityStore` cache and reuse it across queries.

Two selection signals are combined:

1. **Structural** — given the resolved seed titles, look up which
   communities contain those entities and fetch their reports. This is
   the cheap, high-precision path that mirrors GraphRAG's local search
   "communities for these entities" lookup.
2. **Semantic** — embed the question and rank community summaries by
   cosine similarity against precomputed summary embeddings. This is
   the DRIFT-style primer path that catches questions where the
   structural lookup is empty or weak (no seeds resolved, off-topic
   seeds, etc.).

Both signals are unioned, deduped by community id, and capped at
``top_k`` reports ordered structural-first then by cosine score.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from graphrag_llm.embedding import LLMEmbedding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-wide cache
# ---------------------------------------------------------------------------


@dataclass
class CommunityStore:
    """In-memory index of community reports and their entity membership.

    ``title_to_communities`` maps an entity *title* (uppercased to match
    the graph convention) to the list of community ids containing it.
    ``report_by_community`` maps a community id to a dict with the
    report metadata (title, summary, full_content, rank, level, size).
    ``summary_embeddings`` is a parallel ``(community_ids, matrix)``
    pair lazily built on first semantic call.
    """

    title_to_communities: dict[str, list[int]] = field(default_factory=dict)
    report_by_community: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Lazy-built on first semantic lookup.
    summary_community_ids: list[int] | None = None
    summary_matrix: np.ndarray | None = None
    _embed_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def num_reports(self) -> int:
        return len(self.report_by_community)


_STORE_CACHE: dict[str, CommunityStore] = {}


def _ragtest_output_dir() -> Path:
    """Resolve the parquet output directory.

    Order of precedence:
      1. ``BT_COMMUNITY_REPORTS_DIR`` env var.
      2. ``GRAPHRAG_ROOT`` env var + ``output``.
      3. ``./.ragtest/output``.
    """
    env_dir = os.getenv("BT_COMMUNITY_REPORTS_DIR")
    if env_dir:
        return Path(env_dir)
    root = os.getenv("GRAPHRAG_ROOT")
    if root:
        return Path(root) / "output"
    return Path(".ragtest") / "output"


def load_community_store(
    output_dir: Path | None = None,
    *,
    refresh: bool = False,
) -> CommunityStore | None:
    """Load (or fetch from cache) the community store for a project.

    Returns ``None`` if the parquet artefacts are missing — callers
    should treat that as "communities unavailable" and skip the
    community-aware retrieval branch.
    """
    out = output_dir or _ragtest_output_dir()
    key = str(out.resolve())
    if not refresh and key in _STORE_CACHE:
        return _STORE_CACHE[key]

    reports_path = out / "community_reports.parquet"
    communities_path = out / "communities.parquet"
    entities_path = out / "entities.parquet"
    if not reports_path.exists() or not communities_path.exists():
        logger.info(
            "Community store unavailable: %s or %s missing",
            reports_path, communities_path,
        )
        return None

    reports_df = pd.read_parquet(reports_path)
    communities_df = pd.read_parquet(communities_path)

    # Build report lookup keyed by community id.
    report_by_community: dict[int, dict[str, Any]] = {}
    for row in reports_df.itertuples(index=False):
        cid = getattr(row, "community", None)
        if cid is None:
            continue
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        report_by_community[cid_int] = {
            "community": cid_int,
            "level": getattr(row, "level", None),
            "title": getattr(row, "title", "") or "",
            "summary": getattr(row, "summary", "") or "",
            "full_content": getattr(row, "full_content", "") or "",
            "rank": getattr(row, "rank", None),
            "period": getattr(row, "period", None),
            "size": getattr(row, "size", None),
        }

    # Build the entity → community mapping. ``communities_df.entity_ids``
    # contains UUIDs; we need the human-readable *title* to match the
    # seeds resolved against Neo4j. Look those up from entities.parquet.
    title_to_communities: dict[str, list[int]] = {}
    if entities_path.exists():
        entities_df = pd.read_parquet(entities_path, columns=["id", "title"])
        id_to_title: dict[str, str] = dict(
            zip(entities_df["id"].astype(str), entities_df["title"].astype(str))
        )

        for row in communities_df.itertuples(index=False):
            cid = getattr(row, "community", None)
            ent_ids = getattr(row, "entity_ids", None)
            if cid is None or ent_ids is None:
                continue
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                continue
            if cid_int not in report_by_community:
                continue
            try:
                ent_iter = list(ent_ids)
            except TypeError:
                continue
            for eid in ent_iter:
                title = id_to_title.get(str(eid))
                if not title:
                    continue
                title_to_communities.setdefault(title, []).append(cid_int)

    store = CommunityStore(
        title_to_communities=title_to_communities,
        report_by_community=report_by_community,
    )
    _STORE_CACHE[key] = store
    logger.info(
        "Community store loaded: %d reports, %d titles mapped, dir=%s",
        store.num_reports, len(store.title_to_communities), out,
    )
    return store


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _cosine_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between ``query_vec`` and ``matrix``."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    norms = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ q) / norms


async def _ensure_summary_embeddings(
    store: CommunityStore,
    embedding_model: "LLMEmbedding",
) -> None:
    """Embed every community summary once and cache the matrix."""
    if store.summary_matrix is not None:
        return
    async with store._embed_lock:
        if store.summary_matrix is not None:
            return
        community_ids = sorted(store.report_by_community.keys())
        texts: list[str] = []
        for cid in community_ids:
            r = store.report_by_community[cid]
            # Concatenate title + summary; full_content tends to be
            # repetitive and inflates the embedding cost.
            t = (r.get("title") or "").strip()
            s = (r.get("summary") or "").strip()
            blob = f"{t}\n\n{s}" if t else s
            texts.append(blob or " ")
        # Embed in one batched call. Reuse the same helper the indexing
        # pipeline uses to stay under OpenAI's per-request caps.
        from graphrag.bt_graphrag.temporal_extraction.embedding_enrichment import (
            _embed_in_batches,
        )

        vectors = await _embed_in_batches(texts, embedding_model)
        # Drop community ids whose embedding came back empty (shouldn't
        # happen since we substituted blanks with " ", but be defensive).
        kept_ids: list[int] = []
        kept_vecs: list[list[float]] = []
        for cid, vec in zip(community_ids, vectors):
            if vec:
                kept_ids.append(cid)
                kept_vecs.append(vec)
        store.summary_community_ids = kept_ids
        store.summary_matrix = (
            np.asarray(kept_vecs, dtype=np.float32) if kept_vecs else None
        )
        logger.info(
            "Community store: embedded %d / %d summaries",
            len(kept_ids), len(community_ids),
        )


async def select_relevant_communities(
    *,
    store: CommunityStore | None,
    question: str,
    seed_titles: list[str],
    embedding_model: "LLMEmbedding | None" = None,
    top_k: int = 3,
    score_threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """Pick up to ``top_k`` community reports relevant to a question.

    Combines a structural lookup (communities containing any seed) with
    an optional semantic lookup (question-to-summary cosine). Returns
    each report enriched with ``_selection_score`` and
    ``_selection_source`` (``"structural"``, ``"semantic"``, or
    ``"both"``) for transparency.
    """
    if store is None or store.num_reports == 0 or top_k <= 0:
        return []

    # 1. Structural — communities containing any seed.
    structural_ids: dict[int, float] = {}
    if seed_titles:
        for title in seed_titles:
            if not title:
                continue
            for cid in store.title_to_communities.get(title, []):
                # Smaller communities (fewer entities) are usually more
                # focused; penalise huge ones slightly to break ties.
                size = store.report_by_community.get(cid, {}).get("size") or 1
                score = 1.0 / (1.0 + max(int(size or 1) - 1, 0) / 50.0)
                if cid not in structural_ids or structural_ids[cid] < score:
                    structural_ids[cid] = score

    # 2. Semantic — top-K reports by question-to-summary cosine.
    semantic_ids: dict[int, float] = {}
    if embedding_model is not None and question.strip():
        await _ensure_summary_embeddings(store, embedding_model)
        if store.summary_matrix is not None and store.summary_community_ids:
            try:
                resp = await embedding_model.embedding_async(input=[question])
                q_vec = list(getattr(resp, "embeddings", []) or [[]])[0]
            except Exception as exc:  # noqa: BLE001
                logger.debug("Community semantic embedding failed: %s", exc)
                q_vec = []
            if q_vec:
                sims = _cosine_matrix(
                    np.asarray(q_vec, dtype=np.float32), store.summary_matrix,
                )
                # Pick a generous over-fetch so the post-threshold
                # ranking has enough survivors.
                n_fetch = min(len(sims), max(top_k * 5, 10))
                top_idx = np.argpartition(-sims, n_fetch - 1)[:n_fetch]
                for idx in top_idx:
                    score = float(sims[idx])
                    if score < score_threshold:
                        continue
                    cid = store.summary_community_ids[idx]
                    semantic_ids[cid] = score

    if not structural_ids and not semantic_ids:
        return []

    # Merge: structural first (precision), then fill with semantic.
    chosen: list[dict[str, Any]] = []
    seen: set[int] = set()
    for cid, score in sorted(structural_ids.items(), key=lambda x: -x[1]):
        if cid in seen or len(chosen) >= top_k:
            break
        rep = dict(store.report_by_community[cid])
        rep["_selection_source"] = (
            "both" if cid in semantic_ids else "structural"
        )
        rep["_selection_score"] = max(score, semantic_ids.get(cid, 0.0))
        chosen.append(rep)
        seen.add(cid)
    for cid, score in sorted(semantic_ids.items(), key=lambda x: -x[1]):
        if cid in seen or len(chosen) >= top_k:
            break
        rep = dict(store.report_by_community[cid])
        rep["_selection_source"] = "semantic"
        rep["_selection_score"] = score
        chosen.append(rep)
        seen.add(cid)

    return chosen


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_community_reports_for_llm(
    reports: list[dict[str, Any]],
    *,
    max_chars_per_report: int = 1500,
) -> str:
    """Render the selected community reports for the synthesis prompt."""
    if not reports:
        return ""
    lines: list[str] = []
    for i, r in enumerate(reports, 1):
        title = (r.get("title") or "(untitled community)").strip()
        body = (r.get("summary") or r.get("full_content") or "").strip()
        if len(body) > max_chars_per_report:
            body = body[: max_chars_per_report - 1].rstrip() + "…"
        source = r.get("_selection_source", "?")
        score = r.get("_selection_score")
        try:
            score_str = f"{float(score):.2f}" if score is not None else "?"
        except (TypeError, ValueError):
            score_str = "?"
        lines.append(
            f"[Community {i} — source={source}, score={score_str}] {title}\n{body}"
        )
    return "\n\n".join(lines)
