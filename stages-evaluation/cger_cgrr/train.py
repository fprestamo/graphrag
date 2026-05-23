# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Threshold search: tune CGER and CGRR cosine thresholds on the train split.

Loads the JSON files written by ``extract.py --split train``:

    data/train-extracted/entities.json
    data/train-extracted/relationships.json

…together with the manually authored train ground truth:

    data/train-ground-true/entity_resolution.json
    data/train-ground-true/relationship_resolution.json

Searches independently for the best ``cger_cosine_threshold`` and
``cgrr_cosine_threshold`` using simulated annealing (the two stages
don't share state, so they're optimized in two separate SA runs).
All other knobs (``CGER_PHASE_B_TOP_K``, ``CGER_CANDIDATE_TOP_K``,
``CGRR_CANDIDATE_TOP_K``, ``NEO4J_VECTOR_DIMENSIONS``,
``COMPLETION_MODEL``) are held at the same fixed values used by
``evaluate.py``.

The SA objective is ``F1 - lambda * (llm_calls / max_llm_calls)``,
which discourages threshold->0 (where every candidate is verified
by the LLM and F1 is trivially perfect but the run is expensive).
``max_llm_calls`` is a loose upper bound on Phase-B LLM calls
(num entities for CGER, num unique relation types for CGRR).

Writes the full search trace + best thresholds to:

    data/train-results.json

Usage:
    python stages-evaluation/cger_cgrr/train.py

Optional flags:
    --stages cger,cgrr         # which stages to tune (default: both)
    --iters 10                 # SA iterations per stage (default: 10)
    --seed 42                  # RNG seed for SA (default: 42)
    --lambda-llm 0.3           # LLM-usage penalty weight (default: 0.3)
    --grid-points 5            # pre-scan grid size per stage (default: 5)

Requirements:
    - ``extract.py --split train`` has been run first.
    - Train ground truth has been authored at data/train-ground-true/.
    - ``GRAPHRAG_API_KEY`` (or ``OPENAI_API_KEY``) defined in the env
      or in the project-root ``.env`` file.
    - Neo4j running at neo4j://127.0.0.1:7687 with the same two
      databases ``evaluate.py`` requires (``cgrreval``, ``cgerbatch``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Awaitable, Callable

import pandas as pd

from graphrag.bt_graphrag.entity_resolution.cger import resolve_entities
from graphrag.bt_graphrag.entity_resolution.cgrr import resolve_relationships
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig
from graphrag_llm.config.types import LLMProviderType


# ---------------------------------------------------------------------------
# Configuration  (held fixed across the search)
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
TRAIN_EXTRACTED_DIR = HERE / "data" / "train-extracted"
TRAIN_GROUND_TRUTH_DIR = HERE / "data" / "train-ground-true"
RESULTS_PATH = HERE / "data" / "train-results.json"
ENV_PATH = PROJECT_ROOT / ".env"

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
TEST_DB = "cgrreval"
CGER_TEMP_DB = "cgerbatch"

COMPLETION_MODEL = "gpt-4.1-mini"

CGER_CANDIDATE_TOP_K = 5
CGER_PHASE_B_TOP_K = 5
CGRR_CANDIDATE_TOP_K = 5
NEO4J_VECTOR_DIMENSIONS = 3072

# ---------------------------------------------------------------------------
# Simulated-annealing knobs
# ---------------------------------------------------------------------------

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.95
INITIAL_THRESHOLD = 0.75   # SA start point
INITIAL_TEMPERATURE = 0.10  # in units of F1 (objective is in [0, 1])
COOLING_ALPHA = 0.85        # T <- T * alpha per accepted/rejected step
NEIGHBOR_RADIUS_INIT = 0.15  # max neighborhood jump at T = T_0
NEIGHBOR_RADIUS_MIN = 0.01   # neighborhood never shrinks below this
ROUND_TO = 3                 # cache key precision for F1 memoization

# Weight of the LLM-usage penalty in the SA objective:
#   score = F1 - LLM_PENALTY_LAMBDA * (llm_calls / max_llm_calls)
# Overridable via --lambda-llm. Higher = stronger pressure away from
# low thresholds (which would otherwise trivially maximize F1 by
# routing every candidate through the LLM).
LLM_PENALTY_LAMBDA = 0.3

# Number of thresholds in the coarse pre-scan that precedes SA. Spread
# uniformly across [THRESHOLD_MIN, THRESHOLD_MAX]; the best one seeds
# SA's starting point. Guards against degenerate local optima at the
# bounds (e.g. CGRR converging to high thresholds where F1=0 and
# llm_calls=0, so score=0 looks "best" locally).
GRID_SCAN_POINTS = 5


# ---------------------------------------------------------------------------
# Env / clients (mirror evaluate.py)
# ---------------------------------------------------------------------------

def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _api_key() -> str:
    _load_env_file(ENV_PATH)
    key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print(f"[ERROR] GRAPHRAG_API_KEY not found in env or {ENV_PATH}.",
              file=sys.stderr)
        sys.exit(1)
    return key


def _completion():
    return create_completion(
        ModelConfig(
            type=LLMProviderType.LiteLLM,
            model_provider="openai",
            model=COMPLETION_MODEL,
            api_key=_api_key(),
        )
    )


# ---------------------------------------------------------------------------
# Ground-truth + scoring helpers  (duplicated from evaluate.py for isolation)
# ---------------------------------------------------------------------------

def _load_clusters(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] Ground truth file missing: {path}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("clusters", [])


def _pairs_from_clusters(clusters: list[dict], alias_key: str = "aliases") -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for cluster in clusters:
        items = list(dict.fromkeys(cluster.get(alias_key, [])))
        for a, b in combinations(items, 2):
            if a != b:
                pairs.add(frozenset({a, b}))
    return pairs


def _pairs_from_merge_map(merge_map: dict[str, str]) -> set[frozenset[str]]:
    groups: dict[str, set[str]] = {}
    for alias, canonical in merge_map.items():
        groups.setdefault(canonical, set()).add(canonical)
        groups[canonical].add(alias)

    pairs: set[frozenset[str]] = set()
    for members in groups.values():
        for a, b in combinations(sorted(members), 2):
            pairs.add(frozenset({a, b}))
    return pairs


def _f1(predicted: set[frozenset[str]], truth: set[frozenset[str]]) -> tuple[float, float, float]:
    tp = predicted & truth
    p = len(tp) / len(predicted) if predicted else 0.0
    r = len(tp) / len(truth) if truth else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Stage-evaluation callbacks  (one call = one F1 measurement at a threshold)
# ---------------------------------------------------------------------------

def _make_config(cger_threshold: float, cgrr_threshold: float) -> BTGraphRAGConfig:
    return BTGraphRAGConfig(
        enabled=True,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        neo4j_database=TEST_DB,
        # CGER
        cger_enabled=True,
        cger_cosine_threshold=cger_threshold,
        cger_candidate_top_k=CGER_CANDIDATE_TOP_K,
        cger_phase_b_temp_db=CGER_TEMP_DB,
        # CGRR
        cgrr_enabled=True,
        cgrr_cosine_threshold=cgrr_threshold,
        cgrr_candidate_top_k=CGRR_CANDIDATE_TOP_K,
        neo4j_vector_dimensions=NEO4J_VECTOR_DIMENSIONS,
    )


def _count_llm_calls(phase_b_log: list[dict]) -> int:
    """Count Phase-B LLM verdicts from a CGER/CGRR phase_b_log.

    Both pipelines tag LLM-driven entries with a ``decision`` field
    that starts with ``"LLM_"`` (``LLM_MERGE``, ``LLM_DIFFERENT_ENTITY``,
    ``LLM_DIFFERENT_TEMPORAL`` for CGER; ``LLM_NORMALIZE``,
    ``LLM_DIFFERENT`` for CGRR). Non-LLM entries (e.g. ``BELOW_THRESHOLD``)
    are excluded.
    """
    return sum(
        1 for entry in phase_b_log
        if str(entry.get("decision", "")).startswith("LLM_")
    )


async def _eval_cger(
    threshold: float,
    entities_df: pd.DataFrame,
    truth_pairs: set[frozenset[str]],
    model,
    driver,
) -> tuple[float, float, float, int]:
    config = _make_config(threshold, cgrr_threshold=0.85)  # cgrr unused here
    _, merge_map, phase_b_log = await resolve_entities(
        new_entities=entities_df,
        existing_entities=pd.DataFrame(),
        config=config,
        model=model,
        driver=driver,
        phase_b_top_k=CGER_PHASE_B_TOP_K,
    )
    p, r, f1 = _f1(_pairs_from_merge_map(merge_map), truth_pairs)
    return p, r, f1, _count_llm_calls(phase_b_log)


async def _eval_cgrr(
    threshold: float,
    rels_df: pd.DataFrame,
    truth_pairs: set[frozenset[str]],
    model,
    driver,
) -> tuple[float, float, float, int]:
    config = _make_config(cger_threshold=0.85, cgrr_threshold=threshold)  # cger unused here
    async with driver.session(database=TEST_DB) as session:
        await session.run("MATCH (n) DETACH DELETE n")
        _, normalize_map, phase_b_log = await resolve_relationships(
            relationships_df=rels_df,
            config=config,
            session=session,
            model=model,
        )
    p, r, f1 = _f1(_pairs_from_merge_map(normalize_map), truth_pairs)
    return p, r, f1, _count_llm_calls(phase_b_log)


# ---------------------------------------------------------------------------
# Simulated annealing  (1-D, bounded, with cached evaluations)
# ---------------------------------------------------------------------------

async def _simulated_annealing(
    label: str,
    objective: Callable[[float], Awaitable[tuple[float, float, float, int]]],
    iters: int,
    rng: random.Random,
    max_llm_calls: int,
    lambda_llm: float,
    n_grid_points: int = GRID_SCAN_POINTS,
) -> dict:
    """Maximize ``F1 - lambda * llm_rate`` with simulated annealing.

    ``llm_rate = llm_calls / max_llm_calls`` is bounded approximately
    in [0, 1] for the Phase-B-only training regime. The penalty
    discourages collapsing to low thresholds where the LLM is invoked
    on every candidate.

    Returns a dict with the best (threshold, P, R, F1, llm_calls,
    score), the full trace, and the per-threshold cache.
    """
    cache: dict[float, tuple[float, float, float, int]] = {}

    def composite(f1: float, llm_calls: int) -> float:
        rate = (llm_calls / max_llm_calls) if max_llm_calls > 0 else 0.0
        return f1 - lambda_llm * rate

    async def f(theta: float) -> tuple[float, float, float, int]:
        key = round(theta, ROUND_TO)
        if key in cache:
            return cache[key]
        scores = await objective(key)
        cache[key] = scores
        return scores

    print(f"\n{'=' * 70}")
    print(f"  [{label}] Simulated annealing  "
          f"iters={iters}  T0={INITIAL_TEMPERATURE}  alpha={COOLING_ALPHA}")
    print(f"  Bounds=[{THRESHOLD_MIN}, {THRESHOLD_MAX}]  "
          f"grid_points={n_grid_points}")
    print(f"  Objective: F1 - {lambda_llm} * (llm_calls / {max_llm_calls})")
    print(f"{'=' * 70}")

    # --- Coarse grid pre-scan -------------------------------------------
    # Evaluate evenly-spaced thresholds across the bounds, then start SA
    # from whichever scored best. This breaks degenerate local optima at
    # the high-threshold end (F1=0, llm_calls=0 -> score=0 "wins" if SA
    # never explores far enough to reach productive thresholds).
    grid_points = [
        THRESHOLD_MIN + i * (THRESHOLD_MAX - THRESHOLD_MIN) / (n_grid_points - 1)
        for i in range(n_grid_points)
    ] if n_grid_points >= 2 else [INITIAL_THRESHOLD]
    if INITIAL_THRESHOLD not in grid_points:
        grid_points.append(INITIAL_THRESHOLD)
    grid_points = sorted(set(round(g, ROUND_TO) for g in grid_points))

    print(f"  [{label}] Grid pre-scan ({len(grid_points)} points):")
    trace: list[dict] = []
    grid_results: list[dict] = []
    grid_best: tuple[float, float, float, float, int, float] | None = None
    for gi, gtheta in enumerate(grid_points):
        gp, gr, gf1, gllm = await f(gtheta)
        gscore = composite(gf1, gllm)
        entry = {
            "phase": "grid",
            "iter": gi,
            "threshold": round(gtheta, ROUND_TO),
            "precision": gp, "recall": gr, "f1": gf1,
            "llm_calls": gllm, "llm_rate": gllm / max(1, max_llm_calls),
            "score": gscore,
        }
        trace.append(entry)
        grid_results.append(entry)
        is_grid_best = grid_best is None or gscore > grid_best[5] + 1e-9
        marker = "  <- grid best" if is_grid_best else ""
        if is_grid_best:
            grid_best = (gtheta, gp, gr, gf1, gllm, gscore)
        print(f"    grid {gi}  theta={gtheta:.3f}  P={gp:.3f} R={gr:.3f} "
              f"F1={gf1:.3f}  llm={gllm}/{max_llm_calls} score={gscore:.3f}"
              f"{marker}")

    assert grid_best is not None
    current = grid_best[0]
    cur_p, cur_r, cur_f1, cur_llm, cur_score = grid_best[1:]
    best = (current, cur_p, cur_r, cur_f1, cur_llm, cur_score)

    # --- Simulated annealing seeded from grid best ----------------------
    trace.append({
        "phase": "sa",
        "iter": 0, "threshold": round(current, ROUND_TO),
        "precision": cur_p, "recall": cur_r, "f1": cur_f1,
        "llm_calls": cur_llm, "llm_rate": cur_llm / max(1, max_llm_calls),
        "score": cur_score,
        "accepted": True, "temperature": INITIAL_TEMPERATURE,
        "is_best_so_far": True,
    })
    print(f"\n  [{label}] SA start (seeded from grid):")
    print(f"  iter   0  theta={current:.3f}  P={cur_p:.3f} R={cur_r:.3f} "
          f"F1={cur_f1:.3f} llm={cur_llm}/{max_llm_calls} "
          f"score={cur_score:.3f}   <- start")

    temperature = INITIAL_TEMPERATURE
    for step in range(1, iters + 1):
        radius = max(NEIGHBOR_RADIUS_MIN,
                     NEIGHBOR_RADIUS_INIT * temperature / INITIAL_TEMPERATURE)
        proposal = current + rng.uniform(-radius, radius)
        proposal = max(THRESHOLD_MIN, min(THRESHOLD_MAX, proposal))

        prop_p, prop_r, prop_f1, prop_llm = await f(proposal)
        prop_score = composite(prop_f1, prop_llm)
        delta = prop_score - cur_score  # we maximize, so positive delta is good

        if delta >= 0:
            accept = True
            reason = "improve"
        else:
            # Metropolis: accept worse with prob exp(delta / T)  (delta < 0)
            accept_p = math.exp(delta / max(temperature, 1e-9))
            accept = rng.random() < accept_p
            reason = f"metropolis p={accept_p:.3f}"

        is_new_best = prop_score > best[5] + 1e-9
        if is_new_best:
            best = (proposal, prop_p, prop_r, prop_f1, prop_llm, prop_score)

        trace.append({
            "phase": "sa",
            "iter": step,
            "threshold": round(proposal, ROUND_TO),
            "precision": prop_p, "recall": prop_r, "f1": prop_f1,
            "llm_calls": prop_llm, "llm_rate": prop_llm / max(1, max_llm_calls),
            "score": prop_score,
            "accepted": accept, "reason": reason,
            "temperature": temperature, "radius": radius,
            "is_best_so_far": is_new_best,
        })

        marker = " ** NEW BEST" if is_new_best else ""
        print(f"  iter {step:3d}  theta={proposal:.3f}  "
              f"P={prop_p:.3f} R={prop_r:.3f} F1={prop_f1:.3f} "
              f"llm={prop_llm}/{max_llm_calls} score={prop_score:.3f}  "
              f"T={temperature:.4f}  r={radius:.3f}  "
              f"{'accept' if accept else 'reject'} ({reason}){marker}")

        if accept:
            current = proposal
            cur_p, cur_r, cur_f1, cur_llm, cur_score = (
                prop_p, prop_r, prop_f1, prop_llm, prop_score,
            )

        temperature *= COOLING_ALPHA

    print(f"\n  [{label}] best: theta={best[0]:.3f}  "
          f"P={best[1]:.3f} R={best[2]:.3f} F1={best[3]:.3f}  "
          f"llm={best[4]}/{max_llm_calls} score={best[5]:.3f}  "
          f"({len(cache)} unique evals)")

    return {
        "best": {
            "threshold": best[0],
            "precision": best[1],
            "recall": best[2],
            "f1": best[3],
            "llm_calls": best[4],
            "llm_rate": best[4] / max(1, max_llm_calls),
            "score": best[5],
        },
        "grid": {
            "points": grid_results,
            "best": {
                "threshold": grid_best[0],
                "precision": grid_best[1],
                "recall": grid_best[2],
                "f1": grid_best[3],
                "llm_calls": grid_best[4],
                "score": grid_best[5],
            },
        },
        "trace": trace,
        "cache": {
            f"{k:.3f}": {
                "precision": v[0], "recall": v[1], "f1": v[2],
                "llm_calls": v[3],
                "llm_rate": v[3] / max(1, max_llm_calls),
                "score": composite(v[2], v[3]),
            }
            for k, v in sorted(cache.items())
        },
        "iters": iters,
        "unique_evals": len(cache),
        "max_llm_calls": max_llm_calls,
        "lambda_llm": lambda_llm,
        "grid_points_evaluated": len(grid_results),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(stages: list[str], iters: int, seed: int,
               lambda_llm: float, grid_points: int) -> None:
    entities_path = TRAIN_EXTRACTED_DIR / "entities.json"
    rels_path = TRAIN_EXTRACTED_DIR / "relationships.json"
    if not entities_path.exists() or not rels_path.exists():
        print(f"[ERROR] Missing {entities_path} or {rels_path}. "
              f"Run: python {Path(__file__).name.replace('train', 'extract')} --split train",
              file=sys.stderr)
        sys.exit(1)

    entities_df = pd.read_json(entities_path)
    rels_df = pd.read_json(rels_path)
    print(f"[LOAD] {len(entities_df)} train entities, {len(rels_df)} train relationships")

    truth_entity_clusters = _load_clusters(TRAIN_GROUND_TRUTH_DIR / "entity_resolution.json")
    truth_rel_clusters = _load_clusters(TRAIN_GROUND_TRUTH_DIR / "relationship_resolution.json")
    truth_entity_pairs = _pairs_from_clusters(truth_entity_clusters)
    truth_rel_pairs = _pairs_from_clusters(truth_rel_clusters)
    print(f"[LOAD] train ground truth: {len(truth_entity_pairs)} entity pair(s), "
          f"{len(truth_rel_pairs)} relation-type pair(s)")

    print(f"\n[CONFIG] (fixed across search)")
    print(f"  CGER_CANDIDATE_TOP_K       = {CGER_CANDIDATE_TOP_K}")
    print(f"  CGER_PHASE_B_TOP_K         = {CGER_PHASE_B_TOP_K}")
    print(f"  CGRR_CANDIDATE_TOP_K       = {CGRR_CANDIDATE_TOP_K}")
    print(f"  NEO4J_VECTOR_DIMENSIONS    = {NEO4J_VECTOR_DIMENSIONS}")
    print(f"  COMPLETION_MODEL           = {COMPLETION_MODEL}")
    print(f"  LLM_PENALTY_LAMBDA         = {lambda_llm}")
    print(f"  GRID_SCAN_POINTS           = {grid_points}")
    print(f"  stages={stages}  iters={iters}  seed={seed}")

    # Loose upper bounds on Phase-B LLM verdicts per evaluation. CGER
    # invokes the LLM at most once per entity; CGRR at most once per
    # unique relation type (the first canonical-seeded type is free).
    cger_max_llm = max(1, len(entities_df))
    cgrr_max_llm = max(1, rels_df["relation_type"].nunique())

    rng = random.Random(seed)
    model = _completion()
    started = time.time()

    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    results: dict[str, object] = {
        "config": {
            "cger_candidate_top_k": CGER_CANDIDATE_TOP_K,
            "cger_phase_b_top_k": CGER_PHASE_B_TOP_K,
            "cgrr_candidate_top_k": CGRR_CANDIDATE_TOP_K,
            "neo4j_vector_dimensions": NEO4J_VECTOR_DIMENSIONS,
            "completion_model": COMPLETION_MODEL,
            "sa": {
                "iters": iters,
                "seed": seed,
                "initial_threshold": INITIAL_THRESHOLD,
                "initial_temperature": INITIAL_TEMPERATURE,
                "cooling_alpha": COOLING_ALPHA,
                "neighbor_radius_init": NEIGHBOR_RADIUS_INIT,
                "neighbor_radius_min": NEIGHBOR_RADIUS_MIN,
                "threshold_bounds": [THRESHOLD_MIN, THRESHOLD_MAX],
                "lambda_llm": lambda_llm,
                "cger_max_llm_calls": cger_max_llm,
                "cgrr_max_llm_calls": cgrr_max_llm,
                "grid_scan_points": grid_points,
            },
        },
        "train_data": {
            "entities": len(entities_df),
            "relationships": len(rels_df),
            "entity_truth_pairs": len(truth_entity_pairs),
            "relation_truth_pairs": len(truth_rel_pairs),
        },
    }
    try:
        if "cger" in stages:
            async def cger_obj(theta: float) -> tuple[float, float, float, int]:
                return await _eval_cger(theta, entities_df, truth_entity_pairs,
                                        model, driver)
            results["cger"] = await _simulated_annealing(
                "CGER", cger_obj, iters, rng,
                max_llm_calls=cger_max_llm, lambda_llm=lambda_llm,
                n_grid_points=grid_points,
            )

        if "cgrr" in stages:
            async def cgrr_obj(theta: float) -> tuple[float, float, float, int]:
                return await _eval_cgrr(theta, rels_df, truth_rel_pairs,
                                        model, driver)
            results["cgrr"] = await _simulated_annealing(
                "CGRR", cgrr_obj, iters, rng,
                max_llm_calls=cgrr_max_llm, lambda_llm=lambda_llm,
                n_grid_points=grid_points,
            )
    finally:
        await driver.close()

    results["elapsed_seconds"] = round(time.time() - started, 2)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    print()
    print("=" * 70)
    print("  Best thresholds (train)")
    print("=" * 70)
    if "cger" in stages:
        b = results["cger"]["best"]  # type: ignore[index]
        print(f"  CGER_COSINE_THRESHOLD = {b['threshold']:.3f}   "
              f"(P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}  "
              f"llm={b['llm_calls']}/{cger_max_llm} score={b['score']:.3f})")
    if "cgrr" in stages:
        b = results["cgrr"]["best"]  # type: ignore[index]
        print(f"  CGRR_COSINE_THRESHOLD = {b['threshold']:.3f}   "
              f"(P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}  "
              f"llm={b['llm_calls']}/{cgrr_max_llm} score={b['score']:.3f})")
    print(f"  elapsed: {results['elapsed_seconds']}s")
    print(f"  full trace + cache written to: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tune CGER/CGRR cosine thresholds on the train split via simulated annealing.")
    p.add_argument("--stages", default="cger,cgrr",
                   help="Comma-separated stages to tune. Default: cger,cgrr")
    p.add_argument("--iters", type=int, default=10,
                   help="Simulated-annealing iterations per stage. Default: 10")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility. Default: 42")
    p.add_argument("--lambda-llm", type=float, default=LLM_PENALTY_LAMBDA,
                   dest="lambda_llm",
                   help=f"LLM-usage penalty weight in the SA objective "
                        f"F1 - lambda*(llm_calls/max_llm_calls). "
                        f"Default: {LLM_PENALTY_LAMBDA}")
    p.add_argument("--grid-points", type=int, default=GRID_SCAN_POINTS,
                   dest="grid_points",
                   help=f"Number of uniformly-spaced thresholds in the "
                        f"coarse pre-scan that seeds SA. Larger = more "
                        f"robust against degenerate local optima, but "
                        f"costs that many extra evaluations per stage. "
                        f"Default: {GRID_SCAN_POINTS}")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stages = [s.strip().lower() for s in args.stages.split(",") if s.strip()]
    invalid = [s for s in stages if s not in ("cger", "cgrr")]
    if invalid:
        print(f"[ERROR] Unknown stage(s): {invalid}. Allowed: cger, cgrr", file=sys.stderr)
        sys.exit(1)
    if args.lambda_llm < 0:
        print(f"[ERROR] --lambda-llm must be >= 0 (got {args.lambda_llm})", file=sys.stderr)
        sys.exit(1)
    if args.grid_points < 1:
        print(f"[ERROR] --grid-points must be >= 1 (got {args.grid_points})", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main(stages, args.iters, args.seed, args.lambda_llm,
                     args.grid_points))
