"""Metaheuristic hyperparameter optimizer for CGRR scorers.

Same DE / SA / PSO algorithms as the CGER optimizer but searches the CGRR
config space (bm25_weight, semantic_weight, endpoint_weight, merge_threshold,
llm_threshold_low).
"""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from graphrag.bt_graphrag.models.config import BTGraphRAGConfig


# ---------------------------------------------------------------------------
# Search spaces  (one per CGRR scorer)
# ---------------------------------------------------------------------------

@dataclass
class ParamBound:
    name: str
    low: float
    high: float
    config_field: str


# All CGRR scorers share the same threshold + weight search space.
# The weights are only meaningful for scorers that actually use them;
# the objective function normalises them so unused weights cause no harm.
_COMMON = [
    ParamBound("bm25_weight",       0.05, 0.90, "cgrr_bm25_weight"),
    ParamBound("semantic_weight",   0.05, 0.90, "cgrr_semantic_weight"),
    ParamBound("endpoint_weight",   0.00, 0.60, "cgrr_endpoint_weight"),
    ParamBound("merge_threshold",   0.35, 0.95, "cgrr_merge_threshold"),
    ParamBound("llm_threshold_low", 0.15, 0.80, "cgrr_llm_threshold_low"),
]

SEARCH_SPACES: dict[str, list[ParamBound]] = {
    "composite_3signal": _COMMON,
    "bm25_only":         [ParamBound("merge_threshold", 0.35, 0.99, "cgrr_merge_threshold"),
                          ParamBound("llm_threshold_low", 0.15, 0.80, "cgrr_llm_threshold_low")],
    "semantic_only":     [ParamBound("merge_threshold", 0.10, 0.90, "cgrr_merge_threshold"),
                          ParamBound("llm_threshold_low", 0.05, 0.70, "cgrr_llm_threshold_low")],
    "type_and_endpoint": [ParamBound("bm25_weight",     0.05, 0.90, "cgrr_bm25_weight"),
                          ParamBound("endpoint_weight", 0.05, 0.90, "cgrr_endpoint_weight"),
                          ParamBound("merge_threshold", 0.35, 0.95, "cgrr_merge_threshold"),
                          ParamBound("llm_threshold_low", 0.15, 0.80, "cgrr_llm_threshold_low")],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_params(vec: list[float], space: list[ParamBound]) -> BTGraphRAGConfig:
    cfg = BTGraphRAGConfig()
    for i, b in enumerate(space):
        setattr(cfg, b.config_field, b.low + vec[i] * (b.high - b.low))
    if cfg.cgrr_llm_threshold_low >= cfg.cgrr_merge_threshold:
        cfg.cgrr_llm_threshold_low = cfg.cgrr_merge_threshold - 0.05
    return cfg


def normalize_cgrr_weights(cfg: BTGraphRAGConfig) -> BTGraphRAGConfig:
    total = cfg.cgrr_bm25_weight + cfg.cgrr_semantic_weight + cfg.cgrr_endpoint_weight
    if total > 0:
        cfg.cgrr_bm25_weight     /= total
        cfg.cgrr_semantic_weight /= total
        cfg.cgrr_endpoint_weight /= total
    return cfg


def vec_to_params(vec: list[float], space: list[ParamBound]) -> dict[str, float]:
    return {b.name: round(b.low + vec[i] * (b.high - b.low), 4) for i, b in enumerate(space)}


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _objective(vec, scorer_name, scorer_fn, gt, space):
    from graphrag.bt_graphrag.evaluation.cgrr.harness import evaluate_scorer
    cfg = decode_params(vec, space)
    if scorer_name == "composite_3signal":
        cfg = normalize_cgrr_weights(cfg)
    return -evaluate_scorer(scorer_fn, scorer_name, gt, cfg).f1


# ---------------------------------------------------------------------------
# DE / SA / PSO  (identical structure to cger.optimizer)
# ---------------------------------------------------------------------------

def _de(obj, ndim, pop=30, iters=80, F=0.8, CR=0.9, seed=42):
    rng = random.Random(seed)
    pop_ = [[rng.random() for _ in range(ndim)] for _ in range(pop)]
    fit = [obj(ind) for ind in pop_]; hist = []
    for gen in range(1, iters + 1):
        for i in range(pop):
            idxs = [x for x in range(pop) if x != i]
            a, b, c = rng.sample(idxs, 3); j0 = rng.randint(0, ndim - 1)
            trial = [max(0., min(1., pop_[a][d] + F*(pop_[b][d]-pop_[c][d]))
                         if (rng.random() < CR or d == j0) else pop_[i][d])
                     for d in range(ndim)]
            ft = obj(trial)
            if ft <= fit[i]: pop_[i], fit[i] = trial, ft
        if gen % 10 == 0:
            hist.append({"iter": gen, "best_f1": round(-min(fit), 4)})
            print(f"      [DE] gen {gen}/{iters}: F1={-min(fit):.4f}")
    best = min(range(pop), key=lambda i: fit[i])
    return pop_[best], -fit[best], hist


def _sa(obj, ndim, iters=1500, T0=1.0, Tend=0.001, step=0.1, seed=42):
    rng = random.Random(seed)
    cur = [rng.random() for _ in range(ndim)]
    cur_c = obj(cur); best = list(cur); best_c = cur_c
    alpha = (Tend/T0)**(1.0/iters); T = T0; hist = []
    for it in range(1, iters + 1):
        nb = [max(0., min(1., cur[d] + rng.gauss(0, step))) for d in range(ndim)]
        nb_c = obj(nb); delta = nb_c - cur_c
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            cur, cur_c = nb, nb_c
        if cur_c < best_c: best, best_c = list(cur), cur_c
        T *= alpha
        if it % 200 == 0:
            hist.append({"iter": it, "best_f1": round(-best_c, 4), "T": round(T, 6)})
            print(f"      [SA] iter {it}/{iters}: F1={-best_c:.4f}, T={T:.5f}")
    return best, -best_c, hist


def _pso(obj, ndim, n=25, iters=80, w=0.7, c1=1.5, c2=1.5, seed=42):
    rng = random.Random(seed)
    pos = [[rng.random() for _ in range(ndim)] for _ in range(n)]
    vel = [[rng.uniform(-0.1, 0.1) for _ in range(ndim)] for _ in range(n)]
    pb = [list(p) for p in pos]; pbc = [obj(p) for p in pos]
    gi = min(range(n), key=lambda i: pbc[i]); gb = list(pb[gi]); gbc = pbc[gi]
    hist = []
    for it in range(1, iters + 1):
        for i in range(n):
            for d in range(ndim):
                r1, r2 = rng.random(), rng.random()
                vel[i][d] = w*vel[i][d] + c1*r1*(pb[i][d]-pos[i][d]) + c2*r2*(gb[d]-pos[i][d])
                pos[i][d] = max(0., min(1., pos[i][d]+vel[i][d]))
            c = obj(pos[i])
            if c < pbc[i]:
                pb[i], pbc[i] = list(pos[i]), c
                if c < gbc: gb, gbc = list(pos[i]), c
        if it % 10 == 0:
            hist.append({"iter": it, "best_f1": round(-gbc, 4)})
            print(f"      [PSO] iter {it}/{iters}: F1={-gbc:.4f}")
    return gb, -gbc, hist


# ---------------------------------------------------------------------------
# Public result + entry point
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    scorer_name: str
    component: str
    method: str
    best_params: dict[str, float]
    best_f1: float
    elapsed_seconds: float
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scorer_name": self.scorer_name,
            "component": self.component,
            "method": self.method,
            "best_params": self.best_params,
            "best_f1": round(self.best_f1, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "history": self.history,
        }


async def optimize(
    scorer_name: str,
    ground_truth: dict | None = None,
    ground_truth_path: str | None = None,
    methods: list[str] | None = None,
    de_iters: int = 80, sa_iters: int = 1500, pso_iters: int = 80,
    seed: int = 42,
) -> list[OptimizationResult]:
    """Find best hyperparameters for a CGRR scorer via metaheuristics."""
    import json
    from graphrag.bt_graphrag.evaluation.cgrr.scorers import SCORER_REGISTRY

    if ground_truth is None:
        if ground_truth_path:
            with open(ground_truth_path) as f:
                ground_truth = json.load(f)
        else:
            from graphrag.bt_graphrag.evaluation.cgrr.ground_truth import build_ground_truth
            ground_truth = build_ground_truth()

    space = SEARCH_SPACES[scorer_name]
    scorer_fn = SCORER_REGISTRY[scorer_name]
    ndim = len(space)

    def obj(v): return _objective(v, scorer_name, scorer_fn, ground_truth, space)

    results: list[OptimizationResult] = []
    for method in (methods or ["de", "sa", "pso"]):
        print(f"\n    [CGRR Optimizer] {method.upper()} for '{scorer_name}' ({ndim} dims)…")
        t0 = time.time()
        if method == "de":
            bv, bf, hist = _de(obj, ndim, iters=de_iters, seed=seed)
        elif method == "sa":
            bv, bf, hist = _sa(obj, ndim, iters=sa_iters, seed=seed)
        else:
            bv, bf, hist = _pso(obj, ndim, iters=pso_iters, seed=seed)
        elapsed = time.time() - t0

        params = vec_to_params(bv, space)
        if scorer_name == "composite_3signal":
            tw = sum(params.get(b.name, 0) for b in space[:3])
            if tw > 0:
                for b in space[:3]:
                    params[f"{b.name}_norm"] = round(params[b.name] / tw, 4)

        results.append(OptimizationResult(
            scorer_name=scorer_name, component="CGRR",
            method=method, best_params=params,
            best_f1=bf, elapsed_seconds=elapsed, history=hist,
        ))
        print(f"    [CGRR Optimizer] {method.upper()} done → F1={bf:.4f} in {elapsed:.1f}s")
    return results
