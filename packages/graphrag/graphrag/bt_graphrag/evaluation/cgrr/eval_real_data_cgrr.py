# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CGRR Evaluation with Real Data and Metaheuristic Hyperparameter Optimisation.

This script:

  1. Loads GRAPHRAG_API_KEY from ragtest/.env (or environment).
  2. Builds real LLM (gpt-4.1-mini) and embedder (text-embedding-3-large) from
     the default graphrag config.
  3. Extracts relationships from every .txt file under ragtest/input/cgrr/ using
     the GraphExtractor (same module the full indexing pipeline uses).
  4. Generates a cross-document ground-truth dataset by asking the LLM whether
     every pair of extracted relationship types refers to the SAME semantic
     relationship or a DIFFERENT one (SAME / DIFFERENT).
  5. Evaluates all four CGRR scorers:
       • compute_relationship_composite_score    (3-signal)
       • bm25_only_relationship_scorer
       • semantic_only_relationship_scorer
       • type_and_endpoint_relationship_scorer
  6. Uses Differential Evolution (a population-based metaheuristic available in
     scipy) to find the hyperparameter configuration that maximises F1 with LLM.
     The objective is:
         maximize   F1_with_llm  −  lambda * llm_call_rate
  7. Prints a full report and saves results to a timestamped JSON file.

Usage:
    python packages/graphrag/graphrag/bt_graphrag/evaluation/cgrr/eval_real_data_cgrr.py

    Optional env vars:
        GRAPHRAG_API_KEY   – OpenAI API key (or set in ragtest/.env)
        CGRR_INPUT_DIR     – override the default ragtest/input/cgrr directory
        CGRR_MAX_PAIRS     – max labelled pairs to collect (default: unlimited)
        CGRR_LAMBDA        – lambda for the metaheuristic objective (default: 0.2)
        CGRR_DE_POPSIZE    – DE population size multiplier (default: 15)
        CGRR_DE_MAXITER    – DE max iterations (default: 300)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installing the package
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[7]
_PACKAGES  = _REPO_ROOT / "packages" / "graphrag"
if str(_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_PACKAGES))

# ---------------------------------------------------------------------------
# Imports from graphrag
# ---------------------------------------------------------------------------

from graphrag.bt_graphrag.entity_resolution.scorers import (
    bm25_only_relationship_scorer,
    compute_relationship_composite_score,
    semantic_only_relationship_scorer,
    type_and_endpoint_relationship_scorer,
)
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT
from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig
from graphrag_llm.config.types import LLMProviderType
from graphrag_llm.embedding import create_embedding

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_env(env_file: Path | None = None) -> None:
    """Load GRAPHRAG_API_KEY from ragtest/.env if it exists."""
    candidates = [
        env_file,
        _REPO_ROOT / "ragtest" / ".env",
        Path.cwd() / "ragtest" / ".env",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            with candidate.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        os.environ.setdefault(key.strip(), val.strip())
            print(f"[env] Loaded variables from {candidate}", file=sys.stderr)
            return
    print("[env] No .env file found — using environment variables as-is.", file=sys.stderr)


def _real_llm(model: str = "gpt-4.1-mini"):
    api_key = os.environ.get("GRAPHRAG_API_KEY", "")
    if not api_key:
        print("[WARN] GRAPHRAG_API_KEY not set — LLM calls will fail.", file=sys.stderr)
    cfg = ModelConfig(
        type=LLMProviderType.LiteLLM,
        model_provider="openai",
        model=model,
        api_key=api_key,
    )
    return create_completion(cfg)


def _real_embedder(model: str = "text-embedding-3-large"):
    api_key = os.environ.get("GRAPHRAG_API_KEY", "")
    if not api_key:
        print("[WARN] GRAPHRAG_API_KEY not set — embedding calls will fail.", file=sys.stderr)
    cfg = ModelConfig(
        type=LLMProviderType.LiteLLM,
        model_provider="openai",
        model=model,
        api_key=api_key,
    )
    return create_embedding(cfg)


# ---------------------------------------------------------------------------
# Relationship extraction from txt files
# ---------------------------------------------------------------------------

async def extract_relationships_from_files(
    input_dir: Path,
    llm_model,
    entity_types: list[str] | None = None,
    max_gleanings: int = 0,
) -> list[dict]:
    """Run GraphExtractor on every .txt file in *input_dir*.

    Returns a flat list of relationship dicts, each augmented with:
      - ``source_file``   — the originating file name
      - ``id``            — a new UUID
      - ``relation_type`` — uppercased relation_type if present, else derived from description
    """
    if entity_types is None:
        entity_types = ["organization", "person", "geo", "event"]

    extractor = GraphExtractor(
        model=llm_model,
        prompt=GRAPH_EXTRACTION_PROMPT,
        max_gleanings=max_gleanings,
    )

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    all_rels: list[dict] = []
    for txt_path in txt_files:
        print(f"  Extracting from {txt_path.name} …", file=sys.stderr)
        text = txt_path.read_text(encoding="utf-8")
        _, rels_df = await extractor(
            text=text,
            entity_types=entity_types,
            source_id=txt_path.stem,
        )
        for _, row in rels_df.iterrows():
            rel_type = str(row.get("relation_type", "")).strip().upper()
            if not rel_type:
                # Derive a simple relation type label from the description
                desc = str(row.get("description", ""))
                words = desc.upper().split()[:3]
                rel_type = "_".join(w.strip(".,;") for w in words) if words else "RELATED_TO"
            rel = {
                "id":          str(uuid4()),
                "source":      str(row.get("source", "")).strip().upper(),
                "target":      str(row.get("target", "")).strip().upper(),
                "relation_type": rel_type,
                "description": str(row.get("description", "")),
                "weight":      float(row.get("weight", 1.0)),
                "source_id":   str(row.get("source_id", txt_path.stem)),
                "source_file": txt_path.name,
            }
            all_rels.append(rel)

    print(f"  Extracted {len(all_rels)} relationships from {len(txt_files)} files.",
          file=sys.stderr)
    return all_rels


# ---------------------------------------------------------------------------
# Ground-truth labelling via LLM
# ---------------------------------------------------------------------------

_LABEL_PROMPT = """\
You are an expert knowledge-graph analyst.
Determine whether the two relationship types below represent the SAME semantic relationship.
Consider: synonyms, paraphrases, and name variants count as SAME.
Reply with exactly one word: SAME or DIFFERENT.

Relationship A:
  Type       : {type_a}
  Description: {desc_a}
  Between    : {src_a} → {tgt_a}

Relationship B:
  Type       : {type_b}
  Description: {desc_b}
  Between    : {src_b} → {tgt_b}
"""


async def _llm_label_rel_pair(rel_a: dict, rel_b: dict, llm_model) -> str:
    """Ask the LLM whether rel_a and rel_b represent the same relationship type."""
    from graphrag_llm.utils import CompletionMessagesBuilder

    prompt = _LABEL_PROMPT.format(
        type_a=rel_a.get("relation_type", ""),
        desc_a=rel_a.get("description", ""),
        src_a=rel_a.get("source", ""),
        tgt_a=rel_a.get("target", ""),
        type_b=rel_b.get("relation_type", ""),
        desc_b=rel_b.get("description", ""),
        src_b=rel_b.get("source", ""),
        tgt_b=rel_b.get("target", ""),
    )
    builder  = CompletionMessagesBuilder().add_user_message(prompt)
    response = await llm_model.completion_async(messages=builder.build())
    answer   = response.content.strip().upper()
    if "SAME" in answer:
        return "SAME"
    return "DIFFERENT"


@dataclass
class LabelledRelPair:
    rel_a:       dict
    rel_b:       dict
    label:       str   # "SAME" or "DIFFERENT"
    description: str   # human-readable note


async def build_ground_truth_from_relationships(
    rels: list[dict],
    llm_model,
    max_pairs: int | None = None,
) -> list[LabelledRelPair]:
    """Label cross-document relationship pairs as SAME or DIFFERENT using the LLM.

    Strategy:
    - Pair every relationship from file A with every relationship from file B (cross-document).
    - Skip pairs from the same source file.
    - Cap with *max_pairs* to control API costs.
    """
    by_file: dict[str, list[dict]] = {}
    for rel in rels:
        by_file.setdefault(rel["source_file"], []).append(rel)

    files = list(by_file.keys())
    candidates: list[tuple[dict, dict]] = []

    for i, fa in enumerate(files):
        for fb in files[i + 1:]:
            for ra in by_file[fa]:
                for rb in by_file[fb]:
                    candidates.append((ra, rb))

    if max_pairs is not None:
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(candidates))[:max_pairs]
        candidates = [candidates[i] for i in sorted(idx)]

    print(f"  Labelling {len(candidates)} cross-document relationship pairs with LLM …",
          file=sys.stderr)

    pairs: list[LabelledRelPair] = []
    for k, (ra, rb) in enumerate(candidates):
        label = await _llm_label_rel_pair(ra, rb, llm_model)
        desc  = (f"{ra['relation_type']} ({ra['source_file']}) vs "
                 f"{rb['relation_type']} ({rb['source_file']})")
        pairs.append(LabelledRelPair(rel_a=ra, rel_b=rb, label=label, description=desc))
        if (k + 1) % 10 == 0:
            n_same = sum(1 for p in pairs if p.label == "SAME")
            print(f"    … {k + 1}/{len(candidates)} labelled  "
                  f"({n_same} SAME, {k+1-n_same} DIFFERENT)", file=sys.stderr)

    n_same = sum(1 for p in pairs if p.label == "SAME")
    print(f"  Ground truth: {len(pairs)} pairs ({n_same} SAME, "
          f"{len(pairs)-n_same} DIFFERENT)", file=sys.stderr)
    return pairs


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    scorer_name:       str
    config_label:      str
    merge_threshold:   float
    llm_threshold_low: float
    extra_params:      dict = field(default_factory=dict)
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    llm_same: int = 0
    llm_diff: int = 0
    n_total:  int = 0

    @property
    def llm_calls(self) -> int:
        return self.llm_same + self.llm_diff

    @property
    def llm_call_rate(self) -> float:
        return self.llm_calls / max(self.n_total, 1)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def recall_with_llm(self) -> float:
        return (self.tp + self.llm_same) / max(self.tp + self.fn + self.llm_same, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)

    @property
    def f1_with_llm(self) -> float:
        p = (self.tp + self.llm_same) / max(self.tp + self.llm_same + self.fp, 1)
        r = self.recall_with_llm
        return 2 * p * r / max(p + r, 1e-9)

    @property
    def accuracy_with_llm(self) -> float:
        return (self.tp + self.tn + self.llm_calls) / max(self.n_total, 1)

    def objective(self, lam: float = 0.2) -> float:
        return self.f1_with_llm - lam * self.llm_call_rate


def _evaluate_rel_scorer(
    scorer_fn,
    scorer_name: str,
    config: BTGraphRAGConfig,
    pairs: list[LabelledRelPair],
    merge_threshold: float,
    llm_threshold_low: float,
    config_label: str = "",
    extra_params: dict | None = None,
) -> EvalResult:
    result = EvalResult(
        scorer_name=scorer_name,
        config_label=config_label,
        merge_threshold=merge_threshold,
        llm_threshold_low=llm_threshold_low,
        extra_params=extra_params or {},
        n_total=len(pairs),
    )
    for pair in pairs:
        ra, rb = pair.rel_a, pair.rel_b
        score, _ = scorer_fn(
            ra.get("relation_type", ""),
            ra.get("description", ""),
            ra.get("source", ""),
            ra.get("target", ""),
            rb.get("relation_type", ""),
            rb.get("description", ""),
            rb.get("source", ""),
            rb.get("target", ""),
            config,
        )
        is_same = pair.label == "SAME"
        if score >= merge_threshold:
            if is_same:
                result.tp += 1
            else:
                result.fp += 1
        elif score >= llm_threshold_low:
            if is_same:
                result.llm_same += 1
            else:
                result.llm_diff += 1
        else:
            if is_same:
                result.fn += 1
            else:
                result.tn += 1
    return result


# ---------------------------------------------------------------------------
# Metaheuristic optimisation — Differential Evolution via scipy
# ---------------------------------------------------------------------------

def _run_de_optimise(
    scorer_fn,
    scorer_name: str,
    pairs: list[LabelledRelPair],
    bounds: list[tuple[float, float]],
    param_names: list[str],
    make_config,
    lam: float = 0.2,
    popsize: int = 15,
    maxiter: int = 300,
    seed: int = 42,
) -> tuple[EvalResult, dict]:
    """Run Differential Evolution to maximise the scorer objective."""
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        print("[WARN] scipy not available — falling back to coarse grid search.",
              file=sys.stderr)
        return _fallback_grid_search(scorer_fn, scorer_name, pairs, bounds,
                                     param_names, make_config, lam)

    call_count = [0]

    def _neg_objective(x: np.ndarray) -> float:
        cfg, extra = make_config(list(x))
        mt = float(x[0])
        lt = float(x[1])
        if lt >= mt:
            return 1.0
        result = _evaluate_rel_scorer(
            scorer_fn, scorer_name, cfg, pairs, mt, lt,
            extra_params=extra,
        )
        call_count[0] += 1
        return -result.objective(lam)

    print(f"  Running Differential Evolution (popsize×={popsize}, maxiter={maxiter}) …",
          file=sys.stderr)
    de_result = differential_evolution(
        _neg_objective,
        bounds=bounds,
        seed=seed,
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-5,
        mutation=(0.5, 1.0),
        recombination=0.7,
        workers=1,
        polish=True,
        updating="immediate",
    )

    best_x = list(de_result.x)
    cfg_best, extra_best = make_config(best_x)
    mt_best = float(best_x[0])
    lt_best = float(best_x[1])
    if lt_best >= mt_best:
        lt_best = mt_best - 0.01

    best_result = _evaluate_rel_scorer(
        scorer_fn, scorer_name, cfg_best, pairs,
        mt_best, lt_best,
        config_label=scorer_name,
        extra_params=extra_best,
    )
    best_params = dict(zip(param_names, best_x))
    print(f"  DE finished after {call_count[0]} evaluations. "
          f"Best objective={best_result.objective(lam):.4f}", file=sys.stderr)
    return best_result, best_params


def _fallback_grid_search(scorer_fn, scorer_name, pairs, bounds, param_names, make_config, lam):
    best_obj    = -1.0
    best_result = None
    best_params: dict = {}
    mt_vals    = np.linspace(bounds[0][0], bounds[0][1], 10)
    lt_vals    = np.linspace(bounds[1][0], bounds[1][1], 10)
    other_mids = [(b[0] + b[1]) / 2 for b in bounds[2:]]
    for mt in mt_vals:
        for lt in lt_vals:
            if lt >= mt:
                continue
            x      = [mt, lt] + other_mids
            cfg, extra = make_config(x)
            result = _evaluate_rel_scorer(
                scorer_fn, scorer_name, cfg, pairs, float(mt), float(lt),
                extra_params=extra,
            )
            obj = result.objective(lam)
            if obj > best_obj:
                best_obj    = obj
                best_result = result
                best_params = dict(zip(param_names, x))
    return best_result, best_params


# ---------------------------------------------------------------------------
# Scorer-specific DE setup
# ---------------------------------------------------------------------------

def _optimise_composite(
    pairs: list[LabelledRelPair], lam: float, popsize: int, maxiter: int
) -> tuple[EvalResult, dict]:
    """DE for composite (3-signal) scorer — 5 hyperparameters: mt, lt, w_bm25, w_sem, w_ep."""
    bounds = [
        (0.40, 0.99),  # merge_threshold
        (0.10, 0.75),  # llm_threshold_low
        (0.0,  1.0),   # w_bm25  (raw, normalised)
        (0.0,  1.0),   # w_semantic
        (0.0,  1.0),   # w_endpoint
    ]
    param_names = ["merge_threshold", "llm_threshold_low", "w_bm25", "w_semantic", "w_endpoint"]

    def make_config(x: list[float]):
        raw   = np.array(x[2:], dtype=float)
        total = raw.sum()
        if total < 1e-9:
            raw = np.ones(3) / 3
        else:
            raw = raw / total
        cfg = BTGraphRAGConfig(
            cgrr_bm25_weight=float(raw[0]),
            cgrr_semantic_weight=float(raw[1]),
            cgrr_endpoint_weight=float(raw[2]),
        )
        extra = {
            "w_bm25":     round(float(raw[0]), 4),
            "w_semantic": round(float(raw[1]), 4),
            "w_endpoint": round(float(raw[2]), 4),
        }
        return cfg, extra

    return _run_de_optimise(
        compute_relationship_composite_score, "composite",
        pairs, bounds, param_names, make_config,
        lam=lam, popsize=popsize, maxiter=maxiter,
    )


def _optimise_bm25_only(
    pairs: list[LabelledRelPair], lam: float, popsize: int, maxiter: int
) -> tuple[EvalResult, dict]:
    """DE for bm25_only scorer — 2 hyperparameters: mt, lt."""
    bounds = [(0.40, 0.99), (0.10, 0.75)]
    param_names = ["merge_threshold", "llm_threshold_low"]

    def make_config(_x: list[float]):
        return BTGraphRAGConfig(), {}

    return _run_de_optimise(
        bm25_only_relationship_scorer, "bm25_only",
        pairs, bounds, param_names, make_config,
        lam=lam, popsize=popsize, maxiter=maxiter,
    )


def _optimise_semantic_only(
    pairs: list[LabelledRelPair], lam: float, popsize: int, maxiter: int
) -> tuple[EvalResult, dict]:
    """DE for semantic_only scorer — 2 hyperparameters: mt, lt."""
    bounds = [(0.40, 0.99), (0.10, 0.75)]
    param_names = ["merge_threshold", "llm_threshold_low"]

    def make_config(_x: list[float]):
        return BTGraphRAGConfig(), {}

    return _run_de_optimise(
        semantic_only_relationship_scorer, "semantic_only",
        pairs, bounds, param_names, make_config,
        lam=lam, popsize=popsize, maxiter=maxiter,
    )


def _optimise_type_endpoint(
    pairs: list[LabelledRelPair], lam: float, popsize: int, maxiter: int
) -> tuple[EvalResult, dict]:
    """DE for type_and_endpoint scorer — 4 hyperparameters: mt, lt, w_bm25, w_endpoint."""
    bounds = [
        (0.40, 0.99),  # merge_threshold
        (0.10, 0.75),  # llm_threshold_low
        (0.0,  1.0),   # w_bm25
        (0.0,  1.0),   # w_endpoint
    ]
    param_names = ["merge_threshold", "llm_threshold_low", "w_bm25", "w_endpoint"]

    def make_config(x: list[float]):
        raw   = np.array(x[2:], dtype=float)
        total = raw.sum()
        if total < 1e-9:
            raw = np.ones(2) / 2
        else:
            raw = raw / total
        cfg = BTGraphRAGConfig(
            cgrr_bm25_weight=float(raw[0]),
            cgrr_endpoint_weight=float(raw[1]),
        )
        extra = {
            "w_bm25":     round(float(raw[0]), 4),
            "w_endpoint": round(float(raw[1]), 4),
        }
        return cfg, extra

    return _run_de_optimise(
        type_and_endpoint_relationship_scorer, "type_endpoint",
        pairs, bounds, param_names, make_config,
        lam=lam, popsize=popsize, maxiter=maxiter,
    )


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

LAMBDAS = [0.0, 0.1, 0.2, 0.4, 0.6, 1.0]


def _sep(title: str) -> None:
    print("\n" + "═" * 80)
    print(f"  {title}")
    print("═" * 80)


def _print_best(result: EvalResult, params: dict, scorer_name: str, lam: float) -> None:
    _sep(f"BEST CONFIGURATION — {scorer_name} (lambda={lam})")
    print(f"  merge_threshold   = {result.merge_threshold:.4f}")
    print(f"  llm_threshold_low = {result.llm_threshold_low:.4f}")
    for k, v in (result.extra_params or params).items():
        print(f"  {k:20s}= {v}")
    print(f"  TP / FP / TN / FN = {result.tp} / {result.fp} / {result.tn} / {result.fn}")
    print(f"  LLM zone SAME/DIFF= {result.llm_same} / {result.llm_diff}")
    print(f"  Precision         = {result.precision:.4f}")
    print(f"  Recall            = {result.recall:.4f}")
    print(f"  F1 (auto only)    = {result.f1:.4f}")
    print(f"  F1 (with LLM)     = {result.f1_with_llm:.4f}")
    print(f"  LLM call rate     = {result.llm_call_rate:.1%}  "
          f"({result.llm_calls}/{result.n_total} pairs)")
    print(f"  Objective         = {result.objective(lam):.4f}")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def _result_to_dict(r: EvalResult, params: dict) -> dict:
    return {
        "scorer_name":       r.scorer_name,
        "merge_threshold":   round(r.merge_threshold, 6),
        "llm_threshold_low": round(r.llm_threshold_low, 6),
        "extra_params":      r.extra_params or params,
        "tp": r.tp, "tn": r.tn, "fp": r.fp, "fn": r.fn,
        "llm_same": r.llm_same, "llm_diff": r.llm_diff,
        "n_total":  r.n_total,
        "precision":         round(r.precision, 6),
        "recall":            round(r.recall, 6),
        "f1":                round(r.f1, 6),
        "f1_with_llm":       round(r.f1_with_llm, 6),
        "accuracy_with_llm": round(r.accuracy_with_llm, 6),
        "llm_call_rate":     round(r.llm_call_rate, 6),
        "objectives": {str(lam): round(r.objective(lam), 6) for lam in LAMBDAS},
        "de_params":  params,
    }


def _save_json(
    pairs: list[LabelledRelPair],
    comp_result:   EvalResult, comp_params:   dict,
    bm25_result:   EvalResult, bm25_params:   dict,
    sem_result:    EvalResult, sem_params:    dict,
    ep_result:     EvalResult, ep_params:     dict,
    lam: float,
    output_path: str | None = None,
) -> str:
    if output_path is None:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(__file__).parent / f"eval_cgrr_real_results_{ts}.json")

    gt_summary = [
        {
            "index":      i + 1,
            "rel_type_a": p.rel_a.get("relation_type", ""),
            "rel_type_b": p.rel_b.get("relation_type", ""),
            "label":      p.label,
            "description": p.description,
        }
        for i, p in enumerate(pairs)
    ]

    n_same = sum(1 for p in pairs if p.label == "SAME")
    payload = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "method": "differential_evolution",
            "n_pairs": len(pairs),
            "n_same": n_same,
            "n_different": len(pairs) - n_same,
            "lambda": lam,
        },
        "ground_truth": gt_summary,
        "scorers": {
            "composite_3signal":       _result_to_dict(comp_result, comp_params),
            "bm25_only":               _result_to_dict(bm25_result, bm25_params),
            "semantic_only":           _result_to_dict(sem_result,  sem_params),
            "type_and_endpoint":       _result_to_dict(ep_result,   ep_params),
        },
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\n  📄 Results saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    _load_env()

    input_dir  = Path(os.environ.get("CGRR_INPUT_DIR",
                      str(_REPO_ROOT / "ragtest" / "input" / "cgrr")))
    max_pairs  = int(os.environ.get("CGRR_MAX_PAIRS", "0")) or None
    lam        = float(os.environ.get("CGRR_LAMBDA",   "0.2"))
    de_popsize = int(os.environ.get("CGRR_DE_POPSIZE", "15"))
    de_maxiter = int(os.environ.get("CGRR_DE_MAXITER", "300"))

    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║   CGRR Real-Data Evaluation + Metaheuristic Hyperparameter Optimisation     ║")
    print("║                                                                              ║")
    print("║   Objective = F1(with_llm) − lambda × LLM_call_rate                        ║")
    print("║   Optimiser = Differential Evolution (scipy)                                 ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Input dir : {input_dir}")
    print(f"  Max pairs : {max_pairs or 'unlimited'}")
    print(f"  Lambda    : {lam}")
    print(f"  DE popsize: {de_popsize}×  max iter: {de_maxiter}")

    # ── Step 1: Build real models ────────────────────────────────────────────
    print("\n[1/4] Building LLM and embedding models …", file=sys.stderr)
    llm_model = _real_llm()
    embedder  = _real_embedder()  # noqa: F841  (available for future embedding signals)

    # ── Step 2: Extract relationships from txt files ─────────────────────────
    print("\n[2/4] Extracting relationships from txt files …", file=sys.stderr)
    rels = await extract_relationships_from_files(input_dir, llm_model)

    if not rels:
        print("[ERROR] No relationships extracted — check your txt files and API key.",
              file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Build ground-truth pairs ─────────────────────────────────────
    print("\n[3/4] Labelling cross-document relationship pairs with LLM …", file=sys.stderr)
    pairs = await build_ground_truth_from_relationships(rels, llm_model, max_pairs=max_pairs)

    if len(pairs) < 4:
        print("[ERROR] Too few labelled pairs to run optimisation "
              "(need ≥ 4). Add more txt files or relationships.", file=sys.stderr)
        sys.exit(1)

    # ── Step 4: Metaheuristic optimisation per scorer ────────────────────────
    print("\n[4/4] Running Differential Evolution per scorer …", file=sys.stderr)

    _sep("SCORER 1 — compute_relationship_composite_score (3-signal)")
    comp_result, comp_params = _optimise_composite(pairs, lam, de_popsize, de_maxiter)
    _print_best(comp_result, comp_params, "compute_relationship_composite_score", lam)

    _sep("SCORER 2 — bm25_only_relationship_scorer")
    bm25_result, bm25_params = _optimise_bm25_only(pairs, lam, de_popsize, de_maxiter)
    _print_best(bm25_result, bm25_params, "bm25_only_relationship_scorer", lam)

    _sep("SCORER 3 — semantic_only_relationship_scorer")
    sem_result, sem_params = _optimise_semantic_only(pairs, lam, de_popsize, de_maxiter)
    _print_best(sem_result, sem_params, "semantic_only_relationship_scorer", lam)

    _sep("SCORER 4 — type_and_endpoint_relationship_scorer")
    ep_result, ep_params = _optimise_type_endpoint(pairs, lam, de_popsize, de_maxiter)
    _print_best(ep_result, ep_params, "type_and_endpoint_relationship_scorer", lam)

    # ── Cross-scorer summary ──────────────────────────────────────────────────
    _sep("CROSS-SCORER COMPARISON")
    print(f"\n  {'Scorer':40s} {'F1+LLM':>8} {'LLM%':>7} {'Acc+LLM':>8} {'Objective':>10}")
    print(f"  {'-'*40} {'-'*8} {'-'*7} {'-'*8} {'-'*10}")
    for name, result in [
        ("composite_3signal",   comp_result),
        ("bm25_only",           bm25_result),
        ("semantic_only",       sem_result),
        ("type_and_endpoint",   ep_result),
    ]:
        print(
            f"  {name:40s} {result.f1_with_llm:8.4f} "
            f"{result.llm_call_rate:6.1%} {result.accuracy_with_llm:8.4f} "
            f"{result.objective(lam):10.4f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_json(
        pairs,
        comp_result, comp_params,
        bm25_result, bm25_params,
        sem_result,  sem_params,
        ep_result,   ep_params,
        lam,
    )

    print("\n" + "═" * 80)
    print("  CGRR real-data evaluation complete.")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
