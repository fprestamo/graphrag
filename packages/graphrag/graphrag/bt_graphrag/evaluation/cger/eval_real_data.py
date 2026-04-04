# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CGER Evaluation with Real Data and Metaheuristic Hyperparameter Optimisation.

This script:

  1. Loads GRAPHRAG_API_KEY from ragtest/.env (or environment).
  2. Builds real LLM (gpt-4.1-mini) and embedder (text-embedding-3-large) from
     the default graphrag config.
  3. Extracts entities from every .txt file under ragtest/input/cger/ using the
     GraphExtractor (the same module the full indexing pipeline uses).
  4. Embeds the extracted entities with the real embedding model.
  5. Generates a cross-document ground-truth dataset by asking the LLM whether
     every pair of extracted entities refers to the same real-world entity
     (SAME / DIFFERENT).
  6. Evaluates all three CGER scorers:
       • embedding_only_entity_scorer
       • citation_and_description_entity_scorer
       • compute_entity_composite_score  (5-signal)
  7. Uses Differential Evolution (a population-based metaheuristic available in
     scipy) to find the hyperparameter configuration that maximises F1 with LLM.
     The objective is:
         maximize   F1_with_llm  −  lambda * llm_call_rate
  8. Prints a full report and saves results to a timestamped JSON file.

Usage:
    python packages/graphrag/graphrag/bt_graphrag/evaluation/cger/eval_real_data.py

    Optional env vars:
        GRAPHRAG_API_KEY   – OpenAI API key (or set in ragtest/.env)
        CGER_INPUT_DIR     – override the default ragtest/input/cger directory
        CGER_MAX_PAIRS     – max labelled pairs to collect (default: unlimited)
        CGER_LAMBDA        – lambda for the metaheuristic objective (default: 0.2)
        CGER_DE_POPSIZE    – DE population size multiplier (default: 15)
        CGER_DE_MAXITER    – DE max iterations (default: 300)
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
_SCRIPT_DIR = Path(__file__).resolve().parent   # evaluation/cger/
if str(_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_PACKAGES))

# ---------------------------------------------------------------------------
# Imports from graphrag
# ---------------------------------------------------------------------------

from graphrag.bt_graphrag.entity_resolution.scorers import (
    citation_and_description_entity_scorer,
    compute_entity_composite_score,
    embedding_only_entity_scorer,
)
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.prompts import TEMPORAL_GRAPH_EXTRACTION_PROMPT
from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
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
# Entity extraction from txt files
# ---------------------------------------------------------------------------

async def extract_entities_from_files(
    input_dir: Path,
    llm_model,
    entity_types: list[str] | None = None,
    max_gleanings: int = 0,
) -> list[dict]:
    """Run GraphExtractor on every .txt file in *input_dir*.

    Returns a flat list of entity dicts, each augmented with:
      - ``source_file`` — the originating file name
      - ``id``          — a new UUID
    """
    if entity_types is None:
        entity_types = [
            "organization", "person", "geo", "event",
            "product", "technology",
        ]

    from datetime import datetime, timezone
    doc_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = TEMPORAL_GRAPH_EXTRACTION_PROMPT.replace("{document_date}", doc_date)

    extractor = GraphExtractor(
        model=llm_model,
        prompt=prompt,
        max_gleanings=max_gleanings,
    )

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    all_entities: list[dict] = []
    for txt_path in txt_files:
        print(f"  Extracting from {txt_path.name} …", file=sys.stderr)
        text = txt_path.read_text(encoding="utf-8")
        entities_df, _ = await extractor(
            text=text,
            entity_types=entity_types,
            source_id=txt_path.stem,
        )
        for _, row in entities_df.iterrows():
            entity = {
                "id":          str(uuid4()),
                "title":       row.get("title", ""),
                "type":        row.get("type", ""),
                "description": row.get("description", ""),
                "source_id":   row.get("source_id", txt_path.stem),
                "source_file": txt_path.name,
                # placeholders — filled in by _embed_entities
                "description_embedding": None,
                "text_unit_embedding":   None,
                "active_start": "",
                "active_end":   "",
                "relation_types": [],
            }
            all_entities.append(entity)

    print(f"  Extracted {len(all_entities)} entities from {len(txt_files)} files.",
          file=sys.stderr)
    return all_entities


async def _embed_entities(entities: list[dict], embedder) -> None:
    """Fill *description_embedding* and *text_unit_embedding* via the real model."""
    desc_texts  = [e["description"] for e in entities]
    title_texts = [e["title"]       for e in entities]
    all_texts   = desc_texts + title_texts

    print(f"  Embedding {len(entities)} entities ({len(all_texts)} texts) …",
          file=sys.stderr)
    response   = await embedder.embedding_async(input=all_texts)
    all_vecs: list[list[float]] = response.embeddings

    desc_vecs  = all_vecs[:len(entities)]
    title_vecs = all_vecs[len(entities):]
    for ent, dv, tv in zip(entities, desc_vecs, title_vecs):
        ent["description_embedding"] = dv
        ent["text_unit_embedding"]   = tv


# ---------------------------------------------------------------------------
# Ground-truth labelling via LLM
# ---------------------------------------------------------------------------

_LABEL_PROMPT = """\
You are an expert knowledge-graph analyst.
Determine whether the two entities below refer to the SAME real-world entity.
Reply with exactly one word: SAME or DIFFERENT.

Entity A:
  Name: {name_a}
  Type: {type_a}
  Description: {desc_a}

Entity B:
  Name: {name_b}
  Type: {type_b}
  Description: {desc_b}
"""


async def _llm_label_pair(entity_a: dict, entity_b: dict, llm_model) -> str:
    """Ask the LLM whether *entity_a* and *entity_b* refer to the same entity."""
    from graphrag_llm.utils import CompletionMessagesBuilder

    prompt = _LABEL_PROMPT.format(
        name_a=entity_a.get("title", ""),
        type_a=entity_a.get("type", ""),
        desc_a=entity_a.get("description", ""),
        name_b=entity_b.get("title", ""),
        type_b=entity_b.get("type", ""),
        desc_b=entity_b.get("description", ""),
    )
    builder  = CompletionMessagesBuilder().add_user_message(prompt)
    response = await llm_model.completion_async(messages=builder.build())
    answer   = response.content.strip().upper()
    if "SAME" in answer:
        return "SAME"
    return "DIFFERENT"


@dataclass
class LabelledPair:
    entity_a:    dict
    entity_b:    dict
    label:       str   # "SAME" or "DIFFERENT"
    description: str   # human-readable note


async def build_ground_truth_from_entities(
    entities: list[dict],
    llm_model,
    max_pairs: int | None = None,
) -> list[LabelledPair]:
    """Label cross-document entity pairs as SAME or DIFFERENT using the LLM.

    Strategy:
    - Pair every entity from file A with every entity from file B (cross-document).
    - Skip pairs where both entities come from the same source file (intra-doc).
    - Skip pairs whose types are obviously incompatible (person vs organization, etc.)
    - Use the LLM to assign SAME / DIFFERENT.

    The *max_pairs* limit caps API costs during development.
    """
    # Group by source file
    by_file: dict[str, list[dict]] = {}
    for ent in entities:
        by_file.setdefault(ent["source_file"], []).append(ent)

    files = list(by_file.keys())
    candidates: list[tuple[dict, dict]] = []

    # Build cross-document candidate pairs
    incompatible_groups = [
        {"person"},
        {"organization", "geo"},
        {"event", "concept", "product"},
    ]

    def _type_group(t: str) -> int:
        for g_idx, g in enumerate(incompatible_groups):
            if t in g:
                return g_idx
        return -1

    for i, fa in enumerate(files):
        for fb in files[i + 1:]:
            for ea in by_file[fa]:
                for eb in by_file[fb]:
                    # Skip clearly incompatible types
                    ta = ea.get("type", "").lower()
                    tb = eb.get("type", "").lower()
                    if ta and tb and ta != tb:
                        ga, gb = _type_group(ta), _type_group(tb)
                        if ga != gb and ga != -1 and gb != -1:
                            continue
                    candidates.append((ea, eb))

    if max_pairs is not None:
        # Shuffle deterministically so we get a representative sample
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(candidates))[:max_pairs]
        candidates = [candidates[i] for i in sorted(idx)]

    print(f"  Labelling {len(candidates)} cross-document entity pairs with LLM …",
          file=sys.stderr)

    pairs: list[LabelledPair] = []
    for k, (ea, eb) in enumerate(candidates):
        label = await _llm_label_pair(ea, eb, llm_model)
        desc  = f"{ea['title']} ({ea['source_file']}) vs {eb['title']} ({eb['source_file']})"
        pairs.append(LabelledPair(entity_a=ea, entity_b=eb, label=label, description=desc))
        if (k + 1) % 10 == 0:
            n_same = sum(1 for p in pairs if p.label == "SAME")
            print(f"    … {k + 1}/{len(candidates)} labelled  "
                  f"({n_same} SAME, {k+1-n_same} DIFFERENT)", file=sys.stderr)

    n_same = sum(1 for p in pairs if p.label == "SAME")
    print(f"  Ground truth: {len(pairs)} pairs ({n_same} SAME, "
          f"{len(pairs)-n_same} DIFFERENT)", file=sys.stderr)
    return pairs


# ---------------------------------------------------------------------------
# Evaluation engine (same as in eval_scorer_hyperparams.py)
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


def _evaluate_scorer(
    scorer_fn,
    scorer_name: str,
    config: BTGraphRAGConfig,
    pairs: list[LabelledPair],
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
        score, _ = scorer_fn(pair.entity_a, pair.entity_b, config)
        is_same  = pair.label == "SAME"
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
    pairs: list[LabelledPair],
    bounds: list[tuple[float, float]],
    param_names: list[str],
    make_config,  # Callable[[list[float]], tuple[BTGraphRAGConfig, dict]]
    lam: float = 0.2,
    popsize: int = 15,
    maxiter: int = 300,
    seed: int = 42,
) -> tuple[EvalResult, dict]:
    """Run Differential Evolution to maximise the scorer objective.

    Parameters
    ----------
    scorer_fn:   the scorer callable (ea, eb, config) → (float, dict)
    scorer_name: human-readable label
    pairs:       labelled ground-truth pairs
    bounds:      [(low, high)] for each parameter
    param_names: names matching *bounds*
    make_config: function from parameter list → (BTGraphRAGConfig, extra_params dict)
    lam:         lambda for the objective  F1+LLM − lam * LLM_rate
    popsize:     DE population size multiplier
    maxiter:     maximum DE iterations
    seed:        random seed for reproducibility
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        print("[WARN] scipy not available — falling back to coarse grid search.",
              file=sys.stderr)
        return _fallback_grid_search(scorer_fn, scorer_name, pairs, bounds,
                                     param_names, make_config, lam)

    call_count = [0]

    def _neg_objective(x: np.ndarray) -> float:
        """Negative objective (DE minimises)."""
        cfg, extra = make_config(list(x))
        mt = float(x[0])
        lt = float(x[1])
        # Enforce lt < mt (soft constraint — penalise violation)
        if lt >= mt:
            return 1.0
        result = _evaluate_scorer(
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
    # Clamp lt < mt
    if lt_best >= mt_best:
        lt_best = mt_best - 0.01

    best_result = _evaluate_scorer(
        scorer_fn, scorer_name, cfg_best, pairs,
        mt_best, lt_best,
        config_label=scorer_name,
        extra_params=extra_best,
    )
    best_params = dict(zip(param_names, best_x))
    print(f"  DE finished after {call_count[0]} evaluations. "
          f"Best objective={best_result.objective(lam):.4f}", file=sys.stderr)
    return best_result, best_params


def _fallback_grid_search(
    scorer_fn, scorer_name, pairs, bounds, param_names, make_config, lam
):
    """Coarse 10×10 grid search fallback when scipy is unavailable."""
    best_obj = -1.0
    best_result = None
    best_params: dict = {}
    mt_vals = np.linspace(bounds[0][0], bounds[0][1], 10)
    lt_vals = np.linspace(bounds[1][0], bounds[1][1], 10)
    other_mids = [(b[0] + b[1]) / 2 for b in bounds[2:]]
    for mt in mt_vals:
        for lt in lt_vals:
            if lt >= mt:
                continue
            x = [mt, lt] + other_mids
            cfg, extra = make_config(x)
            result = _evaluate_scorer(
                scorer_fn, scorer_name, cfg, pairs, float(mt), float(lt),
                extra_params=extra,
            )
            obj = result.objective(lam)
            if obj > best_obj:
                best_obj = obj
                best_result = result
                best_params = dict(zip(param_names, x))
    return best_result, best_params


# ---------------------------------------------------------------------------
# Scorer-specific DE setup
# ---------------------------------------------------------------------------

def _optimise_embedding_only(
    pairs: list[LabelledPair], lam: float, popsize: int, maxiter: int
) -> tuple[EvalResult, dict]:
    """DE for embedding_only scorer — 2 hyperparameters: mt, lt."""
    bounds = [(0.40, 0.99), (0.10, 0.75)]
    param_names = ["merge_threshold", "llm_threshold_low"]

    def make_config(_x: list[float]):
        return BTGraphRAGConfig(), {}

    return _run_de_optimise(
        embedding_only_entity_scorer, "emb_only",
        pairs, bounds, param_names, make_config,
        lam=lam, popsize=popsize, maxiter=maxiter,
    )


def _optimise_citation_desc(
    pairs: list[LabelledPair], lam: float, popsize: int, maxiter: int
) -> tuple[EvalResult, dict]:
    """DE for citation_and_description scorer — 3 hyperparameters: mt, lt, w_desc.

    w_cite is derived as (1 - w_desc) so the weights always sum to 1.
    The dynamic scorer is threaded through via a mutable container to avoid
    stale closure issues inside the DE callback.
    """
    bounds = [(0.40, 0.99), (0.10, 0.75), (0.0, 1.0)]
    param_names = ["merge_threshold", "llm_threshold_low", "w_desc"]

    # Mutable container so the DE callback always calls the freshest scorer.
    holder: dict = {"w_desc": 0.5, "w_cite": 0.5}

    def _dyn_scorer(ea, eb, cfg):
        return citation_and_description_entity_scorer(
            ea, eb, cfg,
            w_desc=holder["w_desc"],
            w_cite=holder["w_cite"],
        )

    def make_config(x: list[float]):
        w_desc = float(np.clip(x[2], 0.0, 1.0))
        w_cite = 1.0 - w_desc
        holder["w_desc"] = w_desc
        holder["w_cite"] = w_cite
        return BTGraphRAGConfig(), {"w_desc": round(w_desc, 4), "w_cite": round(w_cite, 4)}

    return _run_de_optimise(
        _dyn_scorer, "cite_desc",
        pairs, bounds, param_names, make_config,
        lam=lam, popsize=popsize, maxiter=maxiter,
    )


def _optimise_composite(
    pairs: list[LabelledPair], lam: float, popsize: int, maxiter: int
) -> tuple[EvalResult, dict]:
    """DE for composite (5-signal) scorer — 7 hyperparameters: mt, lt, w_emb, w_bm25, w_jacc, w_temp, w_rel."""
    # Weights are free-form; we normalise them inside to sum to 1
    bounds = [
        (0.40, 0.99),   # merge_threshold
        (0.10, 0.75),   # llm_threshold_low
        (0.0,  1.0),    # w_emb  (raw, normalised later)
        (0.0,  1.0),    # w_bm25
        (0.0,  1.0),    # w_jacc
        (0.0,  1.0),    # w_temp
        (0.0,  1.0),    # w_rel
    ]
    param_names = ["merge_threshold", "llm_threshold_low",
                   "w_emb", "w_bm25", "w_jacc", "w_temp", "w_rel"]

    def make_config(x: list[float]):
        raw = np.array(x[2:], dtype=float)
        total = raw.sum()
        if total < 1e-9:
            raw = np.ones(5) / 5
        else:
            raw = raw / total
        cfg = BTGraphRAGConfig(
            cger_embedding_weight=float(raw[0]),
            cger_bm25_weight=float(raw[1]),
            cger_jaccard_weight=float(raw[2]),
            cger_temporal_overlap_weight=float(raw[3]),
            cger_relation_context_weight=float(raw[4]),
        )
        extra = {
            "w_emb":  round(float(raw[0]), 4),
            "w_bm25": round(float(raw[1]), 4),
            "w_jacc": round(float(raw[2]), 4),
            "w_temp": round(float(raw[3]), 4),
            "w_rel":  round(float(raw[4]), 4),
        }
        return cfg, extra

    return _run_de_optimise(
        compute_entity_composite_score, "composite",
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
    pairs: list[LabelledPair],
    emb_result: EvalResult, emb_params: dict,
    cite_result: EvalResult, cite_params: dict,
    comp_result: EvalResult, comp_params: dict,
    lam: float,
    output_path: str | None = None,
) -> str:
    if output_path is None:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(__file__).parent / f"eval_real_results_{ts}.json")

    gt_summary = [
        {
            "index":    i + 1,
            "entity_a": p.entity_a.get("title", ""),
            "entity_b": p.entity_b.get("title", ""),
            "label":    p.label,
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
            "embedding_only":           _result_to_dict(emb_result,  emb_params),
            "citation_and_description": _result_to_dict(cite_result, cite_params),
            "composite_5signal":        _result_to_dict(comp_result, comp_params),
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

    _cger_input_env = os.environ.get("CGER_INPUT_DIR", "").strip()
    input_dir = Path(_cger_input_env) if _cger_input_env else None
    if not input_dir or not input_dir.is_dir():
        # Default 1: input/ next to this script (committed sample texts)
        local_input = _SCRIPT_DIR / "input"
        # Default 2: ragtest/input/cger/ (user workspace)
        ragtest_input = _REPO_ROOT / "ragtest" / "input" / "cger"
        input_dir = local_input if local_input.is_dir() else ragtest_input
    max_pairs  = int(os.environ.get("CGER_MAX_PAIRS", "0")) or None
    lam        = float(os.environ.get("CGER_LAMBDA",   "0.2"))
    de_popsize = int(os.environ.get("CGER_DE_POPSIZE", "15"))
    de_maxiter = int(os.environ.get("CGER_DE_MAXITER", "300"))

    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║   CGER Real-Data Evaluation + Metaheuristic Hyperparameter Optimisation     ║")
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
    embedder  = _real_embedder()

    # ── Step 2: Extract entities from txt files ──────────────────────────────
    print("\n[2/4] Extracting entities from txt files …", file=sys.stderr)
    entities = await extract_entities_from_files(input_dir, llm_model)
    await _embed_entities(entities, embedder)

    if not entities:
        print("[ERROR] No entities extracted — check your txt files and API key.",
              file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Build ground-truth pairs ─────────────────────────────────────
    print("\n[3/4] Labelling cross-document entity pairs with LLM …", file=sys.stderr)
    pairs = await build_ground_truth_from_entities(entities, llm_model, max_pairs=max_pairs)

    if len(pairs) < 4:
        print("[ERROR] Too few labelled pairs to run optimisation "
              "(need ≥ 4). Add more txt files or entities.", file=sys.stderr)
        sys.exit(1)

    # ── Step 4: Metaheuristic optimisation per scorer ────────────────────────
    print("\n[4/4] Running Differential Evolution per scorer …", file=sys.stderr)

    _sep("SCORER 1 — embedding_only_entity_scorer")
    emb_result, emb_params = _optimise_embedding_only(pairs, lam, de_popsize, de_maxiter)
    _print_best(emb_result, emb_params, "embedding_only_entity_scorer", lam)

    _sep("SCORER 2 — citation_and_description_entity_scorer")
    cite_result, cite_params = _optimise_citation_desc(pairs, lam, de_popsize, de_maxiter)
    _print_best(cite_result, cite_params, "citation_and_description_entity_scorer", lam)

    _sep("SCORER 3 — compute_entity_composite_score (5-signal)")
    comp_result, comp_params = _optimise_composite(pairs, lam, de_popsize, de_maxiter)
    _print_best(comp_result, comp_params, "compute_entity_composite_score", lam)

    # ── Cross-scorer summary ──────────────────────────────────────────────────
    _sep("CROSS-SCORER COMPARISON")
    print(f"\n  {'Scorer':35s} {'F1+LLM':>8} {'LLM%':>7} {'Acc+LLM':>8} {'Objective':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*7} {'-'*8} {'-'*10}")
    for name, result in [
        ("embedding_only",           emb_result),
        ("citation_and_description", cite_result),
        ("composite_5signal",        comp_result),
    ]:
        print(
            f"  {name:35s} {result.f1_with_llm:8.4f} "
            f"{result.llm_call_rate:6.1%} {result.accuracy_with_llm:8.4f} "
            f"{result.objective(lam):10.4f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_json(
        pairs,
        emb_result,  emb_params,
        cite_result, cite_params,
        comp_result, comp_params,
        lam,
    )

    print("\n" + "═" * 80)
    print("  CGER real-data evaluation complete.")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
