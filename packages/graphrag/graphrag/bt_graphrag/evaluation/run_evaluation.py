#!/usr/bin/env python3
"""CGER + CGRR evaluation pipeline — main entry point.

Each component (CGER, CGRR) is completely independent:
  • its own ground truth
  • its own documents / data folder
  • its own scorer registry
  • its own metaheuristic optimizer
  • its own results folder
They all share the same hypothesis-testing engine from ``shared/``.

Usage
-----
# Quick smoke-test with mock embeddings, skip optimization
python -m graphrag.bt_graphrag.evaluation.run_evaluation --mock --skip-optimize

# Run only CGER with real embeddings
python -m graphrag.bt_graphrag.evaluation.run_evaluation --component cger --model text-embedding-3-small

# Full run: both components, all optimizers
python -m graphrag.bt_graphrag.evaluation.run_evaluation --mock
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import time
from typing import Any

EVAL_DIR = pathlib.Path(__file__).parent


# ===========================================================================
# Embedding helpers
# ===========================================================================

async def _get_embedding_fn(use_mock: bool, model_id: str):
    if use_mock:
        from graphrag.bt_graphrag.evaluation.shared.mock_embeddings import mock_embedding_fn
        return mock_embedding_fn

    try:
        from graphrag_llm.embedding import create_embedding_model
        emb = create_embedding_model(model_id=model_id)

        async def _real(texts):
            out = []
            for i in range(0, len(texts), 100):
                out.extend(await emb.embedding_async(texts[i:i + 100]))
            return out
        return _real
    except ImportError:
        import openai
        client = openai.AsyncOpenAI()

        async def _openai(texts):
            out = []
            for i in range(0, len(texts), 100):
                resp = await client.embeddings.create(input=texts[i:i + 100], model=model_id)
                out.extend(d.embedding for d in resp.data)
            return out
        return _openai


# ===========================================================================
# Per-component runners
# ===========================================================================

async def run_cger(
    embedding_fn,
    skip_optimize: bool,
    methods: list[str],
    de_iters: int, sa_iters: int, pso_iters: int,
) -> dict[str, Any]:
    """Full CGER evaluation pipeline — independent data, results, scorers."""
    from graphrag.bt_graphrag.evaluation.cger.ground_truth import (
        build_ground_truth, write_ground_truth,
    )
    from graphrag.bt_graphrag.evaluation.cger.harness import (
        build_entity_dicts, evaluate_scorer,
    )
    from graphrag.bt_graphrag.evaluation.cger.scorers import SCORER_REGISTRY
    from graphrag.bt_graphrag.evaluation.cger.optimizer import (
        optimize, SEARCH_SPACES, decode_params, normalize_weights,
    )
    from graphrag.bt_graphrag.evaluation.shared.hypothesis_testing import run_hypothesis_tests
    from graphrag.bt_graphrag.models.config import BTGraphRAGConfig

    gt_dir      = EVAL_DIR / "cger" / "data" / "ground_truth"
    results_dir = EVAL_DIR / "cger" / "results"
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "━" * 65)
    print("  CGER  — Cross-Graph Entity Resolution")
    print("━" * 65)

    # 1. Ground truth
    write_ground_truth(str(gt_dir))
    gt = build_ground_truth()
    print(f"  Ground truth: {gt['total_positive_pairs']} positive  "
          f"+ {gt['total_negative_pairs']} negative pairs")

    # 2. Document texts for text_unit_embeddings
    doc_dir = EVAL_DIR / "cger" / "data" / "documents"
    doc_texts: dict[str, list[str]] = {}
    if doc_dir.exists():
        from graphrag.bt_graphrag.evaluation.generate_documents import DOCUMENTS, ALIAS_MAP
        for doc in DOCUMENTS:
            for sf in doc["entities_used"]:
                doc_texts.setdefault(sf.upper(), []).append(doc["text"][:500])

    # 3. Entity embeddings
    entities = await build_entity_dicts(gt, embedding_fn, doc_texts)

    # 4. Optimize
    optimized: dict[str, dict] = {}
    opt_log: dict[str, list] = {}

    if not skip_optimize:
        for sname in SCORER_REGISTRY:
            opt_results = await optimize(
                sname, embedding_fn, ground_truth=gt,
                methods=methods,
                de_iters=de_iters, sa_iters=sa_iters, pso_iters=pso_iters,
            )
            best = max(opt_results, key=lambda r: r.best_f1)
            optimized[sname] = best.best_params
            opt_log[sname]   = [r.to_dict() for r in opt_results]
            print(f"  [CGER] Best for {sname}: {best.method.upper()} F1={best.best_f1:.4f}")

    # Helper: rebuild config from best_params
    def _make_config(sname: str) -> BTGraphRAGConfig:
        if sname not in optimized:
            return BTGraphRAGConfig()
        space = SEARCH_SPACES[sname]
        cfg = BTGraphRAGConfig()
        for b in space:
            val = optimized[sname].get(b.name)
            if val is not None:
                setattr(cfg, b.config_field, val)
        if cfg.cger_llm_threshold_low >= cfg.cger_merge_threshold:
            cfg.cger_llm_threshold_low = cfg.cger_merge_threshold - 0.05
        if sname == "composite_5signal":
            cfg = normalize_weights(cfg)
        return cfg

    # 5. Evaluate
    scorer_results: dict[str, Any] = {}
    for sname, sfn in SCORER_REGISTRY.items():
        cfg = _make_config(sname)
        label = "optimized" if sname in optimized else "default"
        print(f"\n  Evaluating [{sname}] ({label}) "
              f"threshold={cfg.cger_merge_threshold:.3f}")
        res = evaluate_scorer(sfn, sname, entities, gt["expected_merges"], cfg)
        scorer_results[sname] = res
        print(f"    P={res.precision:.4f}  R={res.recall:.4f}  "
              f"F1={res.f1:.4f}  Acc={res.accuracy:.4f}  "
              f"TP={res.true_positives}  FP={res.false_positives}  "
              f"FN={res.false_negatives}  TN={res.true_negatives}")

    # 6. Hypothesis tests
    print("\n  [CGER] Running hypothesis tests…")
    report = run_hypothesis_tests(scorer_results, component="CGER")
    _print_test_summary(report)

    # 7. Save
    full = {
        "component": "CGER",
        "optimization": opt_log,
        "evaluations": {n: r.to_dict() for n, r in scorer_results.items()},
        "hypothesis_tests": report.to_dict(),
    }
    (results_dir / "cger_report.json").write_text(json.dumps(full, indent=2))
    for sname, res in scorer_results.items():
        (results_dir / f"pair_details_{sname}.json").write_text(
            json.dumps(res.pair_details, indent=2)
        )
    print(f"\n  [CGER] Results saved to {results_dir}/")
    return full


async def run_cgrr(
    skip_optimize: bool,
    methods: list[str],
    de_iters: int, sa_iters: int, pso_iters: int,
) -> dict[str, Any]:
    """Full CGRR evaluation pipeline — independent data, results, scorers."""
    from graphrag.bt_graphrag.evaluation.cgrr.ground_truth import (
        build_ground_truth, write_ground_truth,
    )
    from graphrag.bt_graphrag.evaluation.cgrr.harness import evaluate_scorer
    from graphrag.bt_graphrag.evaluation.cgrr.scorers import SCORER_REGISTRY
    from graphrag.bt_graphrag.evaluation.cgrr.optimizer import (
        optimize, SEARCH_SPACES, decode_params, normalize_cgrr_weights,
    )
    from graphrag.bt_graphrag.evaluation.shared.hypothesis_testing import run_hypothesis_tests
    from graphrag.bt_graphrag.models.config import BTGraphRAGConfig

    gt_dir      = EVAL_DIR / "cgrr" / "data" / "ground_truth"
    results_dir = EVAL_DIR / "cgrr" / "results"
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "━" * 65)
    print("  CGRR  — Cross-Graph Relationship Resolution")
    print("━" * 65)

    # 1. Ground truth
    write_ground_truth(str(gt_dir))
    gt = build_ground_truth()
    print(f"  Ground truth: {gt['total_positive_pairs']} positive  "
          f"+ {gt['total_negative_pairs']} negative pairs  "
          f"({gt['total_canonical_relations']} canonical relation types)")

    # 2. Optimize
    optimized: dict[str, dict] = {}
    opt_log: dict[str, list] = {}

    if not skip_optimize:
        for sname in SCORER_REGISTRY:
            opt_results = await optimize(
                sname, ground_truth=gt,
                methods=methods,
                de_iters=de_iters, sa_iters=sa_iters, pso_iters=pso_iters,
            )
            best = max(opt_results, key=lambda r: r.best_f1)
            optimized[sname] = best.best_params
            opt_log[sname]   = [r.to_dict() for r in opt_results]
            print(f"  [CGRR] Best for {sname}: {best.method.upper()} F1={best.best_f1:.4f}")

    # Helper: rebuild config from best_params
    def _make_config(sname: str) -> BTGraphRAGConfig:
        if sname not in optimized:
            return BTGraphRAGConfig()
        space = SEARCH_SPACES[sname]
        cfg = BTGraphRAGConfig()
        for b in space:
            val = optimized[sname].get(b.name)
            if val is not None:
                setattr(cfg, b.config_field, val)
        if cfg.cgrr_llm_threshold_low >= cfg.cgrr_merge_threshold:
            cfg.cgrr_llm_threshold_low = cfg.cgrr_merge_threshold - 0.05
        if sname == "composite_3signal":
            cfg = normalize_cgrr_weights(cfg)
        return cfg

    # 3. Evaluate
    scorer_results: dict[str, Any] = {}
    for sname, sfn in SCORER_REGISTRY.items():
        cfg = _make_config(sname)
        label = "optimized" if sname in optimized else "default"
        print(f"\n  Evaluating [{sname}] ({label}) "
              f"threshold={cfg.cgrr_merge_threshold:.3f}")
        res = evaluate_scorer(sfn, sname, gt, cfg)
        scorer_results[sname] = res
        print(f"    P={res.precision:.4f}  R={res.recall:.4f}  "
              f"F1={res.f1:.4f}  Acc={res.accuracy:.4f}  "
              f"TP={res.true_positives}  FP={res.false_positives}  "
              f"FN={res.false_negatives}  TN={res.true_negatives}")

    # 4. Hypothesis tests
    print("\n  [CGRR] Running hypothesis tests…")
    report = run_hypothesis_tests(scorer_results, component="CGRR")
    _print_test_summary(report)

    # 5. Save
    full = {
        "component": "CGRR",
        "optimization": opt_log,
        "evaluations": {n: r.to_dict() for n, r in scorer_results.items()},
        "hypothesis_tests": report.to_dict(),
    }
    (results_dir / "cgrr_report.json").write_text(json.dumps(full, indent=2))
    for sname, res in scorer_results.items():
        (results_dir / f"pair_details_{sname}.json").write_text(
            json.dumps(res.pair_details, indent=2)
        )
    print(f"\n  [CGRR] Results saved to {results_dir}/")
    return full


# ===========================================================================
# Shared helpers
# ===========================================================================

def _print_test_summary(report) -> None:
    for t in report.tests:
        sig = " ***" if t.significant else ""
        print(f"    {t.test_name:20s} {t.scorer_a} vs {t.scorer_b or 'all':30s} "
              f"p={t.p_value:.4f}{sig}")
    print(f"  Summary: best_f1={report.summary['best_by_f1']}  "
          f"best_acc={report.summary['best_by_accuracy']}  "
          f"significant={report.summary['significant']}/{report.summary['total_tests']}")


def _print_final_banner(all_reports: dict[str, dict], total_s: float) -> None:
    print("\n" + "═" * 65)
    print("  FINAL SUMMARY")
    print("═" * 65)
    for comp, rep in all_reports.items():
        print(f"\n  {comp}")
        evals = rep.get("evaluations", {})
        ranking = sorted(evals.items(), key=lambda x: x[1].get("f1", 0), reverse=True)
        for rank, (sname, metrics) in enumerate(ranking, 1):
            tag = " (optimized)" if rep.get("optimization", {}).get(sname) else " (default)"
            print(f"    #{rank}  {sname}{tag}")
            print(f"         F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
                  f"R={metrics['recall']:.4f}  Acc={metrics['accuracy']:.4f}")
        winner_name = ranking[0][0] if ranking else "—"
        winner_f1   = ranking[0][1].get("f1", 0) if ranking else 0
        print(f"  → Winner: {winner_name}  F1={winner_f1:.4f}")
    print(f"\n  Total time: {total_s:.1f}s")
    print("═" * 65)


# ===========================================================================
# CLI
# ===========================================================================

async def main_async(args) -> None:
    t0 = time.time()
    embedding_fn = await _get_embedding_fn(args.mock, args.model)
    components = [c.lower() for c in args.component] if args.component else ["cger", "cgrr"]
    methods = args.methods or ["de", "sa", "pso"]
    all_reports: dict[str, dict] = {}

    if "cger" in components:
        all_reports["CGER"] = await run_cger(
            embedding_fn,
            skip_optimize=args.skip_optimize,
            methods=methods,
            de_iters=args.de_iter,
            sa_iters=args.sa_iter,
            pso_iters=args.pso_iter,
        )

    if "cgrr" in components:
        all_reports["CGRR"] = await run_cgrr(
            skip_optimize=args.skip_optimize,
            methods=methods,
            de_iters=args.de_iter,
            sa_iters=args.sa_iter,
            pso_iters=args.pso_iter,
        )

    _print_final_banner(all_reports, time.time() - t0)


def main():
    p = argparse.ArgumentParser(description="CGER + CGRR scorer evaluation pipeline")
    p.add_argument("--mock",          action="store_true",
                   help="Use deterministic mock embeddings (no API key needed)")
    p.add_argument("--model",         default="text-embedding-3-small",
                   help="Embedding model for real runs")
    p.add_argument("--component",     nargs="+", choices=["cger", "cgrr"],
                   help="Which component(s) to evaluate (default: both)")
    p.add_argument("--skip-optimize", action="store_true",
                   help="Skip metaheuristic optimization, use default params")
    p.add_argument("--methods",       nargs="+", choices=["de", "sa", "pso"],
                   help="Optimization methods (default: all three)")
    p.add_argument("--de-iter",  type=int, default=80)
    p.add_argument("--sa-iter",  type=int, default=1500)
    p.add_argument("--pso-iter", type=int, default=80)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
