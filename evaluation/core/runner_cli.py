# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Reusable evaluation harness shared by every benchmark.

Each benchmark only has to:

1. Provide a :class:`BenchmarkSpec` describing how to load its dataset and how
   to score a single prediction.
2. Call :func:`run_benchmark` from its ``evaluate.py`` ``__main__``.

The harness takes care of CLI parsing, JSONL IO, async orchestration,
optional rate-limiting, per-benchmark and aggregate metric computation, and
report generation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from evaluation.core.btgraphrag_runner import RunnerConfig, answer_question
from evaluation.core.dataset import (
    EvalPrediction,
    EvalRecord,
    iter_records,
    save_jsonl,
)
from evaluation.core.llm_judge import judge_answer
from evaluation.core.metrics import (
    aggregate_metrics,
    crag_truthfulness_score,
    exact_match,
    token_f1,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark specification
# ---------------------------------------------------------------------------

ScoringFn = Callable[[EvalRecord, str, "JudgeLabel"], dict[str, float]]
"""Per-example scorer signature: (record, prediction, judge_label) -> metrics."""

JudgeLabel = str  # "correct" | "incorrect" | "missing"


def default_scorer(
    record: EvalRecord, prediction: str, judge_label: JudgeLabel
) -> dict[str, float]:
    """Default per-example scorer: EM, F1 and judge-derived correctness."""
    golds = [record.answer, *record.aliases]
    return {
        "exact_match": exact_match(prediction, golds),
        "f1": token_f1(prediction, golds),
        "judge_correct": 1.0 if judge_label == "correct" else 0.0,
        "judge_missing": 1.0 if judge_label == "missing" else 0.0,
        "judge_incorrect": 1.0 if judge_label == "incorrect" else 0.0,
    }


@dataclass
class BenchmarkSpec:
    name: str
    """Benchmark name (used in output filenames and reports)."""

    default_dataset: Path
    """Path to the default JSONL dataset file (relative to repo root or absolute)."""

    description: str = ""
    scorer: ScoringFn = default_scorer
    use_llm_judge: bool = True
    """Whether to call the LLM judge (``False`` skips and uses the heuristic)."""

    extra_report_fn: Callable[[list[EvalPrediction]], dict[str, Any]] | None = None
    """Optional benchmark-specific report extension (e.g. per-query-type accuracy)."""


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------


async def _process_record(
    record: EvalRecord,
    spec: BenchmarkSpec,
    runner_cfg: RunnerConfig,
) -> EvalPrediction:
    qtime = None
    if record.question_time:
        try:
            qtime = datetime.fromisoformat(record.question_time.replace("Z", "+00:00"))
        except ValueError:
            qtime = None

    result = await answer_question(
        record.question, config=runner_cfg, question_time=qtime
    )
    prediction = result.get("answer", "") or ""

    judge = await judge_answer(
        record.question,
        record.answer,
        prediction,
        use_llm=spec.use_llm_judge,
    )

    metrics = spec.scorer(record, prediction, judge.label)
    return EvalPrediction(
        qid=record.qid,
        question=record.question,
        gold_answer=record.answer,
        prediction=prediction,
        judge_label=judge.label,
        judge_reason=judge.reason,
        metrics=metrics,
        context=dict(record.context),
        raw=result.get("raw", {}),
    )


async def _run_async(
    spec: BenchmarkSpec,
    dataset_path: Path,
    out_dir: Path,
    *,
    limit: int | None,
    concurrency: int,
    runner_cfg: RunnerConfig,
) -> dict[str, Any]:
    records = list(iter_records(dataset_path))
    if limit is not None:
        records = records[:limit]

    if not records:
        raise ValueError(f"No records found in dataset {dataset_path}")

    sem = asyncio.Semaphore(max(concurrency, 1))

    async def _bounded(rec: EvalRecord) -> EvalPrediction:
        async with sem:
            return await _process_record(rec, spec, runner_cfg)

    started = time.time()
    predictions: list[EvalPrediction] = await asyncio.gather(
        *(_bounded(r) for r in records)
    )
    elapsed = time.time() - started

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    save_jsonl(pred_path, predictions)

    aggregated = aggregate_metrics(p.metrics for p in predictions)
    crag_stats = crag_truthfulness_score(p.judge_label or "" for p in predictions)

    report: dict[str, Any] = {
        "benchmark": spec.name,
        "dataset": str(dataset_path),
        "num_examples": len(predictions),
        "elapsed_seconds": round(elapsed, 2),
        "metrics": aggregated,
        "crag_truthfulness": crag_stats,
        "runner": {
            "neo4j_database": runner_cfg.neo4j_database,
            "model_id": runner_cfg.model_id,
            "dry_run": runner_cfg.dry_run,
        },
    }

    if spec.extra_report_fn is not None:
        try:
            report["extra"] = spec.extra_report_fn(predictions)
        except Exception as exc:  # pragma: no cover
            logger.warning("extra_report_fn failed: %s", exc)

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Wrote %s and %s", pred_path, report_path)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser(spec: BenchmarkSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Evaluate BT-GraphRAG on the {spec.name} benchmark.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=spec.default_dataset,
        help=f"Path to the {spec.name} JSONL dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/results") / spec.name,
        help="Output directory for predictions.jsonl and report.json.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N records."
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Async concurrency for queries."
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-as-judge and use only the heuristic scorer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not contact Neo4j / the LLM; emit empty predictions (smoke test).",
    )
    return parser


def run_benchmark(spec: BenchmarkSpec) -> None:
    """Entrypoint that benchmarks call from their ``__main__`` block."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser(spec).parse_args()

    if args.no_judge:
        spec.use_llm_judge = False

    runner_cfg = RunnerConfig(dry_run=args.dry_run)

    report = asyncio.run(
        _run_async(
            spec,
            dataset_path=args.dataset,
            out_dir=args.out,
            limit=args.limit,
            concurrency=args.concurrency,
            runner_cfg=runner_cfg,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
