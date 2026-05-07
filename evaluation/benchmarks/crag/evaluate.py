# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Run BT-GraphRAG against CRAG and report the truthfulness score.

Usage::

    python -m evaluation.benchmarks.crag.evaluate \\
        --dataset evaluation/benchmarks/crag/data/crag.jsonl \\
        --out     evaluation/results/crag

Outputs::

    <out>/predictions.jsonl   – per-example predictions and judge labels
    <out>/report.json         – aggregate metrics including the CRAG
                                 truthfulness score and per-domain accuracy
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.core.dataset import EvalPrediction
from evaluation.core.runner_cli import BenchmarkSpec, run_benchmark


def _per_category_breakdown(predictions: list[EvalPrediction]) -> dict[str, Any]:
    """Compute accuracy and truthfulness broken down by CRAG metadata."""
    by_domain: dict[str, list[str]] = defaultdict(list)
    by_qtype: dict[str, list[str]] = defaultdict(list)
    by_dynamics: dict[str, list[str]] = defaultdict(list)

    for p in predictions:
        ctx = p.context or {}
        if "domain" in ctx:
            by_domain[str(ctx["domain"])].append(p.judge_label or "")
        if "question_type" in ctx:
            by_qtype[str(ctx["question_type"])].append(p.judge_label or "")
        if "static_or_dynamic" in ctx:
            by_dynamics[str(ctx["static_or_dynamic"])].append(p.judge_label or "")

    from evaluation.core.metrics import crag_truthfulness_score

    def _agg(buckets: dict[str, list[str]]) -> dict[str, dict[str, float]]:
        return {k: crag_truthfulness_score(v) for k, v in sorted(buckets.items())}

    return {
        "by_domain": _agg(by_domain),
        "by_question_type": _agg(by_qtype),
        "by_static_or_dynamic": _agg(by_dynamics),
    }


SPEC = BenchmarkSpec(
    name="crag",
    description="Comprehensive RAG Benchmark (Meta, 2024) – truthfulness score.",
    default_dataset=Path("evaluation/benchmarks/crag/data/crag.jsonl"),
    use_llm_judge=True,
    extra_report_fn=_per_category_breakdown,
)


if __name__ == "__main__":
    run_benchmark(SPEC)
