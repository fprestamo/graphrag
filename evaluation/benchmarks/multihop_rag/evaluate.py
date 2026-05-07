# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Run BT-GraphRAG against the MultiHop-RAG benchmark."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.core.dataset import EvalPrediction
from evaluation.core.metrics import aggregate_metrics, crag_truthfulness_score
from evaluation.core.runner_cli import BenchmarkSpec, run_benchmark


def _per_query_type(predictions: list[EvalPrediction]) -> dict[str, Any]:
    metrics_by: dict[str, list[dict[str, float]]] = defaultdict(list)
    labels_by: dict[str, list[str]] = defaultdict(list)
    for p in predictions:
        ctx = p.context or {}
        qt = str(ctx.get("query_type") or "unknown")
        metrics_by[qt].append(p.metrics)
        labels_by[qt].append(p.judge_label or "")
    return {
        "by_query_type": {
            k: {
                **aggregate_metrics(metrics_by[k]),
                "truthfulness": crag_truthfulness_score(labels_by[k]),
            }
            for k in sorted(metrics_by)
        }
    }


SPEC = BenchmarkSpec(
    name="multihop_rag",
    description="MultiHop-RAG news multi-hop QA – accuracy by query type.",
    default_dataset=Path("evaluation/benchmarks/multihop_rag/data/multihop_rag.jsonl"),
    extra_report_fn=_per_query_type,
)


if __name__ == "__main__":
    run_benchmark(SPEC)
