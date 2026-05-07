# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Run BT-GraphRAG against MuSiQue (answerable dev split)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.core.dataset import EvalPrediction
from evaluation.core.metrics import aggregate_metrics
from evaluation.core.runner_cli import BenchmarkSpec, run_benchmark


def _per_hop_breakdown(predictions: list[EvalPrediction]) -> dict[str, Any]:
    by_hops: dict[str, list[dict[str, float]]] = defaultdict(list)
    for p in predictions:
        ctx = p.context or {}
        n = ctx.get("num_hops")
        if n:
            by_hops[f"{n}-hop"].append(p.metrics)
    return {"by_num_hops": {k: aggregate_metrics(v) for k, v in sorted(by_hops.items())}}


SPEC = BenchmarkSpec(
    name="musique",
    description="MuSiQue 2-/3-/4-hop QA – EM, token-F1 and judge accuracy.",
    default_dataset=Path("evaluation/benchmarks/musique/data/musique.jsonl"),
    extra_report_fn=_per_hop_breakdown,
)


if __name__ == "__main__":
    run_benchmark(SPEC)
