# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Run BT-GraphRAG against HotpotQA distractor dev set."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.core.dataset import EvalPrediction
from evaluation.core.metrics import aggregate_metrics
from evaluation.core.runner_cli import BenchmarkSpec, run_benchmark


def _per_type_breakdown(predictions: list[EvalPrediction]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_level: dict[str, list[dict[str, float]]] = defaultdict(list)
    for p in predictions:
        ctx = p.context or {}
        t = ctx.get("type")
        l = ctx.get("level")
        if t:
            by_type[str(t)].append(p.metrics)
        if l:
            by_level[str(l)].append(p.metrics)

    return {
        "by_type": {k: aggregate_metrics(v) for k, v in sorted(by_type.items())},
        "by_level": {k: aggregate_metrics(v) for k, v in sorted(by_level.items())},
    }


SPEC = BenchmarkSpec(
    name="hotpotqa",
    description="HotpotQA distractor multi-hop QA – EM, token-F1 and judge accuracy.",
    default_dataset=Path("evaluation/benchmarks/hotpotqa/data/hotpotqa.jsonl"),
    extra_report_fn=_per_type_breakdown,
)


if __name__ == "__main__":
    run_benchmark(SPEC)
