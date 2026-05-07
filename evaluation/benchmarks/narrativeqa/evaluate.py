# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Run BT-GraphRAG against the NarrativeQA test split.

NarrativeQA is a long-form QA benchmark, so the headline metric is token-F1
combined with the LLM-as-judge correctness label.  Optional ROUGE-L is
reported when the ``rouge-score`` package is available.
"""

from __future__ import annotations

import logging
from pathlib import Path

from evaluation.core.dataset import EvalPrediction, EvalRecord
from evaluation.core.metrics import (
    aggregate_metrics,
    exact_match,
    token_f1,
)
from evaluation.core.runner_cli import BenchmarkSpec, run_benchmark

logger = logging.getLogger(__name__)


def _try_rouge_l(pred: str, golds: list[str]) -> float | None:
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    best = 0.0
    for g in golds:
        if not g:
            continue
        score = scorer.score(g, pred)["rougeL"].fmeasure
        if score > best:
            best = score
    return best


def narrative_scorer(
    record: EvalRecord, prediction: str, judge_label: str
) -> dict[str, float]:
    golds = [record.answer, *record.aliases]
    metrics: dict[str, float] = {
        "exact_match": exact_match(prediction, golds),
        "f1": token_f1(prediction, golds),
        "judge_correct": 1.0 if judge_label == "correct" else 0.0,
        "judge_missing": 1.0 if judge_label == "missing" else 0.0,
        "judge_incorrect": 1.0 if judge_label == "incorrect" else 0.0,
    }
    rouge_l = _try_rouge_l(prediction, golds)
    if rouge_l is not None:
        metrics["rouge_l"] = rouge_l
    return metrics


SPEC = BenchmarkSpec(
    name="narrativeqa",
    description="NarrativeQA long-form QA – token-F1, ROUGE-L (optional) and judge accuracy.",
    default_dataset=Path("evaluation/benchmarks/narrativeqa/data/narrativeqa.jsonl"),
    scorer=narrative_scorer,
)


if __name__ == "__main__":
    run_benchmark(SPEC)
