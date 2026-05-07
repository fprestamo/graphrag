# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Shared evaluation utilities (runner, dataset, metrics, LLM-as-judge)."""

from evaluation.core.dataset import EvalRecord, EvalPrediction, load_jsonl, save_jsonl
from evaluation.core.metrics import (
    exact_match,
    token_f1,
    accuracy,
    crag_truthfulness_score,
    aggregate_metrics,
)

__all__ = [
    "EvalRecord",
    "EvalPrediction",
    "load_jsonl",
    "save_jsonl",
    "exact_match",
    "token_f1",
    "accuracy",
    "crag_truthfulness_score",
    "aggregate_metrics",
]
