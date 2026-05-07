# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Lexical metrics shared across benchmarks.

These are *complementary* to the LLM-as-judge scoring (:mod:`evaluation.core.llm_judge`):
EM / F1 are cheap and reproducible while the LLM judge captures semantic
equivalence required by long-form generation benchmarks (CRAG, NarrativeQA).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Normalization (SQuAD / HotpotQA / MuSiQue style)
# ---------------------------------------------------------------------------


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation/articles and collapse whitespace.

    Mirrors the normalization used by SQuAD, HotpotQA and MuSiQue eval scripts
    so the numbers we report are directly comparable to published baselines.
    """
    if s is None:
        return ""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


# ---------------------------------------------------------------------------
# Per-example metrics
# ---------------------------------------------------------------------------


def exact_match(prediction: str, golds: str | Sequence[str]) -> float:
    """1.0 if the normalized prediction matches any gold answer, else 0.0."""
    if isinstance(golds, str):
        golds = [golds]
    pred = normalize_answer(prediction)
    return 1.0 if any(pred == normalize_answer(g) for g in golds) else 0.0


def token_f1(prediction: str, golds: str | Sequence[str]) -> float:
    """Maximum token-level F1 score across the provided gold answers."""
    if isinstance(golds, str):
        golds = [golds]
    pred_toks = _tokens(prediction)
    if not pred_toks and all(not _tokens(g) for g in golds):
        return 1.0
    best = 0.0
    for g in golds:
        gold_toks = _tokens(g)
        if not pred_toks or not gold_toks:
            continue
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


def accuracy(predictions: Iterable[str], golds: Iterable[str | Sequence[str]]) -> float:
    """Mean exact-match accuracy across a stream of (prediction, gold) pairs."""
    n = 0
    s = 0.0
    for p, g in zip(predictions, golds):
        s += exact_match(p, g)
        n += 1
    return s / n if n else 0.0


# ---------------------------------------------------------------------------
# CRAG truthfulness score
# ---------------------------------------------------------------------------
#
# CRAG (Yang et al., 2024) defines the truthfulness score as
#
#     S = ( #correct - #hallucinated ) / #total
#
# where each answer is labelled by an LLM judge as one of ``correct``,
# ``incorrect`` (= hallucinated) or ``missing`` (refusal / "I don't know").
#
# See: https://github.com/facebookresearch/CRAG


def crag_truthfulness_score(labels: Iterable[str]) -> dict[str, float]:
    """Compute CRAG-style truthfulness statistics from judge labels."""
    counts = {"correct": 0, "incorrect": 0, "missing": 0}
    total = 0
    for lbl in labels:
        total += 1
        key = (lbl or "").strip().lower()
        if key not in counts:
            # Treat unknown labels conservatively as ``incorrect``.
            key = "incorrect"
        counts[key] += 1
    if total == 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "hallucination_rate": 0.0,
            "missing_rate": 0.0,
            "truthfulness_score": 0.0,
        }
    return {
        "total": total,
        "accuracy": counts["correct"] / total,
        "hallucination_rate": counts["incorrect"] / total,
        "missing_rate": counts["missing"] / total,
        "truthfulness_score": (counts["correct"] - counts["incorrect"]) / total,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_metrics(per_example: Iterable[dict[str, float]]) -> dict[str, float]:
    """Average each metric key across a stream of per-example metric dicts."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for m in per_example:
        for k, v in m.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            sums[k] = sums.get(k, 0.0) + fv
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums if counts[k]}
