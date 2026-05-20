"""SQuAD-style and TimeQA-native metrics + accuracy aggregation."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable
from statistics import mean
from typing import Any

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str, *, remove_articles: bool = True) -> str:
    """SQuAD-style normalization. Set remove_articles=False for TimeQA-native."""
    s = (text or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    if remove_articles:
        s = _ARTICLES_RE.sub(" ", s)
    return _WHITESPACE_RE.sub(" ", s).strip()


def _to_list(golds: Any) -> list[str]:
    if isinstance(golds, str):
        return [golds]
    return [g for g in (golds or []) if isinstance(g, str)]


def _em_single(pred_norm: str, gold_norm: str) -> float:
    return 1.0 if pred_norm and pred_norm == gold_norm else 0.0


def _f1_single(pred_norm: str, gold_norm: str) -> float:
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, golds: Any, *, remove_articles: bool = True) -> float:
    golds_list = _to_list(golds)
    if not golds_list:
        return 0.0
    p = _normalize(pred, remove_articles=remove_articles)
    return max(_em_single(p, _normalize(g, remove_articles=remove_articles)) for g in golds_list)


def token_f1(pred: str, golds: Any, *, remove_articles: bool = True) -> float:
    golds_list = _to_list(golds)
    if not golds_list:
        return 0.0
    p = _normalize(pred, remove_articles=remove_articles)
    return max(_f1_single(p, _normalize(g, remove_articles=remove_articles)) for g in golds_list)


def timeqa_native(pred: str, golds: Any) -> dict[str, float]:
    """Official TimeQA normalization: lower + strip punctuation only (keeps articles)."""
    return {
        "em_native": exact_match(pred, golds, remove_articles=False),
        "f1_native": token_f1(pred, golds, remove_articles=False),
    }


def truthfulness(labels: Iterable[str]) -> dict[str, float]:
    total = 0
    correct = 0
    for lab in labels:
        total += 1
        if lab == "correct":
            correct += 1
    if total == 0:
        return {"total": 0, "accuracy": 0.0}
    return {
        "total": total,
        "accuracy": correct / total,
        "incorrect_rate": (total - correct) / total,
    }


def aggregate(per_example: Iterable[dict[str, float]]) -> dict[str, float]:
    """Average a stream of metric dicts (per-example). Non-numeric values are skipped."""
    bucket: dict[str, list[float]] = {}
    for row in per_example:
        for k, v in (row or {}).items():
            if isinstance(v, bool):
                v = float(v)
            if isinstance(v, (int, float)):
                bucket.setdefault(k, []).append(float(v))
    return {k: mean(vs) for k, vs in bucket.items() if vs}
