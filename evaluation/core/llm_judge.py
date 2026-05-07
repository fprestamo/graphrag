# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLM-as-judge scoring used by every benchmark.

The judge classifies a (question, gold answer, prediction) triple as one of:

* ``correct``    – the prediction is semantically equivalent to the gold answer.
* ``incorrect``  – the prediction states something that contradicts the gold
  answer (a *hallucination* in CRAG terminology).
* ``missing``    – the prediction declines to answer (e.g. "I don't know" /
  empty / refusal).  Counted neutrally by the CRAG truthfulness score.

This protocol matches the one used by the CRAG benchmark
(https://github.com/facebookresearch/CRAG).  It is reused for the other
free-form benchmarks so that all results are directly comparable.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are an impartial answer-grading judge.

Given a question, the GOLD answer and a SYSTEM answer, decide whether the
SYSTEM answer is:

- "correct"   : it is semantically equivalent to the GOLD answer (paraphrases,
                synonyms, equivalent units and equivalent levels of detail are
                acceptable).
- "incorrect" : it gives a definite answer that contradicts the GOLD answer or
                contains a clear factual error (a hallucination).
- "missing"   : it refuses to answer, says it does not know, returns an empty
                string, or only restates the question without committing to a
                value.

Respond with a single JSON object on one line, with keys "label" (one of
"correct", "incorrect", "missing") and "reason" (a short justification, <= 30
words). Do not output anything else.

Question: {question}
GOLD answer: {gold}
SYSTEM answer: {prediction}
"""


@dataclass
class JudgeResult:
    label: str
    reason: str
    raw: str = ""


# ---------------------------------------------------------------------------
# Heuristic fallback (used when no LLM client is available, e.g. in CI)
# ---------------------------------------------------------------------------


_REFUSAL_PATTERNS = re.compile(
    r"\b(i\s+(?:do(?:n['’]t)?|cannot|can['’]t)\s+(?:know|answer|tell)|"
    r"insufficient (?:information|context)|no\s+(?:information|answer|data)|"
    r"unknown|n/?a|not\s+(?:available|sure|enough))\b",
    re.IGNORECASE,
)


def _heuristic_label(prediction: str, gold: str) -> JudgeResult:
    """Cheap fallback classifier; only used when no LLM judge is configured."""
    from evaluation.core.metrics import normalize_answer, token_f1

    pred = (prediction or "").strip()
    if not pred or _REFUSAL_PATTERNS.search(pred):
        return JudgeResult("missing", "empty / refusal", raw="heuristic")

    np, ng = normalize_answer(pred), normalize_answer(gold)
    if np == ng or (ng and ng in np) or (np and np in ng):
        return JudgeResult("correct", "string overlap", raw="heuristic")

    if token_f1(pred, gold) >= 0.6:
        return JudgeResult("correct", "high token-F1", raw="heuristic")

    return JudgeResult("incorrect", "no overlap with gold", raw="heuristic")


# ---------------------------------------------------------------------------
# LLM-backed judge
# ---------------------------------------------------------------------------


async def _build_judge_llm():
    """Best-effort construction of an LLM client used as a judge."""
    model_id = os.getenv("BTG_JUDGE_MODEL_ID") or os.getenv("BTG_MODEL_ID")
    if not model_id:
        return None
    try:
        from graphrag_llm.factory import ModelFactory  # type: ignore

        if ModelFactory.is_supported_model(model_id):
            return ModelFactory.create_chat_model(model_id)
    except Exception as exc:  # pragma: no cover - env dependent
        logger.warning("Could not build LLM judge (%s); using heuristic.", exc)
    return None


def _parse_judge_response(text: str) -> JudgeResult:
    """Extract the JSON object emitted by the judge prompt."""
    if not text:
        return JudgeResult("missing", "empty judge response", raw=text)
    # Greedy JSON extraction – the model occasionally wraps the JSON in prose.
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return JudgeResult("incorrect", "unparseable judge response", raw=text)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return JudgeResult("incorrect", "invalid JSON from judge", raw=text)

    label = str(data.get("label", "")).strip().lower()
    if label not in {"correct", "incorrect", "missing"}:
        label = "incorrect"
    return JudgeResult(label, str(data.get("reason", ""))[:200], raw=text)


async def judge_answer(
    question: str,
    gold: str,
    prediction: str,
    *,
    use_llm: bool = True,
) -> JudgeResult:
    """Score a single prediction. Falls back to a heuristic when needed."""
    if not use_llm:
        return _heuristic_label(prediction, gold)

    llm = await _build_judge_llm()
    if llm is None:
        return _heuristic_label(prediction, gold)

    try:
        from graphrag_llm.utils import CompletionMessagesBuilder  # type: ignore

        prompt = JUDGE_PROMPT.format(
            question=question, gold=gold, prediction=prediction or "(empty)"
        )
        messages = CompletionMessagesBuilder().add_user_message(prompt).build()
        response = await llm.completion_async(messages=messages)
        return _parse_judge_response(response.content or "")
    except Exception as exc:  # pragma: no cover - env dependent
        logger.warning("Judge LLM call failed (%s); falling back to heuristic.", exc)
        return _heuristic_label(prediction, gold)
