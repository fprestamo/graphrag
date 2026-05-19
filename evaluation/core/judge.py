"""LLM judge for open-ended answers. Falls back to a heuristic if no LLM is available."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from evaluation.core.metrics import exact_match, token_f1

logger = logging.getLogger(__name__)

_REFUSAL_PATTERNS = (
    "i don't know",
    "i do not know",
    "no information",
    "insufficient",
    "cannot determine",
    "can't determine",
    "unable to answer",
    "n/a",
)

_JUDGE_PROMPT = """You are a strict evaluator of question-answering systems.

Question: {question}
Reference answer: {gold}
Candidate answer: {prediction}

Decide whether the candidate answer is:
  - "correct": semantically matches the reference (paraphrases, alias names, equivalent dates).
  - "missing": refuses or says it does not know / has no information.
  - "incorrect": says something concrete that contradicts the reference or is unrelated.

Reply with EXACTLY one line of JSON:
{{"label": "correct|incorrect|missing", "reason": "<=30 words"}}
"""


@dataclass
class JudgeResult:
    label: str  # "correct" | "incorrect" | "missing"
    reason: str
    raw: str = ""


def _heuristic(question: str, gold: str, prediction: str) -> JudgeResult:
    pred = (prediction or "").strip()
    if not pred:
        return JudgeResult("missing", "empty prediction")
    low = pred.lower()
    if any(pat in low for pat in _REFUSAL_PATTERNS):
        return JudgeResult("missing", "refusal pattern")
    em = exact_match(pred, [gold])
    f1 = token_f1(pred, [gold])
    if em >= 1.0 or f1 >= 0.6:
        return JudgeResult("correct", f"heuristic em={em:.2f} f1={f1:.2f}")
    if f1 <= 0.0:
        return JudgeResult("incorrect", "no token overlap")
    return JudgeResult("incorrect", f"low overlap f1={f1:.2f}")


_JSON_RE = re.compile(r"\{[^{}]*\}")


def _parse_label(raw: str) -> JudgeResult | None:
    if not raw:
        return None
    candidates = _JSON_RE.findall(raw)
    if not candidates:
        candidates = [raw]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        label = str(data.get("label", "")).strip().lower()
        if label in {"correct", "incorrect", "missing"}:
            reason = str(data.get("reason", "")).strip()
            return JudgeResult(label, reason, raw=raw)
    return None


def _build_llm():
    """Return an LLM completion object, or None if anything fails."""
    model_id = os.getenv("BTG_JUDGE_MODEL_ID") or os.getenv("BTG_MODEL_ID")
    if not model_id:
        return None
    try:  # noqa: SIM105 — explicit broad try so any factory hiccup degrades to heuristic
        from graphrag_llm.completion import create_completion
        from graphrag_llm.config import ModelConfig
        from graphrag_llm.config.types import LLMProviderType
    except ImportError as exc:
        logger.debug("graphrag_llm not importable: %s", exc)
        return None
    api_key = os.getenv("GRAPHRAG_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    try:
        cfg = ModelConfig(
            type=LLMProviderType.LiteLLM,
            model_provider=os.getenv("BTG_JUDGE_PROVIDER", "openai"),
            model=model_id,
            api_key=api_key,
        )
        return create_completion(cfg)
    except Exception as exc:  # noqa: BLE001 — judge must never raise
        logger.debug("Could not build LLM judge: %s", exc)
        return None


async def judge_answer(
    question: str,
    gold: str,
    prediction: str,
    *,
    use_llm: bool = True,
) -> JudgeResult:
    if not use_llm:
        return _heuristic(question, gold, prediction)

    llm = _build_llm()
    if llm is None:
        return _heuristic(question, gold, prediction)

    try:
        from graphrag_llm.utils import CompletionMessagesBuilder

        prompt = _JUDGE_PROMPT.format(
            question=question.strip(),
            gold=(gold or "").strip(),
            prediction=(prediction or "").strip(),
        )
        messages = CompletionMessagesBuilder().add_user_message(prompt).build()
        response = await llm.completion_async(messages=messages)
        raw = getattr(response, "content", "") or ""
        parsed = _parse_label(raw)
        if parsed is not None:
            return parsed
        logger.debug("Judge LLM returned unparseable output: %r", raw[:200])
    except Exception as exc:  # noqa: BLE001
        logger.debug("LLM judge call failed: %s", exc)

    return _heuristic(question, gold, prediction)
