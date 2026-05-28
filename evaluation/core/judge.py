"""LLM judge for open-ended answers. Falls back to a heuristic if no LLM is available."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from evaluation.core.metrics import exact_match, token_f1

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """You evaluate a candidate answer against a reference for a question-answering benchmark.

Question: {question}
Reference answer: {gold}
Candidate answer: {prediction}

Your task is to decide whether the candidate, as a response to the question, names the same entity / value as the reference.

GROUND RULES:
- Treat the REFERENCE as ground truth. Do NOT contradict it with outside knowledge — if you think the reference is factually wrong, you must still judge against it.
- Do NOT introduce facts that appear in neither the prediction nor the reference. Compare only what is on the page.
- The question's date/scope anchor matters only insofar as the reference uses it. Do not penalise the candidate for naming the reference entity even if a side detail (month, exact day, alternate name) differs from the question.

LABEL "correct" WHEN:
- The candidate names the same entity/value as the reference, allowing for:
    * paraphrases ("U.S. Senator" vs "Senate"; "Governor of Kentucky" vs "governors")
    * aliases, abbreviations, full names ("Ajax" vs "AFC Ajax"; "TTC" vs "Toronto Transit Commission")
    * alternate spellings / transliterations ("Feodorovna" vs "Fedorovna")
    * adding a country or location qualifier the reference omits (cities/orgs inside the reference's country)
    * refining the reference with a more specific sub-type (when the question allows it)
    * sub/super-organization equivalence ONLY when the question's grain leaves them interchangeable; if the question pins a specific sub-unit and the candidate names the parent (or vice versa), that is INCORRECT
- The candidate names the reference entity AND adds non-contradictory context (extra dates, role detail, related entities) — extra correct information never makes a right answer wrong.
- For "after T?" questions: the candidate names the reference entity, even if it also mentions a later successor in chronological order. The reference being one item from a sequence does not make a sequence-naming answer wrong.
- For "before T?" / "in T?" questions: the candidate names the reference entity for the asked time, even if it also gives the predecessor's tenure as context — as long as the reference entity is identified as the one true at T.
- The candidate hedges linguistically ("reportedly", "likely", "according to sources") but still commits to the reference entity. Hedging without contradiction is still a match.

LABEL "incorrect" WHEN:
- The candidate names a DIFFERENT primary entity than the reference (and the two are not aliases/paraphrases of each other).
- The candidate explicitly negates the reference ("X was not Y", "did not hold the position", "no spouse at that time").
- The candidate refuses or says the information is unknown / unavailable / not in the data, without committing to the reference entity.
- The candidate is empty.
- For exclusive-relation questions (spouse, single position, current team) the candidate names a DIFFERENT entity as the active one at the asked time, even if the reference entity appears somewhere else in the answer as a non-active mention.

EDGE CASES:
- If the candidate names multiple entities and the reference is one of them, label "correct" UNLESS the candidate explicitly designates a different one as THE answer to the asked slot.
- If the candidate gives a parent/child organisation instead of the exact reference, label "incorrect" when they refer to distinct entities (e.g. parent broadcaster vs specific channel) and "correct" when they are interchangeable for the question's grain.
- "Late <decade>s" or other time phrases: interpret them as the reference does. If the reference's entity is the candidate's entity and the candidate's dates are consistent with the reference's interpretation, label "correct".

Reply with EXACTLY one line of JSON:
{{"label": "correct|incorrect", "reason": "<=30 words"}}
"""


@dataclass
class JudgeResult:
    label: str  # "correct" | "incorrect"
    reason: str
    raw: str = ""


def _heuristic(question: str, gold: str, prediction: str) -> JudgeResult:
    pred = (prediction or "").strip()
    if not pred:
        return JudgeResult("incorrect", "empty prediction")
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
        if label in {"correct", "incorrect"}:
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
