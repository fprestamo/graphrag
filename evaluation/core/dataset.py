# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Common dataset record / IO helpers used by all benchmarks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class EvalRecord:
    """A single benchmark question.

    Attributes:
        qid: stable question identifier (benchmark-specific).
        question: natural-language question.
        answer: gold answer (string).  For datasets with multiple gold answers
            (e.g. NarrativeQA) the canonical one is stored here and the rest in
            ``aliases``.
        aliases: alternative gold answers accepted as correct.
        question_time: ISO-8601 timestamp at which the question is asked
            (used by BT-GraphRAG's temporal query when available).
        context: optional additional benchmark-specific context (supporting
            facts, sub-questions, query type, ...).
    """

    qid: str
    question: str
    answer: str
    aliases: list[str] = field(default_factory=list)
    question_time: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalRecord":
        return cls(
            qid=str(data["qid"]),
            question=str(data["question"]),
            answer=str(data.get("answer", "")),
            aliases=list(data.get("aliases", []) or []),
            question_time=data.get("question_time"),
            context=dict(data.get("context", {}) or {}),
        )


@dataclass
class EvalPrediction:
    """A single prediction emitted by the system under test."""

    qid: str
    question: str
    gold_answer: str
    prediction: str
    judge_label: str | None = None
    """One of ``correct`` / ``incorrect`` / ``missing``  when an LLM judge is used."""

    judge_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    """Original benchmark context copied from :class:`EvalRecord` (domain,
    question type, num_hops, ...).  Used by per-category report breakdowns."""

    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file as a list of dicts (skips blank lines)."""
    p = Path(path)
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    """Write an iterable of dataclasses or dicts to JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            if hasattr(r, "to_dict"):
                d = r.to_dict()
            elif hasattr(r, "__dict__") and not isinstance(r, dict):
                d = asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r.__dict__)
            else:
                d = dict(r)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def iter_records(path: str | Path) -> Iterator[EvalRecord]:
    """Iterate over a JSONL benchmark dataset as :class:`EvalRecord` objects."""
    for raw in load_jsonl(path):
        yield EvalRecord.from_dict(raw)
