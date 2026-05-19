"""JSONL-backed dataset primitives shared by all stages of the harness."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalRecord:
    qid: str
    question: str
    answer: str
    aliases: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalRecord":
        return cls(
            qid=str(d["qid"]),
            question=str(d["question"]),
            answer=str(d.get("answer", "")),
            aliases=list(d.get("aliases", []) or []),
            context=dict(d.get("context", {}) or {}),
        )


@dataclass
class EvalPrediction:
    qid: str
    question: str
    gold_answer: str
    prediction: str
    judge_label: str | None = None
    judge_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalPrediction":
        return cls(
            qid=str(d["qid"]),
            question=str(d.get("question", "")),
            gold_answer=str(d.get("gold_answer", "")),
            prediction=str(d.get("prediction", "")),
            judge_label=d.get("judge_label"),
            judge_reason=d.get("judge_reason"),
            metrics=dict(d.get("metrics", {}) or {}),
            context=dict(d.get("context", {}) or {}),
        )


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            if hasattr(rec, "to_dict"):
                rec = rec.to_dict()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def iter_records(path: str | Path) -> Iterator[EvalRecord]:
    for d in load_jsonl(path):
        yield EvalRecord.from_dict(d)
