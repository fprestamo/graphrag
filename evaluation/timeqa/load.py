"""TimeQA loader: raw JSON (easy/hard) → JSONL + per-entity .txt corpus."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from evaluation.core.dataset import EvalRecord, save_jsonl

logger = logging.getLogger(__name__)

_ENTITY_RE = re.compile(r"^(/wiki/[^#]+)")
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_FILENAME_MAX = 120


def _read_raw(path: Path) -> list[dict[str, Any]]:
    """Accept both a JSON array and JSONL."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: top-level JSON must be a list")
        return data
    out: list[dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i}: {exc}") from exc
    return out


def _flatten_targets(targets: Any) -> list[str]:
    out: list[str] = []
    if not targets:
        return out
    if isinstance(targets, str):
        return [targets]
    for t in targets:
        if isinstance(t, str):
            s = t.strip()
            if s:
                out.append(s)
        elif isinstance(t, dict):
            val = t.get("text") or t.get("answer") or t.get("value")
            if isinstance(val, str) and val.strip():
                out.append(val.strip())
    return out


def _entity_id(idx: str) -> str:
    m = _ENTITY_RE.match(idx or "")
    return m.group(1) if m else (idx or "")


def _filename_for(entity_id: str) -> str:
    stem = entity_id.replace("/wiki/", "", 1)
    stem = _SAFE_RE.sub("_", stem).strip("_") or "doc"
    if len(stem) > _FILENAME_MAX:
        stem = stem[:_FILENAME_MAX]
    return f"{stem}.txt"


def _corpus_text(raw: dict[str, Any]) -> str:
    ctx = raw.get("context")
    if isinstance(ctx, str) and ctx.strip():
        return ctx.strip() + "\n"
    paragraphs = raw.get("paragraphs") or []
    parts: list[str] = []
    for p in paragraphs:
        if not isinstance(p, dict):
            continue
        title = (p.get("title") or "").strip()
        text = (p.get("text") or "").strip()
        if title or text:
            parts.append(f"{title}\n{text}".strip())
    return "\n\n".join(parts).strip() + "\n" if parts else ""


def _records_from_split(
    raws: Iterable[dict[str, Any]],
    split: str,
    corpus_dir: Path,
    seen: dict[str, str],
    max_entities: int | None = None,
) -> Iterable[EvalRecord]:
    split_entities: set[str] = set()
    for raw in raws:
        idx = str(raw.get("idx") or raw.get("id") or "")
        if not idx:
            continue
        question = (raw.get("question") or "").strip()
        targets = _flatten_targets(raw.get("targets"))
        if not question or not targets:
            continue
        entity_id = _entity_id(idx)
        if max_entities is not None and entity_id not in split_entities:
            if len(split_entities) >= max_entities:
                continue
            split_entities.add(entity_id)
        filename = seen.get(entity_id)
        if filename is None:
            filename = _filename_for(entity_id)
            text = _corpus_text(raw)
            if text:
                out_path = corpus_dir / filename
                out_path.write_text(text, encoding="utf-8")
                seen[entity_id] = filename
            else:
                seen[entity_id] = filename  # avoid retrying empty contexts
        yield EvalRecord(
            qid=idx,
            question=question,
            answer=targets[0],
            aliases=targets[1:],
            context={
                "split": split,
                "entity_id": entity_id,
                "doc_id": filename,
            },
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.timeqa.load",
        description="Convert TimeQA JSON dumps into JSONL + corpus .txt files.",
    )
    parser.add_argument("--input-easy", type=Path, required=False)
    parser.add_argument("--input-hard", type=Path, required=False)
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("evaluation/timeqa/data/timeqa.jsonl"),
    )
    parser.add_argument(
        "--out-corpus",
        type=Path,
        default=Path("evaluation/timeqa/data/corpus"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--entities-per-split",
        type=int,
        default=None,
        help="Keep only the first N unique entities (/wiki/...) from each split.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.input_easy and not args.input_hard:
        parser.error("provide at least --input-easy or --input-hard")

    args.out_corpus.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    records: list[EvalRecord] = []

    for split, in_path in (("easy", args.input_easy), ("hard", args.input_hard)):
        if not in_path:
            continue
        if not in_path.exists():
            print(f"[warn] {in_path} does not exist, skipping", file=sys.stderr)
            continue
        raws = _read_raw(in_path)
        records.extend(
            _records_from_split(
                raws, split, args.out_corpus, seen, args.entities_per_split
            )
        )

    if args.limit is not None and args.limit >= 0:
        records = records[: args.limit]

    save_jsonl(args.out_jsonl, records)
    logger.info(
        "Wrote %d records to %s and %d unique docs to %s",
        len(records),
        args.out_jsonl,
        len(seen),
        args.out_corpus,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
