"""TimeQA loader (hard split, open-domain RAG protocol).

Builds a single unified corpus from every unique Wikipedia entity referenced in
`hard.json`. All pages go into one directory (`out-corpus/`), one `.txt` per
entity. The indexer chunks/embeds that pool once; at query time the system has
to retrieve the relevant chunks itself — the correct page is never disclosed.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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


def _build_records(
    raws: Iterable[dict[str, Any]],
    corpus_dir: Path,
    max_entities: int | None = None,
    seed: int | None = None,
) -> tuple[list[EvalRecord], int]:
    """Emit one EvalRecord per question and write the unified corpus.

    Every unique /wiki/... entity contributes one .txt to `corpus_dir`. All
    questions share that single pool — there is no per-entity corpus.
    """
    raws = list(raws)

    selected: set[str] | None = None
    if max_entities is not None:
        all_entities: list[str] = []
        seen_entities: set[str] = set()
        for raw in raws:
            idx = str(raw.get("idx") or raw.get("id") or "")
            if not idx:
                continue
            eid = _entity_id(idx)
            if eid and eid not in seen_entities:
                seen_entities.add(eid)
                all_entities.append(eid)
        rng = random.Random(seed)
        if max_entities < len(all_entities):
            selected = set(rng.sample(all_entities, max_entities))
        else:
            selected = set(all_entities)

    records: list[EvalRecord] = []
    seen: dict[str, str] = {}  # entity_id -> filename

    for raw in raws:
        idx = str(raw.get("idx") or raw.get("id") or "")
        if not idx:
            continue

        entity_id = _entity_id(idx)
        if selected is not None and entity_id not in selected:
            continue

        # Corpus: write the page once per unique entity, regardless of whether
        # any of its questions are answerable. Unanswerable-only pages still
        # belong in the open-domain retrieval pool as distractors.
        filename = seen.get(entity_id)
        if filename is None:
            filename = _filename_for(entity_id)
            text = _corpus_text(raw)
            if text:
                (corpus_dir / filename).write_text(text, encoding="utf-8")
            seen[entity_id] = filename

        # Records: only answerable items (targets != [""] after flattening).
        question = (raw.get("question") or "").strip()
        targets = _flatten_targets(raw.get("targets"))
        if not question or not targets:
            continue

        records.append(
            EvalRecord(
                qid=idx,
                question=question,
                answer=targets[0],
                aliases=targets[1:],
                context={"entity_id": entity_id, "doc_id": filename},
            )
        )

    return records, len(seen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.timeqa.load",
        description=(
            "Convert TimeQA hard.json into JSONL + a unified corpus of "
            "Wikipedia pages (open-domain RAG protocol)."
        ),
    )
    parser.add_argument(
        "--input-hard",
        type=Path,
        default=Path("evaluation/timeqa/data/raw/hard.json"),
        help="Path to TimeQA hard.json (JSON array or JSONL).",
    )
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
        "--max-entities",
        type=int,
        default=None,
        help="Keep N unique entities sampled at random (for quick smoke tests).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for --max-entities sampling. Omit for nondeterministic.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.input_hard.exists():
        print(f"[error] {args.input_hard} does not exist", file=sys.stderr)
        return 2

    args.out_corpus.mkdir(parents=True, exist_ok=True)

    raws = _read_raw(args.input_hard)
    records, doc_count = _build_records(
        raws, args.out_corpus, args.max_entities, args.seed
    )

    if args.limit is not None and args.limit >= 0:
        records = records[: args.limit]

    save_jsonl(args.out_jsonl, records)
    logger.info(
        "Wrote %d records to %s and %d docs to %s",
        len(records),
        args.out_jsonl,
        doc_count,
        args.out_corpus,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
