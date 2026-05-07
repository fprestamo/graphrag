# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Convert the public CRAG release into the harness's JSONL schema.

CRAG (https://github.com/facebookresearch/CRAG) is distributed as a
HuggingFace dataset (``CRAG/crag``) with one record per question and the
following fields::

    interaction_id       -> qid
    query                -> question
    answer               -> answer
    query_time           -> question_time   (ISO-8601)
    domain, question_type, static_or_dynamic -> stored under ``context``

Running this script requires the optional ``datasets`` dependency::

    pip install -r evaluation/requirements.txt
    python -m evaluation.benchmarks.crag.load_dataset \\
        --out evaluation/benchmarks/crag/data/crag.jsonl --limit 500

If ``datasets`` is not installed, the script can also ingest a CRAG JSONL file
that has already been downloaded manually (``--input <file.jsonl>``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

from evaluation.core.dataset import EvalRecord, save_jsonl

logger = logging.getLogger(__name__)


CRAG_HF_REPO = "CRAG/crag"
CRAG_HF_SPLIT = "validation"


def _to_record(raw: dict[str, Any]) -> EvalRecord:
    return EvalRecord(
        qid=str(raw.get("interaction_id") or raw.get("id") or raw.get("qid")),
        question=str(raw.get("query") or raw.get("question") or ""),
        answer=str(raw.get("answer") or raw.get("ground_truth") or ""),
        question_time=raw.get("query_time") or raw.get("question_time"),
        context={
            k: raw[k]
            for k in ("domain", "question_type", "static_or_dynamic", "split")
            if k in raw
        },
    )


def _from_huggingface(limit: int | None) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The 'datasets' package is required to download CRAG. "
            "Install it with `pip install -r evaluation/requirements.txt` "
            "or pass --input <file.jsonl>."
        ) from exc

    logger.info("Loading %s [%s] from HuggingFace…", CRAG_HF_REPO, CRAG_HF_SPLIT)
    ds = load_dataset(CRAG_HF_REPO, split=CRAG_HF_SPLIT)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield row


def _from_jsonl(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/benchmarks/crag/data/crag.jsonl"),
        help="Output JSONL path (default: %(default)s).",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional pre-downloaded CRAG JSONL file (skips HuggingFace).",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Keep only the first N records."
    )
    args = p.parse_args(argv)

    if args.input is not None:
        rows = _from_jsonl(args.input, args.limit)
    else:
        rows = _from_huggingface(args.limit)

    records = [_to_record(r) for r in rows]
    save_jsonl(args.out, records)
    logger.info("Wrote %d records to %s", len(records), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
