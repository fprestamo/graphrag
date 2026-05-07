# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Convert the NarrativeQA test split into the harness's JSONL schema.

NarrativeQA (https://huggingface.co/datasets/deepmind/narrativeqa) ships two
reference answers per question.  We store the first one as ``answer`` and the
second as an alias so token-F1 / EM accept either.

Requires the ``datasets`` package (``pip install -r evaluation/requirements.txt``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from evaluation.core.dataset import EvalRecord, save_jsonl

logger = logging.getLogger(__name__)


def _to_record(idx: int, raw: dict[str, Any]) -> EvalRecord:
    answers = raw.get("answers") or []
    texts = [a.get("text") if isinstance(a, dict) else str(a) for a in answers]
    texts = [t for t in texts if t]
    primary = texts[0] if texts else ""
    aliases = texts[1:]
    q = raw.get("question")
    if isinstance(q, dict):
        q = q.get("text", "")
    doc = raw.get("document") or {}
    return EvalRecord(
        qid=str(raw.get("id") or f"narrativeqa-{idx:05d}"),
        question=str(q or ""),
        answer=str(primary),
        aliases=list(aliases),
        context={"document_id": doc.get("id") if isinstance(doc, dict) else None},
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/benchmarks/narrativeqa/data/narrativeqa.jsonl"),
    )
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args(argv)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The 'datasets' package is required to download NarrativeQA. "
            "Install it with `pip install -r evaluation/requirements.txt`."
        ) from exc

    logger.info("Loading deepmind/narrativeqa [%s]…", args.split)
    ds = load_dataset("deepmind/narrativeqa", split=args.split)
    records: list[EvalRecord] = []
    for i, row in enumerate(ds):
        if args.limit is not None and i >= args.limit:
            break
        records.append(_to_record(i, row))
    save_jsonl(args.out, records)
    logger.info("Wrote %d records to %s", len(records), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
