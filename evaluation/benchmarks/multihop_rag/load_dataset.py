# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Convert the MultiHop-RAG dataset to the harness's JSONL schema.

MultiHop-RAG (https://github.com/yixuantt/MultiHop-RAG) is distributed as a
single JSON list under ``dataset/MultiHopRAG.json``.  Each record has::

    query, answer, question_type, evidence_list

We map ``question_type`` into ``context["query_type"]`` so the per-type
breakdown in :mod:`evaluation.benchmarks.multihop_rag.evaluate` works.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any

from evaluation.core.dataset import EvalRecord, save_jsonl

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/MultiHopRAG.json"
)


def _to_record(idx: int, raw: dict[str, Any]) -> EvalRecord:
    return EvalRecord(
        qid=str(raw.get("qid") or raw.get("id") or f"mhrag-{idx:05d}"),
        question=str(raw.get("query") or raw.get("question") or ""),
        answer=str(raw.get("answer") or ""),
        context={
            "query_type": raw.get("question_type") or raw.get("query_type"),
            "num_evidence": len(raw.get("evidence_list") or []),
        },
    )


def _download(dest: Path) -> None:
    """Download the official MultiHop-RAG dataset to ``dest``.

    The URL is hardcoded to :data:`DATASET_URL` to avoid SSRF-style misuse
    of this helper.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", DATASET_URL, dest)
    with urllib.request.urlopen(DATASET_URL) as resp, dest.open("wb") as f:  # noqa: S310
        f.write(resp.read())


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/benchmarks/multihop_rag/data/multihop_rag.jsonl"),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Pre-downloaded MultiHopRAG.json (skips network).",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    if args.input is not None:
        src = args.input
    else:
        src = args.out.parent / "MultiHopRAG.json"
        if not src.exists():
            _download(src)

    raw = json.loads(src.read_text(encoding="utf-8"))
    if args.limit is not None:
        raw = raw[: args.limit]
    records = [_to_record(i, r) for i, r in enumerate(raw)]
    save_jsonl(args.out, records)
    logger.info("Wrote %d records to %s", len(records), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
