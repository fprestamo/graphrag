# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Convert the public HotpotQA dev set into the harness's JSONL schema.

HotpotQA (https://hotpotqa.github.io/) is distributed as a single JSON file.
The dev set used by every published baseline is::

    http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

Each record contains::

    _id, question, answer, type, level, supporting_facts, context
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

HOTPOT_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def _to_record(raw: dict[str, Any]) -> EvalRecord:
    return EvalRecord(
        qid=str(raw["_id"]),
        question=str(raw["question"]),
        answer=str(raw.get("answer", "")),
        context={
            "type": raw.get("type"),
            "level": raw.get("level"),
            "supporting_facts": raw.get("supporting_facts", []),
        },
    )


def _download(dest: Path) -> None:
    """Download the official HotpotQA dev set to ``dest``.

    The URL is hardcoded to :data:`HOTPOT_DEV_URL` to avoid SSRF-style misuse
    of this helper.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", HOTPOT_DEV_URL, dest)
    with urllib.request.urlopen(HOTPOT_DEV_URL) as resp, dest.open("wb") as f:  # noqa: S310
        f.write(resp.read())


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/benchmarks/hotpotqa/data/hotpotqa.jsonl"),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Pre-downloaded HotpotQA JSON file (skips network).",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    if args.input is not None:
        src = args.input
    else:
        src = args.out.parent / "hotpot_dev_distractor_v1.json"
        if not src.exists():
            _download(src)

    raw = json.loads(src.read_text(encoding="utf-8"))
    if args.limit is not None:
        raw = raw[: args.limit]
    records = [_to_record(r) for r in raw]
    save_jsonl(args.out, records)
    logger.info("Wrote %d records to %s", len(records), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
