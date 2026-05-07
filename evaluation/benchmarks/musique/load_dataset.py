# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Convert the public MuSiQue release into the harness's JSONL schema.

MuSiQue (https://github.com/StonyBrookNLP/musique) is distributed as a JSONL
file (e.g. ``musique_ans_v1.0_dev.jsonl``).  Each record contains::

    id, question, answer, answer_aliases, paragraphs, question_decomposition

The dataset must be downloaded manually from the official repo's release page
(or via the ``musique`` HuggingFace mirror) – it cannot be redistributed here.
Pass the local file with ``--input``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from evaluation.core.dataset import EvalRecord, save_jsonl

logger = logging.getLogger(__name__)


def _to_record(raw: dict[str, Any]) -> EvalRecord:
    return EvalRecord(
        qid=str(raw.get("id") or raw.get("_id")),
        question=str(raw["question"]),
        answer=str(raw.get("answer") or ""),
        aliases=list(raw.get("answer_aliases") or []),
        context={
            "num_hops": len(raw.get("question_decomposition") or []) or None,
            "answerable": raw.get("answerable", True),
        },
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the official MuSiQue JSONL file (e.g. musique_ans_v1.0_dev.jsonl).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/benchmarks/musique/data/musique.jsonl"),
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    records: list[EvalRecord] = []
    with args.input.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            records.append(_to_record(json.loads(line)))

    save_jsonl(args.out, records)
    logger.info("Wrote %d records to %s", len(records), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
