# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Run every benchmark in :mod:`evaluation.benchmarks` and emit a summary.

Example::

    python -m evaluation.run_all --limit 50 --dry-run
    python -m evaluation.run_all --benchmarks crag hotpotqa --out evaluation/results
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
from pathlib import Path
from typing import Any

from evaluation.core.btgraphrag_runner import RunnerConfig
from evaluation.core.runner_cli import _run_async  # internal helper, OK within package

logger = logging.getLogger(__name__)


# Registry of available benchmarks → import path of their `SPEC` object.
BENCHMARKS: dict[str, str] = {
    "crag":          "evaluation.benchmarks.crag.evaluate:SPEC",
    "hotpotqa":      "evaluation.benchmarks.hotpotqa.evaluate:SPEC",
    "musique":       "evaluation.benchmarks.musique.evaluate:SPEC",
    "multihop_rag":  "evaluation.benchmarks.multihop_rag.evaluate:SPEC",
    "narrativeqa":   "evaluation.benchmarks.narrativeqa.evaluate:SPEC",
}


def _load_spec(dotted: str):
    mod_path, attr = dotted.split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


async def _run_one(name: str, args, runner_cfg: RunnerConfig) -> dict[str, Any]:
    spec = _load_spec(BENCHMARKS[name])
    if args.no_judge:
        spec.use_llm_judge = False
    dataset = spec.default_dataset
    if not dataset.exists():
        logger.warning("Skipping %s – dataset not found at %s", name, dataset)
        return {"benchmark": name, "skipped": True, "reason": f"missing {dataset}"}

    out_dir = args.out / name
    return await _run_async(
        spec,
        dataset_path=dataset,
        out_dir=out_dir,
        limit=args.limit,
        concurrency=args.concurrency,
        runner_cfg=runner_cfg,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(BENCHMARKS),
        default=sorted(BENCHMARKS),
    )
    p.add_argument("--out", type=Path, default=Path("evaluation/results"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    runner_cfg = RunnerConfig(dry_run=args.dry_run)
    args.out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"benchmarks": {}}
    for name in args.benchmarks:
        logger.info("=== Running benchmark: %s ===", name)
        try:
            summary["benchmarks"][name] = asyncio.run(_run_one(name, args, runner_cfg))
        except Exception as exc:
            logger.exception("Benchmark %s failed", name)
            summary["benchmarks"][name] = {"error": str(exc)}

    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Wrote combined summary to %s", summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
