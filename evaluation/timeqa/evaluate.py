"""Single CLI: run BT-GraphRAG, vanilla GraphRAG, or both on TimeQA and report results."""

from __future__ import annotations

import os

# Silence LiteLLM's verbose request/response dumps before it gets imported anywhere.
os.environ.setdefault("LITELLM_LOG", "WARNING")

import argparse
import asyncio
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from evaluation.core.dataset import (
    EvalPrediction,
    EvalRecord,
    iter_records,
    save_jsonl,
)
from evaluation.core.judge import judge_answer
from evaluation.core.metrics import (
    aggregate,
    exact_match,
    timeqa_native,
    token_f1,
    truthfulness,
)
from evaluation.core.runners import BTConfig, VanillaConfig, bt_answer, vanilla_answer

logger = logging.getLogger(__name__)

SYSTEMS = ("btgraphrag", "graphrag")
SCOPES = ("global", "per_entity")


def _entity_name(rec: EvalRecord) -> str:
    eid = str(rec.context.get("entity_id", "")).strip()
    if eid.startswith("/wiki/"):
        eid = eid[len("/wiki/"):]
    return eid.replace("_", " ").strip()


def _scoped_question(rec: EvalRecord, scope: str) -> str:
    """In `per_entity` scope we prepend the Wikipedia entity name as a soft anchor.

    This emulates "scope retrieval to this entity" without requiring a separate
    per-entity index. The underlying graph is still the global one; the entity
    name simply biases retrieval. Useful as an upper-bound / extraction-only signal.
    """
    if scope == "per_entity":
        ent = _entity_name(rec)
        if ent:
            return f"In the context of {ent}, {rec.question}"
    return rec.question


@dataclass
class _RunOpts:
    use_judge: bool
    dry_run: bool
    concurrency: int
    bt_cfg: BTConfig
    vanilla_cfg: VanillaConfig


# ---------------------------------------------------------------------------
# Per-record scoring
# ---------------------------------------------------------------------------


def _default_scorer(
    rec: EvalRecord, prediction: str, judge_label: str | None
) -> dict[str, float]:
    golds = [rec.answer, *rec.aliases] if rec.answer else list(rec.aliases)
    em = exact_match(prediction, golds)
    f1 = token_f1(prediction, golds)
    native = timeqa_native(prediction, golds)
    metrics: dict[str, float] = {
        "exact_match": em,
        "f1": f1,
        **native,
    }
    if judge_label is not None:
        metrics["judge_correct"] = 1.0 if judge_label == "correct" else 0.0
        metrics["judge_missing"] = 1.0 if judge_label == "missing" else 0.0
        metrics["judge_incorrect"] = 1.0 if judge_label == "incorrect" else 0.0
    return metrics


def _truncate(s: str, n: int = 80) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


async def _eval_one(
    system: str,
    rec: EvalRecord,
    opts: _RunOpts,
    sem: asyncio.Semaphore,
    scope: str,
) -> EvalPrediction:
    async with sem:
        started = time.perf_counter()
        query = _scoped_question(rec, scope)
        if system == "btgraphrag":
            result = await bt_answer(query, config=opts.bt_cfg)
        else:
            result = await vanilla_answer(query, config=opts.vanilla_cfg)

        prediction = (result.get("answer") or "").strip()

        # Judge always sees the original (unscoped) question — we're evaluating
        # whether the answer is correct, not whether the scoped query was good.
        judge = await judge_answer(
            rec.question, rec.answer, prediction, use_llm=opts.use_judge
        )

        metrics = _default_scorer(rec, prediction, judge.label)
        ctx: dict[str, Any] = {**rec.context, "scope": scope, "raw": result.get("raw", {})}
        if query != rec.question:
            ctx["scoped_question"] = query
        ctx["_elapsed"] = round(time.perf_counter() - started, 2)

        raw = result.get("raw") or {}
        if isinstance(raw, dict):
            if "edges_used" in raw:
                ctx["_edges_used"] = raw.get("edges_used")
            if "error" in raw:
                ctx["_error"] = raw.get("error")
        return EvalPrediction(
            qid=rec.qid,
            question=rec.question,
            gold_answer=rec.answer,
            prediction=prediction,
            judge_label=judge.label,
            judge_reason=judge.reason,
            metrics=metrics,
            context=ctx,
        )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


_REPORT_METRIC_KEYS = {"judge_correct", "judge_missing", "judge_incorrect"}


def _judge_only(metrics: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if k in _REPORT_METRIC_KEYS}


def _build_report(
    system: str,
    dataset_path: Path,
    predictions: list[EvalPrediction],
    elapsed: float,
    scope: str,
) -> dict[str, Any]:
    metrics_all = _judge_only(aggregate(p.metrics for p in predictions))
    labels = [p.judge_label for p in predictions if p.judge_label is not None]
    truth = truthfulness(labels) if labels else {}

    by_split: dict[str, dict[str, Any]] = {}
    split_groups: dict[str, list[EvalPrediction]] = {}
    for p in predictions:
        split = str(p.context.get("split", "unknown"))
        split_groups.setdefault(split, []).append(p)
    for split, items in split_groups.items():
        split_labels = [x.judge_label for x in items if x.judge_label is not None]
        by_split[split] = {
            "num_examples": len(items),
            "metrics": _judge_only(aggregate(x.metrics for x in items)),
            "truthfulness": truthfulness(split_labels) if split_labels else {},
        }

    return {
        "benchmark": "timeqa",
        "scope": scope,
        "system": system,
        "dataset": str(dataset_path),
        "num_examples": len(predictions),
        "elapsed_seconds": round(elapsed, 3),
        "metrics": metrics_all,
        "truthfulness": truth,
        "extra": {"by_split": by_split},
    }


_JUDGE_GLYPH = {"correct": "✓", "incorrect": "✗", "missing": "∅"}


def _log_prediction(idx: int, total: int, system: str, scope: str, p: EvalPrediction) -> None:
    glyph = _JUDGE_GLYPH.get(p.judge_label or "", "·")
    label = (p.judge_label or "n/a").upper()
    reason = p.judge_reason or ""
    judge_line = f"{glyph} {label}" + (f" — {reason}" if reason else "")
    err = p.context.get("_error")
    print()
    print(f"[{system}/{scope}] {idx}/{total}")
    print(f"  Q:     {p.question}")
    print(f"  Pred:  {p.prediction or '<empty>'}")
    print(f"  Gold:  {p.gold_answer or '<none>'}")
    print(f"  Judge: {judge_line}")
    if err:
        print(f"  ERROR: {err}")


def _log_summary(system: str, scope: str, report: dict[str, Any]) -> None:
    m = report.get("metrics") or {}
    truth = report.get("truthfulness") or {}
    n = report.get("num_examples", 0)
    elapsed = report.get("elapsed_seconds", 0.0)
    logger.info("-" * 72)
    logger.info("[%s/%s] SUMMARY — %d examples in %.1fs", system, scope, n, elapsed)
    logger.info(
        "  EM=%.3f  F1=%.3f  EM_native=%.3f  F1_native=%.3f",
        m.get("exact_match", 0.0), m.get("f1", 0.0),
        m.get("em_native", 0.0), m.get("f1_native", 0.0),
    )
    if truth:
        logger.info(
            "  Judge — correct=%.0f%% missing=%.0f%% incorrect=%.0f%%  truthfulness=%.3f",
            100 * truth.get("correct_rate", 0.0),
            100 * truth.get("missing_rate", 0.0),
            100 * truth.get("incorrect_rate", 0.0),
            truth.get("truthfulness", 0.0),
        )
    logger.info("-" * 72)


async def _run_system(
    system: str,
    records: list[EvalRecord],
    out_dir: Path,
    dataset_path: Path,
    opts: _RunOpts,
    scope: str,
) -> dict[str, Any]:
    logger.info("=" * 72)
    logger.info(
        "[%s/%s] %d examples (concurrency=%d, judge=%s, dry_run=%s)",
        system, scope, len(records), opts.concurrency,
        "off" if not opts.use_judge else "on",
        opts.dry_run,
    )
    logger.info("=" * 72)
    sem = asyncio.Semaphore(opts.concurrency)
    started = time.perf_counter()
    total = len(records)
    tasks = [
        asyncio.create_task(_eval_one(system, rec, opts, sem, scope))
        for rec in records
    ]
    preds_by_qid: dict[str, EvalPrediction] = {}
    done_count = 0
    for fut in asyncio.as_completed(tasks):
        pred = await fut
        done_count += 1
        _log_prediction(done_count, total, system, scope, pred)
        preds_by_qid[pred.qid] = pred
    # Preserve original input order in outputs.
    preds = [preds_by_qid[r.qid] for r in records if r.qid in preds_by_qid]
    elapsed = time.perf_counter() - started

    sys_dir = out_dir / system / "timeqa" / scope
    sys_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(sys_dir / "predictions.jsonl", preds)
    report = _build_report(system, dataset_path, list(preds), elapsed, scope)
    (sys_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log_summary(system, scope, report)
    logger.info(
        "[%s/%s] wrote %s and %s",
        system, scope,
        sys_dir / "predictions.jsonl",
        sys_dir / "report.json",
    )
    return report


# ---------------------------------------------------------------------------
# Comparison artifacts (only when two systems were run)
# ---------------------------------------------------------------------------


_COMPARE_METRICS = (
    ("exact_match", "EM"),
    ("f1", "F1"),
    ("em_native", "EM (native)"),
    ("f1_native", "F1 (native)"),
    ("judge_correct", "Judge correct"),
    ("judge_missing", "Judge missing"),
    ("judge_incorrect", "Judge incorrect"),
)


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _write_comparison(
    out_dir: Path, reports: dict[str, dict[str, Any]], scope: str
) -> None:
    bt = reports.get("btgraphrag", {})
    gr = reports.get("graphrag", {})
    bt_m = (bt.get("metrics") or {})
    gr_m = (gr.get("metrics") or {})

    rows: list[dict[str, Any]] = []
    for key, label in _COMPARE_METRICS:
        if key not in bt_m and key not in gr_m:
            continue
        bt_val = bt_m.get(key)
        gr_val = gr_m.get(key)
        delta = None
        if isinstance(bt_val, (int, float)) and isinstance(gr_val, (int, float)):
            delta = bt_val - gr_val
        rows.append(
            {
                "benchmark": "timeqa",
                "scope": scope,
                "metric_key": key,
                "metric": label,
                "graphrag": gr_val,
                "btgraphrag": bt_val,
                "delta": delta,
            }
        )

    compare = {
        "benchmark": "timeqa",
        "scope": scope,
        "systems": {"graphrag": gr, "btgraphrag": bt},
        "headline_metric": "f1_native",
        "rows": rows,
    }
    suffix = f"_{scope}"
    (out_dir / f"compare{suffix}.json").write_text(
        json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (out_dir / f"compare{suffix}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["benchmark", "scope", "metric", "graphrag", "btgraphrag", "delta"]
        )
        for r in rows:
            writer.writerow(
                [
                    r["benchmark"],
                    r["scope"],
                    r["metric"],
                    "" if r["graphrag"] is None else f"{r['graphrag']:.4f}",
                    "" if r["btgraphrag"] is None else f"{r['btgraphrag']:.4f}",
                    "" if r["delta"] is None else f"{r['delta']:+.4f}",
                ]
            )

    lines = [
        f"% Auto-generated by evaluation/timeqa/evaluate.py — scope={scope}, "
        "headline metric: f1_native.",
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Benchmark & Scope & Metric & GraphRAG & BT-GraphRAG & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        gr_cell = "--" if r["graphrag"] is None else f"{r['graphrag']:.3f}"
        bt_cell = "--" if r["btgraphrag"] is None else f"{r['btgraphrag']:.3f}"
        delta_cell = "--" if r["delta"] is None else f"{r['delta']:+.3f}"
        lines.append(
            f"{_latex_escape(r['benchmark'])} & "
            f"{_latex_escape(r['scope'])} & "
            f"{_latex_escape(r['metric'])} & "
            f"{gr_cell} & {bt_cell} & {delta_cell} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (out_dir / f"compare{suffix}.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    logger.info(
        "[%s] wrote comparison: compare%s.{json,csv,tex}",
        scope, suffix,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.timeqa.evaluate",
        description="Evaluate BT-GraphRAG and/or vanilla GraphRAG on TimeQA.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/timeqa/data/timeqa.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/results"),
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=list(SYSTEMS),
        default=list(SYSTEMS),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="DEBUG-level logging (shows internal graphrag/neo4j calls).",
    )
    parser.add_argument("--graphrag-root", type=Path, default=None)
    parser.add_argument("--graphrag-data", type=Path, default=None)
    parser.add_argument(
        "--graphrag-search", choices=["local", "global"], default=None
    )
    parser.add_argument(
        "--scope",
        choices=["global", "per_entity", "both"],
        default="both",
        help=(
            "Evaluation scope. 'global' = raw question against the full index. "
            "'per_entity' = question prepended with the entity name as a soft anchor "
            "(approximates per-entity scoping without separate indexing). "
            "'both' runs each scope and writes separate reports."
        ),
    )
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        return 2

    records = list(iter_records(args.dataset))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        logger.error("No records found in %s", args.dataset)
        return 2

    bt_cfg = BTConfig(dry_run=args.dry_run)
    vanilla_cfg = VanillaConfig(dry_run=args.dry_run)
    if args.graphrag_root is not None:
        vanilla_cfg.root_dir = args.graphrag_root
    if args.graphrag_data is not None:
        vanilla_cfg.data_dir = args.graphrag_data
    if args.graphrag_search is not None:
        vanilla_cfg.search_mode = args.graphrag_search

    opts = _RunOpts(
        use_judge=not args.no_judge,
        dry_run=args.dry_run,
        concurrency=max(1, args.concurrency),
        bt_cfg=bt_cfg,
        vanilla_cfg=vanilla_cfg,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    scopes = list(SCOPES) if args.scope == "both" else [args.scope]

    for scope in scopes:
        reports: dict[str, dict[str, Any]] = {}
        for system in args.systems:
            reports[system] = await _run_system(
                system, records, args.out, args.dataset, opts, scope
            )
        if len(args.systems) >= 2:
            _write_comparison(args.out, reports, scope)

    return 0


_NOISY_LOGGERS = (
    "LiteLLM",
    "litellm",
    "httpx",
    "httpcore",
    "openai",
    "openai._base_client",
    "urllib3",
    "asyncio",
    "neo4j",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    # Silence third-party DEBUG noise even when --verbose is on for our own code.
    third_party_level = logging.DEBUG if args.verbose else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
