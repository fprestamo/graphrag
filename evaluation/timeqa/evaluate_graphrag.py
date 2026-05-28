"""Run vanilla GraphRAG (original Microsoft pipeline) on TimeQA hard.

Indexes the corpus under `.ragtestgraphrag` using the *standard* indexing
method (no bitemporal extension) and answers the same TimeQA questions with
vanilla GraphRAG local/global search. Uses the same chat model + embedding
model as the BT-GraphRAG eval (gpt-4.1-mini / text-embedding-3-large) and
keeps `max_gleanings: 0` in extract_graph — that field already ships at 0 in
the default `poe init` settings.yaml so we just rely on it.
"""

from __future__ import annotations

import os

# Silence LiteLLM's verbose request/response dumps before it gets imported anywhere.
os.environ.setdefault("LITELLM_LOG", "WARNING")

import argparse
import asyncio
import json
import logging
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
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
from evaluation.core.runners import VanillaConfig, vanilla_answer
from evaluation.core.token_tracker import active_system, get_tracker
from evaluation.timeqa.load import main as load_main

logger = logging.getLogger(__name__)

SYSTEM = "graphrag"


@dataclass
class _RunOpts:
    use_judge: bool
    dry_run: bool
    concurrency: int
    vanilla_cfg: VanillaConfig


# ---------------------------------------------------------------------------
# Ragtest setup (init → env → corpus → index) — runs before the query phase.
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_poe(task: str, *extra: str) -> None:
    """Invoke `uv run poe <task> <extra>` from the repo root, streaming output."""
    cmd = ["uv", "run", "poe", task, *extra]
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` exited with code {proc.returncode}"
        )


def _write_ragtest_env(ragtest_root: Path) -> None:
    api_key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GRAPHRAG_API_KEY (or OPENAI_API_KEY) is not set in the environment; "
            "cannot configure the ragtest .env file."
        )
    env_path = ragtest_root / ".env"
    env_path.write_text(f"GRAPHRAG_API_KEY={api_key}\n", encoding="utf-8")
    logger.info("Wrote %s (api key from global env)", env_path)


def _copy_corpus(corpus_dir: Path, ragtest_root: Path) -> int:
    if not corpus_dir.is_dir():
        raise RuntimeError(
            f"Corpus directory not found: {corpus_dir}. "
            "Run `python -m evaluation.timeqa.load` first."
        )
    input_dir = ragtest_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in corpus_dir.glob("*.txt"):
        shutil.copy2(src, input_dir / src.name)
        n += 1
    if n == 0:
        raise RuntimeError(f"No .txt documents found under {corpus_dir}.")
    logger.info("Copied %d documents from %s to %s", n, corpus_dir, input_dir)
    return n


def _enforce_max_gleanings_zero(ragtest_root: Path) -> None:
    """Make sure extract_graph.max_gleanings stays at 0 in the generated settings.yaml."""
    settings_path = ragtest_root / "settings.yaml"
    if not settings_path.is_file():
        return
    text = settings_path.read_text(encoding="utf-8")
    new_text = re.sub(
        r"(extract_graph:\s*(?:\n\s+[^\n]+)*?\n\s+max_gleanings:\s*)\d+",
        r"\g<1>0",
        text,
    )
    if new_text != text:
        settings_path.write_text(new_text, encoding="utf-8")
        logger.info("[setup] pinned extract_graph.max_gleanings to 0 in %s", settings_path)


def _setup_ragtest(
    ragtest_root: Path,
    corpus_dir: Path,
    init_model: str,
    init_embedding: str,
    index_method: str,
) -> None:
    """Wipe the ragtest dir, run `poe init`, write env + corpus, run `poe index`."""
    logger.info("=" * 72)
    logger.info("[setup] preparing ragtest at %s (method=%s)", ragtest_root, index_method)
    logger.info("=" * 72)

    if ragtest_root.exists():
        logger.info("[setup] removing existing %s", ragtest_root)
        shutil.rmtree(ragtest_root)

    _run_poe(
        "init",
        "--root", str(ragtest_root),
        "-m", init_model,
        "-e", init_embedding,
        "--force",
    )
    _write_ragtest_env(ragtest_root)
    _enforce_max_gleanings_zero(ragtest_root)
    _copy_corpus(corpus_dir, ragtest_root)
    _run_poe("index", "--root", str(ragtest_root), "--method", index_method)
    logger.info("[setup] ragtest ready at %s", ragtest_root)


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
        metrics["judge_incorrect"] = 1.0 if judge_label == "incorrect" else 0.0
    return metrics


_PROGRESS_COUNTERS = {"started": 0, "finished": 0, "total": 0}


async def _eval_one(
    rec: EvalRecord,
    opts: _RunOpts,
    sem: asyncio.Semaphore,
) -> EvalPrediction:
    async with sem:
        _PROGRESS_COUNTERS["started"] += 1
        start_idx = _PROGRESS_COUNTERS["started"]
        total = _PROGRESS_COUNTERS["total"] or "?"
        started = time.perf_counter()
        print(
            f"  [q-start] {start_idx:>3d}/{total} "
            f"qid={rec.qid} mode={opts.vanilla_cfg.search_mode}: "
            f"{rec.question[:80]}",
            flush=True,
        )
        result = await vanilla_answer(rec.question, config=opts.vanilla_cfg)
        prediction = (result.get("answer") or "").strip()

        elapsed_search = time.perf_counter() - started
        _PROGRESS_COUNTERS["finished"] += 1
        print(
            f"  [q-done ] {_PROGRESS_COUNTERS['finished']:>3d}/{total} "
            f"qid={rec.qid} {elapsed_search:6.1f}s "
            f"answer={prediction[:60]!r}",
            flush=True,
        )

        judge = await judge_answer(
            rec.question, rec.answer, prediction, use_llm=opts.use_judge
        )

        metrics = _default_scorer(rec, prediction, judge.label)
        ctx: dict[str, Any] = {**rec.context, "raw": result.get("raw", {})}
        ctx["_elapsed"] = round(time.perf_counter() - started, 2)
        raw = result.get("raw") or {}
        if isinstance(raw, dict) and "error" in raw:
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


_REPORT_METRIC_KEYS = {"judge_correct", "judge_incorrect"}


def _judge_only(metrics: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if k in _REPORT_METRIC_KEYS}


_MODEL_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_model_name(name: str) -> str:
    stem = _MODEL_SAFE_RE.sub("_", name).strip("_")
    return stem or "unknown"


def _resolve_model(opts: "_RunOpts") -> str:
    root = Path(opts.vanilla_cfg.root_dir)
    for cfg_name in ("settings.yaml", "settings.yml"):
        cfg_path = root / cfg_name
        if not cfg_path.is_file():
            continue
        try:
            import yaml  # type: ignore[import-untyped]

            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            models = data.get("completion_models") or {}
            for v in models.values():
                if isinstance(v, dict) and v.get("model"):
                    return str(v["model"])
        except Exception:  # noqa: BLE001
            break
    return "unknown"


def _build_report(
    dataset_path: Path,
    predictions: list[EvalPrediction],
    elapsed: float,
    tag: str = "",
    model: str = "",
    search_mode: str = "local",
    index_method: str = "standard",
) -> dict[str, Any]:
    metrics_all = _judge_only(aggregate(p.metrics for p in predictions))
    labels = [p.judge_label for p in predictions if p.judge_label is not None]
    truth = truthfulness(labels) if labels else {}

    report: dict[str, Any] = {
        "benchmark": "timeqa",
        "split": "hard",
        "system": SYSTEM,
        "dataset": str(dataset_path),
        "num_examples": len(predictions),
        "elapsed_seconds": round(elapsed, 3),
        "metrics": metrics_all,
        "truthfulness": truth,
        "search_mode": search_mode,
        "index_method": index_method,
    }
    if tag:
        report["tag"] = tag
    if model:
        report["model"] = model
    return report


_JUDGE_GLYPH = {"correct": "✓", "incorrect": "✗"}


def _log_prediction(
    idx: int,
    total: int,
    p: EvalPrediction,
    tally: dict[str, int] | None = None,
) -> None:
    glyph = _JUDGE_GLYPH.get(p.judge_label or "", "·")
    label = (p.judge_label or "n/a").upper()
    reason = p.judge_reason or ""
    judge_line = f"{glyph} {label}" + (f" — {reason}" if reason else "")
    err = p.context.get("_error")
    print()
    print(f"[{SYSTEM}] {idx}/{total}")
    if tally is not None:
        ok = tally.get("correct", 0)
        bad = tally.get("incorrect", 0)
        other = tally.get("other", 0)
        seen = ok + bad + other
        acc = (ok / seen) if seen else 0.0
        running = f"  Running: ✓ {ok}  ✗ {bad}"
        if other:
            running += f"  · {other}"
        running += f"   (acc={acc:.3f})"
        print(running)
    print(f"  Q:     {p.question}")
    print(f"  Pred:  {p.prediction or '<empty>'}")
    print(f"  Gold:  {p.gold_answer or '<none>'}")
    print(f"  Judge: {judge_line}")
    if err:
        print(f"  ERROR: {err}")


def _log_summary(report: dict[str, Any]) -> None:
    m = report.get("metrics") or {}
    truth = report.get("truthfulness") or {}
    n = report.get("num_examples", 0)
    elapsed = report.get("elapsed_seconds", 0.0)
    logger.info("-" * 72)
    logger.info("[%s] SUMMARY — %d examples in %.1fs", SYSTEM, n, elapsed)
    logger.info(
        "  EM=%.3f  F1=%.3f  EM_native=%.3f  F1_native=%.3f",
        m.get("exact_match", 0.0), m.get("f1", 0.0),
        m.get("em_native", 0.0), m.get("f1_native", 0.0),
    )
    if truth:
        logger.info(
            "  Judge — accuracy=%.3f  (correct=%.0f%%, incorrect=%.0f%%)",
            truth.get("accuracy", 0.0),
            100 * truth.get("accuracy", 0.0),
            100 * truth.get("incorrect_rate", 0.0),
        )
    logger.info("-" * 72)


async def _run_system(
    records: list[EvalRecord],
    out_dir: Path,
    dataset_path: Path,
    opts: _RunOpts,
    tag: str = "",
    index_method: str = "standard",
) -> dict[str, Any]:
    model_raw = _resolve_model(opts)
    model_dir = _safe_model_name(model_raw)
    label = f"{SYSTEM}" + (f"/{tag}" if tag else "") + f"/{model_dir}"
    logger.info("=" * 72)
    logger.info(
        "[%s] %d examples (concurrency=%d, judge=%s, dry_run=%s, search=%s)",
        label, len(records), opts.concurrency,
        "off" if not opts.use_judge else "on",
        opts.dry_run,
        opts.vanilla_cfg.search_mode,
    )
    logger.info("=" * 72)
    sem = asyncio.Semaphore(opts.concurrency)
    started = time.perf_counter()
    total = len(records)
    _PROGRESS_COUNTERS["started"] = 0
    _PROGRESS_COUNTERS["finished"] = 0
    _PROGRESS_COUNTERS["total"] = total
    tasks = [
        asyncio.create_task(_eval_one(rec, opts, sem))
        for rec in records
    ]
    preds_by_qid: dict[str, EvalPrediction] = {}
    done_count = 0
    tally = {"correct": 0, "incorrect": 0, "other": 0}
    for fut in asyncio.as_completed(tasks):
        pred = await fut
        done_count += 1
        if pred.judge_label == "correct":
            tally["correct"] += 1
        elif pred.judge_label == "incorrect":
            tally["incorrect"] += 1
        else:
            tally["other"] += 1
        _log_prediction(done_count, total, pred, tally=tally)
        preds_by_qid[pred.qid] = pred
    preds = [preds_by_qid[r.qid] for r in records if r.qid in preds_by_qid]
    elapsed = time.perf_counter() - started

    sys_dir = out_dir / SYSTEM / "timeqa"
    if tag:
        sys_dir = sys_dir / tag
    sys_dir = sys_dir / model_dir
    sys_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(sys_dir / "predictions.jsonl", preds)
    report = _build_report(
        dataset_path,
        list(preds),
        elapsed,
        tag,
        model_raw,
        search_mode=opts.vanilla_cfg.search_mode,
        index_method=index_method,
    )
    (sys_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log_summary(report)
    logger.info(
        "[%s] wrote %s and %s",
        label,
        sys_dir / "predictions.jsonl",
        sys_dir / "report.json",
    )
    return report


def _system_report_dir(
    out_dir: Path, tag: str, report: dict[str, Any]
) -> Path | None:
    model_raw = (report.get("model") or "unknown").strip() or "unknown"
    model_dir = _safe_model_name(model_raw)
    sys_dir = out_dir / SYSTEM / "timeqa"
    if tag:
        sys_dir = sys_dir / tag
    sys_dir = sys_dir / model_dir
    return sys_dir if sys_dir.is_dir() else None


# ---------------------------------------------------------------------------
# Multi-run summary
# ---------------------------------------------------------------------------


_SUMMARY_METRIC_KEYS = (
    "exact_match", "f1", "em_native", "f1_native",
    "judge_correct", "judge_incorrect",
)
_SUMMARY_TRUTH_KEYS = (
    "accuracy", "correct_rate", "incorrect_rate",
)
_SUMMARY_TOKEN_KEYS = (
    "calls", "prompt_tokens", "completion_tokens", "total_tokens",
)


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def _write_multirun_summary(
    multirun_dir: Path,
    runs: list[dict[str, Any]],
    tag: str,
) -> None:
    summary: dict[str, Any] = {
        "benchmark": "timeqa",
        "split": "hard",
        "system": SYSTEM,
        "tag": tag,
        "num_runs": len(runs),
        "runs": [
            {
                "run": r["run"],
                "out_dir": r["out_dir"],
                "report": {
                    "metrics": r["report"].get("metrics", {}),
                    "truthfulness": r["report"].get("truthfulness", {}),
                    "elapsed_seconds": r["report"].get("elapsed_seconds"),
                    "num_examples": r["report"].get("num_examples"),
                    "model": r["report"].get("model"),
                    "llm_tokens": r["report"].get("llm_tokens", {}),
                },
            }
            for r in runs
        ],
    }

    metric_series: dict[str, list[float]] = {k: [] for k in _SUMMARY_METRIC_KEYS}
    truth_series: dict[str, list[float]] = {k: [] for k in _SUMMARY_TRUTH_KEYS}
    token_series: dict[str, list[float]] = {k: [] for k in _SUMMARY_TOKEN_KEYS}
    elapsed_series: list[float] = []
    for r in runs:
        rep = r["report"]
        m = rep.get("metrics") or {}
        for k in _SUMMARY_METRIC_KEYS:
            if k in m and isinstance(m[k], (int, float)):
                metric_series[k].append(float(m[k]))
        t = rep.get("truthfulness") or {}
        for k in _SUMMARY_TRUTH_KEYS:
            if k in t and isinstance(t[k], (int, float)):
                truth_series[k].append(float(t[k]))
        tok = r["tokens"] or {}
        for k in _SUMMARY_TOKEN_KEYS:
            if k in tok and isinstance(tok[k], (int, float)):
                token_series[k].append(float(tok[k]))
        if isinstance(rep.get("elapsed_seconds"), (int, float)):
            elapsed_series.append(float(rep["elapsed_seconds"]))

    summary["stats"] = {
        "metrics_stats": {k: _mean_std(v) for k, v in metric_series.items() if v},
        "truthfulness_stats": {k: _mean_std(v) for k, v in truth_series.items() if v},
        "token_stats": {k: _mean_std(v) for k, v in token_series.items() if v},
        "elapsed_seconds_stats": _mean_std(elapsed_series),
        "total_tokens_across_runs": int(sum(token_series.get("total_tokens", []))),
    }

    out_path = multirun_dir / "summary.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("[multirun] wrote %s", out_path)
    stats = summary["stats"]
    acc = (stats["truthfulness_stats"].get("accuracy") or {}).get("mean", 0.0)
    acc_std = (stats["truthfulness_stats"].get("accuracy") or {}).get("std", 0.0)
    f1 = (stats["metrics_stats"].get("f1") or {}).get("mean", 0.0)
    toks = (stats["token_stats"].get("total_tokens") or {}).get("mean", 0.0)
    logger.info(
        "[multirun] %s — acc=%.3f±%.3f  f1=%.3f  mean_tokens/run=%.0f  total=%d",
        SYSTEM, acc, acc_std, f1, toks,
        stats.get("total_tokens_across_runs", 0),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.timeqa.evaluate_graphrag",
        description=(
            "Evaluate vanilla (original) GraphRAG on TimeQA hard. Indexes the "
            "corpus under .ragtestgraphrag using the standard method and runs "
            "local/global search to answer the same questions."
        ),
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="DEBUG-level logging.",
    )
    parser.add_argument(
        "--graphrag-root",
        type=Path,
        default=Path(".ragtestgraphrag"),
        help="Root directory for the vanilla GraphRAG project (default: .ragtestgraphrag).",
    )
    parser.add_argument(
        "--search",
        choices=["local", "global"],
        default="local",
        help="GraphRAG search mode to use at query time (default: local).",
    )
    parser.add_argument(
        "--index-method",
        choices=["standard", "fast"],
        default="standard",
        help="Indexing method passed to `python -m graphrag index` (default: standard).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("evaluation/timeqa/data/corpus"),
        help="Directory holding the corpus *.txt files produced by load.py.",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=None,
        help=(
            "Maximum number of unique Wikipedia entities to keep when "
            "regenerating the corpus."
        ),
    )
    parser.add_argument(
        "--load-seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip the load step (assumes the corpus + JSONL already exist).",
    )
    parser.add_argument(
        "--init-model",
        type=str,
        default="gpt-4.1-mini",
        help="Chat model to write into settings.yaml during `poe init`.",
    )
    parser.add_argument(
        "--init-embedding",
        type=str,
        default=os.getenv("BTG_EMBEDDING_MODEL_ID", "text-embedding-3-large"),
        help="Embedding model to write into settings.yaml during `poe init`.",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip ragtest wipe + init + corpus copy + index (use existing index).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional label written into the output path and report.json.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Run the query phase K times; writes <out>/multirun/<ts>/run_<k>/...",
    )
    parser.add_argument(
        "--multirun-tag",
        type=str,
        default="",
        help="Name the multirun directory instead of the auto timestamp.",
    )
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    if not args.skip_load:
        logger.info("=" * 72)
        logger.info("[load] regenerating dataset + corpus")
        logger.info("=" * 72)
        load_argv: list[str] = [
            "--out-jsonl", str(args.dataset),
            "--out-corpus", str(args.corpus),
        ]
        if args.max_entities is not None:
            load_argv += ["--max-entities", str(args.max_entities)]
        if args.load_seed is not None:
            load_argv += ["--seed", str(args.load_seed)]
        rc = load_main(load_argv)
        if rc != 0:
            logger.error("Load step failed (exit code %d)", rc)
            return rc

    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        return 2

    records = list(iter_records(args.dataset))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        logger.error("No records found in %s", args.dataset)
        return 2

    vanilla_cfg = VanillaConfig(dry_run=args.dry_run)
    vanilla_cfg.root_dir = args.graphrag_root
    vanilla_cfg.search_mode = args.search

    if not args.skip_setup and not args.dry_run:
        ragtest_root = Path(vanilla_cfg.root_dir).resolve()
        try:
            _setup_ragtest(
                ragtest_root=ragtest_root,
                corpus_dir=args.corpus.resolve(),
                init_model=args.init_model,
                init_embedding=args.init_embedding,
                index_method=args.index_method,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Setup failed: %s", exc)
            return 2

    opts = _RunOpts(
        use_judge=not args.no_judge,
        dry_run=args.dry_run,
        concurrency=max(1, args.concurrency),
        vanilla_cfg=vanilla_cfg,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = (args.tag or "").strip()
    runs = max(1, int(getattr(args, "runs", 1) or 1))

    tracker = get_tracker()
    multirun_dir: Path | None = None
    if runs > 1:
        multirun_label = (args.multirun_tag or "").strip() or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        multirun_dir = args.out / "multirun" / multirun_label
        multirun_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 72)
        logger.info("[multirun] %d runs → %s", runs, multirun_dir)
        logger.info("=" * 72)

    per_run_records: list[dict[str, Any]] = []
    for k in range(1, runs + 1):
        run_out = (multirun_dir / f"run_{k}") if multirun_dir else args.out
        run_out.mkdir(parents=True, exist_ok=True)
        if runs > 1:
            logger.info("=" * 72)
            logger.info("[multirun] run %d/%d → %s", k, runs, run_out)
            logger.info("=" * 72)

        tracker.reset()
        with active_system(SYSTEM):
            report = await _run_system(
                records, run_out, args.dataset, opts, tag, args.index_method
            )
        snap = tracker.snapshot().get(SYSTEM, {})
        report["llm_tokens"] = snap
        sys_dir = _system_report_dir(run_out, tag, report)
        if sys_dir is not None:
            (sys_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        per_run_records.append({
            "run": k,
            "out_dir": str(run_out),
            "report": report,
            "tokens": snap,
        })

    if multirun_dir is not None:
        _write_multirun_summary(multirun_dir, per_run_records, tag)

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
)

_VERY_NOISY_LOGGERS = (
    "graphrag.query.context_builder.community_context",
    "graphrag.query.structured_search.local_search.mixed_context",
    "graphrag.query.structured_search.basic_search.basic_context",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    third_party_level = logging.DEBUG if args.verbose else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)
    if not args.verbose:
        for name in _VERY_NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
