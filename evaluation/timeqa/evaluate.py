"""Run BT-GraphRAG and/or vanilla GraphRAG on TimeQA hard (open-domain RAG)."""

from __future__ import annotations

import os

# Silence LiteLLM's verbose request/response dumps before it gets imported anywhere.
os.environ.setdefault("LITELLM_LOG", "WARNING")

import argparse
import asyncio
import csv
import json
import logging
import re
import shutil
import subprocess
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
from evaluation.timeqa.load import main as load_main
from scripts.wipe_neo4j_dbs import DEFAULT_DBS as WIPE_DEFAULT_DBS
from scripts.wipe_neo4j_dbs import main as wipe_neo4j_main

logger = logging.getLogger(__name__)

SYSTEMS = ("btgraphrag", "graphrag")


@dataclass
class _RunOpts:
    use_judge: bool
    dry_run: bool
    concurrency: int
    bt_cfg: BTConfig
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
    """Overwrite <ragtest_root>/.env with GRAPHRAG_API_KEY from the global env."""
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
    """Copy every *.txt under `corpus_dir` into <ragtest_root>/input/."""
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


def _setup_ragtest(
    ragtest_root: Path,
    corpus_dir: Path,
    init_model: str,
    init_embedding: str,
) -> None:
    """Wipe the ragtest dir, run `poe init`, write env + corpus, run `poe index`."""
    logger.info("=" * 72)
    logger.info("[setup] preparing ragtest at %s", ragtest_root)
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
    _copy_corpus(corpus_dir, ragtest_root)
    _run_poe("index", "--root", str(ragtest_root))
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


async def _eval_one(
    system: str,
    rec: EvalRecord,
    opts: _RunOpts,
    sem: asyncio.Semaphore,
) -> EvalPrediction:
    async with sem:
        started = time.perf_counter()
        if system == "btgraphrag":
            result = await bt_answer(rec.question, config=opts.bt_cfg)
        else:
            result = await vanilla_answer(rec.question, config=opts.vanilla_cfg)

        prediction = (result.get("answer") or "").strip()

        judge = await judge_answer(
            rec.question, rec.answer, prediction, use_llm=opts.use_judge
        )

        metrics = _default_scorer(rec, prediction, judge.label)
        ctx: dict[str, Any] = {**rec.context, "raw": result.get("raw", {})}
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


_REPORT_METRIC_KEYS = {"judge_correct", "judge_incorrect"}


def _judge_only(metrics: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if k in _REPORT_METRIC_KEYS}


_MODEL_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_model_name(name: str) -> str:
    stem = _MODEL_SAFE_RE.sub("_", name).strip("_")
    return stem or "unknown"


def _resolve_model(system: str, opts: "_RunOpts") -> str:
    """Best-effort identification of the model used to answer questions."""
    if system == "btgraphrag":
        return (opts.bt_cfg.model_id or os.getenv("BTG_MODEL_ID") or "unknown").strip() or "unknown"
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
    system: str,
    dataset_path: Path,
    predictions: list[EvalPrediction],
    elapsed: float,
    tag: str = "",
    model: str = "",
) -> dict[str, Any]:
    metrics_all = _judge_only(aggregate(p.metrics for p in predictions))
    labels = [p.judge_label for p in predictions if p.judge_label is not None]
    truth = truthfulness(labels) if labels else {}

    report: dict[str, Any] = {
        "benchmark": "timeqa",
        "split": "hard",
        "system": system,
        "dataset": str(dataset_path),
        "num_examples": len(predictions),
        "elapsed_seconds": round(elapsed, 3),
        "metrics": metrics_all,
        "truthfulness": truth,
    }
    if tag:
        report["tag"] = tag
    if model:
        report["model"] = model
    return report


_JUDGE_GLYPH = {"correct": "✓", "incorrect": "✗"}


def _log_prediction(idx: int, total: int, system: str, p: EvalPrediction) -> None:
    glyph = _JUDGE_GLYPH.get(p.judge_label or "", "·")
    label = (p.judge_label or "n/a").upper()
    reason = p.judge_reason or ""
    judge_line = f"{glyph} {label}" + (f" — {reason}" if reason else "")
    err = p.context.get("_error")
    print()
    print(f"[{system}] {idx}/{total}")
    print(f"  Q:     {p.question}")
    print(f"  Pred:  {p.prediction or '<empty>'}")
    print(f"  Gold:  {p.gold_answer or '<none>'}")
    print(f"  Judge: {judge_line}")
    if err:
        print(f"  ERROR: {err}")


def _log_summary(system: str, report: dict[str, Any]) -> None:
    m = report.get("metrics") or {}
    truth = report.get("truthfulness") or {}
    n = report.get("num_examples", 0)
    elapsed = report.get("elapsed_seconds", 0.0)
    logger.info("-" * 72)
    logger.info("[%s] SUMMARY — %d examples in %.1fs", system, n, elapsed)
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
    system: str,
    records: list[EvalRecord],
    out_dir: Path,
    dataset_path: Path,
    opts: _RunOpts,
    tag: str = "",
) -> dict[str, Any]:
    model_raw = _resolve_model(system, opts)
    model_dir = _safe_model_name(model_raw)
    label = f"{system}" + (f"/{tag}" if tag else "") + f"/{model_dir}"
    logger.info("=" * 72)
    logger.info(
        "[%s] %d examples (concurrency=%d, judge=%s, dry_run=%s)",
        label, len(records), opts.concurrency,
        "off" if not opts.use_judge else "on",
        opts.dry_run,
    )
    logger.info("=" * 72)
    sem = asyncio.Semaphore(opts.concurrency)
    started = time.perf_counter()
    total = len(records)
    tasks = [
        asyncio.create_task(_eval_one(system, rec, opts, sem))
        for rec in records
    ]
    preds_by_qid: dict[str, EvalPrediction] = {}
    done_count = 0
    for fut in asyncio.as_completed(tasks):
        pred = await fut
        done_count += 1
        _log_prediction(done_count, total, system, pred)
        preds_by_qid[pred.qid] = pred
    # Preserve original input order in outputs.
    preds = [preds_by_qid[r.qid] for r in records if r.qid in preds_by_qid]
    elapsed = time.perf_counter() - started

    sys_dir = out_dir / system / "timeqa"
    if tag:
        sys_dir = sys_dir / tag
    sys_dir = sys_dir / model_dir
    sys_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(sys_dir / "predictions.jsonl", preds)
    report = _build_report(
        system, dataset_path, list(preds), elapsed, tag, model_raw
    )
    (sys_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log_summary(system, report)
    logger.info(
        "[%s] wrote %s and %s",
        label,
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
    out_dir: Path, reports: dict[str, dict[str, Any]], tag: str = ""
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
                "metric_key": key,
                "metric": label,
                "graphrag": gr_val,
                "btgraphrag": bt_val,
                "delta": delta,
            }
        )

    compare = {
        "benchmark": "timeqa",
        "split": "hard",
        "systems": {"graphrag": gr, "btgraphrag": bt},
        "headline_metric": "f1_native",
        "rows": rows,
    }
    if tag:
        compare["tag"] = tag
    suffix = f"_{tag}" if tag else ""
    compare_dir = out_dir / tag if tag else out_dir
    compare_dir.mkdir(parents=True, exist_ok=True)
    (compare_dir / f"compare{suffix}.json").write_text(
        json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (compare_dir / f"compare{suffix}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["benchmark", "metric", "graphrag", "btgraphrag", "delta"])
        for r in rows:
            writer.writerow(
                [
                    r["benchmark"],
                    r["metric"],
                    "" if r["graphrag"] is None else f"{r['graphrag']:.4f}",
                    "" if r["btgraphrag"] is None else f"{r['btgraphrag']:.4f}",
                    "" if r["delta"] is None else f"{r['delta']:+.4f}",
                ]
            )

    lines = [
        "% Auto-generated by evaluation/timeqa/evaluate.py — headline metric: f1_native.",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Benchmark & Metric & GraphRAG & BT-GraphRAG & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        gr_cell = "--" if r["graphrag"] is None else f"{r['graphrag']:.3f}"
        bt_cell = "--" if r["btgraphrag"] is None else f"{r['btgraphrag']:.3f}"
        delta_cell = "--" if r["delta"] is None else f"{r['delta']:+.3f}"
        lines.append(
            f"{_latex_escape(r['benchmark'])} & "
            f"{_latex_escape(r['metric'])} & "
            f"{gr_cell} & {bt_cell} & {delta_cell} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (compare_dir / f"compare{suffix}.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    logger.info(
        "wrote comparison: %s/compare%s.{json,csv,tex}", compare_dir, suffix,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.timeqa.evaluate",
        description="Evaluate BT-GraphRAG and/or vanilla GraphRAG on TimeQA hard.",
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
    parser.add_argument("--concurrency", type=int, default=100)
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
            "Maximum number of unique Wikipedia entities (corpus *.txt files) "
            "the load step should keep. Forwarded to "
            "`python -m evaluation.timeqa.load --max-entities`."
        ),
    )
    parser.add_argument(
        "--load-seed",
        type=int,
        default=None,
        help="Seed for --max-entities sampling in the load step.",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip the load step (assumes the corpus + JSONL already exist).",
    )
    parser.add_argument(
        "--wipe-dbs",
        type=str,
        default=None,
        help=(
            "Comma-separated Neo4j databases to DROP+CREATE before indexing. "
            f"Defaults to all six known DBs ({','.join(WIPE_DEFAULT_DBS)}). "
            "Pass an empty string to disable, or a subset like "
            "'btgraphrag,etcdreval'."
        ),
    )
    parser.add_argument(
        "--skip-wipe-dbs",
        action="store_true",
        help="Skip the Neo4j wipe step entirely.",
    )
    parser.add_argument(
        "--init-model",
        type=str,
        default="gpt-4.1",
        help=(
            "Chat model to write into settings.yaml during `poe init`. "
            "Hardcoded to gpt-4.1 so vanilla GraphRAG's model is "
            "independent of BTG_MODEL_ID; override explicitly if needed."
        ),
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
        help=(
            "Optional label for this run. When set, outputs are written under "
            "<out>/<system>/timeqa/<tag>/<model>/ and comparison files under "
            "<out>/<tag>/compare_<tag>.{json,csv,tex}. The tag is also recorded "
            "inside report.json."
        ),
    )
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    # Load step: regenerate <args.dataset> and <args.corpus> from hard.json.
    # Always runs unless --skip-load; --max-entities caps the corpus size.
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

    bt_cfg = BTConfig(dry_run=args.dry_run)
    vanilla_cfg = VanillaConfig(dry_run=args.dry_run)
    if args.graphrag_root is not None:
        vanilla_cfg.root_dir = args.graphrag_root
    if args.graphrag_data is not None:
        vanilla_cfg.data_dir = args.graphrag_data
    if args.graphrag_search is not None:
        vanilla_cfg.search_mode = args.graphrag_search

    # Neo4j wipe: DROP+CREATE the target databases so `poe index` writes into
    # a clean store. Defaults to just the BT-GraphRAG database this eval uses.
    if not args.skip_wipe_dbs and not args.dry_run:
        if args.wipe_dbs is None:
            wipe_targets = list(WIPE_DEFAULT_DBS)
        else:
            wipe_targets = [d.strip() for d in args.wipe_dbs.split(",") if d.strip()]
        if wipe_targets:
            logger.info("=" * 72)
            logger.info("[wipe] resetting Neo4j databases: %s", wipe_targets)
            logger.info("=" * 72)
            rc = await wipe_neo4j_main(
                uri=bt_cfg.neo4j_uri,
                user=bt_cfg.neo4j_user,
                password=bt_cfg.neo4j_password,
                dbs=wipe_targets,
                dry_run=False,
            )
            if rc != 0:
                logger.warning("Neo4j wipe finished with warnings (rc=%d)", rc)

    # Setup phase: wipe the ragtest dir, run `poe init`, drop in the global
    # API key, copy the corpus produced by load.py, then `poe index`. The
    # bitemporal indexing pipeline emits both the parquet outputs vanilla
    # GraphRAG queries and the Neo4j graph BT-GraphRAG queries, so a single
    # index run powers both systems in the query phase below.
    if not args.skip_setup and not args.dry_run:
        ragtest_root = Path(vanilla_cfg.root_dir).resolve()
        try:
            _setup_ragtest(
                ragtest_root=ragtest_root,
                corpus_dir=args.corpus.resolve(),
                init_model=args.init_model,
                init_embedding=args.init_embedding,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Setup failed: %s", exc)
            return 2

    opts = _RunOpts(
        use_judge=not args.no_judge,
        dry_run=args.dry_run,
        concurrency=max(1, args.concurrency),
        bt_cfg=bt_cfg,
        vanilla_cfg=vanilla_cfg,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = (args.tag or "").strip()

    # Share a single Neo4j driver across all BT-GraphRAG calls. Each driver
    # owns its own connection pool, so creating one per question at
    # concurrency=100 spawns 100 simultaneous pools and Neo4j drops
    # connections ("Failed to read from defunct connection").
    shared_bt_driver = None
    if "btgraphrag" in args.systems and not args.dry_run:
        try:
            from neo4j import AsyncGraphDatabase

            shared_bt_driver = AsyncGraphDatabase.driver(
                bt_cfg.neo4j_uri,
                auth=(bt_cfg.neo4j_user, bt_cfg.neo4j_password),
                max_connection_pool_size=max(50, opts.concurrency),
            )
            bt_cfg.driver = shared_bt_driver
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create shared Neo4j driver: %s", exc)

    try:
        reports: dict[str, dict[str, Any]] = {}
        for system in args.systems:
            reports[system] = await _run_system(
                system, records, args.out, args.dataset, opts, tag
            )
        if len(args.systems) >= 2:
            _write_comparison(args.out, reports, tag)
    finally:
        if shared_bt_driver is not None:
            try:
                await shared_bt_driver.close()
            except Exception:  # noqa: BLE001
                pass

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
