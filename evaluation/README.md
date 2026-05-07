# BT-GraphRAG evaluation pipeline

End-to-end evaluation harness for **BT-GraphRAG** against public Q&A and
RAG benchmarks.  Each benchmark lives in its own subfolder under
[`benchmarks/`](./benchmarks) and ships everything needed to reproduce its
results: dataset loader, configuration, evaluation script and README.

```
evaluation/
├── core/                       # shared utilities (runner, judge, metrics, CLI)
│   ├── btgraphrag_runner.py    #   thin async wrapper over bt_graphrag query
│   ├── llm_judge.py            #   correct / incorrect / missing classifier
│   ├── metrics.py              #   EM, token-F1, CRAG truthfulness, …
│   ├── dataset.py              #   common JSONL schema (EvalRecord)
│   └── runner_cli.py           #   shared CLI used by every benchmark
├── benchmarks/
│   ├── crag/                   # Comprehensive RAG (Meta, 2024)
│   ├── hotpotqa/               # HotpotQA distractor multi-hop QA
│   ├── musique/                # MuSiQue 2-/3-/4-hop QA
│   ├── multihop_rag/           # MultiHop-RAG (news, temporal split)
│   └── narrativeqa/            # NarrativeQA long-form QA
├── run_all.py                  # orchestrator – runs every benchmark
├── requirements.txt            # optional deps (datasets, rouge-score, …)
└── README.md
```

## Quick start

```bash
# Optional: install the loaders/scorers that need extra packages
pip install -r evaluation/requirements.txt

# Bring up Neo4j / btgraphrag as usual (see ragtest/PIPELINE_INSTRUCTIONS.md)
export NEO4J_URI="neo4j://127.0.0.1:7687"
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=12345678
export NEO4J_DATABASE=btgraphrag
export BTG_MODEL_ID="<your model id from settings.yaml>"

# 1. Convert one of the public datasets into the harness JSONL schema
python -m evaluation.benchmarks.crag.load_dataset \
    --out evaluation/benchmarks/crag/data/crag.jsonl --limit 200

# 2. Evaluate BT-GraphRAG on it
python -m evaluation.benchmarks.crag.evaluate \
    --dataset evaluation/benchmarks/crag/data/crag.jsonl \
    --out     evaluation/results/crag

# 3. Or run every benchmark in one go
python -m evaluation.run_all --limit 100 --out evaluation/results
```

For a smoke test that does **not** require a running Neo4j or LLM, pass
`--dry-run` (predictions become empty strings, useful to validate the dataset
plumbing in CI):

```bash
python -m evaluation.run_all --benchmarks crag --limit 5 --dry-run
```

## How a benchmark works

Every benchmark is a Python package that defines a single
[`BenchmarkSpec`](./core/runner_cli.py) and delegates the heavy lifting to the
shared harness:

```python
# evaluation/benchmarks/<name>/evaluate.py
from evaluation.core.runner_cli import BenchmarkSpec, run_benchmark

SPEC = BenchmarkSpec(
    name="<name>",
    default_dataset=Path("evaluation/benchmarks/<name>/data/<name>.jsonl"),
    use_llm_judge=True,
)

if __name__ == "__main__":
    run_benchmark(SPEC)
```

The harness handles CLI parsing, async orchestration, JSONL IO, the LLM judge
and report generation.  Adding a new benchmark therefore boils down to:

1. Create `evaluation/benchmarks/<name>/` with `__init__.py`, `config.yaml`,
   `README.md`, `load_dataset.py` and `evaluate.py`.
2. In `load_dataset.py`, convert the upstream dataset into the JSONL schema
   defined by [`EvalRecord`](./core/dataset.py)
   (`qid`, `question`, `answer`, optional `aliases`, `question_time`, `context`).
3. In `evaluate.py`, instantiate a `BenchmarkSpec` and call `run_benchmark`.
4. Register the benchmark in `evaluation/run_all.py::BENCHMARKS`.

## Outputs

Each benchmark writes two files under its `--out` directory:

| File                | Description                                                  |
|---------------------|--------------------------------------------------------------|
| `predictions.jsonl` | One line per question with prediction, judge label, metrics. |
| `report.json`       | Aggregate metrics + CRAG truthfulness + per-category breakdowns. |

`run_all.py` additionally writes `evaluation/results/summary.json`
combining every benchmark's report.

## Metric: CRAG truthfulness

Following the [CRAG paper](https://github.com/facebookresearch/CRAG), every
free-form answer is graded by an LLM judge as `correct`, `incorrect` or
`missing`, and the headline metric is

```
truthfulness = (#correct − #incorrect) / #total
```

`missing` (refusal / "I don't know") is treated neutrally so a model is
incentivised to abstain rather than hallucinate.  When no LLM judge is
configured the harness falls back to a deterministic string-overlap heuristic,
so the pipeline still produces numbers in CI.

## Environment variables

| Variable             | Purpose                                                |
|----------------------|--------------------------------------------------------|
| `NEO4J_URI`          | BT-GraphRAG Neo4j connection URI.                      |
| `NEO4J_USER`         | Neo4j user.                                            |
| `NEO4J_PASSWORD`     | Neo4j password.                                        |
| `NEO4J_DATABASE`     | Neo4j database (default `btgraphrag`).                 |
| `BTG_MODEL_ID`       | LLM id used for answer synthesis.                      |
| `BTG_JUDGE_MODEL_ID` | LLM id used by the judge (defaults to `BTG_MODEL_ID`). |
