# CRAG benchmark

[CRAG](https://github.com/facebookresearch/CRAG) (Comprehensive RAG Benchmark,
Meta 2024) is a Q&A benchmark covering five domains (finance, sports, music,
movie, open) and eight question types (simple, simple-with-condition, set,
comparison, aggregation, multi-hop, post-processing, false-premise).

## Usage

```bash
# 1. Convert the public CRAG release to the harness JSONL schema
python -m evaluation.benchmarks.crag.load_dataset \
    --out evaluation/benchmarks/crag/data/crag.jsonl \
    --limit 500

# 2. Run BT-GraphRAG against it
python -m evaluation.benchmarks.crag.evaluate \
    --dataset evaluation/benchmarks/crag/data/crag.jsonl \
    --out     evaluation/results/crag
```

## Metric

Following the CRAG paper, every answer is graded by an LLM judge as one of
`correct`, `incorrect` or `missing`, and the headline number is the
**truthfulness score**:

```
S = (#correct - #incorrect) / #total
```

The `report.json` produced by `evaluate.py` also contains per-domain and
per-question-type breakdowns.

## Inputs

The harness expects a JSONL file with the schema produced by `load_dataset.py`:

```json
{"qid": "abc123", "question": "...", "answer": "...",
 "question_time": "2024-03-12T15:00:00Z",
 "context": {"domain": "finance", "question_type": "simple"}}
```
