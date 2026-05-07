# HotpotQA benchmark

[HotpotQA](https://hotpotqa.github.io/) is a multi-hop QA dataset where each
question requires reasoning over two Wikipedia paragraphs.  We use the
**distractor** dev split following the official evaluation protocol.

## Usage

```bash
python -m evaluation.benchmarks.hotpotqa.load_dataset \
    --out evaluation/benchmarks/hotpotqa/data/hotpotqa.jsonl \
    --limit 500

python -m evaluation.benchmarks.hotpotqa.evaluate \
    --dataset evaluation/benchmarks/hotpotqa/data/hotpotqa.jsonl \
    --out     evaluation/results/hotpotqa
```

## Metrics

* **Exact-Match** and **token-F1** with the official SQuAD-style normalization
  (lower-case, strip punctuation/articles, collapse whitespace).
* LLM-as-judge correctness (`correct` / `incorrect` / `missing`).
* Per-`type` (bridge / comparison) and per-`level` (easy / medium / hard)
  breakdowns in `report.json`.
