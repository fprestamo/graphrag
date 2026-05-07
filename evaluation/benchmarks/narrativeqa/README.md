# NarrativeQA benchmark

[NarrativeQA](https://huggingface.co/datasets/deepmind/narrativeqa) (Kočiský
et al., 2018) is a long-form QA benchmark over books and movie scripts.  Each
question has **two** human-written reference answers, both of which are
accepted by token-F1 / EM.

## Usage

```bash
python -m evaluation.benchmarks.narrativeqa.load_dataset \
    --out evaluation/benchmarks/narrativeqa/data/narrativeqa.jsonl \
    --limit 200

python -m evaluation.benchmarks.narrativeqa.evaluate \
    --dataset evaluation/benchmarks/narrativeqa/data/narrativeqa.jsonl \
    --out     evaluation/results/narrativeqa
```

## Metrics

* token-F1 against either reference answer (headline number).
* LLM-as-judge correctness.
* **ROUGE-L** is reported when the `rouge-score` package is installed.
