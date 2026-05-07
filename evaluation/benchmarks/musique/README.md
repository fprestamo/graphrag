# MuSiQue benchmark

[MuSiQue](https://github.com/StonyBrookNLP/musique) (Trivedi et al., 2022) is
a multi-hop QA dataset built by composing single-hop questions.  It contains
2-hop, 3-hop and 4-hop chains which makes it a strong stress test for the
temporal / multi-step reasoning of BT-GraphRAG.

We use the **answerable** dev split (`musique_ans_v1.0_dev.jsonl`).

## Usage

```bash
# 1. Download MuSiQue manually from the official repo, then convert it:
python -m evaluation.benchmarks.musique.load_dataset \
    --input /path/to/musique_ans_v1.0_dev.jsonl \
    --out   evaluation/benchmarks/musique/data/musique.jsonl \
    --limit 500

# 2. Evaluate
python -m evaluation.benchmarks.musique.evaluate \
    --dataset evaluation/benchmarks/musique/data/musique.jsonl \
    --out     evaluation/results/musique
```

## Metrics

Same as HotpotQA (EM + F1 + judge), plus a **per-`num_hops` breakdown** in the
report so improvements on deeper chains are visible.
