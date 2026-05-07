# MultiHop-RAG benchmark

[MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) (Tang & Yang, 2024)
is an English news QA benchmark designed specifically to evaluate retrieval-
augmented generation systems on multi-hop questions.  It contains four query
types: **inference**, **comparison**, **temporal** and **null**.

The *temporal* split makes it especially relevant for BT-GraphRAG.

## Usage

```bash
python -m evaluation.benchmarks.multihop_rag.load_dataset \
    --out evaluation/benchmarks/multihop_rag/data/multihop_rag.jsonl \
    --limit 500

python -m evaluation.benchmarks.multihop_rag.evaluate \
    --dataset evaluation/benchmarks/multihop_rag/data/multihop_rag.jsonl \
    --out     evaluation/results/multihop_rag
```

## Metrics

* EM, token-F1 and judge correctness.
* Per `query_type` breakdown (inference / comparison / temporal / null) plus
  a CRAG-style truthfulness score for each type.
