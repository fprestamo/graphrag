# BT-GraphRAG vs GraphRAG — TimeQA harness

Minimal evaluation pipeline. One benchmark (TimeQA), two systems, one report.

## Flow

1. **Place the raw TimeQA dumps**: drop `human_test.easy.json` and `human_test.hard.json`
   into `evaluation/timeqa/data/raw/` (both JSON-array and JSONL are accepted).
2. **Build the dataset + corpus**:

   ```bash
   python -m evaluation.timeqa.load \
       --input-easy  evaluation/timeqa/data/raw/human_test.easy.json \
       --input-hard  evaluation/timeqa/data/raw/human_test.hard.json \
       --out-jsonl   evaluation/timeqa/data/timeqa.jsonl \
       --out-corpus  evaluation/timeqa/data/corpus
   ```
3. **Feed the corpus to the indexers** (one document per unique Wikipedia entity):

   ```bash
   cp evaluation/timeqa/data/corpus/*.txt .ragtest/input/
   ```
4. **Index twice**: once with `bt_graphrag.enabled=true` (BT pipeline + Neo4j) and once
   with `bt_graphrag.enabled=false` (vanilla GraphRAG output parquets). Keep both side-by-side
   so the runners can read each.
5. **Run the evaluation** (default `--scope both` runs the two scopes back-to-back):

   ```bash
   python -m evaluation.timeqa.evaluate --systems btgraphrag graphrag --limit 500
   ```
6. **Inspect outputs** in `evaluation/results/`:
   - `btgraphrag/timeqa/{global,per_entity}/{predictions.jsonl,report.json}`
   - `graphrag/timeqa/{global,per_entity}/{predictions.jsonl,report.json}`
   - `compare_global.{json,csv,tex}` and `compare_per_entity.{json,csv,tex}`
     (only when both systems were run)

## Evaluation scopes

`--scope` controls how the question is presented to the system. Both run against the
**same** global index (the corpus you built in step 3) — we don't reindex per entity.

- `global` — the raw question goes in. Measures retrieval **and** extraction over the
  full graph. This is the realistic RAG setting.
- `per_entity` — the question is prefixed with `"In the context of <entity>, ..."`
  (entity name taken from the TimeQA `/wiki/...` id). Acts as a soft anchor on the
  global index, approximating "scope to this entity's subgraph". Use it as an
  upper-bound / extraction-only signal: if `per_entity > global`, the gap is the
  retrieval cost.
- `both` (default) — runs both scopes and writes separate `compare_*` files.

> True per-entity scoping (a separate GraphRAG index per Wikipedia page) would require
> indexing N workspaces. That's out of scope for this harness; `per_entity` here is a
> query-side anchoring approximation. Reports include `scope` in their JSON so the
> distinction is preserved.

## Environment variables

- `BTG_MODEL_ID` — model used by the BT-GraphRAG answer-synthesis step.
- `BTG_JUDGE_MODEL_ID` — LLM judge model (falls back to `BTG_MODEL_ID`, then to a heuristic).
- `GRAPHRAG_API_KEY` / `OPENAI_API_KEY` — credentials for the above.
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` — BT-GraphRAG Neo4j store.
- `GRAPHRAG_ROOT` (default `.ragtest`), `GRAPHRAG_SEARCH` (`local` or `global`).

Useful flags: `--scope {global,per_entity,both}`, `--dry-run` (no model calls),
`--no-judge` (heuristic only), `--concurrency N`, `--graphrag-root PATH`,
`--graphrag-data PATH`.
