# BT-GraphRAG vs GraphRAG — TimeQA harness

Minimal evaluation pipeline. One benchmark (TimeQA **hard**), two systems, one
report. Follows the open-domain / RAG protocol used by GraphRAG, DyG-RAG, etc.:
**a single unified corpus is built from every unique Wikipedia page referenced
in `hard.json`, indexed once, and the system must retrieve relevant chunks
itself — the correct page is never disclosed at query time.**

## Flow

1. **Place the raw TimeQA dump**: drop `hard.json` into
   `evaluation/timeqa/data/raw/` (both JSON-array and JSONL are accepted).
2. **Build the dataset + unified corpus**:

   ```bash
   python -m evaluation.timeqa.load \
       --input-hard  evaluation/timeqa/data/raw/hard.json \
       --out-jsonl   evaluation/timeqa/data/timeqa.jsonl \
       --out-corpus  evaluation/timeqa/data/corpus
   ```

   The loader writes **one `.txt` per unique `/wiki/...` entity** into a single
   directory. There is no per-entity corpus — every question shares the same
   pool of documents.
3. **Feed the unified corpus to the indexers**:

   ```bash
   cp evaluation/timeqa/data/corpus/*.txt .ragtest/input/
   ```
4. **Index twice**: once with `bt_graphrag.enabled=true` (BT pipeline + Neo4j)
   and once with `bt_graphrag.enabled=false` (vanilla GraphRAG output parquets).
   Both indexes are built over the **same global pool**.
5. **Run the evaluation**:

   ```bash
   python -m evaluation.timeqa.evaluate --systems btgraphrag graphrag --limit 500
   ```

   The raw question goes in untouched — the system retrieves from the global
   index without any hint about which Wikipedia page is correct.
6. **Inspect outputs** in `evaluation/results/`:
   - `btgraphrag/timeqa/<model>/{predictions.jsonl,report.json}`
   - `graphrag/timeqa/<model>/{predictions.jsonl,report.json}`
   - `compare.{json,csv,tex}` (only when both systems were run)

## Environment variables

- `BTG_MODEL_ID` — model used by the BT-GraphRAG answer-synthesis step.
- `BTG_JUDGE_MODEL_ID` — LLM judge model (falls back to `BTG_MODEL_ID`, then to a heuristic).
- `GRAPHRAG_API_KEY` / `OPENAI_API_KEY` — credentials for the above.
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` — BT-GraphRAG Neo4j store.
- `GRAPHRAG_ROOT` (default `.ragtest`), `GRAPHRAG_SEARCH` (`local` or `global`).

Useful flags: `--dry-run` (no model calls), `--no-judge` (heuristic only),
`--concurrency N`, `--graphrag-root PATH`, `--graphrag-data PATH`,
`--tag LABEL` (groups outputs under a sub-directory).
