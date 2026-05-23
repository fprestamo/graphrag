# CGER + CGRR Evaluation

End-to-end evaluation of the two resolution stages:

- **CGER** ([entity_resolution/cger.py](../../packages/graphrag/graphrag/bt_graphrag/entity_resolution/cger.py)) — merges new entities that refer to the same real-world thing as something the graph already has.
- **CGRR** ([entity_resolution/cgrr.py](../../packages/graphrag/graphrag/bt_graphrag/entity_resolution/cgrr.py)) — normalises new relation-type strings that denote the same predicate as an existing canonical type.

The harness drives both modules with the **intra-batch (Phase B)** path
only: the existing graph is held empty, so every alias the extractor
produces has to be caught against another alias in the same batch.

## Layout

    cger_cgrr/
      data/
        corpus/                                # raw input .txt files
          doc1_tech_leaders.txt
          doc2_ai_labs.txt
          doc3_university_affiliations.txt
        extracted/                             # written by extract.py
          text_units.json
          entities.json
          relationships.json
        ground_truth/                          # authored by hand
          entity_resolution.json
          relationship_resolution.json
      extract.py
      evaluate.py

## Workflow

### 1. Extract — corpus → text units → entities + relationships

    python stages-evaluation/cger_cgrr/extract.py

For each `*.txt` in `data/corpus/`:

1. Token-chunk the document into text units (size=800, overlap=100).
2. Run the temporal graph extractor on each chunk (LLM + embeddings).
3. Concatenate results and write them to `data/extracted/` as JSON.

The output is the deterministic input to `evaluate.py` — re-running the
evaluation does not re-spend tokens on extraction unless you delete
`data/extracted/`.

### 2. Fill in the ground truth (manual)

Open `data/ground_truth/entity_resolution.json` and
`data/ground_truth/relationship_resolution.json`.  Each file ships with
one **example** cluster showing the format; replace/extend that cluster
with the actual alias groups present in your extracted output.

Entity ground truth (`entity_resolution.json`):

```json
{
  "clusters": [
    {
      "canonical": "GEOFFREY HINTON",
      "type": "person",
      "aliases": ["GEOFFREY HINTON", "PROF. HINTON", "HINTON"]
    }
  ]
}
```

Relationship ground truth (`relationship_resolution.json`):

```json
{
  "clusters": [
    {
      "canonical": "IS_CEO_OF",
      "aliases": ["IS_CEO_OF", "LEADS", "CHIEF_EXECUTIVE_OF"]
    }
  ]
}
```

Authoring rules:

- Use the strings exactly as they appear in `data/extracted/entities.json`
  / `relationships.json` — entities are UPPERCASE, relation types are
  UPPER_SNAKE_CASE.
- A cluster of one alias is meaningless; only list alias groups of size
  ≥ 2.
- The `canonical` field is the title/type the resolver should collapse
  every alias to. It must itself appear in `aliases`.

### 3. Evaluate — run CGER and CGRR against the ground truth

    python stages-evaluation/cger_cgrr/evaluate.py

Both scripts auto-load `GRAPHRAG_API_KEY` (or `OPENAI_API_KEY`) from the
project-root `.env` file, so no inline export is needed.

Requires Neo4j running at `neo4j://127.0.0.1:7687` with an empty
database named `cgrreval` (CGRR's session parameter is mandatory even
when the graph is empty; the script wipes the DB before running):

    CREATE DATABASE cgrreval;

The script:

1. Loads `extracted/entities.json` and runs `cger.resolve_entities`
   (in-memory mode — no Neo4j needed for CGER itself).
2. Loads `extracted/relationships.json` and runs
   `cgrr.resolve_relationships` against the empty `cgrreval` database.
3. Converts each module's `merge_map` / `normalize_map` into the set of
   alias pairs it implies, intersects with the ground-truth pairs, and
   reports pairwise **precision / recall / F1** plus a verbose listing
   of TPs (correct merges), FPs (over-merges) and FNs (missed merges).
