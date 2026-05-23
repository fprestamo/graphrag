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
        test-corpus/                           # raw input .txt files (eval set)
          doc1_tech_leaders.txt
          doc2_ai_labs.txt
          doc3_university_affiliations.txt
        test-extracted/                        # written by `extract.py --split test`
          text_units.json
          entities.json
          relationships.json
        test-ground-true/                      # authored by hand (eval set)
          entity_resolution.json
          relationship_resolution.json
        train-corpus/                          # raw input .txt files (tuning set)
        train-extracted/                       # written by `extract.py --split train`
        train-ground-true/                     # authored by hand (tuning set)
        train-results.json                     # written by `train.py`
      extract.py
      evaluate.py
      train.py

## Workflow

### 1. Extract — corpus → text units → entities + relationships

    python stages-evaluation/cger_cgrr/extract.py --split test
    python stages-evaluation/cger_cgrr/extract.py --split train   # only if tuning

`--split test` reads `data/test-corpus/` and writes `data/test-extracted/`.
`--split train` reads `data/train-corpus/` and writes `data/train-extracted/`.
The flag defaults to `test`.

For each `*.txt` in the chosen corpus:

1. Token-chunk the document into text units (size=800, overlap=100).
2. Run the temporal graph extractor on each chunk (LLM + embeddings).
3. Concatenate results and write them to the corresponding `*-extracted/`
   folder as JSON.

Re-running `evaluate.py` or `train.py` does not re-spend tokens on
extraction unless you delete the relevant `*-extracted/` folder.

### 2. Fill in the ground truth (manual)

Open `data/test-ground-true/entity_resolution.json` and
`data/test-ground-true/relationship_resolution.json` (and, if tuning,
the matching pair under `data/train-ground-true/`). Each file ships
with one **example** cluster showing the format; replace/extend that
cluster with the actual alias groups present in your extracted output.

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

### 2b. (Alternative) Generate ground truth with an agent

If you'd rather have an LLM agent draft the ground-truth files for you,
hand it one of the two prompts below verbatim — one is hard-coded for
the **test** split, the other for the **train** split. The agent only
needs read access to the corresponding `*-extracted/` folder and write
access to the matching `*-ground-true/` folder.

#### Prompt — TEST split

```text
You are authoring the manual ground truth for an entity-resolution and
relation-type-resolution evaluation harness. Two JSON files already
exist at:

  stages-evaluation/cger_cgrr/data/test-extracted/entities.json
  stages-evaluation/cger_cgrr/data/test-extracted/relationships.json

You must produce two JSON files at:

  stages-evaluation/cger_cgrr/data/test-ground-true/entity_resolution.json
  stages-evaluation/cger_cgrr/data/test-ground-true/relationship_resolution.json

== Schema for entity_resolution.json ==

{
  "_description": "<keep the existing description verbatim>",
  "_format":      "<keep the existing format block verbatim>",
  "clusters": [
    {
      "canonical": "<UPPERCASE_CANONICAL_TITLE>",
      "type":      "<entity type, lowercased, e.g. person|organization|geo>",
      "aliases":   ["<UPPERCASE_ALIAS_1>", "<UPPERCASE_ALIAS_2>", "..."]
    }
  ]
}

== Schema for relationship_resolution.json ==

{
  "_description": "<keep the existing description verbatim>",
  "_format":      "<keep the existing format block verbatim>",
  "clusters": [
    {
      "canonical": "<UPPER_SNAKE_CASE_CANONICAL_TYPE>",
      "aliases":   ["<UPPER_SNAKE_CASE_ALIAS_1>", "<UPPER_SNAKE_CASE_ALIAS_2>", "..."]
    }
  ]
}

== Procedure ==

1. Read entities.json. For each entity object you only need its
   `title`, `type`, and `description` — IGNORE `description_embedding`
   (it is a large float vector you must not load or inspect). Build a
   list of (title, type, description) tuples.

2. Cluster the entities. Two titles belong in the same cluster iff
   they refer to the SAME real-world entity according to their
   descriptions AND share the same `type`. Examples of valid groupings:
     - "GEOFFREY HINTON" + "HINTON" + "PROF. HINTON"     (person)
     - "OPENAI" + "OPEN AI"                              (organization)
     - "UNIVERSITY OF TORONTO" + "U OF T" + "UTORONTO"   (organization)
   Do NOT group entities that are merely related (e.g. an employee and
   their employer) — only literal aliases of the same thing.

3. Read relationships.json. For each row you only need `source`,
   `target`, `description`, and `relation_type` — IGNORE both
   embedding fields.

4. Cluster relation types. Two `relation_type` strings belong in the
   same cluster iff they denote the SAME predicate (verb/role)
   regardless of which subject/object pair they were observed on.
   Use the per-row `description` to disambiguate when the bare type
   name is ambiguous. Examples:
     - "IS_CEO_OF" + "LEADS" + "CHIEF_EXECUTIVE_OF"
     - "EMPLOYED_AT" + "WORKS_AT" + "WORKED_FOR"
     - "FOUNDED" + "ESTABLISHED" + "CO_FOUNDED"
   Do NOT cluster relation types that share a topic but mean different
   things (e.g. "FOUNDED" vs "ACQUIRED").

== Authoring rules (HARD) ==

- Strings inside `aliases` must appear VERBATIM in the source JSON —
  same casing, same punctuation, same underscores. No invented or
  normalised forms.
- Drop clusters of size 1. A cluster with only its canonical and no
  other alias is meaningless and must be omitted.
- `canonical` must itself be one of the strings in its `aliases`
  array. Pick the longest / most complete form as canonical when in
  doubt (e.g. "GEOFFREY HINTON" over "HINTON").
- A given alias string may appear in AT MOST ONE cluster across the
  whole file. If two clusters would share an alias, merge them.
- For the entity file, all aliases inside one cluster must share the
  same `type`. Entities of different types are never in the same
  cluster, even if their titles collide.
- Preserve the existing top-level `_description` and `_format` keys
  byte-for-byte (re-read the current ground-truth files to copy them).
- Output must be valid JSON, UTF-8, two-space indentation, no
  trailing commas.

== Be conservative ==

It is better to omit a borderline cluster than to invent one. The
harness measures pairwise precision/recall against your file, so a
wrong grouping costs more than a missing one. When the descriptions
do not give you high confidence that two titles refer to the same
thing, leave them apart.

Write the two files. Do not modify anything else.
```

#### Prompt — TRAIN split

```text
You are authoring the manual ground truth for an entity-resolution and
relation-type-resolution evaluation harness. Two JSON files already
exist at:

  stages-evaluation/cger_cgrr/data/train-extracted/entities.json
  stages-evaluation/cger_cgrr/data/train-extracted/relationships.json

You must produce two JSON files at:

  stages-evaluation/cger_cgrr/data/train-ground-true/entity_resolution.json
  stages-evaluation/cger_cgrr/data/train-ground-true/relationship_resolution.json

== Schema for entity_resolution.json ==

{
  "_description": "<keep the existing description verbatim>",
  "_format":      "<keep the existing format block verbatim>",
  "clusters": [
    {
      "canonical": "<UPPERCASE_CANONICAL_TITLE>",
      "type":      "<entity type, lowercased, e.g. person|organization|geo>",
      "aliases":   ["<UPPERCASE_ALIAS_1>", "<UPPERCASE_ALIAS_2>", "..."]
    }
  ]
}

== Schema for relationship_resolution.json ==

{
  "_description": "<keep the existing description verbatim>",
  "_format":      "<keep the existing format block verbatim>",
  "clusters": [
    {
      "canonical": "<UPPER_SNAKE_CASE_CANONICAL_TYPE>",
      "aliases":   ["<UPPER_SNAKE_CASE_ALIAS_1>", "<UPPER_SNAKE_CASE_ALIAS_2>", "..."]
    }
  ]
}

== Procedure ==

1. Read entities.json. For each entity object you only need its
   `title`, `type`, and `description` — IGNORE `description_embedding`
   (it is a large float vector you must not load or inspect). Build a
   list of (title, type, description) tuples.

2. Cluster the entities. Two titles belong in the same cluster iff
   they refer to the SAME real-world entity according to their
   descriptions AND share the same `type`. Examples of valid groupings:
     - "JENSEN HUANG" + "MR. HUANG" + "J. HUANG"                          (person)
     - "NVIDIA" + "NVIDIA CORPORATION" + "NVIDIA CORP."                   (organization)
     - "TAIWAN SEMICONDUCTOR MANUFACTURING COMPANY" + "TSMC" + "T.S.M.C." (organization)
     - "AMAZON WEB SERVICES" + "AWS"                                      (organization)
     - "INTERNATIONAL BUSINESS MACHINES" + "IBM"                          (organization)
   Do NOT group entities that are merely related (e.g. a parent
   company and its subsidiary, like "ALPHABET INC." and "GOOGLE", or
   "AMAZON.COM" and "AMAZON WEB SERVICES") — only literal aliases of
   the same thing.

3. Read relationships.json. For each row you only need `source`,
   `target`, `description`, and `relation_type` — IGNORE both
   embedding fields.

4. Cluster relation types. Two `relation_type` strings belong in the
   same cluster iff they denote the SAME predicate (verb/role)
   regardless of which subject/object pair they were observed on.
   Use the per-row `description` to disambiguate when the bare type
   name is ambiguous. Examples:
     - "IS_CEO_OF" + "LEADS" + "HEADS" + "RUNS" + "CHIEF_EXECUTIVE_OF"
     - "FOUNDED" + "CO_FOUNDED" + "ESTABLISHED"
     - "ACQUIRED" + "PURCHASED" + "BOUGHT"
     - "EMPLOYED_AT" + "WORKS_AT"
   Do NOT cluster relation types that share a topic but mean different
   things (e.g. "FOUNDED" vs "ACQUIRED", or "IS_CEO_OF" vs
   "SUCCEEDED").

== Authoring rules (HARD) ==

- Strings inside `aliases` must appear VERBATIM in the source JSON —
  same casing, same punctuation, same underscores. No invented or
  normalised forms.
- Drop clusters of size 1. A cluster with only its canonical and no
  other alias is meaningless and must be omitted.
- `canonical` must itself be one of the strings in its `aliases`
  array. Pick the longest / most complete form as canonical when in
  doubt (e.g. "NVIDIA CORPORATION" over "NVIDIA").
- A given alias string may appear in AT MOST ONE cluster across the
  whole file. If two clusters would share an alias, merge them.
- For the entity file, all aliases inside one cluster must share the
  same `type`. Entities of different types are never in the same
  cluster, even if their titles collide.
- Preserve the existing top-level `_description` and `_format` keys
  byte-for-byte (re-read the current ground-truth files to copy them).
- Output must be valid JSON, UTF-8, two-space indentation, no
  trailing commas.

== Be conservative ==

It is better to omit a borderline cluster than to invent one. The
harness measures pairwise precision/recall against your file, so a
wrong grouping costs more than a missing one. When the descriptions
do not give you high confidence that two titles refer to the same
thing, leave them apart.

Write the two files. Do not modify anything else.
```

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

### 4. (Optional) Tune the cosine thresholds on the train split

    python stages-evaluation/cger_cgrr/extract.py --split train   # once, after adding train docs
    python stages-evaluation/cger_cgrr/train.py

`train.py` searches for the best `CGER_COSINE_THRESHOLD` and
`CGRR_COSINE_THRESHOLD` independently using **simulated annealing**.
All other knobs (`CGER_CANDIDATE_TOP_K`, `CGER_PHASE_B_TOP_K`,
`CGRR_CANDIDATE_TOP_K`, `NEO4J_VECTOR_DIMENSIONS`, `COMPLETION_MODEL`)
are held at the same fixed values used by `evaluate.py`.

Per-stage SA defaults (tweakable in `train.py`):

- start `theta = 0.75`, bounds `[0.30, 0.95]`
- initial temperature `T0 = 0.10`, geometric cooling `alpha = 0.85`
- neighbor radius shrinks with temperature (`0.15 → 0.01`)
- F1 values are cached per rounded threshold so repeated proposals are
  free; only the **unique** (theta) evaluations actually call the LLM.

Default budget is 25 iterations per stage (`--iters 25`), tunable. Pass
`--stages cger` or `--stages cgrr` to tune only one stage.

The full SA trace (every proposal + accept/reject decision), the cache
of F1 values per threshold, and the best `(threshold, P, R, F1)` for
each stage are written to `data/train-results.json`. `evaluate.py` is
**not modified** — once you're happy with the results, copy the values
into `evaluate.py`'s `CGER_COSINE_THRESHOLD` / `CGRR_COSINE_THRESHOLD`
constants by hand.
