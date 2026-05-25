# ETCDR Evaluation

End-to-end evaluation of the Edge-Level Temporal Conflict Detection and
Resolution stage:

- **ETCDR** ([conflict_detection/etcdr.py](../../packages/graphrag/graphrag/bt_graphrag/conflict_detection/etcdr.py))
  — for each new relationship, runs cardinality-aware bidirectional
  conflict queries against the graph and the in-flight batch, classifies
  the conflict, and selects a resolution strategy (NEW_EDGE,
  CORROBORATION, EVOLUTION, CORRECTION, DISAGREEMENT).

This harness does **not** evaluate CGER or CGRR in isolation — there is
a dedicated harness for that ([`../cger_cgrr/`](../cger_cgrr/)). Instead,
it uses authored entity/relation-type ground-truth files to drive the
canonicalisation step so that ETCDR receives a predictable,
ground-truth-aligned batch of edges. There is **no train split**: the
harness ships a single `test-*` set.

## Layout

    etcdr/
      data/
        test-corpus/                           # raw input .txt files
        test-extracted/                        # written by `extract.py`
          text_units.json
          entities.json
          relationships.json
        test-ground-true/                      # authored by hand
          entity_resolution.json               # alias clusters (drives CGER oracle)
          relationship_resolution.json         # alias clusters (drives CGRR oracle)
          conflict_resolution.json             # expected ETCDR strategy per candidate
        test-results.json                      # written by `evaluate.py`
        etcdr_resolution_log.json              # written by `evaluate.py` (ETCDR audit)
      extract.py
      evaluate.py

## Workflow

### 1. Extract — corpus → text units → entities + relationships

    python stages-evaluation/etcdr/extract.py

Reads `data/test-corpus/` and writes `data/test-extracted/`. For each
`*.txt`:

1. Token-chunk the document (size=600, overlap=100).
2. Run the temporal graph extractor on each chunk (LLM + embeddings).
3. Concatenate results and write the consolidated JSON.

Re-running `evaluate.py` does not re-spend tokens on extraction unless
you delete `data/test-extracted/`.

### 2. Fill in the ground truth (manual)

Three files under `data/test-ground-true/` drive the evaluation.

#### `entity_resolution.json` — alias clusters
Same shape as the CGER/CGRR harness:

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

#### `relationship_resolution.json` — alias clusters

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

#### `conflict_resolution.json` — expected ETCDR strategies

Each entry in `expected` is matched against the *n*-th occurrence of its
`(source, relation_type, target)` triple in the **canonicalised** batch
(after the alias clusters above have been applied). Use the
**canonical** strings, not the raw extracted aliases.

```json
{
  "cardinality_overrides": {
    "IS_CEO_OF": "BOTH_EXCLUSIVE",
    "EMPLOYED_AT": "NON_EXCLUSIVE"
  },
  "expected": [
    {
      "source": "ELON MUSK",
      "relation_type": "IS_CEO_OF",
      "target": "TESLA",
      "expected_strategy": "NEW_EDGE",
      "notes": "first sighting of Tesla CEO"
    },
    {
      "source": "ELON MUSK",
      "relation_type": "IS_CEO_OF",
      "target": "TESLA",
      "expected_strategy": "CORROBORATION",
      "notes": "second document confirms it"
    },
    {
      "source": "ROBYN DENHOLM",
      "relation_type": "IS_CEO_OF",
      "target": "TESLA",
      "expected_strategy": "EVOLUTION",
      "notes": "CEO change closes Elon's edge"
    }
  ]
}
```

`expected_strategy` must be one of: `NEW_EDGE`, `CORROBORATION`,
`EVOLUTION`, `CORRECTION`, `DISAGREEMENT`.

`cardinality_overrides` is optional — list it for every relation type
that needs a non-default cardinality. Anything you omit is classified
on the fly by the LLM (same approach as the in-package ETCDR test).

Authoring rules:

- Use the canonical strings exactly as they appear in the resolution
  clusters above (entities UPPERCASE, relation types UPPER_SNAKE_CASE).
- Order matters when the same triple should resolve differently across
  occurrences (e.g. NEW_EDGE then CORROBORATION). The *k*-th entry
  with a given triple is paired with the *k*-th batch row carrying
  that triple.
- Candidates whose triple is not present in `expected` are skipped
  during scoring (they still go through ETCDR; only the comparison
  step ignores them).

### 2b. (Alternative) Generate `conflict_resolution.json` with an agent

If you'd rather have an LLM agent draft the file for you, hand it the
prompt below verbatim. The agent only needs read access to
`data/test-extracted/` and `data/test-ground-true/{entity,relationship}_resolution.json`,
plus write access to `data/test-ground-true/conflict_resolution.json`.

The recommended workflow is:

1. Run `evaluate.py` once with an empty `expected: []`. The harness
   still classifies cardinalities and runs ETCDR end-to-end; the
   resulting `data/test-results.json` lists every candidate's actual
   strategy under `outcomes[]`. That is a strong starting point.
2. Hand the agent the prompt below — it will canonicalise, decide
   cardinalities, and pick expected strategies independently. Compare
   its draft against the actual outcomes to spot ETCDR errors.

#### Prompt — authoring `conflict_resolution.json`

```text
You are authoring the manual ground truth for the ETCDR evaluation
harness. Four JSON files already exist:

  stages-evaluation/etcdr/data/test-extracted/entities.json
  stages-evaluation/etcdr/data/test-extracted/relationships.json
  stages-evaluation/etcdr/data/test-ground-true/entity_resolution.json
  stages-evaluation/etcdr/data/test-ground-true/relationship_resolution.json

You must produce a JSON file at:

  stages-evaluation/etcdr/data/test-ground-true/conflict_resolution.json

== Schema ==

{
  "_description": "<keep the existing description verbatim>",
  "_format":      "<keep the existing format block verbatim>",
  "cardinality_overrides": {
    "<UPPER_SNAKE_CASE_RELATION_TYPE>":
      "<SUBJECT_EXCLUSIVE | OBJECT_EXCLUSIVE | BOTH_EXCLUSIVE | NON_EXCLUSIVE>"
  },
  "expected": [
    {
      "source": "<CANONICAL_SOURCE>",
      "relation_type": "<CANONICAL_RELATION_TYPE>",
      "target": "<CANONICAL_TARGET>",
      "expected_strategy":
        "<NEW_EDGE | CORROBORATION | EVOLUTION | CORRECTION | DISAGREEMENT>",
      "notes": "<one-line human justification>"
    }
  ]
}

== Procedure ==

1. Build the canonicalisation maps:
   - entity_merge_map  := for every cluster in entity_resolution.json,
     map each alias -> canonical title.
   - relation_normalize_map := for every cluster in
     relationship_resolution.json, map each alias -> canonical
     relation_type.

2. Read relationships.json. For each row use only `source`, `target`,
   `relation_type`, `description`, `t_valid_start`, `t_valid_end`,
   `document` and the row order (IGNORE both embedding fields).
   Compute the canonical row:
     canonical_source     = entity_merge_map.get(source, source)
     canonical_target     = entity_merge_map.get(target, target)
     canonical_rel_type   = relation_normalize_map.get(relation_type,
                                                      relation_type)
   Preserve the original on-disk order — the harness feeds candidates
   to ETCDR in exactly this order.

3. Classify every distinct `canonical_rel_type` into one of the four
   cardinalities and emit it under `cardinality_overrides`. Use the
   description text in relationships.json to disambiguate when the
   bare name is ambiguous. Definitions and examples:

   - SUBJECT_EXCLUSIVE — at any time a single subject can hold at
     most one active instance.
       BORN_IN, DIED_IN, IS_NATIONALITY_OF, HAS_BIRTHPLACE.
   - OBJECT_EXCLUSIVE — at any time a single object can be held by
     at most one subject.
       HAS_CAPITAL, HAS_PRESIDENT (one country -> one head).
   - BOTH_EXCLUSIVE — one-to-one in time, exclusive on both sides.
       IS_CEO_OF, IS_PRESIDENT_OF (a company has one CEO; a person
       is CEO of one company at a time).
   - NON_EXCLUSIVE — no exclusivity; multiple coexisting instances
     allowed.
       EMPLOYED_AT, WORKED_FOR, AUTHORED, ATTENDED, FRIEND_OF,
       PART_OF, LOCATED_IN, FATHER_OF, CHILD_OF.

   When in doubt prefer NON_EXCLUSIVE (it is the safe default the
   harness applies for anything you leave unclassified).

4. Walk the canonical rows in on-disk order, maintaining a running
   list `accepted` of rows you have already classified. For each new
   row pick its `expected_strategy` with the following decision tree
   (run the first matching rule):

   a) CORROBORATION — there exists a prior row in `accepted` with
      the SAME canonical triple
        (canonical_source, canonical_rel_type, canonical_target)
      AND the temporal intervals overlap or coincide AND the two
      descriptions assert the same fact (possibly worded
      differently). The new row simply re-states what the old one
      already said.

   b) EVOLUTION — there exists a prior row in `accepted` that
      conflicts with the new row under the cardinality rule:
        - SUBJECT_EXCLUSIVE: same canonical_source + same
          canonical_rel_type, different canonical_target.
        - OBJECT_EXCLUSIVE: same canonical_target + same
          canonical_rel_type, different canonical_source.
        - BOTH_EXCLUSIVE: either of the above.
        - NON_EXCLUSIVE: same (canonical_source, canonical_target)
          but a different canonical_rel_type whose meaning is the
          temporal CONTINUATION/REPLACEMENT of the prior one
          (e.g. WORKS_AT -> LEFT_COMPANY, RUNS -> FORMER_CEO_OF).
      AND the new row's `t_valid_start` is strictly LATER than the
      prior row's `t_valid_start` AND both descriptions are
      internally consistent ("X CEO 2004-2024" + "Y CEO 2024-").

   c) CORRECTION — same conflict shape as (b), but the new row
      RETRACTS the prior row rather than succeeding it: the new
      description explicitly says the prior fact was wrong, or the
      new row comes from a clearly more authoritative source and
      contradicts the prior one over an overlapping valid period.
      The prior row should not have existed in the first place.

   d) DISAGREEMENT — same conflict shape as (b) but neither
      temporal ordering nor authority breaks the tie. Two sources
      assert contradictory facts about the same exclusive slot;
      neither can be cleanly chosen.

   e) NEW_EDGE — fall-through. No prior row in `accepted` conflicts
      with the new row under its cardinality rule.

5. Append the new row to `accepted` UNLESS its expected_strategy is
   CORRECTION (a correction retracts the prior entry; the new row
   takes its place — for the purposes of this procedure, replace
   the prior row in `accepted` with the new one).

6. Emit one `expected` entry per row processed in step 4. Keep the
   entries in the same order you walked the file; the harness
   pairs the k-th entry with a given triple against the k-th batch
   occurrence of that triple.

== Authoring rules (HARD) ==

- `source`, `relation_type`, `target` in every `expected` entry
  MUST be the CANONICAL strings (post-merge), NOT the raw aliases
  the extractor emitted. Anything that does not match a row in the
  canonicalised batch is silently dropped by the scorer.
- Entities are UPPERCASE, relation types are UPPER_SNAKE_CASE.
- `cardinality_overrides` keys are UPPER_SNAKE_CASE canonical
  relation types. List every type you actually depend on for an
  EVOLUTION / CORRECTION / DISAGREEMENT decision; types you omit
  are LLM-classified at runtime, which is fine for clear-cut
  NON_EXCLUSIVE cases but unreliable for exclusive ones.
- Order matters when the same triple appears multiple times. Emit
  the entries in the same order the triple occurs in
  relationships.json so the harness pairs them correctly.
- Do NOT emit `expected` entries for triples that do not actually
  appear in the canonicalised batch — they are reported as
  `unmatched` and add noise to the report.
- Preserve the existing top-level `_description` and `_format`
  keys byte-for-byte (re-read the current conflict_resolution.json
  to copy them).
- Output must be valid JSON, UTF-8, two-space indentation, no
  trailing commas.

== Be conservative ==

It is better to omit a borderline `expected` entry than to invent
one. The scorer treats unmatched entries as informational, but a
wrong `expected_strategy` directly lowers accuracy. When in doubt:
- temporal ordering unclear -> DISAGREEMENT (not EVOLUTION/CORRECTION).
- cardinality unclear        -> NON_EXCLUSIVE.
- triple uncertain to recur  -> emit only the first occurrence.

Write the file. Do not modify anything else.
```

### 3. Evaluate — canonicalise, run ETCDR, score

    python stages-evaluation/etcdr/evaluate.py

The script auto-loads `GRAPHRAG_API_KEY` (or `OPENAI_API_KEY`) from the
project-root `.env`. It requires Neo4j running at
`neo4j://127.0.0.1:7687` with three databases:

    CREATE DATABASE etcdreval;
    CREATE DATABASE cgerbatch;   -- if not already created by cger_cgrr eval
    CREATE DATABASE cgrrbatch;   -- if not already created by cger_cgrr eval

`evaluate.py` does the following:

1. **Canonicalisation pre-step.** Runs the real CGER and CGRR pipelines
   with the cosine LLM-trigger threshold pinned to `0.0` and a
   *ground-truth oracle LLM* (`GroundTruthOracleCompletion` in
   `evaluate.py`) that consults `entity_resolution.json` /
   `relationship_resolution.json` and answers `SAME` iff the two
   strings sit in the same cluster (otherwise `DIFFERENT_ENTITY` /
   `DIFFERENT`). Every non-CGER / non-CGRR prompt (ETCDR's Decision
   Router, cardinality classifier) is transparently forwarded to a
   real LLM. The net effect is the merge_map / normalize_map mirror
   the GT clusters, while exercising the actual CGER/CGRR code paths.

2. **Cardinality classification.** For each canonical relation type in
   the batch, the harness checks `conflict_resolution.json
   .cardinality_overrides` first. Anything not pinned there is passed
   to an LLM classifier (same prompt the production pipeline uses).

3. **ETCDR pass.** Iterates the canonicalised relationships in the
   order they appear in `relationships.json`, calling
   `detect_and_resolve(candidate, session, accepted_batch=...)` for
   each one with the running `accepted_batch` accumulator. Neo4j is
   wiped at the start so all conflicts surface as intra-batch
   conflicts; the strategy chosen for each candidate is recorded.

4. **Scoring.** Pairs each entry in `conflict_resolution.json
   .expected` with the appropriate batch outcome by triple+occurrence
   (see authoring rules above) and reports:
   - `accuracy = correct / matched`
   - per-strategy confusion matrix
   - verbose listing of mismatches (expected vs. actual) and
     unmatched expectations.

Results are written to `data/test-results.json` and the per-candidate
ETCDR audit trail (one entry per resolution) lands in
`data/etcdr_resolution_log.json`.

## Notes & limitations

- **Order sensitivity.** ETCDR is path-dependent: the same triple can
  produce NEW_EDGE first and CORROBORATION later. The harness preserves
  the on-disk row order from `relationships.json`, which is the order
  `extract.py` writes (document-sorted, then chunk-sequential). If the
  extractor changes its emission order, re-author `conflict_resolution
  .json` accordingly.
- **Intra-batch only.** The harness keeps the `etcdreval` Neo4j
  database empty for the whole run, so the *strategy* chosen by ETCDR
  is what we score. The persistence actions (`apply_evolution`,
  `apply_correction`, …) only fire when there are Neo4j conflicts;
  intra-batch resolutions hit the in-memory `apply_intra_batch_*`
  variants. The chosen strategy is identical in either path.
- **Oracle blind spots.** CGER/CGRR only ask the LLM about the
  *best-cosine* candidate per new entity/type. If the cluster member
  with the highest cosine is in a different cluster, the merge is
  missed even with threshold=0. In practice, cluster members tend to
  have the highest mutual cosine, so this is rare; when it bites,
  inspect the oracle log lines and adjust the corpus or the cluster
  contents.
