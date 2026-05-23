# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Stage 2 of the CGER/CGRR evaluation: run resolution and score against ground truth.

Loads the JSON files written by ``extract.py``:

    data/extracted/entities.json
    data/extracted/relationships.json

…together with the manually authored ground truth:

    data/ground_truth/entity_resolution.json
    data/ground_truth/relationship_resolution.json

Then:
  1. Runs ``cger.resolve_entities`` on the extracted entities, using a
     scratch Neo4j database ('cgerbatch') as Phase B's HNSW-backed top-K
     candidate index. The existing graph is empty -> Phase A is skipped.
  2. Runs ``cgrr.resolve_relationships`` on the extracted relationships,
     pointing the required Neo4j session at an empty test database so
     Phase A is skipped and only the intra-batch Phase B runs.
  3. Compares the resulting merge_map / normalize_map against the
     ground-truth clusters and prints pairwise precision / recall / F1
     plus a per-pair diagnostic listing of TPs, FPs and FNs.

Usage:
    python stages-evaluation/cger_cgrr/evaluate.py

Requirements:
    - extract.py has been run first (so the extracted/ JSONs exist).
    - ``GRAPHRAG_API_KEY`` (or ``OPENAI_API_KEY``) defined either in the
      current shell or in the project-root ``.env`` file — the script
      auto-loads ``.env`` before reading the variable.
    - Neo4j running at neo4j://127.0.0.1:7687 with two databases:
        * 'cgrreval'  (CREATE DATABASE cgrreval;)  — empty workspace
          for CGRR; wiped at start so Phase A is a no-op.
        * 'cgerbatch' (CREATE DATABASE cgerbatch;) — scratch workspace
          for CGER Phase B's HNSW-backed top-K retrieval.
          ``CGERBatchDB`` wipes it on entry/exit automatically.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

from graphrag.bt_graphrag.entity_resolution.cger import resolve_entities
from graphrag.bt_graphrag.entity_resolution.cgrr import resolve_relationships
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig
from graphrag_llm.config.types import LLMProviderType


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
EXTRACTED_DIR = HERE / "data" / "extracted"
GROUND_TRUTH_DIR = HERE / "data" / "ground_truth"
ENV_PATH = PROJECT_ROOT / ".env"

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
TEST_DB = "cgrreval"
CGER_TEMP_DB = "cgerbatch"

COMPLETION_MODEL = "gpt-4.1-mini"

# --- CGER tunables (mirror BTGraphRAGConfig defaults; tweak to study effect) ---
CGER_COSINE_THRESHOLD = 0.85
"""Cosine ≥ this triggers the LLM SAME/DIFFERENT_ENTITY/DIFFERENT_TEMPORAL verdict.
Below this, CGER never merges. Lower → more LLM calls, more recall, more risk of FPs."""

CGER_CANDIDATE_TOP_K = 10
"""Per new entity, this many same-type candidates are scored by cosine.
Only matters for Phase A (existing graph); Phase B intra-batch uses its own knob."""

CGER_PHASE_B_TOP_K = 5
"""Per entity in Phase B, this many already-seen batch entities are scored by cosine."""

# --- CGRR tunables ---
CGRR_COSINE_THRESHOLD = 0.85
"""Cosine ≥ this triggers the LLM SAME/DIFFERENT verdict for two relation-type strings."""

CGRR_CANDIDATE_TOP_K = 5
"""Per candidate relation type, this many existing types are kept for scoring."""

NEO4J_VECTOR_DIMENSIONS = 3072
"""Must match the embedding model: 1536 = text-embedding-3-small, 3072 = text-embedding-3-large."""


def _load_env_file(path: Path) -> None:
    """Populate ``os.environ`` from a ``.env`` file without clobbering existing vars."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _api_key() -> str:
    _load_env_file(ENV_PATH)
    key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print(f"[ERROR] GRAPHRAG_API_KEY not found in env or {ENV_PATH}.",
              file=sys.stderr)
        sys.exit(1)
    return key


def _completion():
    return create_completion(
        ModelConfig(
            type=LLMProviderType.LiteLLM,
            model_provider="openai",
            model=COMPLETION_MODEL,
            api_key=_api_key(),
        )
    )


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def _load_clusters(path: Path) -> list[dict]:
    """Read a ground-truth JSON file and return its ``clusters`` list.

    The top-level may contain ``_description`` / ``_format`` keys for
    documentation; we ignore them and only consume ``clusters``.
    """
    if not path.exists():
        print(f"[ERROR] Ground truth file missing: {path}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("clusters", [])


def _pairs_from_clusters(clusters: list[dict], alias_key: str = "aliases") -> set[frozenset[str]]:
    """Return the set of unordered pairs implied by a list of clusters.

    Two strings end up in the same cluster ⇒ they should be merged ⇒ the
    pair is added to the result.  Pairs are stored as frozensets so the
    set comparison is order-independent.
    """
    pairs: set[frozenset[str]] = set()
    for cluster in clusters:
        items = list(dict.fromkeys(cluster.get(alias_key, [])))
        for a, b in combinations(items, 2):
            if a != b:
                pairs.add(frozenset({a, b}))
    return pairs


def _pairs_from_merge_map(merge_map: dict[str, str]) -> set[frozenset[str]]:
    """Reverse a ``merge_map`` (alias -> canonical) into the implied pairs.

    Each canonical accumulates a set of aliases that resolved to it;
    every pair within that set is a predicted merge.
    """
    groups: dict[str, set[str]] = {}
    for alias, canonical in merge_map.items():
        groups.setdefault(canonical, set()).add(canonical)
        groups[canonical].add(alias)

    pairs: set[frozenset[str]] = set()
    for members in groups.values():
        for a, b in combinations(sorted(members), 2):
            pairs.add(frozenset({a, b}))
    return pairs


def _score(predicted: set[frozenset[str]], truth: set[frozenset[str]]) -> dict:
    tp = predicted & truth
    fp = predicted - truth
    fn = truth - predicted
    p = len(tp) / len(predicted) if predicted else 0.0
    r = len(tp) / len(truth) if truth else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "true_positives": sorted(tuple(sorted(pair)) for pair in tp),
        "false_positives": sorted(tuple(sorted(pair)) for pair in fp),
        "false_negatives": sorted(tuple(sorted(pair)) for pair in fn),
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def _print_report(label: str, scores: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Precision : {scores['precision']:.3f}")
    print(f"  Recall    : {scores['recall']:.3f}")
    print(f"  F1        : {scores['f1']:.3f}")
    print(f"  TP={len(scores['true_positives'])}  "
          f"FP={len(scores['false_positives'])}  "
          f"FN={len(scores['false_negatives'])}")
    if scores["true_positives"]:
        print(f"\n  True positives (correctly merged):")
        for pair in scores["true_positives"]:
            print(f"    + {pair[0]}  <->  {pair[1]}")
    if scores["false_positives"]:
        print(f"\n  False positives (merged but shouldn't have):")
        for pair in scores["false_positives"]:
            print(f"    ! {pair[0]}  <->  {pair[1]}")
    if scores["false_negatives"]:
        print(f"\n  False negatives (missed merges):")
        for pair in scores["false_negatives"]:
            print(f"    - {pair[0]}  <->  {pair[1]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    entities_path = EXTRACTED_DIR / "entities.json"
    rels_path = EXTRACTED_DIR / "relationships.json"
    if not entities_path.exists() or not rels_path.exists():
        print(f"[ERROR] Run extract.py first — missing {entities_path} or {rels_path}",
              file=sys.stderr)
        sys.exit(1)

    entities_df = pd.read_json(entities_path)
    rels_df = pd.read_json(rels_path)
    print(f"[LOAD] {len(entities_df)} entities, {len(rels_df)} relationships")

    truth_entity_clusters = _load_clusters(
        GROUND_TRUTH_DIR / "entity_resolution.json"
    )
    truth_rel_clusters = _load_clusters(
        GROUND_TRUTH_DIR / "relationship_resolution.json"
    )
    truth_entity_pairs = _pairs_from_clusters(truth_entity_clusters)
    truth_rel_pairs = _pairs_from_clusters(truth_rel_clusters)
    print(f"[LOAD] ground truth: {len(truth_entity_pairs)} entity merge pair(s), "
          f"{len(truth_rel_pairs)} relation-type merge pair(s)")

    config = BTGraphRAGConfig(
        enabled=True,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        neo4j_database=TEST_DB,
        # CGER
        cger_enabled=True,
        cger_cosine_threshold=CGER_COSINE_THRESHOLD,
        cger_candidate_top_k=CGER_CANDIDATE_TOP_K,
        cger_phase_b_temp_db=CGER_TEMP_DB,
        # CGRR
        cgrr_enabled=True,
        cgrr_cosine_threshold=CGRR_COSINE_THRESHOLD,
        cgrr_candidate_top_k=CGRR_CANDIDATE_TOP_K,
        neo4j_vector_dimensions=NEO4J_VECTOR_DIMENSIONS,
    )

    print(f"\n[CONFIG]")
    print(f"  CGER: cosine≥{CGER_COSINE_THRESHOLD}  top_k={CGER_CANDIDATE_TOP_K}  "
          f"phase_b_top_k={CGER_PHASE_B_TOP_K}")
    print(f"  CGRR: cosine≥{CGRR_COSINE_THRESHOLD}  top_k={CGRR_CANDIDATE_TOP_K}")
    print(f"  LLM:  {COMPLETION_MODEL}  |  vector_dim={NEO4J_VECTOR_DIMENSIONS}")

    model = _completion()

    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        # --- CGER: scratch DB ('cgerbatch') backs Phase B top-K via HNSW ---
        print(f"\n[CGER] Running with empty existing graph "
              f"(Phase B intra-batch via scratch DB '{CGER_TEMP_DB}')…")
        _, cger_merge_map, _ = await resolve_entities(
            new_entities=entities_df,
            existing_entities=pd.DataFrame(),
            config=config,
            model=model,
            driver=driver,
            phase_b_top_k=CGER_PHASE_B_TOP_K,
        )
        print(f"[CGER] merge_map: {len(cger_merge_map)} entries")
        cger_pairs = _pairs_from_merge_map(cger_merge_map)
        cger_scores = _score(cger_pairs, truth_entity_pairs)
        _print_report("CGER evaluation (entity resolution)", cger_scores)

        # --- CGRR: requires a session; point at an empty test DB ---
        async with driver.session(database=TEST_DB) as session:
            print(f"\n[CGRR] Wiping {TEST_DB} (so Phase A finds an empty graph)…")
            await session.run("MATCH (n) DETACH DELETE n")
            print(f"[CGRR] Running with empty existing graph (Phase B intra-batch only)…")
            _, cgrr_normalize_map, _ = await resolve_relationships(
                relationships_df=rels_df,
                config=config,
                session=session,
                model=model,
            )
        print(f"[CGRR] normalize_map: {len(cgrr_normalize_map)} entries")
        cgrr_pairs = _pairs_from_merge_map(cgrr_normalize_map)
        cgrr_scores = _score(cgrr_pairs, truth_rel_pairs)
        _print_report("CGRR evaluation (relationship-type resolution)", cgrr_scores)
    finally:
        await driver.close()

    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  CGER:  P={cger_scores['precision']:.3f}  "
          f"R={cger_scores['recall']:.3f}  F1={cger_scores['f1']:.3f}")
    print(f"  CGRR:  P={cgrr_scores['precision']:.3f}  "
          f"R={cgrr_scores['recall']:.3f}  F1={cgrr_scores['f1']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
