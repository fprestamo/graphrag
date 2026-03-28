# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Standalone CGER (Cross-Graph Entity Resolution) evaluation script.

Creates two batches of entities (existing graph entities + new incoming
entities) and runs resolve_entities() through every code path:

  Scenario 1 — First run (empty graph):  Phase A is skipped; Phase B
               resolves intra-batch duplicates ("SpaceX" / "Space X").

  Scenario 2 — Auto-merge:  "Elon Musk" already in graph; new batch
               contains "Elon Musk" (identical) → score ≥ merge_threshold
               → AUTO-MERGE without calling LLM.

  Scenario 3 — LLM zone: "Tesla, Inc." in graph vs "Tesla Inc" in new
               batch → borderline score → LLM asked → mock returns SAME
               → LLM-confirmed merge.

  Scenario 4 — LLM zone rejected: "OpenAI" vs "Open AI Foundation" →
               borderline → mock LLM returns DIFFERENT → kept separate.

  Scenario 5 — Below threshold: completely different entities → no merge.

  Scenario 6 — Merge map applied to relationships: verifies that source/
               target fields in a relationships DataFrame are rewritten.

Usage:
    python packages/graphrag/graphrag/bt_graphrag/evaluation/cger/test_cger.py

No Neo4j connection required — CGER operates purely on DataFrames.
The evaluation uses a config that zeroes out the embedding weight so no
real embeddings are needed; the remaining four signals (BM25, Jaccard,
temporal overlap, relation context) carry the scoring.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

import os

from graphrag.bt_graphrag.entity_resolution.cger import (
    apply_merge_map_to_relationships,
    resolve_entities,
)
from graphrag.bt_graphrag.entity_resolution.scorers import (
    compute_entity_composite_score,
)
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig
from graphrag_llm.config.types import LLMProviderType

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

UTC = timezone.utc

# Embedding weight = 0 so no real embeddings are needed in the evaluation.
# The remaining weights are scaled so they sum to 1.0.
CONFIG = BTGraphRAGConfig(
    enabled=True,
    # Neo4j not used by CGER — kept for completeness
    neo4j_uri="neo4j://127.0.0.1:7687",
    neo4j_user="neo4j",
    neo4j_password="12345678",
    neo4j_database="cgertest",
    # Scoring — no embeddings
    cger_embedding_weight=0.0,
    cger_bm25_weight=0.40,
    cger_jaccard_weight=0.30,
    cger_temporal_overlap_weight=0.20,
    cger_relation_context_weight=0.10,
    # Thresholds (tuned for non-embedding scoring)
    cger_merge_threshold=0.70,
    cger_llm_threshold_low=0.40,
    cger_llm_threshold_high=0.70,
    cger_candidate_top_k=20,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _entity_row(
    title: str,
    etype: str = "person",
    description: str = "",
    active_start: datetime | None = None,
    active_end: datetime | None = None,
    relation_types: list[str] | None = None,
) -> dict:
    return {
        "id": str(uuid4()),
        "title": title,
        "type": etype,
        "description": description or f"Entity: {title}",
        "active_start": (active_start or _dt(2000)).isoformat(),
        "active_end": (active_end or _dt(9999)).isoformat(),
        "first_seen": _dt(2000).isoformat(),
        "last_seen": _dt(2024).isoformat(),
        "description_embedding": None,   # intentionally empty — embedding weight = 0
        "relation_types": relation_types or [],
    }


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _real_llm():
    """Build a real LiteLLM completion using OpenAI (gpt-4.1-mini).

    Requires the GRAPHRAG_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("GRAPHRAG_API_KEY", "")
    if not api_key:
        print("[WARN] GRAPHRAG_API_KEY not set — LLM calls will fail.", file=sys.stderr)
    cfg = ModelConfig(
        type=LLMProviderType.LiteLLM,
        model_provider="openai",
        model="gpt-4.1-mini",
        api_key=api_key,
    )
    return create_completion(cfg)


def _sep(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def _print_merge_map(merge_map: dict[str, str], label: str = "") -> None:
    if not merge_map:
        print(f"  {label}merge_map : (empty — no merges)")
        return
    print(f"  {label}merge_map :")
    for src, dst in merge_map.items():
        print(f"    '{src}'  →  '{dst}'")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — First run (empty graph)
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_1_empty_graph(model) -> None:
    _sep("SCENARIO 1 — First run: empty graph (Phase A skipped)")

    # Two duplicates in the incoming batch ("SpaceX" / "Space X")
    new_entities = _df([
        _entity_row("Elon Musk",  "person",       "Founder and CEO of multiple companies",
                    _dt(1971), relation_types=["IS_CEO_OF", "FOUNDED"]),
        _entity_row("SpaceX",     "organization", "Private aerospace manufacturer",
                    _dt(2002), relation_types=["HAS_CEO"]),
        _entity_row("Space X",    "organization", "Rocket company founded by Elon Musk",
                    _dt(2002), relation_types=["HAS_CEO"]),
        _entity_row("Tesla",      "organization", "Electric vehicle manufacturer",
                    _dt(2003), relation_types=["IS_CEO_OF"]),
    ])
    existing_entities = _df([])   # empty graph

    print(f"  New entities    : {len(new_entities)}")
    print(f"  Existing entities: 0 (first run)\n")

    resolved, merge_map, phase_b_log = await resolve_entities(
        new_entities=new_entities,
        existing_entities=existing_entities,
        config=CONFIG,
        model=model,
        entity_scorer=compute_entity_composite_score,
    )

    print(f"\n  Resolved entities: {len(resolved)}")
    _print_merge_map(merge_map)
    if phase_b_log:
        print(f"  Phase B log entries: {len(phase_b_log)}")
        for entry in phase_b_log:
            print(f"    [{entry['decision']}] '{entry['entity']}' -> '{entry['best_match']}'  "
                  f"score={entry['best_score']}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Auto-merge: identical entity already in graph
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_2_auto_merge(model) -> None:
    _sep("SCENARIO 2 — Auto-merge: identical name already in graph")

    existing_entities = _df([
        _entity_row("Elon Musk", "person",
                    "Entrepreneur; CEO of Tesla, SpaceX and X",
                    _dt(1971), relation_types=["IS_CEO_OF", "FOUNDED", "OWNS"]),
        _entity_row("Tesla",     "organization",
                    "Electric vehicle and clean energy company",
                    _dt(2003), relation_types=["HAS_CEO", "PRODUCES"]),
        _entity_row("SpaceX",    "organization",
                    "Private aerospace manufacturer and space transport company",
                    _dt(2002), relation_types=["HAS_CEO", "OPERATES"]),
    ])

    # New batch: "Elon Musk" is an exact duplicate; "OpenAI" is brand new
    new_entities = _df([
        _entity_row("Elon Musk", "person",
                    "Technology entrepreneur and CEO",
                    _dt(1971), relation_types=["IS_CEO_OF", "FOUNDED"]),
        _entity_row("OpenAI", "organization",
                    "AI safety and research company",
                    _dt(2015), relation_types=["FOUNDED_BY"]),
    ])

    print(f"  Existing: {len(existing_entities)} entities")
    print(f"  New     : {len(new_entities)} entities\n")

    resolved, merge_map, _ = await resolve_entities(
        new_entities=new_entities,
        existing_entities=existing_entities,
        config=CONFIG,
        model=model,  # should not be called for exact match (auto-merge)
        entity_scorer=compute_entity_composite_score,
    )

    print(f"\n  Resolved entities: {len(resolved)}")
    _print_merge_map(merge_map)
    merged = "Elon Musk" in merge_map
    print(f"\n  → {'✓' if merged else '✗'} 'Elon Musk' auto-merged to graph entity: {merged}")
    print(f"  → {'✓' if 'OpenAI' not in merge_map else '✗'} 'OpenAI' kept as new entity: {'OpenAI' not in merge_map}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — LLM zone: alias confirmed as SAME
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_3_llm_same(model) -> None:
    _sep("SCENARIO 3 — LLM zone: alias verification by real LLM")

    existing_entities = _df([
        _entity_row("Tesla, Inc.", "organization",
                    "Electric vehicle and clean energy company headquartered in Austin TX",
                    _dt(2003), relation_types=["HAS_CEO", "PRODUCES", "IS_TRADED_ON"]),
    ])

    # "Tesla Inc" (no comma) → borderline score → LLM asked → SAME → merged
    new_entities = _df([
        _entity_row("Tesla Inc", "organization",
                    "EV manufacturer and energy storage company",
                    _dt(2003), relation_types=["HAS_CEO", "PRODUCES"]),
    ])

    print(f"  Existing : 'Tesla, Inc.'")
    print(f"  Candidate: 'Tesla Inc'")
    print(f"  LLM      : real (gpt-4.1-mini)\n")

    resolved, merge_map, _ = await resolve_entities(
        new_entities=new_entities,
        existing_entities=existing_entities,
        config=CONFIG,
        model=model,
        entity_scorer=compute_entity_composite_score,
    )

    print(f"\n  Resolved entities: {len(resolved)}")
    _print_merge_map(merge_map)
    merged = "Tesla Inc" in merge_map
    print(f"\n  → {'✓' if merged else 'ℹ️  score fell outside LLM zone or LLM said DIFFERENT,'} "
          f"'Tesla Inc' merged: {merged}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 — LLM zone: alias rejected as DIFFERENT
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_4_llm_different(model) -> None:
    _sep("SCENARIO 4 — LLM zone: alias verification by real LLM")

    existing_entities = _df([
        _entity_row("OpenAI", "organization",
                    "AI safety research lab co-founded by Elon Musk and Sam Altman",
                    _dt(2015), relation_types=["FOUNDED_BY", "DEVELOPS"]),
    ])

    # "Open AI Foundation" — name overlap but different entity
    new_entities = _df([
        _entity_row("Open AI Foundation", "organization",
                    "Non-profit AI research organisation",
                    _dt(2015), relation_types=["FOUNDED_BY"]),
    ])

    print(f"  Existing : 'OpenAI'")
    print(f"  Candidate: 'Open AI Foundation'")
    print(f"  LLM      : real (gpt-4.1-mini)\n")

    resolved, merge_map, _ = await resolve_entities(
        new_entities=new_entities,
        existing_entities=existing_entities,
        config=CONFIG,
        model=model,
        entity_scorer=compute_entity_composite_score,
    )

    print(f"\n  Resolved entities: {len(resolved)}")
    _print_merge_map(merge_map)
    kept = "Open AI Foundation" not in merge_map
    print(f"\n  → {'✓' if kept else '✗'} 'Open AI Foundation' kept separate: {kept}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 — Below threshold: completely different entities
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_5_below_threshold(model) -> None:
    _sep("SCENARIO 5 — Below threshold: no similarity → no merges")

    existing_entities = _df([
        _entity_row("Elon Musk",  "person",       "CEO of Tesla and SpaceX",
                    _dt(1971), relation_types=["IS_CEO_OF"]),
        _entity_row("Tesla",      "organization", "EV manufacturer",
                    _dt(2003), relation_types=["HAS_CEO"]),
    ])

    # Entities with no name or type overlap
    new_entities = _df([
        _entity_row("Nikola Tesla",    "person",       "Serbian-American inventor",
                    _dt(1856), _dt(1943), relation_types=["INVENTED"]),
        _entity_row("General Motors",  "organization", "American multinational automaker",
                    _dt(1908), relation_types=["PRODUCES"]),
        _entity_row("Jeff Bezos",      "person",       "Founder of Amazon",
                    _dt(1964), relation_types=["FOUNDED"]),
    ])

    print(f"  Existing : {[r['title'] for r in existing_entities.to_dict('records')]}")
    print(f"  New      : {[r['title'] for r in new_entities.to_dict('records')]}\n")

    resolved, merge_map, _ = await resolve_entities(
        new_entities=new_entities,
        existing_entities=existing_entities,
        config=CONFIG,
        model=model,
        entity_scorer=compute_entity_composite_score,
    )

    print(f"\n  Resolved entities: {len(resolved)}")
    _print_merge_map(merge_map)
    print(f"\n  → {'✓' if not merge_map else '✗'} No merges produced: {not bool(merge_map)}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6 — Merge map applied to relationships DataFrame
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_6_apply_to_relationships() -> None:
    _sep("SCENARIO 6 — apply_merge_map_to_relationships()")

    merge_map = {
        "Elon Musk (duplicate)": "Elon Musk",
        "Tesla Inc":             "Tesla, Inc.",
    }

    rels_before = pd.DataFrame([
        {"source": "Elon Musk (duplicate)", "target": "Tesla Inc",   "relation_type": "IS_CEO_OF",  "description": "CEO role"},
        {"source": "Elon Musk",             "target": "SpaceX",      "relation_type": "IS_CEO_OF",  "description": "CEO role"},
        {"source": "Elon Musk (duplicate)", "target": "SpaceX",      "relation_type": "FOUNDED",    "description": "Founder"},
        {"source": "OpenAI",                "target": "Tesla Inc",   "relation_type": "PARTNERED",  "description": "Partnership"},
        {"source": "Jeff Bezos",            "target": "Amazon",      "relation_type": "FOUNDED",    "description": "Founder"},
    ])

    print(f"  Merge map   : {merge_map}")
    print(f"\n  Relationships BEFORE ({len(rels_before)} rows):")
    for _, row in rels_before.iterrows():
        print(f"    ({row['source']}) -[{row['relation_type']}]-> ({row['target']})")

    rels_after = apply_merge_map_to_relationships(rels_before, merge_map)

    print(f"\n  Relationships AFTER  ({len(rels_after)} rows):")
    for _, row in rels_after.iterrows():
        print(f"    ({row['source']}) -[{row['relation_type']}]-> ({row['target']})")

    changed = (rels_before["source"] != rels_after["source"]) | (rels_before["target"] != rels_after["target"])
    n_changed = changed.sum()
    print(f"\n  Rows rewritten: {n_changed}")
    print(f"  → {'✓' if n_changed == 3 else '✗'} Expected 3 rows rewritten: {n_changed == 3}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7 — Score breakdown: inspect per-signal scores directly
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_7_score_breakdown() -> None:
    _sep("SCENARIO 7 — Score breakdown: per-signal inspection")

    pairs = [
        ("Elon Musk",  "IS_CEO_OF", _dt(1971), "Elon Musk",       "IS_CEO_OF", _dt(1971), "Exact duplicate"),
        ("Tesla Inc",  "HAS_CEO",   _dt(2003), "Tesla, Inc.",      "HAS_CEO",   _dt(2003), "Minor punctuation diff"),
        ("SpaceX",     "OPERATES",  _dt(2002), "Space Exploration","OPERATES",  _dt(2002), "Partial token match"),
        ("OpenAI",     "DEVELOPS",  _dt(2015), "Microsoft",        "PRODUCES",  _dt(1975), "Unrelated entities"),
    ]

    print(f"  {'Entity A':25s}  {'Entity B':25s}  {'BM25':>6}  {'Jacc':>6}  {'Temp':>6}  {'Composite':>9}  Label")
    print(f"  {'-'*25}  {'-'*25}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*9}  --------")

    for (title_a, rel_a, start_a, title_b, rel_b, start_b, label) in pairs:
        e_a = {
            "title": title_a, "type": "person", "description": f"Entity {title_a}",
            "active_start": start_a.isoformat(), "active_end": _dt(9999).isoformat(),
            "description_embedding": None, "relation_types": [rel_a],
        }
        e_b = {
            "title": title_b, "type": "person", "description": f"Entity {title_b}",
            "active_start": start_b.isoformat(), "active_end": _dt(9999).isoformat(),
            "description_embedding": None, "relation_types": [rel_b],
        }
        score, breakdown = compute_entity_composite_score(e_a, e_b, CONFIG)
        print(
            f"  {title_a:25s}  {title_b:25s}  "
            f"{breakdown.get('bm25_name', 0):6.3f}  "
            f"{breakdown.get('jaccard_name', 0):6.3f}  "
            f"{breakdown.get('temporal_overlap', 0):6.3f}  "
            f"{score:9.4f}  {label}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          CGER Evaluation — Cross-Graph Entity Resolution (real LLM)  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Merge threshold  : {CONFIG.cger_merge_threshold}")
    print(f"  LLM zone low     : {CONFIG.cger_llm_threshold_low}")
    print(f"  LLM              : openai/gpt-4.1-mini (real)")
    print(f"  Weights (emb=0)  : bm25={CONFIG.cger_bm25_weight}  "
          f"jaccard={CONFIG.cger_jaccard_weight}  "
          f"temporal={CONFIG.cger_temporal_overlap_weight}  "
          f"relation={CONFIG.cger_relation_context_weight}")

    model = _real_llm()

    try:
        await run_scenario_1_empty_graph(model)
        await run_scenario_2_auto_merge(model)
        await run_scenario_3_llm_same(model)
        await run_scenario_4_llm_different(model)
        await run_scenario_5_below_threshold(model)
        await run_scenario_6_apply_to_relationships()
        await run_scenario_7_score_breakdown()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise

    print("\n" + "═" * 70)
    print("  All CGER scenarios complete.")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
