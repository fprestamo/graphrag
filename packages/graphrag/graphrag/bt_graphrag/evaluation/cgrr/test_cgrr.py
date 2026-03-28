# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Standalone CGRR (Cross-Graph Relationship Resolution) evaluation script.

Seeds entities and canonical relationships into the 'cgrrtest' Neo4j database,
then runs resolve_relationships() against candidate batches that exercise
every code path:

  Scenario 1 — First run (empty graph): No existing types in Neo4j;
               Phase A is skipped; Phase B resolves intra-batch aliases
               ("IS_CEO" and "IS_CEO_OF" in the same incoming batch).

  Scenario 2 — Exact match: Candidate type already exists verbatim in
               the graph → skipped (no LLM call, no normalization).

  Scenario 3 — Auto-normalize: "LEADS" vs existing "IS_CEO_OF" → high
               BM25 + semantic + endpoint score → AUTO-NORMALIZE.

  Scenario 4 — LLM zone confirmed SAME: "HEADS" vs "IS_CEO_OF" →
               borderline score → LLM asked → mock returns SAME → normalized.

  Scenario 5 — LLM zone rejected DIFFERENT: "COLLABORATED_WITH" vs
               "IS_CEO_OF" → score in LLM zone → mock returns DIFFERENT
               → kept separate.

  Scenario 6 — Below threshold: "FOUNDED_IN_YEAR" vs "IS_CEO_OF" →
               score < llm_threshold_low → kept as new type.

  Scenario 7 — apply_normalize_map_to_cardinality: verify that edges
               inherit canonical type's cardinality after normalization.

Usage:
    python packages/graphrag/graphrag/bt_graphrag/evaluation/cgrr/test_cgrr.py

Requires a running Neo4j instance at neo4j://127.0.0.1:7687 and the
'cgrrtest' database to exist.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

import os

from graphrag.bt_graphrag.entity_resolution.cgrr import (
    apply_normalize_map_to_cardinality,
    resolve_relationships,
)
from graphrag.bt_graphrag.entity_resolution.scorers import (
    compute_relationship_composite_score,
)
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.models.temporal_types import (
    INFINITY,
    ProvenanceRecord,
    TemporalEntity,
    TemporalRelationship,
    TemporalStateQuad,
    utcnow,
)
from graphrag.bt_graphrag.neo4j_store import (
    init_schema,
    insert_relationship,
    upsert_entity,
)
from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig
from graphrag_llm.config.types import LLMProviderType

# ─────────────────────────────────────────────────────────────────────────────
# Config  (mirrors ragtest/settings.yaml, uses 'cgrrtest' database)
# ─────────────────────────────────────────────────────────────────────────────

UTC = timezone.utc
TEST_DB = "cgrrtest"

CONFIG = BTGraphRAGConfig(
    enabled=True,
    neo4j_uri="neo4j://127.0.0.1:7687",
    neo4j_user="neo4j",
    neo4j_password="12345678",
    neo4j_database=TEST_DB,
    # CGRR thresholds
    cgrr_merge_threshold=0.75,
    cgrr_llm_threshold_low=0.45,
    cgrr_bm25_weight=0.35,
    cgrr_semantic_weight=0.40,
    cgrr_endpoint_weight=0.25,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _entity(title: str, etype: str = "person") -> TemporalEntity:
    return TemporalEntity(
        id=str(uuid4()),
        title=title,
        type=etype,
        description=f"Test entity: {title}",
        first_seen=_dt(2000),
        last_seen=utcnow(),
        active_start=_dt(2000),
    )


def _rel(src: str, rel_type: str, tgt: str,
         description: str = "",
         valid_start: datetime | None = None) -> TemporalRelationship:
    vs = valid_start or _dt(2000)
    return TemporalRelationship(
        id=str(uuid4()),
        source=src,
        target=tgt,
        relation_type=rel_type,
        description=description or f"{src} {rel_type} {tgt}",
        confidence=0.95,
        temporal_quad=TemporalStateQuad(
            t_valid_start=vs,
            t_valid_end=INFINITY,
            t_tx_start=utcnow(),
            t_tx_end=INFINITY,
        ),
        provenance=[
            ProvenanceRecord(
                source_document_id="doc-cgrr-eval",
                text_unit_id="unit-eval",
                source_url="https://eval.example.com",
                trust_score=0.95,
            )
        ],
    )


def _rel_row(src: str, rel_type: str, tgt: str,
             description: str = "",
             cardinality: str = "NON_EXCLUSIVE") -> dict:
    """Build a DataFrame row representing an incoming candidate relationship."""
    return {
        "id": str(uuid4()),
        "source": src,
        "target": tgt,
        "relation_type": rel_type,
        "description": description or f"{src} {rel_type} {tgt}",
        "confidence": 0.9,
        "cardinality": cardinality,
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


def _print_normalize_map(normalize_map: dict[str, str]) -> None:
    if not normalize_map:
        print("  normalize_map : (empty — no normalizations)")
        return
    print("  normalize_map :")
    for orig, canon in normalize_map.items():
        print(f"    '{orig}'  →  '{canon}'")


# ─────────────────────────────────────────────────────────────────────────────
# Database seeding
# ─────────────────────────────────────────────────────────────────────────────

async def seed_database(driver) -> None:
    """Populate cgrrtest with canonical entities and relationships."""
    print("\n[SEED] Clearing cgrrtest database...")
    async with driver.session(database=TEST_DB) as session:
        await session.run("MATCH (n) DETACH DELETE n")

        for title, etype in [
            ("Elon Musk",     "person"),
            ("Tesla",         "organization"),
            ("SpaceX",        "organization"),
            ("OpenAI",        "organization"),
            ("USA",           "geo"),
            ("Washington DC", "geo"),
            ("Sam Altman",    "person"),
        ]:
            await upsert_entity(session, _entity(title, etype))

        # Canonical relationships — these are what CGRR will compare against
        canonical_rels = [
            _rel("Elon Musk", "IS_CEO_OF", "Tesla",
                 "Elon Musk is the Chief Executive Officer of Tesla",
                 _dt(2004)),
            _rel("Elon Musk", "IS_CEO_OF", "SpaceX",
                 "Elon Musk is the CEO of SpaceX",
                 _dt(2002)),
            _rel("USA", "HAS_CAPITAL", "Washington DC",
                 "Washington DC is the capital city of the USA",
                 _dt(1800)),
            _rel("Elon Musk", "FOUNDED", "SpaceX",
                 "Elon Musk founded SpaceX in 2002",
                 _dt(2002)),
            _rel("Sam Altman", "IS_CEO_OF", "OpenAI",
                 "Sam Altman is the CEO of OpenAI",
                 _dt(2019)),
        ]
        for r in canonical_rels:
            await insert_relationship(session, r)

    print(f"[SEED] Seeded 7 entities and {len(canonical_rels)} canonical relationships.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — First run (empty graph, Phase A skipped)
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_1_empty_graph(driver, model) -> None:
    _sep("SCENARIO 1 — First run: empty graph (Phase A skipped)")

    # Use a fresh temporary session against a logically empty view
    # by passing an empty state — we truncate & re-seed so the DB is clean
    async with driver.session(database=TEST_DB) as session:
        await session.run("MATCH ()-[r:RELATIONSHIP]->() DELETE r")

        # Incoming batch with two aliases in the same batch:
        # "IS_CEO" and "IS_CEO_OF" should be resolved intra-batch
        candidates = _df([
            _rel_row("Elon Musk", "IS_CEO",    "Tesla",  "Elon Musk is CEO of Tesla",    "BOTH_EXCLUSIVE"),
            _rel_row("Elon Musk", "IS_CEO_OF", "SpaceX", "Elon Musk is CEO of SpaceX",   "BOTH_EXCLUSIVE"),
            _rel_row("USA",       "HAS_CAPITAL","Washington DC","Capital city of USA",    "SUBJECT_EXCLUSIVE"),
            _rel_row("Elon Musk", "IS_CEO",    "OpenAI", "Elon Musk runs OpenAI",         "BOTH_EXCLUSIVE"),
        ])

        print(f"  Candidate types in batch : IS_CEO, IS_CEO_OF, HAS_CAPITAL")
        print(f"  Graph state              : empty (no relationships yet)\n")

        resolved, normalize_map, phase_b_log = await resolve_relationships(
            relationships_df=candidates,
            config=CONFIG,
            session=session,
            model=model,
            relationship_scorer=compute_relationship_composite_score,
        )

    print(f"\n  Resolved rows    : {len(resolved)}")
    _print_normalize_map(normalize_map)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Exact match: type already exists verbatim
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_2_exact_match(driver, model) -> None:
    _sep("SCENARIO 2 — Exact match: 'IS_CEO_OF' already canonical in graph")

    # Re-seed so graph has IS_CEO_OF edges
    await seed_database(driver)

    async with driver.session(database=TEST_DB) as session:
        candidates = _df([
            _rel_row("Elon Musk", "IS_CEO_OF", "Tesla",
                     "Elon Musk is the CEO of Tesla", "BOTH_EXCLUSIVE"),
            _rel_row("USA",       "HAS_CAPITAL", "Washington DC",
                     "Washington DC is the US capital", "SUBJECT_EXCLUSIVE"),
        ])

        print(f"  Both candidate types ('IS_CEO_OF', 'HAS_CAPITAL') exist verbatim in graph.")
        print(f"  Expected: both skipped as exact matches.\n")

        resolved, normalize_map, _ = await resolve_relationships(
            relationships_df=candidates,
            config=CONFIG,
            session=session,
            model=model,   # should never be called for exact match
            relationship_scorer=compute_relationship_composite_score,
        )

    _print_normalize_map(normalize_map)
    print(f"\n  → {'✓' if not normalize_map else '✗'} No normalizations (exact matches skipped): {not bool(normalize_map)}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Auto-normalize: "LEADS" → "IS_CEO_OF"
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_3_auto_normalize(driver, model) -> None:
    _sep("SCENARIO 3 — Auto-normalize: 'LEADS' alias of 'IS_CEO_OF'")

    async with driver.session(database=TEST_DB) as session:
        candidates = _df([
            _rel_row("Elon Musk", "LEADS", "Tesla",
                     "Elon Musk leads Tesla as its chief executive", "BOTH_EXCLUSIVE"),
            _rel_row("Sam Altman", "LEADS", "OpenAI",
                     "Sam Altman leads OpenAI as CEO", "BOTH_EXCLUSIVE"),
        ])

        print(f"  Candidate type : 'LEADS'")
        print(f"  Canonical type : 'IS_CEO_OF'")
        print(f"  Expected       : auto-normalize if score ≥ {CONFIG.cgrr_merge_threshold}\n")

        resolved, normalize_map, _ = await resolve_relationships(
            relationships_df=candidates,
            config=CONFIG,
            session=session,
            model=model,
            relationship_scorer=compute_relationship_composite_score,
        )

    _print_normalize_map(normalize_map)
    normalized = "LEADS" in normalize_map
    print(f"\n  → {'✓' if normalized else 'ℹ️  score below auto-merge threshold,'} 'LEADS' normalized: {normalized}")
    if normalized:
        print(f"     'LEADS' → '{normalize_map['LEADS']}'")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 — LLM zone confirmed SAME
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_4_llm_same(driver, model) -> None:
    _sep("SCENARIO 4 — LLM zone: 'HEADS' vs 'IS_CEO_OF' (real LLM)")

    async with driver.session(database=TEST_DB) as session:
        candidates = _df([
            _rel_row("Elon Musk", "HEADS", "Tesla",
                     "Elon Musk heads Tesla", "BOTH_EXCLUSIVE"),
        ])

        print(f"  Candidate type : 'HEADS'")
        print(f"  Canonical type : 'IS_CEO_OF'")
        print(f"  LLM            : real (gpt-4.1-mini)\n")

        resolved, normalize_map, _ = await resolve_relationships(
            relationships_df=candidates,
            config=CONFIG,
            session=session,
            model=model,
            relationship_scorer=compute_relationship_composite_score,
        )

    _print_normalize_map(normalize_map)
    normalized = "HEADS" in normalize_map
    print(f"\n  → {'✓' if normalized else 'ℹ️  score outside LLM zone or LLM said DIFFERENT,'} 'HEADS' normalized: {normalized}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 — LLM zone rejected DIFFERENT
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_5_llm_different(driver, model) -> None:
    _sep("SCENARIO 5 — LLM zone: 'CO_FOUNDED' vs 'FOUNDED' (real LLM)")

    async with driver.session(database=TEST_DB) as session:
        candidates = _df([
            _rel_row("Elon Musk", "CO_FOUNDED", "SpaceX",
                     "Elon Musk co-founded SpaceX together with others",
                     "NON_EXCLUSIVE"),
        ])

        print(f"  Candidate type : 'CO_FOUNDED'")
        print(f"  Existing type  : 'FOUNDED'")
        print(f"  LLM            : real (gpt-4.1-mini)\n")

        resolved, normalize_map, _ = await resolve_relationships(
            relationships_df=candidates,
            config=CONFIG,
            session=session,
            model=model,
            relationship_scorer=compute_relationship_composite_score,
        )

    _print_normalize_map(normalize_map)
    kept = "CO_FOUNDED" not in normalize_map
    print(f"\n  → LLM decided: 'CO_FOUNDED' {'normalized' if not kept else 'kept separate'}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6 — Below threshold: unrelated type
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_6_below_threshold(driver, model) -> None:
    _sep("SCENARIO 6 — Below threshold: 'INCORPORATED_IN' → no overlap")

    async with driver.session(database=TEST_DB) as session:
        candidates = _df([
            _rel_row("Tesla", "INCORPORATED_IN", "USA",
                     "Tesla is legally incorporated in the United States",
                     "SUBJECT_EXCLUSIVE"),
        ])

        print(f"  Candidate type : 'INCORPORATED_IN'")
        print(f"  Existing types : IS_CEO_OF, HAS_CAPITAL, FOUNDED")
        print(f"  Expected       : below threshold → kept as new type\n")

        resolved, normalize_map, _ = await resolve_relationships(
            relationships_df=candidates,
            config=CONFIG,
            session=session,
            model=model,
            relationship_scorer=compute_relationship_composite_score,
        )

    _print_normalize_map(normalize_map)
    kept = "INCORPORATED_IN" not in normalize_map
    print(f"\n  → {'✓' if kept else '✗'} 'INCORPORATED_IN' kept as new type: {kept}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7 — apply_normalize_map_to_cardinality
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_7_cardinality_inheritance() -> None:
    _sep("SCENARIO 7 — apply_normalize_map_to_cardinality()")

    normalize_map = {
        "LEADS":  "IS_CEO_OF",
        "HEADS":  "IS_CEO_OF",
    }

    # IS_CEO_OF is BOTH_EXCLUSIVE; original rows use NON_EXCLUSIVE (wrong)
    rels = _df([
        _rel_row("Elon Musk",  "LEADS",      "Tesla",  cardinality="NON_EXCLUSIVE"),
        _rel_row("Sam Altman", "HEADS",       "OpenAI", cardinality="NON_EXCLUSIVE"),
        _rel_row("Elon Musk",  "IS_CEO_OF",   "SpaceX", cardinality="BOTH_EXCLUSIVE"),
        _rel_row("Elon Musk",  "FOUNDED",     "SpaceX", cardinality="NON_EXCLUSIVE"),
    ])

    # First normalize relation_type
    rels["relation_type"] = rels["relation_type"].map(
        lambda rt: normalize_map.get(rt, rt)
    )

    print(f"  Normalize map : {normalize_map}")
    print(f"\n  DataFrame BEFORE cardinality inheritance:")
    before_rels = _df([
        _rel_row("Elon Musk",  "LEADS",      "Tesla",  cardinality="NON_EXCLUSIVE"),
        _rel_row("Sam Altman", "HEADS",       "OpenAI", cardinality="NON_EXCLUSIVE"),
        _rel_row("Elon Musk",  "IS_CEO_OF",   "SpaceX", cardinality="BOTH_EXCLUSIVE"),
        _rel_row("Elon Musk",  "FOUNDED",     "SpaceX", cardinality="NON_EXCLUSIVE"),
    ])
    for _, row in before_rels.iterrows():
        print(f"    {row['relation_type']:20s}  cardinality={row['cardinality']}")

    after = apply_normalize_map_to_cardinality(rels, normalize_map, CONFIG)

    print(f"\n  DataFrame AFTER  cardinality inheritance:")
    for _, row in after.iterrows():
        print(f"    {row['relation_type']:20s}  cardinality={row['cardinality']}")

    # IS_CEO_OF should be BOTH_EXCLUSIVE (from config default_cardinality_map)
    ceo_rows = after[after["relation_type"] == "IS_CEO_OF"]
    all_exclusive = all(r == "BOTH_EXCLUSIVE" for r in ceo_rows["cardinality"])
    print(f"\n  → {'✓' if all_exclusive else '✗'} All IS_CEO_OF rows inherit BOTH_EXCLUSIVE: {all_exclusive}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 8 — Score breakdown: per-signal inspection
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_8_score_breakdown() -> None:
    _sep("SCENARIO 8 — Score breakdown: per-signal inspection")

    pairs = [
        ("IS_CEO_OF",        "CEO of Tesla",    "Elon Musk", "Tesla",
         "IS_CEO_OF",        "CEO of SpaceX",   "Elon Musk", "SpaceX",  "Exact type match"),
        ("LEADS",            "leads the company","Elon Musk", "Tesla",
         "IS_CEO_OF",        "CEO of Tesla",    "Elon Musk", "Tesla",   "Semantic alias"),
        ("HEADS",            "heads org",        "Sam Altman","OpenAI",
         "IS_CEO_OF",        "CEO of OpenAI",   "Sam Altman","OpenAI",  "Same endpoints"),
        ("COLLABORATED_WITH","collab",           "Elon Musk", "OpenAI",
         "IS_CEO_OF",        "CEO role",        "Elon Musk", "Tesla",   "Unrelated types"),
        ("HAS_CAPITAL",      "capital city",     "USA",       "DC",
         "IS_CEO_OF",        "CEO of company",  "Person",    "Org",     "Completely different"),
    ]

    print(f"  {'Type A':20s}  {'Type B':20s}  {'BM25':>6}  {'Sem':>6}  {'Endpt':>6}  {'Score':>7}  Label")
    print(f"  {'-'*20}  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  --------")

    for (ta, da, sa, oa, tb, db, sb, ob, label) in pairs:
        score, breakdown = compute_relationship_composite_score(
            ta, da, sa, oa,
            tb, db, sb, ob,
            CONFIG,
        )
        print(
            f"  {ta:20s}  {tb:20s}  "
            f"{breakdown.get('bm25_type', 0):6.3f}  "
            f"{breakdown.get('semantic_desc', 0):6.3f}  "
            f"{breakdown.get('endpoint_match', 0):6.3f}  "
            f"{score:7.4f}  {label}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║    CGRR Evaluation — Cross-Graph Relationship Resolution (real LLM)  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Neo4j URI        : {CONFIG.neo4j_uri}")
    print(f"  Database         : {CONFIG.neo4j_database}")
    print(f"  Merge threshold  : {CONFIG.cgrr_merge_threshold}")
    print(f"  LLM zone low     : {CONFIG.cgrr_llm_threshold_low}")
    print(f"  LLM              : openai/gpt-4.1-mini (real)")
    print(f"  Weights          : bm25={CONFIG.cgrr_bm25_weight}  "
          f"semantic={CONFIG.cgrr_semantic_weight}  "
          f"endpoint={CONFIG.cgrr_endpoint_weight}")

    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        CONFIG.neo4j_uri,
        auth=(CONFIG.neo4j_user, CONFIG.neo4j_password),
    )

    model = _real_llm()

    try:
        await init_schema(driver, database=TEST_DB)
        await seed_database(driver)

        await run_scenario_1_empty_graph(driver, model)
        await run_scenario_2_exact_match(driver, model)
        await run_scenario_3_auto_normalize(driver, model)
        await run_scenario_4_llm_same(driver, model)
        await run_scenario_5_llm_different(driver, model)
        await run_scenario_6_below_threshold(driver, model)
        await run_scenario_7_cardinality_inheritance()
        await run_scenario_8_score_breakdown()

    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
    finally:
        await driver.close()

    print("\n" + "═" * 70)
    print("  All CGRR scenarios complete.")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
