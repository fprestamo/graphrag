# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Standalone ETCDR evaluation script.

Seeds entities and relationships into the 'etcdrtest' Neo4j database,
then runs each conflict-resolution scenario through detect_and_resolve()
and prints the results.

Usage:
    python packages/graphrag/graphrag/bt_graphrag/evaluation/etcdr/test_etcdr.py

Requirements:
    - Neo4j running at neo4j://127.0.0.1:7687  (same as ragtest/settings.yaml)
    - The 'etcdrtest' database must exist (Neo4j Enterprise / Aura, or rename
      your Community default database to 'etcdrtest')
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from uuid import uuid4

from graphrag.bt_graphrag.conflict_detection.etcdr import detect_and_resolve
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.models.temporal_types import (
    INFINITY,
    INFINITY_ISO,
    ProvenanceRecord,
    ResolutionStrategy,
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
import os

from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig
from graphrag_llm.config.types import LLMProviderType

# ─────────────────────────────────────────────────────────────────────────────
# Config  (mirrors ragtest/settings.yaml, but uses 'etcdrtest' database)
# ─────────────────────────────────────────────────────────────────────────────

UTC = timezone.utc
TEST_DB = "etcdrtest"

CONFIG = BTGraphRAGConfig(
    enabled=True,
    neo4j_uri="neo4j://127.0.0.1:7687",
    neo4j_user="neo4j",
    neo4j_password="12345678",
    neo4j_database=TEST_DB,
    etcdr_enabled=True,
    etcdr_confidence_threshold=0.7,
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


def _rel(
    src: str,
    rel_type: str,
    tgt: str,
    valid_start: datetime,
    valid_end: datetime = INFINITY,
    confidence: float = 1.0,
    description: str = "",
) -> TemporalRelationship:
    return TemporalRelationship(
        id=str(uuid4()),
        source=src,
        target=tgt,
        relation_type=rel_type,
        description=description or f"{src} {rel_type} {tgt}",
        confidence=confidence,
        temporal_quad=TemporalStateQuad(
            t_valid_start=valid_start,
            t_valid_end=valid_end,
            t_tx_start=utcnow(),
            t_tx_end=INFINITY,
        ),
        provenance=[
            ProvenanceRecord(
                source_document_id="doc-etcdr-eval-001",
                text_unit_id="unit-001",
                source_url="https://eval.example.com",
                trust_score=confidence,
            )
        ],
    )


def _real_llm():
    """Build a real LiteLLM completion using OpenAI (gpt-4.1-mini).

    Requires the GRAPHRAG_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("GRAPHRAG_API_KEY", "")
    if not api_key:
        print("[WARN] GRAPHRAG_API_KEY not set — LLM calls will fail.", file=sys.stderr)
    config = ModelConfig(
        type=LLMProviderType.LiteLLM,
        model_provider="openai",
        model="gpt-4.1-mini",
        api_key=api_key,
    )
    return create_completion(config)


async def classify_relation_cardinalities(
    relation_types: list[str],
    seed_rels: list[TemporalRelationship],
    config: BTGraphRAGConfig,
    model,
) -> None:
    """Stage 2c equivalent: classify cardinalities via LLM.

    For each unknown relation type, builds context examples from *seed_rels*
    and asks the LLM to classify its cardinality.  Results are stored in
    ``config.relation_cardinality_overrides`` so that ETCDR can look them up.
    """
    from graphrag_llm.utils import CompletionMessagesBuilder

    prompt_template = config.resolved_cardinality_prompt()

    unknown = [
        rt for rt in relation_types
        if rt.upper().replace(" ", "_") not in config.default_cardinality_map
        and rt.upper().replace(" ", "_") not in config.relation_cardinality_overrides
    ]
    if not unknown:
        print("[CARD] All relation types already classified.")
        return

    print(f"[CARD] Classifying {len(unknown)} relation type(s) via LLM...")
    for rt in unknown:
        examples = []
        for r in seed_rels:
            if r.relation_type == rt:
                examples.append(
                    f"  ({r.source}) -[{rt}]-> ({r.target}): "
                    f"{(r.description or '')[:80]}"
                )
        context = "\n".join(examples) if examples else "No examples available"

        prompt = prompt_template.format(
            relation_type=rt,
            context_examples=context,
        )
        messages = CompletionMessagesBuilder().add_user_message(prompt).build()
        response = await model.completion_async(messages=messages)
        raw_answer = response.content.strip()
        answer = raw_answer.upper()

        classification = "NON_EXCLUSIVE"
        for v in ("BOTH_EXCLUSIVE", "SUBJECT_EXCLUSIVE", "OBJECT_EXCLUSIVE", "NON_EXCLUSIVE"):
            if v in answer:
                classification = v
                break

        key = rt.upper().replace(" ", "_")
        config.relation_cardinality_overrides[key] = classification
        print(f"\n  ┌─ {rt}")
        print(f"  │  LLM raw: {raw_answer}")
        print(f"  │  Parsed : {classification}")
        print(f"  └─")

    print(f"[CARD] Done. Overrides: {config.relation_cardinality_overrides}\n")


def _sep(title: str) -> None:
    width = 70
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def _print_result(result, label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    strategy_name = result.strategy.value if result.strategy else "None"
    print(f"  {tag}has_conflicts : {result.has_conflicts}")
    print(f"  {tag}strategy     : {strategy_name}")
    print(f"  {tag}confidence   : {result.confidence:.2f}")
    n_neo4j = len(result.subject_conflicts) + len(result.object_conflicts)
    n_batch = len(result.intra_batch_subject_conflicts) + len(result.intra_batch_object_conflicts)
    if n_neo4j:
        print(f"  {tag}neo4j confl. : {n_neo4j} edge(s) ({len(result.subject_conflicts)} subject, {len(result.object_conflicts)} object)")
    if n_batch:
        print(f"  {tag}batch confl. : {n_batch} edge(s) ({len(result.intra_batch_subject_conflicts)} subject, {len(result.intra_batch_object_conflicts)} object)")
    if result.candidate and hasattr(result.candidate, "status"):
        print(f"  {tag}cand. status : {result.candidate.status}")


# ─────────────────────────────────────────────────────────────────────────────
# Seed database
# ─────────────────────────────────────────────────────────────────────────────

async def seed_database(session) -> None:
    """Wipe etcdrtest and populate with test entities + relationships."""
    print("\n[SEED] Clearing etcdrtest database...")
    await session.run("MATCH (n) DETACH DELETE n")

    entities = [
        ("Elon Musk",      "person"),
        ("Tesla",          "organization"),
        ("SpaceX",         "organization"),
        ("OpenAI",         "organization"),
        ("USA",            "geo"),
        ("Washington DC",  "geo"),
    ]
    for title, etype in entities:
        await upsert_entity(session, _entity(title, etype))
    print(f"[SEED] Upserted {len(entities)} entities.")

    seed_rels = [
        _rel("Elon Musk", "IS_CEO_OF", "Tesla",
             valid_start=_dt(2004), valid_end=_dt(2024),
             description="Elon Musk was CEO of Tesla 2004–2024"),
        _rel("Elon Musk", "IS_CEO_OF", "SpaceX",
             valid_start=_dt(2002),
             description="Elon Musk is the founding CEO of SpaceX"),
        _rel("USA", "HAS_CAPITAL", "Washington DC",
             valid_start=_dt(1800),
             description="Washington DC is the capital of the USA"),
    ]
    for r in seed_rels:
        await insert_relationship(session, r)
    print(f"[SEED] Inserted {len(seed_rels)} seed relationships.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario runners
# ─────────────────────────────────────────────────────────────────────────────

async def run_no_conflict(session, model) -> None:
    _sep("SCENARIO 1 — No conflict (new NON_EXCLUSIVE edge)")
    candidate = _rel(
        "Elon Musk", "WORKED_AT", "OpenAI",
        valid_start=_dt(2015), valid_end=_dt(2018),
        description="Elon Musk co-founded and worked at OpenAI early on",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=0,
    )
    _print_result(result)
    ok = not result.has_conflicts and result.strategy == ResolutionStrategy.CORROBORATION
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected no conflicts + CORROBORATION")


async def run_evolution(session, model) -> None:
    _sep("SCENARIO 2 — EVOLUTION (CEO role change, OBJECT_EXCLUSIVE)")
    # SpaceX already has Elon Musk as CEO. A new person becomes CEO of SpaceX.
    # This is an object-side conflict: two different subjects for the same object.
    await upsert_entity(session, _entity("Gwynne Shotwell", "person"))
    candidate = _rel(
        "Gwynne Shotwell", "IS_CEO_OF", "SpaceX",
        valid_start=_dt(2025),
        confidence=0.95,
        description="Gwynne Shotwell appointed CEO of SpaceX in 2025",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=1,
    )
    _print_result(result)
    ok = result.strategy == ResolutionStrategy.EVOLUTION and result.has_conflicts
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected EVOLUTION with conflicts")

    # Verify Elon's SpaceX CEO edge t_valid_end was closed
    check = await session.run(
        """
        MATCH (s:Entity {title:'Elon Musk'})-[e:RELATIONSHIP]->(o:Entity {title:'SpaceX'})
        WHERE e.relation_type = 'IS_CEO_OF'
        RETURN e.t_valid_end AS t_valid_end ORDER BY e.t_valid_start DESC LIMIT 1
        """
    )
    rec = await check.single()
    closed = rec and rec["t_valid_end"] != INFINITY_ISO
    print(f"  → {'✓ PASS' if closed else '✗ FAIL'}: Elon's SpaceX CEO edge t_valid_end closed = {rec['t_valid_end'] if rec else 'N/A'}")


async def run_correction(session, model) -> None:
    _sep("SCENARIO 3 — CORRECTION (wrong edge retracted)")

    # Plant a deliberately wrong relationship first
    wrong = _rel(
        "Elon Musk", "IS_CEO_OF", "Tesla",
        valid_start=_dt(2023),
        confidence=0.25,
        description="(Wrong) Elon Musk still CEO of Tesla in 2023",
    )
    await insert_relationship(session, wrong)
    print(f"  Planted wrong edge: IS_CEO_OF Tesla @ 2023 (confidence=0.25)")

    candidate = _rel(
        "Elon Musk", "IS_CEO_OF", "Tesla",
        valid_start=_dt(2004), valid_end=_dt(2021),
        confidence=0.98,
        description="Corrected: Elon Musk CEO of Tesla 2004–2021 only",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=2,
    )
    _print_result(result)
    ok = result.strategy == ResolutionStrategy.CORRECTION
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected CORRECTION")

    # Verify at least one edge was retracted (t_tx_end set)
    check = await session.run(
        """
        MATCH ()-[e:RELATIONSHIP]->()
        WHERE e.relation_type = 'IS_CEO_OF' AND e.t_tx_end <> $infinity
        RETURN count(e) AS retracted
        """,
        infinity=INFINITY_ISO,
    )
    rec = await check.single()
    n = rec["retracted"] if rec else 0
    print(f"  → {'✓ PASS' if n >= 1 else '✗ FAIL'}: {n} retracted edge(s) found (t_tx_end closed)")


async def run_corroboration(session, model) -> None:
    _sep("SCENARIO 4 — CORROBORATION (same fact, second source)")

    # Read current support_count
    before_q = await session.run(
        """
        MATCH (s:Entity {title:'USA'})-[e:RELATIONSHIP]->(t:Entity {title:'Washington DC'})
        WHERE e.relation_type = 'HAS_CAPITAL' AND e.t_tx_end = $infinity
        RETURN e.support_count AS cnt
        """,
        infinity=INFINITY_ISO,
    )
    before_rec = await before_q.single()
    before = before_rec["cnt"] if before_rec else 0
    print(f"  support_count before: {before}")

    candidate = _rel(
        "USA", "HAS_CAPITAL", "Washington DC",
        valid_start=_dt(1800),
        confidence=0.99,
        description="Washington DC is the capital of the USA (second source)",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=3,
    )
    _print_result(result)

    # Read new support_count
    after_q = await session.run(
        """
        MATCH (s:Entity {title:'USA'})-[e:RELATIONSHIP]->(t:Entity {title:'Washington DC'})
        WHERE e.relation_type = 'HAS_CAPITAL' AND e.t_tx_end = $infinity
        RETURN e.support_count AS cnt
        """,
        infinity=INFINITY_ISO,
    )
    after_rec = await after_q.single()
    after = after_rec["cnt"] if after_rec else 0
    print(f"  support_count after : {after}")

    ok = result.strategy == ResolutionStrategy.CORROBORATION and after > before
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected CORROBORATION + support_count increment")


async def run_disagreement_low_confidence(session, model) -> None:
    _sep("SCENARIO 5 — DISAGREEMENT (LLM low confidence → possible override)")
    candidate = _rel(
        "Elon Musk", "IS_CEO_OF", "Tesla",
        valid_start=_dt(2022),
        confidence=0.4,
        description="Disputed claim: Elon Musk as CEO of Tesla in 2022",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=4,
    )
    _print_result(result)
    print(f"  → LLM decided: strategy={result.strategy.value if result.strategy else 'None'}, confidence={result.confidence:.2f}")
    print(f"  → (With real LLM, strategy depends on model reasoning)")


async def run_disagreement_explicit(session, model) -> None:
    _sep("SCENARIO 6 — Contested claim (conflicting evidence)")
    candidate = _rel(
        "Elon Musk", "IS_CEO_OF", "SpaceX",
        valid_start=_dt(2020),
        confidence=0.6,
        description="Alternative contested claim about SpaceX CEO in 2020",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=5,
    )
    _print_result(result)
    strategy_name = result.strategy.value if result.strategy else "None"
    disputed = getattr(result.candidate, "status", None) == "disputed"
    print(f"  → LLM decided: strategy={strategy_name}")
    print(f"  → candidate.status = '{getattr(result.candidate, 'status', 'N/A')}' (disputed={disputed})")


async def run_heuristic_no_model(session) -> None:
    _sep("SCENARIO 7 — Heuristic fallback (no LLM model provided)")
    candidate = _rel(
        "Elon Musk", "COLLABORATED_WITH", "OpenAI",
        valid_start=_dt(2016),
        description="Elon Musk collaborated with OpenAI on AI safety research",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=None, edge_index=6,
    )
    _print_result(result)
    ok = result.strategy == ResolutionStrategy.CORROBORATION and not result.has_conflicts
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected no conflict + CORROBORATION (new NON_EXCLUSIVE)")


async def run_late_arrival(session, model) -> None:
    _sep("SCENARIO 8 — Late arrival (t_event=2010, historical conflict query)")
    candidate = _rel(
        "Elon Musk", "IS_CEO_OF", "Tesla",
        valid_start=_dt(2010),
        confidence=0.85,
        description="Late-arriving 2010 document: Elon was CEO of Tesla",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model,
        is_late_arrival=True, t_event=_dt(2010), edge_index=7,
    )
    _print_result(result, label="late@2010")
    print(f"  → {'✓ PASS' if result.strategy is not None else '✗ FAIL'}: strategy produced for late arrival")


async def run_intra_batch_conflict(session, model) -> None:
    _sep("SCENARIO 9 — Intra-batch conflict (two batch edges contradict each other)")
    # Two candidates in the same batch claim different people as CEO of Tesla
    # at overlapping times. With OBJECT_EXCLUSIVE, a company can only have one
    # CEO at a time. The first should be accepted, the second should detect
    # an intra-batch object-side conflict and resolve it.
    await upsert_entity(session, _entity("Tom Zhu", "person"))
    candidate_a = _rel(
        "Elon Musk", "IS_CEO_OF", "Tesla",
        valid_start=_dt(2025),
        confidence=0.9,
        description="Elon Musk appointed CEO of Tesla again in 2025",
    )
    candidate_b = _rel(
        "Tom Zhu", "IS_CEO_OF", "Tesla",
        valid_start=_dt(2025),
        confidence=0.7,
        description="Tom Zhu appointed CEO of Tesla in 2025",
    )
    # Simulate batch: candidate_a is already accepted
    accepted_batch = [candidate_a]

    result = await detect_and_resolve(
        candidate=candidate_b, session=session, config=CONFIG, model=model,
        edge_index=8, accepted_batch=accepted_batch,
    )
    _print_result(result, label="intra-batch")
    ok = result.has_intra_batch_conflicts
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected intra-batch conflict detected")
    print(f"  → strategy={result.strategy.value if result.strategy else 'None'}")
    print(f"  → intra_batch_subject_conflicts={len(result.intra_batch_subject_conflicts)}, "
          f"intra_batch_object_conflicts={len(result.intra_batch_object_conflicts)}")
    # Verify the in-memory mutation on candidate_a
    a_quad = candidate_a.temporal_quad
    a_still_active = a_quad and a_quad.t_valid_end >= INFINITY and a_quad.t_tx_end >= INFINITY
    print(f"  → candidate_a still fully active: {a_still_active}")
    print(f"  → candidate_a status: {candidate_a.status}")


async def run_db_invariants(session) -> None:
    _sep("SCENARIO 10 — Database invariants (SCD2, entity integrity)")

    # All entities present
    result = await session.run("MATCH (n:Entity) RETURN n.title AS title")
    titles = {rec["title"] async for rec in result}
    expected = {"Elon Musk", "Tesla", "SpaceX", "OpenAI", "USA", "Washington DC"}
    missing = expected - titles
    print(f"  Entities found   : {sorted(titles)}")
    print(f"  → {'✓ PASS' if not missing else f'✗ FAIL — missing: {missing}'}: all seeded entities present")

    # Total edge count (SCD2 — no physical deletion)
    count_q = await session.run("MATCH ()-[r:RELATIONSHIP]->() RETURN count(r) AS total")
    rec = await count_q.single()
    total = rec["total"] if rec else 0
    print(f"  Total edges      : {total}  (SCD2: none physically deleted)")
    print(f"  → {'✓ PASS' if total >= 3 else '✗ FAIL'}: at least 3 edges remain")

    # Active edges (both bitemporal ends open)
    active_q = await session.run(
        """
        MATCH (s:Entity)-[e:RELATIONSHIP]->(t:Entity)
        WHERE e.t_tx_end = $infinity AND e.t_valid_end = $infinity
        RETURN s.title AS src, e.relation_type AS rel, t.title AS tgt
        """,
        infinity=INFINITY_ISO,
    )
    active = [(r["src"], r["rel"], r["tgt"]) async for r in active_q]
    print(f"  Active edges     :")
    for src, rel, tgt in active:
        print(f"    ({src}) -[{rel}]-> ({tgt})")
    print(f"  → {'✓ PASS' if active else '✗ FAIL (no active edges)'}: active edges present")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          ETCDR Evaluation — etcdrtest database (real LLM)           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Neo4j URI : {CONFIG.neo4j_uri}")
    print(f"  Database  : {CONFIG.neo4j_database}")
    print(f"  Threshold : {CONFIG.etcdr_confidence_threshold}")
    print(f"  LLM       : openai/gpt-4.1-mini (real)")

    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        CONFIG.neo4j_uri,
        auth=(CONFIG.neo4j_user, CONFIG.neo4j_password),
    )

    model = _real_llm()

    try:
        await init_schema(driver, database=TEST_DB)

        async with driver.session(database=TEST_DB) as session:
            await seed_database(session)

            # Stage 2c equivalent: classify cardinalities via LLM
            # Gather all relation types used across seed + scenario edges
            all_relation_types = [
                "IS_CEO_OF", "HAS_CAPITAL",
                "WORKED_AT", "COLLABORATED_WITH",
            ]
            seed_examples = [
                _rel("Elon Musk", "IS_CEO_OF", "Tesla",
                     valid_start=_dt(2004), description="Elon Musk was CEO of Tesla"),
                _rel("Elon Musk", "IS_CEO_OF", "SpaceX",
                     valid_start=_dt(2002), description="Elon Musk is the founding CEO of SpaceX"),
                _rel("USA", "HAS_CAPITAL", "Washington DC",
                     valid_start=_dt(1800), description="Washington DC is the capital of the USA"),
                _rel("Elon Musk", "WORKED_AT", "OpenAI",
                     valid_start=_dt(2015), description="Elon Musk co-founded and worked at OpenAI"),
                _rel("Elon Musk", "COLLABORATED_WITH", "OpenAI",
                     valid_start=_dt(2016), description="Elon Musk collaborated with OpenAI on AI safety"),
            ]
            await classify_relation_cardinalities(
                all_relation_types, seed_examples, CONFIG, model,
            )

            await run_no_conflict(session, model)
            await run_evolution(session, model)
            await run_correction(session, model)
            await run_corroboration(session, model)
            await run_disagreement_low_confidence(session, model)
            await run_disagreement_explicit(session, model)
            await run_heuristic_no_model(session)
            await run_late_arrival(session, model)
            await run_intra_batch_conflict(session, model)
            await run_db_invariants(session)

    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
    finally:
        await driver.close()

    print("\n" + "═" * 70)
    print("  All scenarios complete.")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
