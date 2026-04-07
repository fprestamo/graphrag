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
    tx_start: datetime | None = None,
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
            t_tx_start=tx_start if tx_start is not None else utcnow(),
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
    ok = not result.has_conflicts and result.strategy == ResolutionStrategy.NEW_EDGE
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected no conflicts + NEW_EDGE")


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
    # Seed an active Tesla CEO edge from a different person so there IS a conflict.
    # Previous scenarios closed all existing Tesla CEO edges, so we plant a fresh one.
    await upsert_entity(session, _entity("Mary Johnson", "person"))
    ref = _rel(
        "Mary Johnson", "IS_CEO_OF", "Tesla",
        valid_start=_dt(2022),
        confidence=0.5,
        description="Mary Johnson controversially named CEO of Tesla in 2022",
    )
    await insert_relationship(session, ref)

    # Candidate: same role, same time, low confidence → expect conflict + LLM routing
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
    ok = result.has_conflicts
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: conflict detected (Mary vs Elon, both Tesla CEO @2022)")
    print(f"  → LLM decided: strategy={result.strategy.value if result.strategy else 'None'}, confidence={result.confidence:.2f}")


async def run_disagreement_explicit(session, model) -> None:
    _sep("SCENARIO 6 — Contested claim (conflicting evidence, expect DISAGREEMENT)")
    # Seed Gwynne as active SpaceX CEO (the edge scenario 2 resolved but never inserted).
    await upsert_entity(session, _entity("Gwynne Shotwell", "person"))
    ref = _rel(
        "Gwynne Shotwell", "IS_CEO_OF", "SpaceX",
        valid_start=_dt(2025),
        confidence=0.9,
        description="Gwynne Shotwell appointed CEO of SpaceX in 2025",
    )
    await insert_relationship(session, ref)

    # Candidate: Elon re-claims SpaceX CEO at the same time with lower confidence
    candidate = _rel(
        "Elon Musk", "IS_CEO_OF", "SpaceX",
        valid_start=_dt(2025),
        confidence=0.5,
        description="Contested: Elon Musk still acting as SpaceX CEO in 2025",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=5,
    )
    _print_result(result)
    ok = result.has_conflicts
    strategy_name = result.strategy.value if result.strategy else "None"
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: conflict detected (Gwynne vs Elon, both SpaceX CEO @2025)")
    print(f"  → LLM decided: strategy={strategy_name}, candidate.status={getattr(result.candidate, 'status', 'N/A')}")


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
    ok = result.strategy == ResolutionStrategy.NEW_EDGE and not result.has_conflicts
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected no conflict + NEW_EDGE (new NON_EXCLUSIVE)")


async def run_late_arrival(session, model) -> None:
    _sep("SCENARIO 8 — Late arrival (t_event=2010, historical conflict query)")
    # The late-arrival query filters t_tx_end = INFINITY (currently believed)
    # AND t_valid_start <= t_event AND t_valid_end > t_event.
    # We seed a reference edge with valid period 2004–2024 so it overlaps 2010.
    # The seeded edge from seed_database() has t_valid_end=2024, so the same-pair
    # query for OBJECT_EXCLUSIVE (S0 pass) will find it.
    # No special tx_start trick needed — the query now uses t_tx_end = INFINITY.
    candidate = _rel(
        "Elon Musk", "IS_CEO_OF", "Tesla",
        valid_start=_dt(2010), valid_end=_dt(2012),
        confidence=0.85,
        description="Late-arriving 2010 document: Elon was CEO of Tesla",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model,
        is_late_arrival=True, t_event=_dt(2010), edge_index=7,
    )
    _print_result(result, label="late@2010")
    # With the fixed late-arrival query (t_tx_end = INFINITY), the seeded
    # Elon IS_CEO_OF Tesla @2004-2024 edge overlaps t_event=2010 and should
    # be found via the S0 same-pair pass → conflict expected.
    ok = result.has_conflicts and result.strategy is not None
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: late-arrival conflict detected against historical edge")


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


async def run_subject_crosstype_conflict(session, model) -> None:
    _sep("SCENARIO 11 — Subject-side cross-type conflict (SUBJECT_EXCLUSIVE, different rel names)")
    # Seed an existing edge with a different type name than the candidate.
    # The cross-type subject-side query (run_subject_any_type_query) must surface it
    # because the subject already holds a leadership role — even though the type names differ.
    await upsert_entity(session, _entity("Alice Johnson", "person"))
    await upsert_entity(session, _entity("Acme Corp", "organization"))
    existing = _rel(
        "Alice Johnson", "IS_DIRECTOR_OF", "Acme Corp",
        valid_start=_dt(2020),
        description="Alice Johnson serves as Director of Acme Corp since 2020",
    )
    await insert_relationship(session, existing)

    candidate = _rel(
        "Alice Johnson", "LEADS", "Acme Corp",
        valid_start=_dt(2023),
        confidence=0.9,
        description="Alice Johnson is leading Acme Corp from 2023 onward",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=10,
    )
    _print_result(result, label="subj-crosstype")
    ok = result.has_conflicts and len(result.subject_conflicts) > 0
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected subject-side conflict via cross-type query (IS_DIRECTOR_OF ≠ LEADS)")
    print(f"  → subject_conflicts={len(result.subject_conflicts)}, strategy={result.strategy.value if result.strategy else 'None'}")


async def run_object_crosstype_conflict(session, model) -> None:
    _sep("SCENARIO 12 — Object-side cross-type conflict (OBJECT_EXCLUSIVE, different rel names)")
    # Seed Bob as president of Beta Corp.  A new person (Carol) arrives with relation type
    # RUNS — different name, same cardinality class.  run_object_any_type_query must find Bob.
    await upsert_entity(session, _entity("Bob Smith", "person"))
    await upsert_entity(session, _entity("Carol Davis", "person"))
    await upsert_entity(session, _entity("Beta Corp", "organization"))
    existing = _rel(
        "Bob Smith", "IS_PRESIDENT_OF", "Beta Corp",
        valid_start=_dt(2018),
        description="Bob Smith is president of Beta Corp since 2018",
    )
    await insert_relationship(session, existing)

    candidate = _rel(
        "Carol Davis", "RUNS", "Beta Corp",
        valid_start=_dt(2023),
        confidence=0.88,
        description="Carol Davis runs Beta Corp starting 2023",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=11,
    )
    _print_result(result, label="obj-crosstype")
    ok = result.has_conflicts and len(result.object_conflicts) > 0
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected object-side conflict via cross-type query (IS_PRESIDENT_OF ≠ RUNS)")
    print(f"  → object_conflicts={len(result.object_conflicts)}, strategy={result.strategy.value if result.strategy else 'None'}")


async def run_both_exclusive_crosstype_subject(session, model) -> None:
    _sep("SCENARIO 13a — BOTH_EXCLUSIVE cross-type: subject side (same subject, different rel name)")
    # Seed Dave as ruler of Gamma Nation.  Candidate is the same subject with COMMANDS.
    # BOTH_EXCLUSIVE triggers run_subject_any_type_query → finds IS_RULER_OF.
    await upsert_entity(session, _entity("Dave Evans", "person"))
    await upsert_entity(session, _entity("Gamma Nation", "geo"))
    existing = _rel(
        "Dave Evans", "IS_RULER_OF", "Gamma Nation",
        valid_start=_dt(2010),
        description="Dave Evans has ruled Gamma Nation since 2010",
    )
    await insert_relationship(session, existing)

    candidate = _rel(
        "Dave Evans", "COMMANDS", "Gamma Nation",
        valid_start=_dt(2023),
        confidence=0.85,
        description="Dave Evans commands Gamma Nation from 2023",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=12,
    )
    _print_result(result, label="both-subj-crosstype")
    ok = result.has_conflicts and len(result.subject_conflicts) > 0
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected BOTH_EXCLUSIVE subject-side cross-type conflict")
    print(f"  → subject_conflicts={len(result.subject_conflicts)}, object_conflicts={len(result.object_conflicts)}, "
          f"strategy={result.strategy.value if result.strategy else 'None'}")


async def run_both_exclusive_crosstype_object(session, model) -> None:
    _sep("SCENARIO 13b — BOTH_EXCLUSIVE cross-type: object side (different subject, different rel name)")
    # Uses independent entities — NOT shared with 13a so the active edge is not already closed.
    # Victor Reign IS_RULER_OF Delta Empire (BOTH_EXCLUSIVE, active).
    # Eve Foster arrives with COMMANDS to the same nation — different subject AND different type.
    # run_object_any_type_query must find Victor's IS_RULER_OF edge.
    await upsert_entity(session, _entity("Victor Reign", "person"))
    await upsert_entity(session, _entity("Eve Foster", "person"))
    await upsert_entity(session, _entity("Delta Empire", "geo"))
    existing = _rel(
        "Victor Reign", "IS_RULER_OF", "Delta Empire",
        valid_start=_dt(2005),
        description="Victor Reign has ruled Delta Empire since 2005",
    )
    await insert_relationship(session, existing)

    candidate = _rel(
        "Eve Foster", "COMMANDS", "Delta Empire",
        valid_start=_dt(2024),
        confidence=0.8,
        description="Eve Foster commands Delta Empire from 2024",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=13,
    )
    _print_result(result, label="both-obj-crosstype")
    ok = result.has_conflicts and len(result.object_conflicts) > 0
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected BOTH_EXCLUSIVE object-side cross-type conflict")
    print(f"  → subject_conflicts={len(result.subject_conflicts)}, object_conflicts={len(result.object_conflicts)}, "
          f"strategy={result.strategy.value if result.strategy else 'None'}")


async def run_non_exclusive_crosstype_conflict(session, model) -> None:
    _sep("SCENARIO 14 — NON_EXCLUSIVE cross-type source-target conflict (same pair, opposing types)")
    # Seed Frank working for Delta Inc.  Candidate LEFT_JOB_AT uses the same (source, target)
    # pair but contradicts the existing edge.  run_source_target_query (second pass for NON_EXCLUSIVE)
    # must surface the WORKS_FOR edge even though the relation type differs.
    await upsert_entity(session, _entity("Frank Garcia", "person"))
    await upsert_entity(session, _entity("Delta Inc", "organization"))
    existing = _rel(
        "Frank Garcia", "WORKS_FOR", "Delta Inc",
        valid_start=_dt(2015),
        description="Frank Garcia works for Delta Inc since 2015",
    )
    await insert_relationship(session, existing)

    candidate = _rel(
        "Frank Garcia", "LEFT_JOB_AT", "Delta Inc",
        valid_start=_dt(2023),
        confidence=0.9,
        description="Frank Garcia left his job at Delta Inc in 2023",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model, edge_index=14,
    )
    _print_result(result, label="non-excl-crosstype")
    # NON_EXCLUSIVE source-target cross-type conflicts land in subject_conflicts
    ok = result.has_conflicts and len(result.subject_conflicts) > 0
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected NON_EXCLUSIVE cross-type conflict via source-target query (WORKS_FOR ≠ LEFT_JOB_AT)")
    print(f"  → subject_conflicts={len(result.subject_conflicts)}, strategy={result.strategy.value if result.strategy else 'None'}")


async def run_intra_batch_subject_crosstype(session, model) -> None:
    _sep("SCENARIO 15 — Intra-batch cross-type SUBJECT_EXCLUSIVE conflict")
    # Both edges are in the same batch — no Neo4j seed needed.
    # Accepted batch has IS_DIRECTOR_OF; candidate brings LEADS.
    # find_intra_batch_subject_any_type_conflicts must catch this.
    batch_edge = _rel(
        "Grace Hill", "IS_DIRECTOR_OF", "Epsilon Corp",
        valid_start=_dt(2021),
        confidence=0.9,
        description="Grace Hill is Director of Epsilon Corp",
    )
    accepted_batch = [batch_edge]

    candidate = _rel(
        "Grace Hill", "LEADS", "Epsilon Corp",
        valid_start=_dt(2021),
        confidence=0.85,
        description="Grace Hill leads Epsilon Corp — extracted from a different sentence",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model,
        edge_index=15, accepted_batch=accepted_batch,
    )
    _print_result(result, label="ib-subj-crosstype")
    ok = result.has_intra_batch_conflicts and len(result.intra_batch_subject_conflicts) > 0
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected intra-batch subject cross-type conflict (IS_DIRECTOR_OF ≠ LEADS)")
    print(f"  → intra_batch_subject_conflicts={len(result.intra_batch_subject_conflicts)}, "
          f"strategy={result.strategy.value if result.strategy else 'None'}")


async def run_intra_batch_object_crosstype(session, model) -> None:
    _sep("SCENARIO 16 — Intra-batch cross-type OBJECT_EXCLUSIVE conflict")
    # Accepted batch has Henry Jones IS_PRESIDENT_OF Zeta Corp.
    # Candidate Ivy King RUNS Zeta Corp — different subject, different type, same target.
    # find_intra_batch_object_any_type_conflicts must surface Henry's edge.
    batch_edge = _rel(
        "Henry Jones", "IS_PRESIDENT_OF", "Zeta Corp",
        valid_start=_dt(2019),
        confidence=0.92,
        description="Henry Jones is President of Zeta Corp",
    )
    accepted_batch = [batch_edge]

    candidate = _rel(
        "Ivy King", "RUNS", "Zeta Corp",
        valid_start=_dt(2022),
        confidence=0.78,
        description="Ivy King runs Zeta Corp from 2022",
    )

    result = await detect_and_resolve(
        candidate=candidate, session=session, config=CONFIG, model=model,
        edge_index=16, accepted_batch=accepted_batch,
    )
    _print_result(result, label="ib-obj-crosstype")
    ok = result.has_intra_batch_conflicts and len(result.intra_batch_object_conflicts) > 0
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}: expected intra-batch object cross-type conflict (IS_PRESIDENT_OF ≠ RUNS)")
    print(f"  → intra_batch_object_conflicts={len(result.intra_batch_object_conflicts)}, "
          f"strategy={result.strategy.value if result.strategy else 'None'}")


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

    # Pre-set cardinalities for the cross-type scenarios (11-16) so the LLM
    # classification stage skips them and the tests are fully deterministic.
    CONFIG.relation_cardinality_overrides.update({
        # Scenario 11 + 15 — SUBJECT_EXCLUSIVE cross-type
        "IS_DIRECTOR_OF": "SUBJECT_EXCLUSIVE",
        "LEADS": "SUBJECT_EXCLUSIVE",
        # Scenario 12 + 16 — OBJECT_EXCLUSIVE cross-type
        "IS_PRESIDENT_OF": "OBJECT_EXCLUSIVE",
        "RUNS": "OBJECT_EXCLUSIVE",
        # Scenario 13a + 13b — BOTH_EXCLUSIVE cross-type
        "IS_RULER_OF": "BOTH_EXCLUSIVE",
        "COMMANDS": "BOTH_EXCLUSIVE",
        # Scenario 14 — NON_EXCLUSIVE cross-type (same pair, opposing semantics)
        "WORKS_FOR": "NON_EXCLUSIVE",
        "LEFT_JOB_AT": "NON_EXCLUSIVE",
    })

    try:
        await init_schema(driver, database=TEST_DB)

        async with driver.session(database=TEST_DB) as session:
            await seed_database(session)

            # Stage 2c equivalent: classify cardinalities via LLM.
            # Only unknown types reach the LLM; the cross-type ones above are skipped.
            all_relation_types = [
                "IS_CEO_OF", "HAS_CAPITAL",
                "WORKED_AT", "COLLABORATED_WITH",
                "IS_DIRECTOR_OF", "LEADS",
                "IS_PRESIDENT_OF", "RUNS",
                "IS_RULER_OF", "COMMANDS",
                "WORKS_FOR", "LEFT_JOB_AT",
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
            # Cross-type conflict scenarios (11-16)
            await run_subject_crosstype_conflict(session, model)
            await run_object_crosstype_conflict(session, model)
            await run_both_exclusive_crosstype_subject(session, model)
            await run_both_exclusive_crosstype_object(session, model)
            await run_non_exclusive_crosstype_conflict(session, model)
            await run_intra_batch_subject_crosstype(session, model)
            await run_intra_batch_object_crosstype(session, model)
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
