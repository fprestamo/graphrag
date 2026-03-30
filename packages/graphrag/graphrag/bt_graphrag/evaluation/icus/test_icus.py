# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Standalone ICUS (Incremental Community Update & Selective Summarization) evaluation.

Evaluates BT-GraphRAG Stages 5–6:
  Stage 5 — Incremental Community Update:
      Re-runs Leiden only on the k-hop neighborhood of modified entities.
  Stage 6 — Selective LLM Summarization:
      Regenerates community reports only for communities marked as stale.

This script covers two evaluation planes:

  PLANE A — CORRECTNESS
      Verifies that every code path in community_update.py behaves as
      specified, using pure-Python DataFrames (no Neo4j required).

      Scenario 1  identify_stale_communities(): modified entities → correct stale IDs
      Scenario 2  identify_stale_communities(): no modified entities → all fresh
      Scenario 3  identify_stale_communities(): all entities modified → all stale
      Scenario 4  identify_stale_communities(): entity_ids as comma-string (legacy format)
      Scenario 5  filter_stale_communities(): returns only stale communities
      Scenario 6  filter_stale_communities(): empty stale list → empty DataFrame
      Scenario 7  merge_community_reports(): stale replaced, fresh kept
      Scenario 8  merge_community_reports(): empty existing → use new reports only
      Scenario 9  merge_community_reports(): empty new reports → keep existing
      Scenario 10 run_incremental_community_update(): no-Neo4j fallback (full rebuild)
      Scenario 11 annotate_community_reports_temporal(): adds column, no crash

  PLANE B — QUALITY METRICS
      Measures the quality of incremental detection and selective
      summarization using precision, recall, F1, and efficiency ratios.

      Metric 1  Stale-detection Recall    — are all truly-stale communities caught?
      Metric 2  Stale-detection Precision — are non-stale communities left alone?
      Metric 3  Stale-detection F1        — harmonic mean of Precision × Recall
      Metric 4  LLM Savings Rate          — fraction of communities not regenerated
      Metric 5  k-Hop Expansion Ratio     — neighbourhood growth per hop
      Metric 6  Merge Correctness Score   — reports correctly kept / replaced

Usage:
    python packages/graphrag/graphrag/bt_graphrag/evaluation/icus/test_icus.py

No Neo4j connection required — Stage 5 falls back gracefully when no driver
is provided, and Stage 6 is pure DataFrame logic.
"""

from __future__ import annotations

import asyncio
import math
import sys
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pandas as pd

from graphrag.bt_graphrag.community_update import (
    annotate_community_reports_temporal,
    filter_stale_communities,
    identify_stale_communities,
    merge_community_reports,
    run_incremental_community_update,
)
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = BTGraphRAGConfig(
    enabled=True,
    neo4j_uri="neo4j://127.0.0.1:7687",
    neo4j_user="neo4j",
    neo4j_password="12345678",
    neo4j_database="icustest",
    community_update_k_hop=2,
)

# ─────────────────────────────────────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    name: str
    passed: bool
    details: str = ""
    metric: float | None = None


RESULTS: list[ScenarioResult] = []


def _check(name: str, condition: bool, details: str = "", metric: float | None = None) -> None:
    icon = "✓" if condition else "✗"
    status = "PASS" if condition else "FAIL"
    print(f"  [{icon}] {status}  {name}")
    if details:
        for line in details.strip().splitlines():
            print(f"          {line}")
    RESULTS.append(ScenarioResult(name=name, passed=condition, details=details, metric=metric))


# ─────────────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────────────

def _community(
    cid: str | int,
    entity_ids: list[str],
    level: int = 0,
) -> dict:
    return {
        "id": str(cid),
        "community": int(cid) if str(cid).isdigit() else 0,
        "level": level,
        "parent": -1,
        "children": [],
        "title": f"Community {cid}",
        "entity_ids": entity_ids,
        "relationship_ids": [],
        "text_unit_ids": [],
        "period": None,
        "size": len(entity_ids),
        "human_readable_id": int(cid) if str(cid).isdigit() else 0,
    }


def _entity(title: str, eid: str | None = None) -> dict:
    return {
        "id": eid or str(uuid4()),
        "title": title,
        "type": "person",
        "description": f"Entity: {title}",
        "text_unit_ids": [],
        "human_readable_id": 0,
        "degree": 1,
        "frequency": 1,
    }


def _report(community_id: str | int, summary: str = "Summary") -> dict:
    return {
        "id": str(uuid4()),
        "community": int(community_id) if str(community_id).isdigit() else 0,
        "human_readable_id": int(community_id) if str(community_id).isdigit() else 0,
        "level": 0,
        "parent": -1,
        "children": [],
        "title": f"Community {community_id} Report",
        "summary": summary,
        "full_content": f"Full content of community {community_id}",
        "rating": 7.5,
        "rating_explanation": "Good",
        "findings": [],
        "full_content_json": "{}",
        "period": None,
        "size": 5,
    }


def _make_communities_df(specs: list[tuple]) -> pd.DataFrame:
    """Build communities DataFrame from list of (id, entity_ids) tuples."""
    rows = [_community(cid, eids) for cid, eids in specs]
    return pd.DataFrame(rows)


def _make_entities_df(specs: list[tuple]) -> pd.DataFrame:
    """Build entities DataFrame from list of (id, title) tuples."""
    rows = [{"id": eid, "title": title, "type": "person",
             "description": "", "text_unit_ids": [],
             "human_readable_id": i, "degree": 1, "frequency": 1}
            for i, (eid, title) in enumerate(specs)]
    return pd.DataFrame(rows)


def _make_reports_df(community_ids: list) -> pd.DataFrame:
    return pd.DataFrame([_report(cid) for cid in community_ids])


# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# PLANE A — CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def _sep(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


# ─── Scenario 1 ──────────────────────────────────────────────────────────────

def run_scenario_1_stale_detection_basic() -> None:
    _sep("SCENARIO 1 — identify_stale_communities(): basic stale detection")

    e_id_a, e_id_b, e_id_c, e_id_d, e_id_e = (str(uuid4()) for _ in range(5))

    # Community 0 has entities A, B
    # Community 1 has entities C, D
    # Community 2 has entities E
    communities_df = _make_communities_df([
        ("0", [e_id_a, e_id_b]),
        ("1", [e_id_c, e_id_d]),
        ("2", [e_id_e]),
    ])

    # Only entity C was modified → community 1 should be stale
    modified_ids = {e_id_c}
    stale = identify_stale_communities(communities_df, modified_ids)
    stale_set = set(stale)

    _check(
        "Community 1 (contains modified entity C) is stale",
        "1" in stale_set,
        f"stale_set={stale_set}",
    )
    _check(
        "Community 0 (unmodified entities A, B) is fresh",
        "0" not in stale_set,
        f"stale_set={stale_set}",
    )
    _check(
        "Community 2 (unmodified entity E) is fresh",
        "2" not in stale_set,
        f"stale_set={stale_set}",
    )
    _check(
        "Exactly 1 stale community returned",
        len(stale) == 1,
        f"len(stale)={len(stale)}",
    )


# ─── Scenario 2 ──────────────────────────────────────────────────────────────

def run_scenario_2_no_modifications() -> None:
    _sep("SCENARIO 2 — identify_stale_communities(): no modified entities → all fresh")

    e_ids = [str(uuid4()) for _ in range(6)]
    communities_df = _make_communities_df([
        ("0", e_ids[:2]),
        ("1", e_ids[2:4]),
        ("2", e_ids[4:]),
    ])

    stale = identify_stale_communities(communities_df, modified_entity_ids=set())

    _check(
        "No stale communities when no entities were modified",
        len(stale) == 0,
        f"stale={stale}",
    )


# ─── Scenario 3 ──────────────────────────────────────────────────────────────

def run_scenario_3_all_modified() -> None:
    _sep("SCENARIO 3 — identify_stale_communities(): all entities modified → all stale")

    e_ids = [str(uuid4()) for _ in range(6)]
    communities_df = _make_communities_df([
        ("0", e_ids[:2]),
        ("1", e_ids[2:4]),
        ("2", e_ids[4:]),
    ])

    stale = identify_stale_communities(communities_df, modified_entity_ids=set(e_ids))

    _check(
        "All 3 communities stale when all entities were modified",
        set(stale) == {"0", "1", "2"},
        f"stale_set={set(stale)}",
    )


# ─── Scenario 4 ──────────────────────────────────────────────────────────────

def run_scenario_4_legacy_string_entity_ids() -> None:
    _sep("SCENARIO 4 — identify_stale_communities(): entity_ids stored as comma-string")

    e_id_x, e_id_y, e_id_z = str(uuid4()), str(uuid4()), str(uuid4())

    # Simulate legacy format where entity_ids is a single comma-separated string
    communities_df = pd.DataFrame([
        {
            "id": "10",
            "entity_ids": f"{e_id_x},{e_id_y}",  # comma-delimited string
        },
        {
            "id": "11",
            "entity_ids": e_id_z,
        },
    ])

    stale = identify_stale_communities(communities_df, {e_id_x})
    stale_set = set(stale)

    _check(
        "Community 10 stale (entity_ids as comma-string, entity x modified)",
        "10" in stale_set,
        f"stale_set={stale_set}",
    )
    _check(
        "Community 11 fresh (entity z not modified)",
        "11" not in stale_set,
        f"stale_set={stale_set}",
    )


# ─── Scenario 5 ──────────────────────────────────────────────────────────────

def run_scenario_5_filter_stale_communities() -> None:
    _sep("SCENARIO 5 — filter_stale_communities(): returns only stale rows")

    e_ids = [str(uuid4()) for _ in range(6)]
    communities_df = _make_communities_df([
        ("0", e_ids[:2]),
        ("1", e_ids[2:4]),
        ("2", e_ids[4:]),
    ])
    existing_reports = _make_reports_df(["0", "1", "2"])

    stale_ids = ["1"]
    filtered = filter_stale_communities(communities_df, existing_reports, stale_ids)

    _check(
        "filter_stale_communities() returns exactly 1 row",
        len(filtered) == 1,
        f"len(filtered)={len(filtered)}",
    )
    _check(
        "The returned row has id == '1'",
        str(filtered.iloc[0]["id"]) == "1",
        f"id={filtered.iloc[0]['id']}",
    )


# ─── Scenario 6 ──────────────────────────────────────────────────────────────

def run_scenario_6_filter_empty_stale_list() -> None:
    _sep("SCENARIO 6 — filter_stale_communities(): empty stale list → empty DataFrame")

    e_ids = [str(uuid4()) for _ in range(4)]
    communities_df = _make_communities_df([("0", e_ids[:2]), ("1", e_ids[2:])])
    existing_reports = _make_reports_df(["0", "1"])

    filtered = filter_stale_communities(communities_df, existing_reports, stale_community_ids=[])

    _check(
        "Empty stale list produces empty filtered DataFrame",
        filtered.empty,
        f"len(filtered)={len(filtered)}",
    )


# ─── Scenario 7 ──────────────────────────────────────────────────────────────

def run_scenario_7_merge_reports_incremental() -> None:
    _sep("SCENARIO 7 — merge_community_reports(): stale replaced, fresh kept")

    existing = _make_reports_df(["0", "1", "2"])
    # Overwrite community 1's summary so we can detect the swap
    existing.loc[existing["community"] == 1, "summary"] = "OLD summary for community 1"

    new_for_stale = pd.DataFrame([{
        **_report("1"),
        "summary": "NEW summary for community 1",
        "community": 1,
    }])

    merged = merge_community_reports(existing, new_for_stale, stale_community_ids=["1"])

    # The merged result should have 3 reports total
    _check(
        "Merged report count equals total communities (3)",
        len(merged) == 3,
        f"len(merged)={len(merged)}",
    )
    # Community 1 report should carry the NEW summary
    c1_rows = merged[merged["community"] == 1]
    _check(
        "Community 1 report replaced with new summary",
        not c1_rows.empty and "NEW" in str(c1_rows.iloc[0]["summary"]),
        f"summary='{c1_rows.iloc[0]['summary'] if not c1_rows.empty else 'NOT FOUND'}'",
    )
    # Community 0 and 2 should be kept unchanged
    c0_rows = merged[merged["community"] == 0]
    c2_rows = merged[merged["community"] == 2]
    _check(
        "Communities 0 and 2 reports are kept from existing",
        not c0_rows.empty and not c2_rows.empty,
        f"c0 present={not c0_rows.empty}, c2 present={not c2_rows.empty}",
    )


# ─── Scenario 8 ──────────────────────────────────────────────────────────────

def run_scenario_8_merge_empty_existing() -> None:
    _sep("SCENARIO 8 — merge_community_reports(): empty existing → use new reports only")

    existing = pd.DataFrame()
    new_reports = _make_reports_df(["0", "1"])

    merged = merge_community_reports(existing, new_reports, stale_community_ids=["0", "1"])

    _check(
        "All new reports preserved when existing is empty",
        len(merged) == 2,
        f"len(merged)={len(merged)}",
    )


# ─── Scenario 9 ──────────────────────────────────────────────────────────────

def run_scenario_9_merge_empty_new() -> None:
    _sep("SCENARIO 9 — merge_community_reports(): empty new reports → keep existing")

    existing = _make_reports_df(["0", "1", "2"])
    new_reports = pd.DataFrame()

    merged = merge_community_reports(existing, new_reports, stale_community_ids=["1"])

    _check(
        "Existing reports kept when new reports is empty",
        len(merged) == 3,
        f"len(merged)={len(merged)}",
    )


# ─── Scenario 10 ─────────────────────────────────────────────────────────────

async def run_scenario_10_no_neo4j_full_rebuild() -> None:
    _sep("SCENARIO 10 — run_incremental_community_update(): no Neo4j → full rebuild")

    e_ids = [str(uuid4()) for _ in range(6)]
    communities_df = _make_communities_df([
        ("0", e_ids[:2]),
        ("1", e_ids[2:4]),
        ("2", e_ids[4:]),
    ])
    entities_df = _make_entities_df([(e, f"Entity {i}") for i, e in enumerate(e_ids)])
    relationships_df = pd.DataFrame()

    updated_communities, stale_ids = await run_incremental_community_update(
        communities_df=communities_df,
        entities_df=entities_df,
        relationships_df=relationships_df,
        config=CONFIG,
        driver=None,           # No Neo4j → triggers full rebuild path
        last_update_time=None,
    )

    _check(
        "Stage 5 returns updated communities DataFrame (not None)",
        updated_communities is not None and isinstance(updated_communities, pd.DataFrame),
    )
    _check(
        "No-Neo4j path marks ALL communities as stale (full rebuild)",
        set(stale_ids) == {"0", "1", "2"},
        f"stale_ids={stale_ids}",
    )
    _check(
        "Community count unchanged after Stage 5",
        len(updated_communities) == 3,
        f"len(updated_communities)={len(updated_communities)}",
    )


# ─── Scenario 11 ─────────────────────────────────────────────────────────────

def run_scenario_11_annotate_temporal() -> None:
    _sep("SCENARIO 11 — annotate_community_reports_temporal(): adds column, no crash")

    reports = _make_reports_df(["0", "1"])
    entities = pd.DataFrame([_entity("Alice"), _entity("Bob")])
    relationships = pd.DataFrame()

    annotated = annotate_community_reports_temporal(reports, entities, relationships)

    _check(
        "annotate returns a DataFrame (not None)",
        isinstance(annotated, pd.DataFrame),
    )
    _check(
        "Row count unchanged after annotation",
        len(annotated) == 2,
        f"len(annotated)={len(annotated)}",
    )
    _check(
        "temporal_evolution column present in result",
        "temporal_evolution" in annotated.columns,
        f"columns={list(annotated.columns)}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# PLANE B — QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityMetrics:
    """Container for Stages 5–6 quality metrics."""
    stale_recall: float = 0.0
    stale_precision: float = 0.0
    stale_f1: float = 0.0
    llm_savings_rate: float = 0.0
    k_hop_expansion_ratio: float = 0.0
    merge_correctness_score: float = 0.0
    run_details: dict[str, Any] = field(default_factory=dict)


def _compute_quality_metrics(
    communities_df: pd.DataFrame,
    true_stale_ids: set[str],
    predicted_stale_ids: list[str],
    total_entities: int,
    neighborhood_size_per_hop: list[int],
    merged_reports: pd.DataFrame,
    expected_replaced_ids: set[str],
    expected_kept_ids: set[str],
    original_reports: pd.DataFrame,
) -> QualityMetrics:
    """Compute all quality metrics for Stages 5–6."""
    predicted_set = set(predicted_stale_ids)
    total_communities = len(communities_df)

    # ── Stale-detection metrics ────────────────────────────────────────────
    tp = len(true_stale_ids & predicted_set)
    fp = len(predicted_set - true_stale_ids)
    fn = len(true_stale_ids - predicted_set)

    recall = tp / max(len(true_stale_ids), 1)
    precision = tp / max(len(predicted_set), 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-9))

    # ── LLM savings ────────────────────────────────────────────────────────
    # Proportion of community reports that were NOT regenerated.
    # Mirrors TG-RAG metric: "1.6M vs 30M tokens for incremental vs full"
    fresh_count = total_communities - len(predicted_stale_ids)
    llm_savings_rate = fresh_count / max(total_communities, 1)

    # ── k-Hop expansion ratio ──────────────────────────────────────────────
    # For each hop, the fraction of the entity space covered grows.
    # We compute average fractional increase per hop.
    if len(neighborhood_size_per_hop) >= 2 and total_entities > 0:
        ratios = []
        for i in range(1, len(neighborhood_size_per_hop)):
            prev = neighborhood_size_per_hop[i - 1]
            curr = neighborhood_size_per_hop[i]
            ratios.append((curr - prev) / max(total_entities - prev, 1))
        k_hop_expansion_ratio = sum(ratios) / len(ratios)
    else:
        k_hop_expansion_ratio = 0.0

    # ── Merge correctness score ────────────────────────────────────────────
    # Fraction of communities whose report is correctly handled:
    #   - replaced communities must carry the new report
    #   - kept communities must carry the original report
    correct = 0
    total_checked = len(expected_replaced_ids) + len(expected_kept_ids)

    if not merged_reports.empty and "community" in merged_reports.columns:
        merged_index = {
            str(row["community"]): row["summary"]
            for _, row in merged_reports.iterrows()
        }
        original_index = {
            str(row["community"]): row["summary"]
            for _, row in original_reports.iterrows()
            if "community" in original_reports.columns
        }

        for cid in expected_replaced_ids:
            # The replaced report should NOT match the original summary
            if cid in merged_index and cid in original_index:
                if merged_index[cid] != original_index[cid]:
                    correct += 1
                # If there was no original to compare, count as correct (new report)
            elif cid in merged_index:
                correct += 1

        for cid in expected_kept_ids:
            # The kept report SHOULD match the original summary
            if cid in merged_index and cid in original_index:
                if merged_index[cid] == original_index[cid]:
                    correct += 1

    merge_correctness_score = correct / max(total_checked, 1)

    return QualityMetrics(
        stale_recall=recall,
        stale_precision=precision,
        stale_f1=f1,
        llm_savings_rate=llm_savings_rate,
        k_hop_expansion_ratio=k_hop_expansion_ratio,
        merge_correctness_score=merge_correctness_score,
        run_details={
            "total_communities": total_communities,
            "true_stale_count": len(true_stale_ids),
            "predicted_stale_count": len(predicted_stale_ids),
            "tp": tp, "fp": fp, "fn": fn,
            "fresh_communities": fresh_count,
        },
    )


def run_quality_metrics_evaluation() -> QualityMetrics:
    _sep("PLANE B — QUALITY METRICS EVALUATION")

    # ── Build a realistic test corpus ──────────────────────────────────────
    #   20 entities split across 5 communities (4 entities each).
    #   3 of the 5 communities are truly stale (modified entities inside).
    #   We simulate a 2-hop neighborhood expansion that covers 8 entities.
    NUM_ENTITIES = 20
    NUM_COMMUNITIES = 5
    e_ids = [str(uuid4()) for _ in range(NUM_ENTITIES)]

    # Group entities into communities
    community_specs = []
    for i in range(NUM_COMMUNITIES):
        start = i * (NUM_ENTITIES // NUM_COMMUNITIES)
        end = start + (NUM_ENTITIES // NUM_COMMUNITIES)
        community_specs.append((str(i), e_ids[start:end]))

    communities_df = _make_communities_df(community_specs)

    # Ground truth: communities 0, 1, and 3 have modified entities
    true_stale_ids = {"0", "1", "3"}

    # The modified entity IDs are one representative from each stale community
    modified_entity_ids = {
        e_ids[0],   # from community 0
        e_ids[4],   # from community 1
        e_ids[12],  # from community 3
    }

    # Stage 5: identify stale communities
    predicted_stale = identify_stale_communities(communities_df, modified_entity_ids)

    # Simulate k-hop expansion sizes (hop 0 = seed, hop 1, hop 2)
    hop_sizes = [3, 8, 12]  # realistic neighborhood growth

    # Stage 6: existing reports for all 5 communities
    existing_reports = _make_reports_df([str(i) for i in range(NUM_COMMUNITIES)])
    # Mark stale reports with "OLD" summary so we can detect replacement
    for cid in true_stale_ids:
        mask = existing_reports["community"] == int(cid)
        existing_reports.loc[mask, "summary"] = f"OLD summary for community {cid}"

    # Generate new reports only for true stale communities
    new_report_rows = []
    for cid in true_stale_ids:
        row = _report(cid)
        row["summary"] = f"NEW summary for community {cid}"
        row["community"] = int(cid)
        new_report_rows.append(row)
    new_reports = pd.DataFrame(new_report_rows)

    # Merge: replace stale, keep fresh
    merged = merge_community_reports(existing_reports, new_reports, list(true_stale_ids))

    # Fresh community IDs (should be kept from existing)
    all_ids = {str(i) for i in range(NUM_COMMUNITIES)}
    expected_kept_ids = all_ids - true_stale_ids

    # ── Compute metrics ────────────────────────────────────────────────────
    metrics = _compute_quality_metrics(
        communities_df=communities_df,
        true_stale_ids=true_stale_ids,
        predicted_stale_ids=predicted_stale,
        total_entities=NUM_ENTITIES,
        neighborhood_size_per_hop=hop_sizes,
        merged_reports=merged,
        expected_replaced_ids=true_stale_ids,
        expected_kept_ids=expected_kept_ids,
        original_reports=existing_reports,
    )

    # ── Print metrics table ────────────────────────────────────────────────
    d = metrics.run_details
    print(f"\n  {'Corpus':}")
    print(f"    Total communities          : {d['total_communities']}")
    print(f"    Total entities             : {NUM_ENTITIES}")
    print(f"    Truly stale communities    : {d['true_stale_count']}  {sorted(true_stale_ids)}")
    print(f"    Predicted stale            : {d['predicted_stale_count']}  {sorted(predicted_stale)}")
    print(f"    TP={d['tp']}  FP={d['fp']}  FN={d['fn']}")

    print(f"\n  {'Metric':<40} {'Value':>10}  Threshold  Status")
    print(f"  {'-'*40}  {'-'*10}  {'-'*9}  {'-'*6}")

    def _row(label: str, value: float, threshold: float, higher_better: bool = True) -> str:
        passed = (value >= threshold) if higher_better else (value <= threshold)
        pct = f"{value * 100:.1f}%"
        icon = "✓" if passed else "✗"
        thr_str = f"≥ {threshold * 100:.0f}%" if higher_better else f"≤ {threshold * 100:.0f}%"
        return f"  {label:<40}  {pct:>10}  {thr_str:>9}  [{icon}]"

    print(_row("Metric 1 — Stale-detection Recall",    metrics.stale_recall,           0.90))
    print(_row("Metric 2 — Stale-detection Precision",  metrics.stale_precision,         0.90))
    print(_row("Metric 3 — Stale-detection F1",         metrics.stale_f1,                0.90))
    print(_row("Metric 4 — LLM Savings Rate",           metrics.llm_savings_rate,        0.30))
    print(_row("Metric 5 — Merge Correctness Score",    metrics.merge_correctness_score, 0.90))
    print(f"  {'Metric 5b — k-Hop Expansion Ratio':<40}  {metrics.k_hop_expansion_ratio:>10.3f}  {'(info)':>9}  [ ]")

    # ── Threshold assertions ───────────────────────────────────────────────
    _check(
        "Metric 1: Stale-detection Recall ≥ 90%",
        metrics.stale_recall >= 0.90,
        f"recall={metrics.stale_recall:.3f}",
        metric=metrics.stale_recall,
    )
    _check(
        "Metric 2: Stale-detection Precision ≥ 90%",
        metrics.stale_precision >= 0.90,
        f"precision={metrics.stale_precision:.3f}",
        metric=metrics.stale_precision,
    )
    _check(
        "Metric 3: Stale-detection F1 ≥ 90%",
        metrics.stale_f1 >= 0.90,
        f"f1={metrics.stale_f1:.3f}",
        metric=metrics.stale_f1,
    )
    _check(
        "Metric 4: LLM Savings Rate ≥ 30%",
        metrics.llm_savings_rate >= 0.30,
        f"savings={metrics.llm_savings_rate:.3f} "
        f"({d['fresh_communities']} fresh / {d['total_communities']} total)",
        metric=metrics.llm_savings_rate,
    )
    _check(
        "Metric 5: Merge Correctness Score ≥ 90%",
        metrics.merge_correctness_score >= 0.90,
        f"score={metrics.merge_correctness_score:.3f}",
        metric=metrics.merge_correctness_score,
    )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# PLANE C — EDGE-CASE STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def run_edge_cases() -> None:
    _sep("PLANE C — EDGE-CASE STRESS TESTS")

    # ── Empty communities DataFrame ────────────────────────────────────────
    stale = identify_stale_communities(pd.DataFrame(columns=["id", "entity_ids"]), {str(uuid4())})
    _check("Empty communities → zero stale IDs", len(stale) == 0)

    # ── Single-entity community, that entity modified ──────────────────────
    single_id = str(uuid4())
    df = pd.DataFrame([{"id": "99", "entity_ids": [single_id]}])
    stale = identify_stale_communities(df, {single_id})
    _check("Single-entity community stale when its entity is modified", "99" in set(stale))

    # ── Communities with no entity_ids field ──────────────────────────────
    df_no_eids = pd.DataFrame([{"id": "5"}, {"id": "6"}])
    stale = identify_stale_communities(df_no_eids, {str(uuid4())})
    _check("Communities without entity_ids field → zero stale (safe)", len(stale) == 0)

    # ── filter_stale with communities DataFrame using 'community' (int) id col
    df_int = pd.DataFrame([
        {"community": 0, "id": "0", "entity_ids": []},
        {"community": 1, "id": "1", "entity_ids": []},
    ])
    filtered = filter_stale_communities(df_int, _make_reports_df(["0", "1"]), ["1"])
    _check(
        "filter_stale_communities() handles 'id' column with stale_id='1'",
        len(filtered) == 1,
        f"len={len(filtered)}",
    )

    # ── merge_community_reports() preserves column schema ─────────────────
    existing = _make_reports_df(["0", "1", "2"])
    new_r = pd.DataFrame([_report("1")])
    merged = merge_community_reports(existing, new_r, stale_community_ids=["1"])
    _check(
        "Merged report has 'community' column",
        "community" in merged.columns,
    )
    _check(
        "Merged report count is 3 (2 kept + 1 replaced)",
        len(merged) == 3,
        f"len(merged)={len(merged)}",
    )

    # ── Large-scale: 100 communities, 10 modified ──────────────────────────
    n = 100
    big_ids = [str(uuid4()) for _ in range(n)]
    big_communities = _make_communities_df([(str(i), [big_ids[i]]) for i in range(n)])
    modified_10 = set(big_ids[:10])
    stale_big = identify_stale_communities(big_communities, modified_10)
    _check(
        "Large scale: 10 modified entities → exactly 10 stale communities (1 entity/community)",
        len(stale_big) == 10,
        f"stale_count={len(stale_big)}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary() -> None:
    _sep("EVALUATION SUMMARY")
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r.passed)
    failed = total - passed

    print(f"\n  Scenarios run  : {total}")
    print(f"  Passed         : {passed}")
    print(f"  Failed         : {failed}")

    if failed:
        print(f"\n  FAILED checks:")
        for r in RESULTS:
            if not r.passed:
                print(f"    ✗  {r.name}")
                if r.details:
                    for line in r.details.strip().splitlines():
                        print(f"       {line}")

    pct = passed / total * 100 if total else 0
    print(f"\n  Overall pass rate: {pct:.1f}%")

    # Collect quality metrics
    quality_results = [r for r in RESULTS if r.metric is not None]
    if quality_results:
        print(f"\n  Quality metrics summary:")
        for r in quality_results:
            pct_val = f"{r.metric * 100:.1f}%" if r.metric is not None else "n/a"
            icon = "✓" if r.passed else "✗"
            print(f"    [{icon}] {r.name:<48} {pct_val:>8}")

    print()
    return failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> bool:
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   ICUS Evaluation — Incremental Community Update &                   ║")
    print("║                     Selective Summarization (Stages 5–6)             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  k-hop radius : {CONFIG.community_update_k_hop}")
    print(f"  Neo4j        : not required (pure-DataFrame evaluation)")

    # Plane A — Correctness
    run_scenario_1_stale_detection_basic()
    run_scenario_2_no_modifications()
    run_scenario_3_all_modified()
    run_scenario_4_legacy_string_entity_ids()
    run_scenario_5_filter_stale_communities()
    run_scenario_6_filter_empty_stale_list()
    run_scenario_7_merge_reports_incremental()
    run_scenario_8_merge_empty_existing()
    run_scenario_9_merge_empty_new()
    await run_scenario_10_no_neo4j_full_rebuild()
    run_scenario_11_annotate_temporal()

    # Plane B — Quality Metrics
    run_quality_metrics_evaluation()

    # Plane C — Edge Cases
    run_edge_cases()

    # Summary
    all_passed = _print_summary()
    return all_passed


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
