"""CGRR evaluation harness.

Converts the relationship ground truth into relationship dicts, runs every
scorer against every test pair, and returns ScorerResult objects that the
shared hypothesis-testing module can consume directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from graphrag.bt_graphrag.entity_resolution.scorers import RelationshipScorer
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig


# ---------------------------------------------------------------------------
# Result dataclass  (mirrors cger.harness.ScorerResult but for relationships)
# ---------------------------------------------------------------------------

@dataclass
class ScorerResult:
    scorer_name: str
    component: str = "CGRR"
    config_snapshot: dict = field(default_factory=dict)
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    total_pairs: int = 0
    elapsed_seconds: float = 0.0
    pair_details: list[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = (self.true_positives + self.true_negatives
                 + self.false_positives + self.false_negatives)
        return (self.true_positives + self.true_negatives) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "scorer_name": self.scorer_name,
            "component": self.component,
            "config_snapshot": self.config_snapshot,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "total_pairs": self.total_pairs,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


# ---------------------------------------------------------------------------
# Build relationship dicts from ground truth
# ---------------------------------------------------------------------------

def build_relationship_dicts(ground_truth: dict) -> list[dict]:
    """Return a flat list of relationship dicts, one per expected pair entry.

    Each dict contains:
      alias, relation_type (same as alias), description, source, target,
      canonical_id, should_normalize.
    """
    canonical_descs = ground_truth.get("canonical_descriptions", {})
    alias_endpoints = ground_truth.get("alias_endpoints", {})

    rows: list[dict] = []
    seen: set[str] = set()
    alias_map = ground_truth.get("alias_map", {})

    for alias in alias_map:
        if alias in seen:
            continue
        seen.add(alias)
        cid = alias_map[alias]
        src, tgt = alias_endpoints.get(alias, ("", ""))
        rows.append({
            "alias": alias,
            "relation_type": alias,
            "description": canonical_descs.get(cid, ""),
            "source": src,
            "target": tgt,
            "canonical_id": cid,
        })
    return rows


# ---------------------------------------------------------------------------
# Evaluate a single scorer
# ---------------------------------------------------------------------------

def evaluate_scorer(
    scorer: RelationshipScorer,
    scorer_name: str,
    ground_truth: dict,
    config: BTGraphRAGConfig,
) -> ScorerResult:
    """Score every pair in ``expected_normalizations`` and compute metrics."""
    result = ScorerResult(
        scorer_name=scorer_name,
        config_snapshot={
            "merge_threshold":   config.cgrr_merge_threshold,
            "llm_threshold_low": config.cgrr_llm_threshold_low,
            "bm25_weight":       config.cgrr_bm25_weight,
            "semantic_weight":   config.cgrr_semantic_weight,
            "endpoint_weight":   config.cgrr_endpoint_weight,
        },
    )

    alias_endpoints    = ground_truth.get("alias_endpoints", {})
    canonical_descs    = ground_truth.get("canonical_descriptions", {})
    alias_map          = ground_truth.get("alias_map", {})
    expected           = ground_truth["expected_normalizations"]

    t0 = time.time()

    for pair in expected:
        a, b = pair["alias_a"], pair["alias_b"]
        should = pair["should_normalize"]

        # Build relationship dicts on the fly from stored fields
        src_a, tgt_a = alias_endpoints.get(a, (pair.get("source_a", ""), pair.get("target_a", "")))
        src_b, tgt_b = alias_endpoints.get(b, (pair.get("source_b", ""), pair.get("target_b", "")))
        cid_a = alias_map.get(a, "")
        cid_b = alias_map.get(b, "")
        desc_a = canonical_descs.get(cid_a, pair.get("desc_a", ""))
        desc_b = canonical_descs.get(cid_b, pair.get("desc_b", ""))

        score, breakdown = scorer(
            a, desc_a, src_a, tgt_a,
            b, desc_b, src_b, tgt_b,
            config,
        )

        predicted = score >= config.cgrr_merge_threshold
        in_llm_zone = config.cgrr_llm_threshold_low <= score < config.cgrr_merge_threshold

        if   should and predicted:     result.true_positives  += 1; outcome = "TP"
        elif should and not predicted: result.false_negatives += 1; outcome = "FN"
        elif not should and predicted: result.false_positives += 1; outcome = "FP"
        else:                          result.true_negatives  += 1; outcome = "TN"

        result.pair_details.append({
            "alias_a": a, "alias_b": b,
            "should_normalize": should,
            "score": round(score, 4),
            "predicted_normalize": predicted,
            "in_llm_zone": in_llm_zone,
            "outcome": outcome,
            "breakdown": {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in breakdown.items()},
        })
        result.total_pairs += 1

    result.elapsed_seconds = time.time() - t0
    return result
