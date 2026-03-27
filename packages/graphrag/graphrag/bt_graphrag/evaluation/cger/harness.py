"""CGER evaluation harness.

Converts ground-truth alias data into entity dicts with embeddings,
runs every scorer against every test pair, and returns ScorerResult objects.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from graphrag.bt_graphrag.entity_resolution.scorers import EntityScorer
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScorerResult:
    scorer_name: str
    component: str = "CGER"
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
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
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
# Build entity dicts with embeddings from ground truth
# ---------------------------------------------------------------------------

async def build_entity_dicts(
    ground_truth: dict,
    embedding_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
    doc_texts: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return {surface_form_upper: entity_dict} with embeddings populated."""
    alias_map = ground_truth["alias_map"]
    canonical_map = {ce["id"]: ce for ce in ground_truth["canonical_entities"]}
    surface_forms = sorted(set(alias_map.keys()))

    descriptions = [
        canonical_map.get(alias_map[sf], {}).get("description", sf)
        for sf in surface_forms
    ]

    print(f"    [CGER] Embedding {len(descriptions)} entity descriptions…")
    desc_embeddings = await embedding_fn(descriptions)

    # Build text_unit_embeddings from document texts that cite each surface form
    surface_to_texts: dict[str, list[str]] = defaultdict(list)
    if doc_texts:
        for sf, texts in doc_texts.items():
            surface_to_texts[sf.upper()].extend(texts)

    all_texts: list[str] = []
    text_idx: dict[str, int] = {}
    for sf in surface_forms:
        for txt in surface_to_texts.get(sf, []):
            if txt not in text_idx:
                text_idx[txt] = len(all_texts)
                all_texts.append(txt)

    text_embeddings: list[list[float]] = []
    if all_texts:
        print(f"    [CGER] Embedding {len(all_texts)} document text units…")
        text_embeddings = await embedding_fn(all_texts)

    entities: dict[str, dict] = {}
    for i, sf in enumerate(surface_forms):
        cid = alias_map[sf]
        canonical = canonical_map.get(cid, {})

        vecs = [text_embeddings[text_idx[t]] for t in surface_to_texts.get(sf, []) if t in text_idx]
        if vecs:
            dim = len(vecs[0])
            tu_emb: list[float] = [sum(v[d] for v in vecs) / len(vecs) for d in range(dim)]
        else:
            tu_emb = []

        entities[sf] = {
            "title": sf,
            "type": canonical.get("type", "UNKNOWN"),
            "description": descriptions[i],
            "description_embedding": desc_embeddings[i] if i < len(desc_embeddings) else [],
            "text_unit_embedding": tu_emb,
            "active_start": None,
            "active_end": None,
            "relation_types": [],
            "canonical_id": cid,
        }

    return entities


# ---------------------------------------------------------------------------
# Evaluate a single scorer
# ---------------------------------------------------------------------------

def evaluate_scorer(
    scorer: EntityScorer,
    scorer_name: str,
    entities: dict[str, dict],
    expected_merges: list[dict],
    config: BTGraphRAGConfig,
) -> ScorerResult:
    result = ScorerResult(
        scorer_name=scorer_name,
        config_snapshot={
            "merge_threshold":    config.cger_merge_threshold,
            "llm_threshold_low":  config.cger_llm_threshold_low,
            "embedding_weight":   config.cger_embedding_weight,
            "bm25_weight":        config.cger_bm25_weight,
            "jaccard_weight":     config.cger_jaccard_weight,
            "temporal_weight":    config.cger_temporal_overlap_weight,
            "relation_weight":    config.cger_relation_context_weight,
        },
    )
    t0 = time.time()

    for pair in expected_merges:
        sa, sb = pair["surface_a"].upper(), pair["surface_b"].upper()
        ea, eb = entities.get(sa), entities.get(sb)
        if ea is None or eb is None:
            continue

        score, breakdown = scorer(ea, eb, config)
        predicted = score >= config.cger_merge_threshold
        should   = pair["should_merge"]

        if   should and predicted:     result.true_positives  += 1; outcome = "TP"
        elif should and not predicted: result.false_negatives += 1; outcome = "FN"
        elif not should and predicted: result.false_positives += 1; outcome = "FP"
        else:                          result.true_negatives  += 1; outcome = "TN"

        result.pair_details.append({
            "surface_a": sa, "surface_b": sb,
            "should_merge": should, "score": round(score, 4),
            "predicted_merge": predicted,
            "in_llm_zone": config.cger_llm_threshold_low <= score < config.cger_merge_threshold,
            "outcome": outcome,
            "breakdown": {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in breakdown.items()},
        })
        result.total_pairs += 1

    result.elapsed_seconds = time.time() - t0
    return result
