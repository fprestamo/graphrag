# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Stage 2 of the ETCDR evaluation: canonicalise then resolve conflicts.

Pipeline:

  1. **Load** the extracted entities + relationships written by
     ``extract.py`` (``data/test-extracted/``) together with the manually
     authored ground truth (``data/test-ground-true/``).

  2. **Canonicalise** entities and relation-type strings. The user-facing
     contract is "use the ground truth files directly to merge"; we
     implement this by running the real CGER/CGRR pipelines with the
     cosine trigger threshold pinned to 0.0 and a *ground-truth oracle
     LLM* that consults ``entity_resolution.json`` /
     ``relationship_resolution.json`` and answers ``SAME`` iff the two
     strings sit in the same cluster (and ``DIFFERENT_ENTITY`` /
     ``DIFFERENT`` otherwise). With threshold=0 every pre-filtered
     candidate triggers the oracle, so the resulting merge_map /
     normalize_map mirrors the ground-truth clusters end-to-end. The
     same oracle transparently delegates any non-CGER / non-CGRR
     prompt (e.g. ETCDR's Decision Router, cardinality classification)
     to a real LLM.

  3. **Classify cardinalities** for every relation type present in the
     canonicalised batch. Types listed in
     ``conflict_resolution.json::cardinality_overrides`` are taken
     verbatim; anything else falls through to an LLM classifier
     (analogous to Stage 2c in ``test_etcdr.py``).

  4. **Run ETCDR** on the canonicalised batch. We keep the Neo4j
     database empty and feed every relationship through
     ``detect_and_resolve`` in the order it appears in
     ``relationships.json``, accumulating an ``accepted_batch`` so
     intra-batch conflicts surface. The strategy chosen for each
     candidate is recorded.

  5. **Score** the actual strategies against
     ``conflict_resolution.json::expected``. Each expected entry is
     matched against the n-th occurrence of its
     (source, relation_type, target) triple in the batch (1st GT entry
     -> 1st triple occurrence, etc.). We report:
       - strategy accuracy (matches / scored)
       - per-strategy confusion matrix
       - a verbose listing of mismatches.

Usage:
    python stages-evaluation/etcdr/evaluate.py

Requirements:
    - ``extract.py`` has been run first (so the extracted JSONs exist).
    - ``GRAPHRAG_API_KEY`` (or ``OPENAI_API_KEY``) defined in the env or
      in the project-root ``.env`` file.
    - Neo4j running at ``neo4j://127.0.0.1:7687`` with three databases:
        * ``etcdreval``  — empty workspace; wiped on entry.
        * ``cgerbatch``  — scratch DB for CGER Phase B top-K retrieval.
        * ``cgrrbatch``  — scratch DB for CGRR Phase B top-K retrieval.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from graphrag.bt_graphrag.conflict_detection.etcdr import (
    detect_and_resolve,
    flush_resolution_log,
)
from graphrag.bt_graphrag.entity_resolution.cger import (
    apply_merge_map_to_relationships,
    resolve_entities,
)
from graphrag.bt_graphrag.entity_resolution.cgrr import resolve_relationships
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.models.temporal_types import (
    INFINITY,
    MINUS_INFINITY,
    ProvenanceRecord,
    RelationCardinality,
    ResolutionStrategy,
    TemporalRelationship,
    TemporalStateQuad,
    utcnow,
)
from graphrag.bt_graphrag.neo4j_store import ETCDRBatchDB, init_schema
from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig
from graphrag_llm.config.types import LLMProviderType
from graphrag_llm.utils import create_completion_response


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
EXTRACTED_DIR = HERE / "data" / "test-extracted"
GROUND_TRUTH_DIR = HERE / "data" / "test-ground-true"
RESULTS_PATH = HERE / "data" / "test-results.json"
RESOLUTION_LOG_DIR = HERE / "data"
ENV_PATH = PROJECT_ROOT / ".env"

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
TEST_DB = "etcdreval"
CGER_TEMP_DB = "cgerbatch"
CGRR_TEMP_DB = "cgrrbatch"
ETCDR_TEMP_DB = "etcdrbatch"

COMPLETION_MODEL = "gpt-4.1-mini"

# Oracle canonicalisation: threshold=0 ⇒ every cosine-best candidate triggers
# the LLM, and the oracle's verdict is determined entirely by the GT clusters.
# top_k is intentionally huge so the candidate pool covers the whole batch
# (the oracle still picks the best-cosine candidate to verify, so the cluster
# member with the highest cosine to the new entity is what gets compared —
# in practice this matches the GT for any well-formed cluster).
CANON_COSINE_THRESHOLD = 0.0
CANON_TOP_K = 10000
CANON_PHASE_B_TOP_K = 10000

NEO4J_VECTOR_DIMENSIONS = 3072

ETCDR_CONFIDENCE_THRESHOLD = 0.7
ETCDR_TOPK = 5
"""Per candidate, how many top-cosine existing edges get routed through the
Decision Router. See ``BTGraphRAGConfig.etcdr_topk``."""

ETCDR_COSINE_THRESHOLD = 0.5
"""Cosine floor below which existing edges are filtered out before top-K
(same-pair entries bypass this floor — they are duplicate checks). See
``BTGraphRAGConfig.etcdr_cosine_threshold``."""


# ---------------------------------------------------------------------------
# .env loading + LLM factories
# ---------------------------------------------------------------------------

def _load_env_file(path: Path) -> None:
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


def _real_completion():
    return create_completion(
        ModelConfig(
            type=LLMProviderType.LiteLLM,
            model_provider="openai",
            model=COMPLETION_MODEL,
            api_key=_api_key(),
        )
    )


# ---------------------------------------------------------------------------
# Ground-truth oracle LLM
# ---------------------------------------------------------------------------

# Regex tied to the CGER_VERIFICATION_PROMPT format in entity_resolution/cger.py.
_CGER_BLOCK_RE = re.compile(
    r"Entity\s+([AB])\s*:\s*\n\s*-\s*Name\s*:\s*(.+?)\n",
    re.IGNORECASE,
)
# Regex tied to the CGRR_VERIFICATION_PROMPT format in entity_resolution/cgrr.py.
_CGRR_BLOCK_RE = re.compile(
    r"Relationship\s+([AB])\s*:\s*\n\s*-\s*Relation\s+Type\s*:\s*(.+?)\n",
    re.IGNORECASE,
)


class GroundTruthOracleCompletion:
    """LLM facade that answers CGER/CGRR verification prompts from ground truth.

    For prompts matching the CGER ``Entity A:`` / ``Entity B:`` template,
    returns ``SAME`` iff both entity titles sit in the same cluster of
    ``entity_clusters``, otherwise ``DIFFERENT_ENTITY``.

    For prompts matching the CGRR ``Relationship A:`` / ``Relationship B:``
    template, returns ``SAME`` iff both relation_type strings sit in the
    same cluster of ``relation_clusters``, otherwise ``DIFFERENT``.

    Every other prompt (ETCDR decision router, cardinality classifier, …)
    is forwarded verbatim to ``fallback_model``.

    The class only needs to expose ``completion_async`` because CGER,
    CGRR and ETCDR all call it through that single method.
    """

    def __init__(
        self,
        entity_clusters: list[dict[str, Any]],
        relation_clusters: list[dict[str, Any]],
        fallback_model: Any,
    ) -> None:
        self._fallback = fallback_model
        # Build alias -> canonical maps and "cluster id" lookups for O(1) membership.
        self._entity_cluster_id: dict[str, int] = {}
        for idx, c in enumerate(entity_clusters):
            for alias in c.get("aliases", []):
                self._entity_cluster_id[str(alias).strip().upper()] = idx
        self._relation_cluster_id: dict[str, int] = {}
        for idx, c in enumerate(relation_clusters):
            for alias in c.get("aliases", []):
                self._relation_cluster_id[str(alias).strip().upper()] = idx

        # Telemetry
        self.cger_calls = 0
        self.cger_same = 0
        self.cger_diff = 0
        self.cgrr_calls = 0
        self.cgrr_same = 0
        self.cgrr_diff = 0
        self.fallback_calls = 0

    @property
    def metrics_store(self) -> Any:  # pragma: no cover — duck-typed
        return getattr(self._fallback, "metrics_store", None)

    @property
    def tokenizer(self) -> Any:  # pragma: no cover — duck-typed
        return getattr(self._fallback, "tokenizer", None)

    def _user_prompt(self, messages: Any) -> str:
        """Extract the user prompt content from a CompletionMessagesBuilder output."""
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            for m in messages:
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                if role == "user":
                    content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
                    if isinstance(content, str):
                        return content
        return ""

    def _route(self, prompt: str) -> str | None:
        """Classify a prompt as 'cger', 'cgrr', or fallback (None)."""
        if "Entity A:" in prompt and "Entity B:" in prompt and "Name:" in prompt:
            return "cger"
        if (
            "Relationship A:" in prompt
            and "Relationship B:" in prompt
            and "Relation Type:" in prompt
        ):
            return "cgrr"
        return None

    def _cger_verdict(self, prompt: str) -> str:
        names: dict[str, str] = {}
        for m in _CGER_BLOCK_RE.finditer(prompt):
            label, name = m.group(1).upper(), m.group(2).strip()
            names[label] = name
        a = names.get("A", "").strip().upper()
        b = names.get("B", "").strip().upper()
        ida = self._entity_cluster_id.get(a)
        idb = self._entity_cluster_id.get(b)
        if ida is not None and idb is not None and ida == idb:
            self.cger_same += 1
            return "SAME"
        self.cger_diff += 1
        return "DIFFERENT_ENTITY"

    def _cgrr_verdict(self, prompt: str) -> str:
        types: dict[str, str] = {}
        for m in _CGRR_BLOCK_RE.finditer(prompt):
            label, rel_type = m.group(1).upper(), m.group(2).strip()
            types[label] = rel_type
        a = types.get("A", "").strip().upper()
        b = types.get("B", "").strip().upper()
        ida = self._relation_cluster_id.get(a)
        idb = self._relation_cluster_id.get(b)
        if ida is not None and idb is not None and ida == idb:
            self.cgrr_same += 1
            return "SAME"
        self.cgrr_diff += 1
        return "DIFFERENT"

    async def completion_async(self, **kwargs: Any) -> Any:
        prompt = self._user_prompt(kwargs.get("messages"))
        route = self._route(prompt)
        if route == "cger":
            self.cger_calls += 1
            return create_completion_response(self._cger_verdict(prompt))
        if route == "cgrr":
            self.cgrr_calls += 1
            return create_completion_response(self._cgrr_verdict(prompt))
        self.fallback_calls += 1
        return await self._fallback.completion_async(**kwargs)

    def completion(self, **kwargs: Any) -> Any:  # pragma: no cover — sync path unused
        prompt = self._user_prompt(kwargs.get("messages"))
        route = self._route(prompt)
        if route == "cger":
            self.cger_calls += 1
            return create_completion_response(self._cger_verdict(prompt))
        if route == "cgrr":
            self.cgrr_calls += 1
            return create_completion_response(self._cgrr_verdict(prompt))
        self.fallback_calls += 1
        return self._fallback.completion(**kwargs)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_ground_truth(path: Path, required_keys: list[str]) -> dict[str, Any]:
    if not path.exists():
        print(f"[ERROR] Ground truth file missing: {path}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    for k in required_keys:
        data.setdefault(k, [] if k != "cardinality_overrides" else {})
    return data


def _parse_dt(raw: Any, *, default: datetime) -> datetime:
    """Parse a temporal field, returning ``default`` for missing/invalid input.

    Use ``default=MINUS_INFINITY`` for ``t_valid_start`` (left-open interval
    sentinel: "unknown start") and ``default=INFINITY`` for ``t_valid_end``
    (right-open interval sentinel: "unknown end / still true"). Keeping the
    two endpoints asymmetric preserves the invariant ``t_v^s <= t_v^e`` and
    matches the bitemporal contract used by ETCDR's apply_evolution table.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return default
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str):
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return default
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return default


def _build_temporal_relationship(row: dict[str, Any]) -> TemporalRelationship:
    """Materialise a TemporalRelationship from one extracted JSON row.

    Uses the canonical source/target/relation_type strings already in
    place (the caller is responsible for applying the merge maps).
    Missing temporal endpoints are coerced to the bitemporal sentinels
    (MINUS_INFINITY for unknown start, INFINITY for unknown end) so the
    candidate enters ETCDR with the same semantics the rest of the
    pipeline assumes.
    """
    quad = TemporalStateQuad(
        t_valid_start=_parse_dt(row.get("t_valid_start"), default=MINUS_INFINITY),
        t_valid_end=_parse_dt(row.get("t_valid_end"), default=INFINITY),
        t_tx_start=utcnow(),
        t_tx_end=INFINITY,
    )
    provenance = [
        ProvenanceRecord(
            source_document_id=str(row.get("document") or "unknown"),
            text_unit_id=str(row.get("source_id") or ""),
            source_url=None,
            trust_score=1.0,
        )
    ]
    raw_emb = row.get("description_embedding")
    description_embedding: list[float] | None
    if raw_emb is None:
        description_embedding = None
    else:
        # extract.py persists the embedding as a list / numpy array depending
        # on the pandas roundtrip; coerce to plain list[float] so it serialises
        # cleanly through Neo4j vector indexes.
        description_embedding = [float(v) for v in raw_emb]
    return TemporalRelationship(
        id=str(uuid4()),
        source=str(row.get("source", "")),
        target=str(row.get("target", "")),
        relation_type=str(row.get("relation_type", "")),
        description=row.get("description") or "",
        weight=float(row.get("weight") or 1.0),
        confidence=1.0,
        description_embedding=description_embedding,
        temporal_quad=quad,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Canonicalisation step (CGER + CGRR with oracle)
# ---------------------------------------------------------------------------

async def _canonicalise(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    config: BTGraphRAGConfig,
    oracle: GroundTruthOracleCompletion,
    driver: Any,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Run CGER then CGRR using the ground-truth oracle.

    Returns the canonicalised relationships DataFrame plus the two maps
    so they can be reported and serialised.
    """
    print("\n" + "=" * 78)
    print("  CANONICALISATION — CGER + CGRR with ground-truth oracle")
    print("=" * 78)

    # CGER — empty existing graph, intra-batch Phase B only
    _, cger_merge_map, _ = await resolve_entities(
        new_entities=entities_df,
        existing_entities=pd.DataFrame(),
        config=config,
        model=oracle,
        driver=driver,
        phase_b_top_k=CANON_PHASE_B_TOP_K,
    )
    print(f"\n[CGER] merge_map: {len(cger_merge_map)} alias(es) -> canonical")

    # Apply CGER's merge map to source/target before CGRR runs, so the
    # candidate edges fed to CGRR already reference canonical entities.
    relationships_df = apply_merge_map_to_relationships(
        relationships_df, cger_merge_map,
    )

    async with driver.session(database=TEST_DB) as session:
        print(f"\n[CGRR] Wiping {TEST_DB} so Phase A is a no-op…")
        await session.run("MATCH (n) DETACH DELETE n")
        relationships_df, cgrr_normalize_map, _ = await resolve_relationships(
            relationships_df=relationships_df,
            config=config,
            session=session,
            model=oracle,
            driver=driver,
            phase_b_top_k=CANON_PHASE_B_TOP_K,
        )
    print(f"[CGRR] normalize_map: {len(cgrr_normalize_map)} alias(es) -> canonical")

    print("\n[ORACLE] tally:")
    print(f"  CGER verifications: {oracle.cger_calls}  "
          f"(SAME={oracle.cger_same}, DIFFERENT_ENTITY={oracle.cger_diff})")
    print(f"  CGRR verifications: {oracle.cgrr_calls}  "
          f"(SAME={oracle.cgrr_same}, DIFFERENT={oracle.cgrr_diff})")
    print(f"  Fallback (real-LLM) calls so far: {oracle.fallback_calls}")

    return relationships_df, cger_merge_map, cgrr_normalize_map


# ---------------------------------------------------------------------------
# Cardinality classification
# ---------------------------------------------------------------------------

async def _classify_cardinalities(
    relation_types: list[str],
    sample_rels: pd.DataFrame,
    config: BTGraphRAGConfig,
    model: Any,
) -> None:
    """Populate config.relation_cardinality_overrides for any unknown types.

    Mirrors ``classify_relation_cardinalities`` from
    ``packages/.../evaluation/etcdr/test_etcdr.py`` but reads the
    examples directly out of the canonicalised DataFrame.
    """
    from graphrag_llm.utils import CompletionMessagesBuilder

    prompt_template = config.resolved_cardinality_prompt()

    unknown = [
        rt for rt in relation_types
        if rt.upper().replace(" ", "_") not in config.default_cardinality_map
        and rt.upper().replace(" ", "_") not in config.relation_cardinality_overrides
    ]
    if not unknown:
        print("[CARD] All relation types already classified via overrides.")
        return

    print(f"\n[CARD] Classifying {len(unknown)} relation type(s) via LLM:")
    for rt in unknown:
        rows = sample_rels[sample_rels["relation_type"] == rt].head(3)
        examples = "\n".join(
            f"  ({r['source']}) -[{rt}]-> ({r['target']}): "
            f"{(str(r.get('description', '')) or '')[:80]}"
            for _, r in rows.iterrows()
        ) or "No examples available"

        prompt = prompt_template.format(
            relation_type=rt,
            context_examples=examples,
        )
        messages = CompletionMessagesBuilder().add_user_message(prompt).build()
        response = await model.completion_async(messages=messages)
        raw = response.content.strip()
        answer = raw.upper()

        classification = "NON_EXCLUSIVE"
        for v in ("BOTH_EXCLUSIVE", "SUBJECT_EXCLUSIVE", "OBJECT_EXCLUSIVE", "NON_EXCLUSIVE"):
            if v in answer:
                classification = v
                break
        key = rt.upper().replace(" ", "_")
        config.relation_cardinality_overrides[key] = classification
        print(f"  - {rt:<28s} -> {classification}")


# ---------------------------------------------------------------------------
# ETCDR loop
# ---------------------------------------------------------------------------

async def _run_etcdr(
    relationships_df: pd.DataFrame,
    config: BTGraphRAGConfig,
    model: Any,
    driver: Any,
) -> list[dict[str, Any]]:
    """Feed every canonicalised relationship through detect_and_resolve.

    The main ``etcdreval`` DB is held empty for the whole pass; all
    conflicts surface as intra-batch conflicts against the running batch.
    Intra-batch retrieval and mutations flow through the scratch
    ``etcdrbatch`` DB (an ``ETCDRBatchDB`` context) so Phase B uses the
    same Neo4j-backed top-K path as Phase A.

    Returns a list of {triple, strategy, confidence} dicts in the same
    order as the input DataFrame, ready for scoring.
    """
    print("\n" + "=" * 78)
    print(f"  ETCDR — processing {len(relationships_df)} canonicalised candidates")
    print("=" * 78)

    accepted: list[TemporalRelationship] = []
    outcomes: list[dict[str, Any]] = []

    # detect_and_resolve runs the cardinality-aware Cypher against the main
    # DB even when it's empty (Phase A returns zero hits but the queries
    # still execute). The scratch DB receives every accepted batch edge so
    # Phase B can do its own indexed lookup + top-K cosine retrieval.
    async with driver.session(database=TEST_DB) as session:
        await session.run("MATCH (n) DETACH DELETE n")
        async with ETCDRBatchDB(
            driver=driver,
            db_name=config.etcdr_phase_b_temp_db,
            vector_dimensions=config.neo4j_vector_dimensions,
        ) as batch_db:
            for i, row in enumerate(relationships_df.to_dict("records")):
                candidate = _build_temporal_relationship(row)
                result = await detect_and_resolve(
                    candidate=candidate,
                    session=session,
                    config=config,
                    model=model,
                    edge_index=i,
                    accepted_batch=accepted,
                    batch_db=batch_db,
                )
                strategy = (
                    result.strategy.value
                    if result.strategy is not None else "UNKNOWN"
                )
                outcomes.append({
                    "index": i,
                    "source": candidate.source,
                    "relation_type": candidate.relation_type,
                    "target": candidate.target,
                    "strategy": strategy,
                    "confidence": round(result.confidence, 4),
                    "has_neo4j_conflicts": result.has_neo4j_conflicts,
                    "has_intra_batch_conflicts": result.has_intra_batch_conflicts,
                })
                # Only accept non-retracted candidates into the rolling
                # batch so future conflict queries reflect the
                # post-resolution state. Add to BOTH the in-memory list
                # (for downstream reporting) and the scratch DB (so Phase
                # B's Cypher actually sees them on the next iteration).
                if candidate.status != "retracted":
                    accepted.append(candidate)
                    await batch_db.add_relationship(candidate)

    return outcomes


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _match_expected(
    outcomes: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    merge_map: dict[str, str] | None = None,
    normalize_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Pair each expected entry with the n-th matching outcome triple.

    Triple = (source, relation_type, target). The k-th expected entry
    with a given triple is paired with the k-th outcome row carrying
    that same triple. Expected entries with no matching outcome are
    reported as 'unmatched'.

    ``merge_map`` / ``normalize_map`` are CGER/CGRR's alias→canonical
    maps. Outcomes are emitted post-canonicalisation, so the GT triples
    (authored against extractor-original strings) must be translated
    through the same maps before the lookup, or any cluster whose CGRR
    canonical differs from the GT canonical (e.g. EMPLOYED_AT → WORKS_FOR)
    silently goes unmatched.
    """
    merge_map = merge_map or {}
    normalize_map = normalize_map or {}

    def _canon(source: str, rel: str, target: str) -> tuple[str, str, str]:
        return (
            merge_map.get(source, source),
            normalize_map.get(rel, rel),
            merge_map.get(target, target),
        )

    triple_to_outcome_indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, o in enumerate(outcomes):
        key = (o["source"], o["relation_type"], o["target"])
        triple_to_outcome_indices[key].append(idx)

    cursors: dict[tuple[str, str, str], int] = defaultdict(int)
    pairs: list[dict[str, Any]] = []
    for exp in expected:
        raw_key = (
            str(exp.get("source", "")),
            str(exp.get("relation_type", "")),
            str(exp.get("target", "")),
        )
        key = _canon(*raw_key)
        candidates = triple_to_outcome_indices.get(key, [])
        cursor = cursors[key]
        if cursor < len(candidates):
            outcome = outcomes[candidates[cursor]]
            cursors[key] = cursor + 1
            pairs.append({
                "expected": exp,
                "expected_canonical": {
                    "source": key[0],
                    "relation_type": key[1],
                    "target": key[2],
                },
                "outcome": outcome,
                "matched": True,
            })
        else:
            pairs.append({
                "expected": exp,
                "expected_canonical": {
                    "source": key[0],
                    "relation_type": key[1],
                    "target": key[2],
                },
                "outcome": None,
                "matched": False,
            })
    return pairs


def _score(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    matched_pairs = [p for p in pairs if p["matched"]]
    correct = sum(
        1 for p in matched_pairs
        if p["outcome"]["strategy"] == p["expected"]["expected_strategy"]
    )

    confusion: dict[str, Counter] = defaultdict(Counter)
    for p in matched_pairs:
        exp_strat = p["expected"]["expected_strategy"]
        act_strat = p["outcome"]["strategy"]
        confusion[exp_strat][act_strat] += 1

    return {
        "total_expected": len(pairs),
        "matched": len(matched_pairs),
        "unmatched": len(pairs) - len(matched_pairs),
        "correct": correct,
        "accuracy": (correct / len(matched_pairs)) if matched_pairs else 0.0,
        "confusion_matrix": {
            exp: dict(actual_counts) for exp, actual_counts in confusion.items()
        },
    }


def _print_report(
    pairs: list[dict[str, Any]],
    scores: dict[str, Any],
    outcomes: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 78)
    print("  ETCDR EVALUATION REPORT")
    print("=" * 78)
    print(f"  Expected resolutions: {scores['total_expected']}")
    print(f"  Matched (triple found in batch): {scores['matched']}")
    print(f"  Unmatched (no candidate produced for this triple): {scores['unmatched']}")
    print(f"  Correct strategy: {scores['correct']}/{scores['matched']}  "
          f"(accuracy={scores['accuracy']:.3f})")

    if scores["confusion_matrix"]:
        print(f"\n  Confusion matrix (expected -> actual):")
        for exp_strat, actuals in sorted(scores["confusion_matrix"].items()):
            actuals_str = ", ".join(
                f"{a}={n}" for a, n in sorted(actuals.items())
            )
            print(f"    {exp_strat:<14s} -> {actuals_str}")

    # Show all mismatches and unmatched entries (these are what the user
    # actually needs to inspect).
    mismatches = [
        p for p in pairs
        if p["matched"]
        and p["outcome"]["strategy"] != p["expected"]["expected_strategy"]
    ]
    if mismatches:
        print(f"\n  Strategy mismatches ({len(mismatches)}):")
        for p in mismatches:
            exp = p["expected"]
            out = p["outcome"]
            note = exp.get("notes") or ""
            print(f"    - ({exp['source']}) -[{exp['relation_type']}]-> "
                  f"({exp['target']})")
            print(f"      expected={exp['expected_strategy']}  "
                  f"actual={out['strategy']}  "
                  f"confidence={out['confidence']}")
            if note:
                print(f"      note: {note}")

    unmatched = [p for p in pairs if not p["matched"]]
    if unmatched:
        print(f"\n  Unmatched expected entries ({len(unmatched)}):")
        for p in unmatched:
            exp = p["expected"]
            print(f"    - ({exp['source']}) -[{exp['relation_type']}]-> "
                  f"({exp['target']})  expected={exp['expected_strategy']}")

    print(f"\n  Strategy distribution across all {len(outcomes)} candidates:")
    dist = Counter(o["strategy"] for o in outcomes)
    for strat, n in dist.most_common():
        print(f"    {strat:<14s} : {n}")


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

    entity_gt = _load_ground_truth(
        GROUND_TRUTH_DIR / "entity_resolution.json",
        required_keys=["clusters"],
    )
    relation_gt = _load_ground_truth(
        GROUND_TRUTH_DIR / "relationship_resolution.json",
        required_keys=["clusters"],
    )
    conflict_gt = _load_ground_truth(
        GROUND_TRUTH_DIR / "conflict_resolution.json",
        required_keys=["cardinality_overrides", "expected"],
    )
    print(f"[LOAD] ground truth: "
          f"{len(entity_gt['clusters'])} entity cluster(s), "
          f"{len(relation_gt['clusters'])} relation-type cluster(s), "
          f"{len(conflict_gt['expected'])} expected conflict resolution(s)")

    config = BTGraphRAGConfig(
        enabled=True,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        neo4j_database=TEST_DB,
        # CGER (oracle-driven canonicalisation)
        cger_enabled=True,
        cger_cosine_threshold=CANON_COSINE_THRESHOLD,
        cger_candidate_top_k=CANON_TOP_K,
        cger_phase_b_temp_db=CGER_TEMP_DB,
        # CGRR (oracle-driven canonicalisation)
        cgrr_enabled=True,
        cgrr_cosine_threshold=CANON_COSINE_THRESHOLD,
        cgrr_candidate_top_k=CANON_TOP_K,
        cgrr_phase_b_temp_db=CGRR_TEMP_DB,
        # ETCDR
        etcdr_enabled=True,
        etcdr_confidence_threshold=ETCDR_CONFIDENCE_THRESHOLD,
        etcdr_topk=ETCDR_TOPK,
        etcdr_cosine_threshold=ETCDR_COSINE_THRESHOLD,
        etcdr_phase_b_temp_db=ETCDR_TEMP_DB,
        relation_cardinality_overrides={
            k.upper().replace(" ", "_"): v
            for k, v in (conflict_gt.get("cardinality_overrides") or {}).items()
        },
        neo4j_vector_dimensions=NEO4J_VECTOR_DIMENSIONS,
    )

    real_model = _real_completion()
    oracle = GroundTruthOracleCompletion(
        entity_clusters=entity_gt["clusters"],
        relation_clusters=relation_gt["clusters"],
        fallback_model=real_model,
    )

    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        await init_schema(driver, database=TEST_DB,
                          vector_dimensions=NEO4J_VECTOR_DIMENSIONS)

        # 1) Canonicalise entities + relationships using the oracle.
        canon_rels_df, merge_map, normalize_map = await _canonicalise(
            entities_df, rels_df, config, oracle, driver,
        )

        # 2) Classify cardinalities for every relation type in the batch.
        relation_types = sorted(canon_rels_df["relation_type"].astype(str).unique())
        await _classify_cardinalities(
            relation_types, canon_rels_df, config, real_model,
        )

        # 3) Run ETCDR over the canonicalised batch.
        outcomes = await _run_etcdr(canon_rels_df, config, real_model, driver)

    finally:
        await driver.close()

    flushed = flush_resolution_log(RESOLUTION_LOG_DIR)
    if flushed:
        print(f"\n[LOG] ETCDR resolution log -> {flushed}")

    # 4) Score against ground truth. The GT is authored against the
    #    extractor-original triples; the outcomes are emitted post-CGER/CGRR,
    #    so we canonicalise the GT side through the same maps before matching.
    pairs = _match_expected(
        outcomes,
        conflict_gt["expected"],
        merge_map=merge_map,
        normalize_map=normalize_map,
    )
    scores = _score(pairs)
    _print_report(pairs, scores, outcomes)

    # 5) Persist
    results = {
        "config": {
            "canon_cosine_threshold": CANON_COSINE_THRESHOLD,
            "canon_top_k": CANON_TOP_K,
            "canon_phase_b_top_k": CANON_PHASE_B_TOP_K,
            "neo4j_vector_dimensions": NEO4J_VECTOR_DIMENSIONS,
            "etcdr_confidence_threshold": ETCDR_CONFIDENCE_THRESHOLD,
            "completion_model": COMPLETION_MODEL,
        },
        "test_data": {
            "entities": int(len(entities_df)),
            "relationships": int(len(rels_df)),
            "entity_truth_clusters": len(entity_gt["clusters"]),
            "relation_truth_clusters": len(relation_gt["clusters"]),
            "conflict_truth_entries": len(conflict_gt["expected"]),
        },
        "canonicalisation": {
            "entity_merge_map": merge_map,
            "relation_normalize_map": normalize_map,
            "oracle": {
                "cger_calls": oracle.cger_calls,
                "cger_same": oracle.cger_same,
                "cger_diff": oracle.cger_diff,
                "cgrr_calls": oracle.cgrr_calls,
                "cgrr_same": oracle.cgrr_same,
                "cgrr_diff": oracle.cgrr_diff,
                "fallback_calls": oracle.fallback_calls,
            },
        },
        "cardinality_overrides_used": dict(config.relation_cardinality_overrides),
        "outcomes": outcomes,
        "scoring": {
            **scores,
            "pairs": [
                {
                    "expected": p["expected"],
                    "expected_canonical": p.get("expected_canonical"),
                    "outcome": p["outcome"],
                    "matched": p["matched"],
                }
                for p in pairs
            ],
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[SAVE] results -> {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
