# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Cross-Graph Entity Resolution (CGER).

Resolves newly extracted entities against the existing graph in two
steps:

1. **Cosine pre-filter.** For each new entity, the top-K candidates by
   description-embedding cosine similarity are retrieved (Neo4j vector
   index or in-memory fallback). The best-scoring candidate of the same
   type whose cosine similarity exceeds ``cger_cosine_threshold``
   triggers the next step.
2. **LLM verification with temporal context.** The LLM receives both
   entities' names, types, descriptions and *active periods*, and must
   answer one of:

   * ``SAME`` — merge the two entities into one.
   * ``DIFFERENT_ENTITY`` — they refer to different real-world things;
     keep them apart.
   * ``DIFFERENT_TEMPORAL`` — they describe the same referent, but the
     temporal gap between them is large enough that fusing the two
     periods would destroy meaningful state (e.g. "Apple 1985" vs
     "Apple 2024"); keep them apart.

Only ``SAME`` produces a merge. The temporal split is not folded into
the score itself; the LLM is the single component responsible for
weighing the temporal evidence.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import pandas as pd

from graphrag.bt_graphrag.entity_resolution.scorers import (
    EntityScorer,
    cosine_similarity,
    description_cosine_entity_scorer,
)
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


def _normalize_type(t: Any) -> str:
    """Normalize an entity-type string for case-insensitive comparison."""
    if t is None:
        return ""
    return str(t).strip().lower()


def _types_match(a: Any, b: Any) -> bool:
    """Whether two entities share the same entity type.

    Empty/unknown type on either side is treated as a non-match: CGER never
    merges across types, so an entity with no declared type cannot be safely
    paired with anything.
    """
    na = _normalize_type(a)
    nb = _normalize_type(b)
    if not na or not nb:
        return False
    return na == nb


# ---------------------------------------------------------------------------
# LLM verification with temporal context
# ---------------------------------------------------------------------------

CGER_VERIFICATION_PROMPT = """You are a strict entity-resolution expert. Two candidate entities are presented below, together with their **active periods**. Decide whether they should be merged.

Your default answer is DIFFERENT_ENTITY. Only answer SAME when the evidence is unambiguous (abbreviation/alias of the same thing, alternate spelling, transliteration, punctuation/casing variant, or the same proper noun across languages — e.g. CATALUNYA = CATALONIA, BYU = BRIGHAM YOUNG UNIVERSITY, SOEHARTO = SUHARTO).

There are three possible verdicts:

1. **SAME** — they refer to the same real-world entity AND their active periods are close enough that fusing them does not erase meaningful state. Merge them.
2. **DIFFERENT_ENTITY** — they refer to **different real-world things**. Keep them separate. Examples:
   - Generic vs specific ("HIGH SCHOOL" vs "SOUTHWEST HIGH SCHOOL").
   - Events/seasons/editions naming distinct years ("1996 NFL SEASON" vs "1997 NFL SEASON").
   - Shared tokens, different organizations ("KRUNG THAI BANK F.C." vs "KRUNG THAI BANK").
   - Country vs nationality/language ("ENGLAND" vs "ENGLISH").
   - Different scope/qualifier that changes the referent ("MINISTER OF CULTURE" vs "MINISTER OF CULTURE, SPORTS AND TOURISM").
   - Disagreeing entity types.
3. **DIFFERENT_TEMPORAL** — they describe the **same referent** but their active periods are separated by a temporally significant gap, such that merging them would destroy the distinction between two states of that referent. Use this verdict when:
   - The two periods do not overlap and the gap between them is on the order of years or longer for slow-changing referents (e.g. organisations, countries, roles), or on the order of months for fast-changing ones (e.g. sports squads, governments).
   - The descriptions describe states that are clearly inconsistent with being a single snapshot (e.g. "APPLE 1985: home-computer company led by Steve Jobs" vs "APPLE 2024: trillion-dollar consumer-electronics multinational led by Tim Cook").
   - In doubt about whether the gap is significant, prefer DIFFERENT_TEMPORAL over SAME — keeping the two states separate is recoverable; merging them is not.

It IS safe to answer SAME when:
- One name is an abbreviation/acronym of the other and active periods overlap or are adjacent.
- They differ only in punctuation, casing, accents, apostrophe style, or whitespace.
- They are the same proper noun in different languages/transliterations and descriptions agree.
- The active periods overlap and descriptions corroborate the same referent.

Entity A:
- Name: {name_a}
- Type: {type_a}
- Description: {desc_a}
- Active Period: {period_a}

Entity B:
- Name: {name_b}
- Type: {type_b}
- Description: {desc_b}
- Active Period: {period_b}

Answer with EXACTLY ONE of the following tokens and nothing else:
- SAME
- DIFFERENT_ENTITY
- DIFFERENT_TEMPORAL

Answer:"""


async def llm_verify_entity_match(
    entity_a: dict[str, Any],
    entity_b: dict[str, Any],
    model: "LLMCompletion",
) -> str:
    """Use the LLM to decide whether two entities should be merged.

    Returns one of ``"SAME"``, ``"DIFFERENT_ENTITY"`` or
    ``"DIFFERENT_TEMPORAL"``. Any unrecognised response is treated as
    ``"DIFFERENT_ENTITY"`` (conservative default — never merge on
    ambiguous evidence).
    """
    from graphrag_llm.utils import CompletionMessagesBuilder

    def _period(e: dict) -> str:
        start = e.get("active_start", "unknown")
        end = e.get("active_end", "present")
        if end == math.inf:
            end = "present"
        return f"{start} to {end}"

    prompt = CGER_VERIFICATION_PROMPT.format(
        name_a=entity_a.get("title", ""),
        type_a=entity_a.get("type", ""),
        desc_a=entity_a.get("description", "")[:500],
        period_a=_period(entity_a),
        name_b=entity_b.get("title", ""),
        type_b=entity_b.get("type", ""),
        desc_b=entity_b.get("description", "")[:500],
        period_b=_period(entity_b),
    )

    messages = CompletionMessagesBuilder().add_user_message(prompt).build()
    response = await model.completion_async(messages=messages)
    answer = response.content.strip().upper()

    valid = {"SAME", "DIFFERENT_ENTITY", "DIFFERENT_TEMPORAL"}
    first_verdict = next(
        (tok for tok in answer.replace(":", " ").replace(",", " ").split()
         if tok in valid),
        "",
    )
    return first_verdict if first_verdict in valid else "DIFFERENT_ENTITY"


# ---------------------------------------------------------------------------
# Main CGER Pipeline
# ---------------------------------------------------------------------------


async def resolve_entities(
    new_entities: pd.DataFrame,
    existing_entities: pd.DataFrame,
    config: BTGraphRAGConfig,
    model: "LLMCompletion | None" = None,
    entity_scorer: EntityScorer = description_cosine_entity_scorer,
    candidate_map: dict[str, list[dict[str, Any]]] | None = None,
    driver: "AsyncDriver | None" = None,
    phase_b_top_k: int = 10,
) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    """Resolve new entities against the existing graph.

    For each new entity, retrieves the top-K most similar existing
    entities, picks the best same-type candidate, and — if its cosine
    similarity exceeds ``config.cger_cosine_threshold`` — defers the
    merge decision to the LLM, which receives both entities' active
    periods as context.

    Returns:
        resolved_entities: The new entities DataFrame with merged IDs
        merge_map: Dict mapping merged entity titles to their canonical titles
        phase_b_log: Debug log entries from Phase B (intra-batch) resolution
    """
    merge_map: dict[str, str] = {}
    phase_b_log: list[dict[str, Any]] = []

    if new_entities.empty:
        print("    [CGER] Skipping — no new entities to process")
        return new_entities, merge_map, phase_b_log

    llm_merges = 0
    llm_diff_entity = 0
    llm_diff_temporal = 0
    no_match_count = 0
    below_threshold_count = 0

    # --- Phase A: Resolve new entities against existing graph ---
    existing_records: list[dict[str, Any]] = (
        existing_entities.to_dict("records") if not existing_entities.empty else []
    )

    if not existing_records:
        print(f"    [CGER] Phase A: Skipping — graph is empty (first run); "
              f"proceeding to intra-batch Phase B for {len(new_entities)} entities")
    else:
        top_k = config.cger_candidate_top_k
        source = "Neo4j vector index" if candidate_map else "in-memory cosine"
        print(f"\n    [CGER] Phase A: Resolving {len(new_entities)} new entities "
              f"against {len(existing_records)} existing (top-K={top_k}, source={source})")
        print(f"    [CGER] LLM trigger: cosine >= {config.cger_cosine_threshold}")
        print()

        for idx, new_row in new_entities.iterrows():
            new_entity = dict(new_row)
            best_score = 0.0
            best_match: dict[str, Any] | None = None

            entity_title = str(new_entity.get("title", "?"))
            new_type = new_entity.get("type")

            # --- Select candidates for this entity ---
            # Hard type-gate: CGER only compares entities of the same type.
            if candidate_map and entity_title in candidate_map:
                candidates = [
                    c for c in candidate_map[entity_title]
                    if _types_match(new_type, c.get("type"))
                ]
            else:
                same_type_existing = [
                    e for e in existing_records
                    if _types_match(new_type, e.get("type"))
                ]
                new_emb = new_entity.get("description_embedding") or []
                if new_emb:
                    scored_candidates: list[tuple[float, dict[str, Any]]] = []
                    for existing in same_type_existing:
                        ex_emb = existing.get("description_embedding") or []
                        cos = cosine_similarity(new_emb, ex_emb) if ex_emb else 0.0
                        scored_candidates.append((cos, existing))
                    scored_candidates.sort(key=lambda x: x[0], reverse=True)
                    candidates = [rec for _, rec in scored_candidates[:top_k]]
                else:
                    candidates = same_type_existing

            # --- Cosine scoring on the narrowed candidate set ---
            for existing in candidates:
                score, _ = entity_scorer(new_entity, existing, config)
                if score > best_score:
                    best_score = score
                    best_match = existing

            if best_match is None:
                no_match_count += 1
                print(f"    [{idx}] '{entity_title[:35]}' — no candidates found")
                continue

            match_title = str(best_match.get("title", "?"))

            if best_score >= config.cger_cosine_threshold and model is not None:
                print(f"    [{idx}] LLM: '{entity_title[:30]}' vs '{match_title[:30]}'  cosine={best_score:.4f}")
                verdict = await llm_verify_entity_match(new_entity, best_match, model)
                if verdict == "SAME":
                    merge_map[new_entity["title"]] = best_match["title"]
                    llm_merges += 1
                    print(f"         LLM verdict: SAME -> MERGED")
                    logger.info(
                        "CGER: LLM-confirmed merge '%s' -> '%s' (cosine=%.3f)",
                        new_entity["title"], best_match["title"], best_score,
                    )
                elif verdict == "DIFFERENT_TEMPORAL":
                    llm_diff_temporal += 1
                    print(f"         LLM verdict: DIFFERENT_TEMPORAL -> KEPT SEPARATE (same referent, temporal gap)")
                    logger.info(
                        "CGER: LLM kept temporally separate '%s' vs '%s' (cosine=%.3f)",
                        new_entity["title"], best_match["title"], best_score,
                    )
                else:
                    llm_diff_entity += 1
                    print(f"         LLM verdict: DIFFERENT_ENTITY -> KEPT SEPARATE")
                    logger.info(
                        "CGER: LLM rejected merge '%s' vs '%s' (cosine=%.3f, verdict=%s)",
                        new_entity["title"], best_match["title"], best_score, verdict,
                    )
            else:
                below_threshold_count += 1
                if best_score > 0.3:
                    print(f"    [{idx}] BELOW THRESHOLD: '{entity_title[:30]}' best='{match_title[:30]}'  cosine={best_score:.4f}")

    # Apply Phase A merge map to new entities
    if merge_map:
        new_entities = new_entities.copy()
        new_entities["title"] = new_entities["title"].map(
            lambda t: merge_map.get(t, t)
        )

    # --- Phase B: Intra-batch resolution (new vs new) ---
    unmerged = new_entities[~new_entities["title"].isin(merge_map.values())]

    intra_llm_merges = 0
    intra_diff_entity = 0
    intra_diff_temporal = 0

    if len(unmerged) > 1:
        unmerged_records = [dict(r) for _, r in unmerged.iterrows()]
        use_temp_db = driver is not None and bool(config.cger_phase_b_temp_db)
        batch_db: Any = None
        if use_temp_db:
            from graphrag.bt_graphrag.neo4j_store import CGERBatchDB
            batch_db = CGERBatchDB(
                driver=driver,  # type: ignore[arg-type]
                db_name=config.cger_phase_b_temp_db,
                vector_dimensions=config.neo4j_vector_dimensions,
            )

        print(f"\n    [CGER] Phase B: Intra-batch resolution "
              f"({len(unmerged)} unmerged entities, top_k={phase_b_top_k}, "
              f"backend={'neo4j_temp_db' if use_temp_db else 'in_memory'})")

        intra_merge_map: dict[str, str] = {}
        seen_entities: list[dict[str, Any]] = []
        seen_by_title: dict[str, dict[str, Any]] = {}
        kept_titles: set[str] = set()

        async def _pick_candidates(
            entity: dict[str, Any], entity_title: str,
        ) -> list[dict[str, Any]]:
            """Return same-name + top-K-embedding candidates for one entity."""
            entity_type = entity.get("type")
            name_lower = entity_title.strip().lower()
            emb = entity.get("description_embedding") or []

            if batch_db is not None and emb:
                cand_refs = await batch_db.find_candidates(
                    embedding=list(emb),
                    name_lower=name_lower,
                    entity_type=entity_type,
                    top_k=phase_b_top_k,
                    exclude_title=entity_title,
                    allowed_titles=kept_titles,
                )
                return [
                    seen_by_title[c["title"]]
                    for c in cand_refs
                    if c["title"] in seen_by_title
                ]

            # In-memory fallback: same-name (case-insensitive) ∪ top-K cosine
            same_type_seen = [
                s for s in seen_entities
                if _types_match(entity_type, s.get("type"))
            ]
            same_name = [
                s for s in same_type_seen
                if str(s.get("title", "")).strip().lower() == name_lower
            ]
            if emb:
                scored: list[tuple[float, dict[str, Any]]] = []
                for s in same_type_seen:
                    s_emb = s.get("description_embedding") or []
                    if not s_emb:
                        continue
                    scored.append((cosine_similarity(list(emb), list(s_emb)), s))
                scored.sort(key=lambda x: x[0], reverse=True)
                topk = [rec for _, rec in scored[:phase_b_top_k]]
            else:
                topk = []
            pool: dict[str, dict[str, Any]] = {}
            for s in same_name + topk:
                pool[str(s.get("title", ""))] = s
            return list(pool.values())

        async def _phase_b_loop() -> None:
            nonlocal intra_llm_merges, intra_diff_entity, intra_diff_temporal
            for entity in unmerged_records:
                entity_title = str(entity.get("title", ""))

                if entity_title in merge_map:
                    continue

                if not seen_entities:
                    seen_entities.append(entity)
                    seen_by_title[entity_title] = entity
                    kept_titles.add(entity_title)
                    continue

                best_score = 0.0
                best_match: dict[str, Any] | None = None

                entity_type = entity.get("type")
                candidate_pool = await _pick_candidates(entity, entity_title)

                for seen in candidate_pool:
                    if not _types_match(entity_type, seen.get("type")):
                        continue
                    score, _ = entity_scorer(entity, seen, config)
                    if score > best_score:
                        best_score = score
                        best_match = seen

                match_title = str(best_match.get("title", "?")) if best_match else ""

                if (best_match is not None
                        and best_score >= config.cger_cosine_threshold
                        and model is not None):
                    print(f"    [INTRA] LLM: '{entity_title[:30]}' vs '{match_title[:30]}'  "
                          f"cosine={best_score:.4f}  (pool={len(candidate_pool)})")
                    verdict = await llm_verify_entity_match(entity, best_match, model)
                    if verdict == "SAME":
                        intra_merge_map[entity_title] = match_title
                        intra_llm_merges += 1
                        phase_b_log.append({
                            "phase": "B_intra_batch",
                            "entity": entity_title,
                            "type": str(entity.get("type", "?")),
                            "best_match": match_title,
                            "best_score": round(best_score, 4),
                            "decision": "LLM_MERGE",
                            "candidate_pool_size": len(candidate_pool),
                            "top_comparisons": [],
                        })
                        print(f"           LLM verdict: SAME -> MERGED")
                        logger.info(
                            "CGER: Intra-batch LLM merge '%s' -> '%s' (cosine=%.3f)",
                            entity_title, match_title, best_score,
                        )
                    elif verdict == "DIFFERENT_TEMPORAL":
                        intra_diff_temporal += 1
                        phase_b_log.append({
                            "phase": "B_intra_batch",
                            "entity": entity_title,
                            "type": str(entity.get("type", "?")),
                            "best_match": match_title,
                            "best_score": round(best_score, 4),
                            "decision": "LLM_DIFFERENT_TEMPORAL",
                            "candidate_pool_size": len(candidate_pool),
                            "top_comparisons": [],
                        })
                        print(f"           LLM verdict: DIFFERENT_TEMPORAL -> KEPT SEPARATE")
                        seen_entities.append(entity)
                        seen_by_title[entity_title] = entity
                        kept_titles.add(entity_title)
                    else:
                        intra_diff_entity += 1
                        phase_b_log.append({
                            "phase": "B_intra_batch",
                            "entity": entity_title,
                            "type": str(entity.get("type", "?")),
                            "best_match": match_title,
                            "best_score": round(best_score, 4),
                            "decision": "LLM_DIFFERENT_ENTITY",
                            "candidate_pool_size": len(candidate_pool),
                            "top_comparisons": [],
                        })
                        print(f"           LLM verdict: DIFFERENT_ENTITY -> KEPT SEPARATE")
                        seen_entities.append(entity)
                        seen_by_title[entity_title] = entity
                        kept_titles.add(entity_title)
                else:
                    if best_score > 0.3:
                        phase_b_log.append({
                            "phase": "B_intra_batch",
                            "entity": entity_title,
                            "type": str(entity.get("type", "?")),
                            "best_match": match_title,
                            "best_score": round(best_score, 4),
                            "decision": "BELOW_THRESHOLD",
                            "candidate_pool_size": len(candidate_pool),
                            "top_comparisons": [],
                        })
                        print(f"    [INTRA] BELOW: '{entity_title[:30]}' best='{match_title[:30]}'  "
                              f"cosine={best_score:.4f}  (pool={len(candidate_pool)})")
                    seen_entities.append(entity)
                    seen_by_title[entity_title] = entity
                    kept_titles.add(entity_title)

        if batch_db is not None:
            try:
                async with batch_db:
                    await batch_db.bulk_load(unmerged_records)
                    print(f"    [CGER] Phase B: temp Neo4j DB '{batch_db.db_name}' "
                          f"loaded with {len(unmerged_records)} entities")
                    await _phase_b_loop()
            except Exception as temp_err:
                logger.warning(
                    "CGER Phase B: temp Neo4j DB failed (%s); "
                    "falling back to in-memory candidate selection",
                    temp_err,
                )
                print(f"    [CGER] Phase B: temp DB unavailable ({temp_err}); "
                      f"falling back to in-memory")
                batch_db = None
                intra_merge_map.clear()
                seen_entities.clear()
                seen_by_title.clear()
                kept_titles.clear()
                intra_llm_merges = intra_diff_entity = intra_diff_temporal = 0
                await _phase_b_loop()
        else:
            await _phase_b_loop()

        if intra_merge_map:
            new_entities = new_entities.copy()
            new_entities["title"] = new_entities["title"].map(
                lambda t: intra_merge_map.get(t, t)
            )
            merge_map.update(intra_merge_map)
        else:
            print(f"    [CGER] Intra-batch: no additional merges found")
    else:
        print(f"\n    [CGER] Phase B: Intra-batch skipped (0-1 unmerged entities)")

    # Deduplicate entity rows created by Phase A/B merging
    _before_dedup = len(new_entities)
    new_entities = new_entities.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    _dedup_dropped = _before_dedup - len(new_entities)
    if _dedup_dropped:
        print(f"\n    [CGER] Deduplicated {_dedup_dropped} entity row(s) after merging")

    # --- Verification Summary ---
    print(f"\n    {'=' * 55}")
    print(f"    CGER VERIFICATION SUMMARY")
    print(f"    {'=' * 55}")
    print(f"    Total new entities processed:   {len(new_entities)}")
    print(f"    Existing entities compared:     {len(existing_records)}")
    print(f"    Top-K candidates per entity:    {config.cger_candidate_top_k}")
    print(f"    Cosine LLM trigger:             {config.cger_cosine_threshold}")
    print(f"    Phase A LLM merges (SAME):                {llm_merges}")
    print(f"    Phase A LLM diff entity:                  {llm_diff_entity}")
    print(f"    Phase A LLM diff temporal:                {llm_diff_temporal}")
    print(f"    Phase A below threshold:                  {below_threshold_count}")
    print(f"    Phase A no candidates:                    {no_match_count}")
    print(f"    Phase B intra LLM merges (SAME):          {intra_llm_merges}")
    print(f"    Phase B intra LLM diff entity:            {intra_diff_entity}")
    print(f"    Phase B intra LLM diff temporal:          {intra_diff_temporal}")
    print(f"    Total merges (A+B):             {len(merge_map)}")

    circular = False
    for src, dst in merge_map.items():
        if dst in merge_map and merge_map[dst] != dst:
            circular = True
            print(f"    WARNING: Potential chain merge: '{src}' -> '{dst}' -> '{merge_map[dst]}'")
    if not circular:
        print(f"    Circular merge check:           PASS")

    print(f"    {'=' * 55}")

    return new_entities, merge_map, phase_b_log


def apply_merge_map_to_relationships(
    relationships: pd.DataFrame,
    merge_map: dict[str, str],
) -> pd.DataFrame:
    """Update source/target in relationships based on entity merge map."""
    if not merge_map or relationships.empty:
        return relationships

    relationships = relationships.copy()
    relationships["source"] = relationships["source"].map(
        lambda s: merge_map.get(s, s)
    )
    relationships["target"] = relationships["target"].map(
        lambda t: merge_map.get(t, t)
    )
    return relationships
