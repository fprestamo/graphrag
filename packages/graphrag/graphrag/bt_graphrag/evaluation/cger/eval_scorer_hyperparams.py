# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Hyperparameter optimisation for the three CGER entity scorers.

Builds a labelled ground-truth dataset of entity pairs (SAME / DIFFERENT),
then sweeps threshold and weight hyperparameters for each scorer to find
the configuration that:

    * maximises resolution accuracy  (correctly merging SAME, keeping DIFFERENT)
    * minimises LLM calls            (pairs in the LLM-zone cost money)

The combined objective is:

    objective = F1  -  lambda * llm_call_rate

Higher is better.  Several lambda values are swept so you see the full
Pareto frontier between "perfect accuracy, many LLM calls" and "fewer
LLM calls, some accuracy loss".

Scorers evaluated
-----------------
1. embedding_only_entity_scorer          — cosine(description_embedding)
2. citation_and_description_entity_scorer — cosine(desc_emb) + cosine(cite_emb)
3. compute_entity_composite_score        — 5-signal weighted composite

Usage:
    python packages/graphrag/graphrag/bt_graphrag/evaluation/cger/eval_scorer_hyperparams.py

Requires GRAPHRAG_API_KEY environment variable for real LLM verification
of pairs that fall in the LLM zone.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from graphrag.bt_graphrag.entity_resolution.scorers import (
    citation_and_description_entity_scorer,
    compute_entity_composite_score,
    embedding_only_entity_scorer,
)
from graphrag.bt_graphrag.models.config import BTGraphRAGConfig
from graphrag.bt_graphrag.entity_resolution.cger import llm_verify_entity_match
from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig
from graphrag_llm.config.types import LLMProviderType
from graphrag_llm.embedding import create_embedding

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

UTC = timezone.utc
EMB_DIM = 3062          # small dimension keeps the script fast
RNG = np.random.RandomState(42)

# Lambda values controlling accuracy-vs-LLM-cost trade-off
LAMBDAS = [0.0, 0.1, 0.2, 0.4, 0.6, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# Real LLM helper
# ─────────────────────────────────────────────────────────────────────────────

def _real_llm():
    """Build a real LiteLLM completion using OpenAI (gpt-4.1-mini)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Real embedding model helper
# ─────────────────────────────────────────────────────────────────────────────

def _real_embedding_model():
    """Build the embedding model from the graphrag config (text-embedding-3-large via OpenAI)."""
    api_key = os.environ.get("GRAPHRAG_API_KEY", "")
    if not api_key:
        print("[WARN] GRAPHRAG_API_KEY not set — embedding calls will fail.", file=sys.stderr)
    cfg = ModelConfig(
        type=LLMProviderType.LiteLLM,
        model_provider="openai",
        model="text-embedding-3-large",
        api_key=api_key,
    )
    return create_embedding(cfg)


async def _embed_pairs(pairs: list[LabelledPair], embedding_model) -> None:
    """Replace synthetic embeddings in *pairs* with real embeddings.

    For each entity:
    - ``description_embedding`` ← embedding of entity description text
    - ``text_unit_embedding``   ← embedding of entity title (proxy for citation context)

    All texts are embedded in a single batched API call.
    """
    entities: list[dict] = []
    for pair in pairs:
        entities.append(pair.entity_a)
        entities.append(pair.entity_b)

    desc_texts = [e["description"] for e in entities]
    title_texts = [e["title"] for e in entities]
    all_texts = desc_texts + title_texts

    print(f"  Embedding {len(entities)} entities ({len(all_texts)} texts) with real model…",
          file=sys.stderr)
    response = await embedding_model.embedding_async(input=all_texts)
    all_vecs: list[list[float]] = response.embeddings

    desc_vecs = all_vecs[: len(entities)]
    title_vecs = all_vecs[len(entities) :]

    for entity, dv, tv in zip(entities, desc_vecs, title_vecs):
        entity["description_embedding"] = dv
        entity["text_unit_embedding"] = tv


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rand_emb(seed: int | None = None) -> list[float]:
    """Unit-norm random embedding vector."""
    rng = np.random.RandomState(seed) if seed is not None else RNG
    v = rng.randn(EMB_DIM)
    return (v / np.linalg.norm(v)).tolist()


def _similar_emb(base: list[float], noise: float = 0.08) -> list[float]:
    """Embedding close to *base* (cosine ≈ 0.92–0.99 depending on noise)."""
    v = np.array(base) + noise * RNG.randn(EMB_DIM)
    return (v / np.linalg.norm(v)).tolist()


def _dt(year: int, month: int = 1, day: int = 1) -> str:
    return datetime(year, month, day, tzinfo=UTC).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth dataset
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LabelledPair:
    entity_a: dict
    entity_b: dict
    label: str            # "SAME" or "DIFFERENT"
    description: str      # human-readable explanation


def _entity(
    title: str,
    etype: str = "person",
    description: str = "",
    active_start: str = "",
    active_end: str = "",
    relation_types: list[str] | None = None,
    desc_emb: list[float] | None = None,
    cite_emb: list[float] | None = None,
) -> dict:
    return {
        "id": str(uuid4()),
        "title": title,
        "type": etype,
        "description": description or f"Entity: {title}",
        "active_start": active_start or _dt(2000),
        "active_end": active_end or _dt(9999),
        "first_seen": _dt(2000),
        "last_seen": _dt(2024),
        "description_embedding": desc_emb,
        "text_unit_embedding": cite_emb,
        "relation_types": relation_types or [],
    }


def build_ground_truth() -> list[LabelledPair]:
    """Create labelled entity pairs with synthetic embeddings.

    Pairs are organized into difficulty tiers to create realistic score overlap
    between SAME and DIFFERENT.  This ensures the LLM-zone threshold actually
    matters and the optimisation is non-trivial.

    Difficulty tiers:
      EASY SAME    — embeddings nearly identical (noise ≤ 0.05), names match
      MEDIUM SAME  — moderate embedding noise (0.25-0.40), names partially match
      HARD SAME    — high embedding noise (0.55-0.80), names very different
      EASY DIFF    — random embeddings, different names
      MEDIUM DIFF  — shared signals (industry/description/relations)
      HARD DIFF    — deliberately similar embeddings + overlapping names
    """
    pairs: list[LabelledPair] = []

    # ── Anchor embeddings (one per "real-world entity") ───────────────────
    emb_elon   = _rand_emb(1)
    emb_tesla  = _rand_emb(2)
    emb_spacex = _rand_emb(3)
    emb_bezos  = _rand_emb(4)
    emb_apple  = _rand_emb(5)
    emb_openai = _rand_emb(6)
    emb_nikola = _rand_emb(7)
    emb_blue   = _rand_emb(8)
    emb_google = _rand_emb(9)
    emb_msft   = _rand_emb(10)
    emb_samsung= _rand_emb(11)
    emb_goldman= _rand_emb(12)

    # Citation embeddings (distinct from desc but correlated for same entity)
    cite_elon   = _similar_emb(emb_elon,  0.15)
    cite_tesla  = _similar_emb(emb_tesla, 0.15)
    cite_spacex = _similar_emb(emb_spacex,0.15)
    cite_bezos  = _similar_emb(emb_bezos, 0.15)
    cite_apple  = _similar_emb(emb_apple, 0.15)
    cite_openai = _similar_emb(emb_openai,0.15)
    cite_google = _similar_emb(emb_google, 0.15)
    cite_msft   = _similar_emb(emb_msft,  0.15)
    cite_samsung= _similar_emb(emb_samsung,0.15)
    cite_goldman= _similar_emb(emb_goldman,0.15)

    # Additional anchors for new pairs
    emb_amazon  = _rand_emb(13)
    emb_alibaba = _rand_emb(14)
    emb_meta    = _rand_emb(15)
    emb_netflix = _rand_emb(16)
    emb_twitter = _rand_emb(17)
    emb_tesla2  = _rand_emb(18)   # Nikola Tesla as person (distinct from org)
    emb_ibm     = _rand_emb(19)
    emb_oracle  = _rand_emb(22)
    emb_uber    = _rand_emb(23)
    emb_airbnb  = _rand_emb(24)
    emb_obama   = _rand_emb(25)
    emb_biden   = _rand_emb(26)
    emb_merkel  = _rand_emb(27)
    emb_newton  = _rand_emb(28)
    emb_einstein= _rand_emb(29)
    emb_harvard = _rand_emb(32)
    emb_mit     = _rand_emb(33)
    emb_stanford= _rand_emb(34)
    emb_who     = _rand_emb(35)
    emb_un      = _rand_emb(36)
    emb_nato    = _rand_emb(37)
    emb_jpmorgan= _rand_emb(38)
    emb_boa     = _rand_emb(39)

    cite_amazon  = _similar_emb(emb_amazon,  0.15)
    cite_alibaba = _similar_emb(emb_alibaba, 0.15)
    cite_meta    = _similar_emb(emb_meta,    0.15)
    cite_netflix = _similar_emb(emb_netflix, 0.15)
    cite_twitter = _similar_emb(emb_twitter, 0.15)
    cite_ibm     = _similar_emb(emb_ibm,     0.15)
    cite_oracle  = _similar_emb(emb_oracle,  0.15)
    cite_uber    = _similar_emb(emb_uber,    0.15)
    cite_airbnb  = _similar_emb(emb_airbnb,  0.15)
    cite_obama   = _similar_emb(emb_obama,   0.15)
    cite_biden   = _similar_emb(emb_biden,   0.15)
    cite_merkel  = _similar_emb(emb_merkel,  0.15)
    cite_newton  = _similar_emb(emb_newton,  0.15)
    cite_einstein= _similar_emb(emb_einstein,0.15)
    cite_harvard = _similar_emb(emb_harvard, 0.15)
    cite_mit     = _similar_emb(emb_mit,     0.15)
    cite_stanford= _similar_emb(emb_stanford,0.15)
    cite_who     = _similar_emb(emb_who,     0.15)
    cite_un      = _similar_emb(emb_un,      0.15)
    cite_nato    = _similar_emb(emb_nato,    0.15)
    cite_jpmorgan= _similar_emb(emb_jpmorgan,0.15)
    cite_boa     = _similar_emb(emb_boa,     0.15)

    # ── EASY SAME pairs (should score > 0.80 on emb scorers) ─────────────

    # 1. Exact duplicate — noise=0.05
    pairs.append(LabelledPair(
        entity_a=_entity("Elon Musk", "person",
                         "Entrepreneur, CEO of Tesla and SpaceX",
                         _dt(1971), relation_types=["IS_CEO_OF", "FOUNDED"],
                         desc_emb=emb_elon, cite_emb=cite_elon),
        entity_b=_entity("Elon Musk", "person",
                         "Technology entrepreneur and CEO",
                         _dt(1971), relation_types=["IS_CEO_OF", "FOUNDED"],
                         desc_emb=_similar_emb(emb_elon, 0.05), cite_emb=_similar_emb(cite_elon, 0.05)),
        label="SAME",
        description="[EASY] Exact name duplicate",
    ))

    # 2. Trivial punctuation — noise=0.03
    pairs.append(LabelledPair(
        entity_a=_entity("Apple Inc", "organization",
                         "Consumer electronics and technology company",
                         _dt(1976), relation_types=["PRODUCES", "HAS_CEO"],
                         desc_emb=emb_apple, cite_emb=cite_apple),
        entity_b=_entity("Apple Inc.", "organization",
                         "Technology company producing iPhone and Mac",
                         _dt(1976), relation_types=["PRODUCES", "HAS_CEO"],
                         desc_emb=_similar_emb(emb_apple, 0.03), cite_emb=_similar_emb(cite_apple, 0.03)),
        label="SAME",
        description="[EASY] Trailing period",
    ))

    # 3. Same name, same type, description drift — noise=0.10
    pairs.append(LabelledPair(
        entity_a=_entity("OpenAI", "organization",
                         "AI safety and research company",
                         _dt(2015), relation_types=["DEVELOPS", "FOUNDED_BY"],
                         desc_emb=emb_openai, cite_emb=cite_openai),
        entity_b=_entity("OpenAI", "organization",
                         "Creator of GPT and ChatGPT, focuses on AGI",
                         _dt(2015), relation_types=["DEVELOPS", "RELEASED"],
                         desc_emb=_similar_emb(emb_openai, 0.10), cite_emb=_similar_emb(cite_openai, 0.10)),
        label="SAME",
        description="[EASY] Same entity, description drift",
    ))

    # ── MEDIUM SAME pairs (should score 0.50-0.75 on emb scorers) ────────

    # 4. Minor punctuation with moderate embedding drift — noise=0.30
    pairs.append(LabelledPair(
        entity_a=_entity("Tesla, Inc.", "organization",
                         "Electric vehicle and clean energy company",
                         _dt(2003), relation_types=["HAS_CEO", "PRODUCES"],
                         desc_emb=emb_tesla, cite_emb=cite_tesla),
        entity_b=_entity("Tesla Inc", "organization",
                         "EV manufacturer and energy storage company",
                         _dt(2003), relation_types=["HAS_CEO", "PRODUCES"],
                         desc_emb=_similar_emb(emb_tesla, 0.30), cite_emb=_similar_emb(cite_tesla, 0.30)),
        label="SAME",
        description="[MEDIUM] Punctuation diff + embedding drift",
    ))

    # 5. Name variant with moderate noise — noise=0.35
    pairs.append(LabelledPair(
        entity_a=_entity("Jeff Bezos", "person",
                         "Founder of Amazon and Blue Origin",
                         _dt(1964), relation_types=["FOUNDED", "IS_CEO_OF"],
                         desc_emb=emb_bezos, cite_emb=cite_bezos),
        entity_b=_entity("Jeffrey P. Bezos", "person",
                         "American entrepreneur, founder of Amazon",
                         _dt(1964), relation_types=["FOUNDED", "IS_CEO_OF"],
                         desc_emb=_similar_emb(emb_bezos, 0.35), cite_emb=_similar_emb(cite_bezos, 0.35)),
        label="SAME",
        description="[MEDIUM] First-name variant + embedding noise",
    ))

    # 6. Middle initial with different description — noise=0.30
    pairs.append(LabelledPair(
        entity_a=_entity("Elon R. Musk", "person",
                         "South African-American entrepreneur",
                         _dt(1971), relation_types=["IS_CEO_OF"],
                         desc_emb=_similar_emb(emb_elon, 0.30), cite_emb=_similar_emb(cite_elon, 0.30)),
        entity_b=_entity("Elon Musk", "person",
                         "Entrepreneur; CEO of Tesla, SpaceX",
                         _dt(1971), relation_types=["IS_CEO_OF", "FOUNDED"],
                         desc_emb=emb_elon, cite_emb=cite_elon),
        label="SAME",
        description="[MEDIUM] Middle initial + embedding drift",
    ))

    # 7. Google naming: "Alphabet Inc." vs "Google" — noise=0.25
    pairs.append(LabelledPair(
        entity_a=_entity("Google", "organization",
                         "Search engine and technology company",
                         _dt(1998), relation_types=["OPERATES", "HAS_CEO", "DEVELOPS"],
                         desc_emb=emb_google, cite_emb=cite_google),
        entity_b=_entity("Alphabet Inc.", "organization",
                         "Parent company of Google and other subsidiaries",
                         _dt(2015), relation_types=["OWNS", "HAS_CEO"],
                         desc_emb=_similar_emb(emb_google, 0.25), cite_emb=_similar_emb(cite_google, 0.25)),
        label="SAME",
        description="[MEDIUM] Subsidiary vs parent (different name, same entity cluster)",
    ))

    # ── HARD SAME pairs (should score 0.30-0.50 on emb scorers) ──────────

    # 8. Abbreviation with HIGH embedding noise — noise=0.55
    pairs.append(LabelledPair(
        entity_a=_entity("SpaceX", "organization",
                         "Private aerospace manufacturer and space transport",
                         _dt(2002), relation_types=["HAS_CEO", "OPERATES", "LAUNCHES"],
                         desc_emb=emb_spacex, cite_emb=cite_spacex),
        entity_b=_entity("Space Exploration Technologies Corp", "organization",
                         "Aerospace company that designs and launches rockets",
                         _dt(2002), relation_types=["HAS_CEO", "OPERATES"],
                         desc_emb=_similar_emb(emb_spacex, 0.55), cite_emb=_similar_emb(cite_spacex, 0.55)),
        label="SAME",
        description="[HARD] Abbreviation + high embedding noise",
    ))

    # 9. Samsung Korean/English — noise=0.65
    pairs.append(LabelledPair(
        entity_a=_entity("Samsung Electronics", "organization",
                         "South Korean multinational electronics corporation",
                         _dt(1969), relation_types=["PRODUCES", "HAS_CEO", "OPERATES_IN"],
                         desc_emb=emb_samsung, cite_emb=cite_samsung),
        entity_b=_entity("Samsung Electronics Co., Ltd.", "organization",
                         "Global semiconductor and consumer electronics manufacturer",
                         _dt(1969), relation_types=["PRODUCES", "MANUFACTURES"],
                         desc_emb=_similar_emb(emb_samsung, 0.65), cite_emb=_similar_emb(cite_samsung, 0.65)),
        label="SAME",
        description="[HARD] Legal name variant + high embedding noise",
    ))

    # 10. Goldman Sachs abbreviation — noise=0.70
    pairs.append(LabelledPair(
        entity_a=_entity("Goldman Sachs", "organization",
                         "American investment bank and financial services company",
                         _dt(1869), relation_types=["UNDERWRITES", "HAS_CEO", "ADVISES"],
                         desc_emb=emb_goldman, cite_emb=cite_goldman),
        entity_b=_entity("The Goldman Sachs Group, Inc.", "organization",
                         "Global investment banking firm headquartered in New York",
                         _dt(1869), relation_types=["UNDERWRITES", "HAS_CEO"],
                         desc_emb=_similar_emb(emb_goldman, 0.70), cite_emb=_similar_emb(cite_goldman, 0.70)),
        label="SAME",
        description="[HARD] Legal name variant + very high embedding noise",
    ))

    # 11. Microsoft naming — noise=0.60
    pairs.append(LabelledPair(
        entity_a=_entity("Microsoft", "organization",
                         "Technology company known for Windows and Office",
                         _dt(1975), relation_types=["PRODUCES", "HAS_CEO", "DEVELOPS"],
                         desc_emb=emb_msft, cite_emb=cite_msft),
        entity_b=_entity("Microsoft Corporation", "organization",
                         "American multinational tech corp producing software and cloud services",
                         _dt(1975), relation_types=["PRODUCES", "DEVELOPS", "ACQUIRED"],
                         desc_emb=_similar_emb(emb_msft, 0.60), cite_emb=_similar_emb(cite_msft, 0.60)),
        label="SAME",
        description="[HARD] Corp suffix + high embedding noise",
    ))

    # ── EASY DIFFERENT pairs (should score < 0.20 on all scorers) ─────────

    # 12. Completely different people
    pairs.append(LabelledPair(
        entity_a=_entity("Elon Musk", "person",
                         "CEO of Tesla and SpaceX",
                         _dt(1971), relation_types=["IS_CEO_OF"],
                         desc_emb=emb_elon, cite_emb=cite_elon),
        entity_b=_entity("Jeff Bezos", "person",
                         "Founder of Amazon",
                         _dt(1964), relation_types=["FOUNDED"],
                         desc_emb=emb_bezos, cite_emb=cite_bezos),
        label="DIFFERENT",
        description="[EASY] Different people entirely",
    ))

    # 13. Substring trap: "AI" != "OpenAI"
    pairs.append(LabelledPair(
        entity_a=_entity("AI", "concept",
                         "Artificial intelligence as a field of study",
                         _dt(1956), relation_types=["STUDIED_BY"],
                         desc_emb=_rand_emb(50), cite_emb=_rand_emb(51)),
        entity_b=_entity("OpenAI", "organization",
                         "AI safety research lab",
                         _dt(2015), relation_types=["DEVELOPS"],
                         desc_emb=emb_openai, cite_emb=cite_openai),
        label="DIFFERENT",
        description="[EASY] Substring trap: 'AI' is not 'OpenAI'",
    ))

    # 14. Different domain, no name overlap
    pairs.append(LabelledPair(
        entity_a=_entity("Goldman Sachs", "organization",
                         "American investment bank",
                         _dt(1869), relation_types=["UNDERWRITES", "ADVISES"],
                         desc_emb=emb_goldman, cite_emb=cite_goldman),
        entity_b=_entity("SpaceX", "organization",
                         "Private aerospace company",
                         _dt(2002), relation_types=["LAUNCHES", "OPERATES"],
                         desc_emb=emb_spacex, cite_emb=cite_spacex),
        label="DIFFERENT",
        description="[EASY] Different domain, no name overlap",
    ))

    # ── MEDIUM DIFFERENT pairs (should score 0.25-0.50) ───────────────────

    # 15. Company vs person with overlapping name "Tesla"
    pairs.append(LabelledPair(
        entity_a=_entity("Tesla", "organization",
                         "Electric vehicle company",
                         _dt(2003), relation_types=["HAS_CEO", "PRODUCES"],
                         desc_emb=emb_tesla, cite_emb=cite_tesla),
        entity_b=_entity("Nikola Tesla", "person",
                         "Serbian-American inventor and electrical engineer",
                         _dt(1856), _dt(1943), relation_types=["INVENTED", "PATENTED"],
                         desc_emb=emb_nikola, cite_emb=_similar_emb(emb_nikola, 0.15)),
        label="DIFFERENT",
        description="[MEDIUM] Company vs person (partial name 'Tesla')",
    ))

    # 16. Same industry, different companies — use somewhat similar embs
    pairs.append(LabelledPair(
        entity_a=_entity("SpaceX", "organization",
                         "Private aerospace company",
                         _dt(2002), relation_types=["LAUNCHES", "OPERATES"],
                         desc_emb=emb_spacex, cite_emb=cite_spacex),
        entity_b=_entity("Blue Origin", "organization",
                         "Private aerospace company founded by Jeff Bezos",
                         _dt(2000), relation_types=["LAUNCHES", "DEVELOPS"],
                         desc_emb=_similar_emb(emb_spacex, 0.50), cite_emb=_similar_emb(cite_spacex, 0.50)),
        label="DIFFERENT",
        description="[MEDIUM] Same industry, shared relation types + semi-similar emb",
    ))

    # 17. GM vs Ford — very similar description, similar embs
    pairs.append(LabelledPair(
        entity_a=_entity("General Motors", "organization",
                         "American multinational automobile manufacturer",
                         _dt(1908), relation_types=["PRODUCES", "HAS_CEO"],
                         desc_emb=_rand_emb(40), cite_emb=_rand_emb(41)),
        entity_b=_entity("Ford Motor Company", "organization",
                         "American multinational automobile manufacturer",
                         _dt(1903), relation_types=["PRODUCES", "HAS_CEO"],
                         desc_emb=_similar_emb(_rand_emb(40), 0.40), cite_emb=_similar_emb(_rand_emb(41), 0.40)),
        label="DIFFERENT",
        description="[MEDIUM] Nearly identical description + moderate emb similarity",
    ))

    # 18. Partial name overlap, different domain
    pairs.append(LabelledPair(
        entity_a=_entity("OpenAI", "organization",
                         "AI safety research lab",
                         _dt(2015), relation_types=["DEVELOPS"],
                         desc_emb=emb_openai, cite_emb=cite_openai),
        entity_b=_entity("Open Doors Foundation", "organization",
                         "Non-profit supporting persecuted Christians",
                         _dt(1955), relation_types=["SUPPORTS", "OPERATES_IN"],
                         desc_emb=_similar_emb(emb_openai, 0.55), cite_emb=_similar_emb(cite_openai, 0.55)),
        label="DIFFERENT",
        description="[MEDIUM] Partial name overlap + semi-similar emb",
    ))

    # ── HARD DIFFERENT pairs (should score 0.50-0.75 — confusing!) ────────

    # 19. Apple Inc vs Apple Records — SIMILAR embedding (high false positive risk!)
    pairs.append(LabelledPair(
        entity_a=_entity("Apple Inc", "organization",
                         "Consumer technology company",
                         _dt(1976), relation_types=["PRODUCES", "HAS_CEO"],
                         desc_emb=emb_apple, cite_emb=cite_apple),
        entity_b=_entity("Apple Records", "organization",
                         "Record label founded by the Beatles",
                         _dt(1968), relation_types=["SIGNED", "PUBLISHED"],
                         desc_emb=_similar_emb(emb_apple, 0.25), cite_emb=_similar_emb(cite_apple, 0.25)),
        label="DIFFERENT",
        description="[HARD] Same first word + HIGH embedding similarity!",
    ))

    # 20. Microsoft vs Microsoft Research — same name prefix, very similar embs
    pairs.append(LabelledPair(
        entity_a=_entity("Microsoft", "organization",
                         "Technology company known for Windows and Office",
                         _dt(1975), relation_types=["PRODUCES", "HAS_CEO", "DEVELOPS"],
                         desc_emb=emb_msft, cite_emb=cite_msft),
        entity_b=_entity("Microsoft Research", "organization",
                         "Research subsidiary of Microsoft focusing on AI and systems",
                         _dt(1991), relation_types=["PUBLISHES", "DEVELOPS", "RESEARCHES"],
                         desc_emb=_similar_emb(emb_msft, 0.15), cite_emb=_similar_emb(cite_msft, 0.15)),
        label="DIFFERENT",
        description="[HARD] Parent vs subsidiary — very similar emb + name overlap!",
    ))

    # 21. Google vs Google DeepMind — same name prefix, similar embs
    pairs.append(LabelledPair(
        entity_a=_entity("Google", "organization",
                         "Search engine and technology company",
                         _dt(1998), relation_types=["OPERATES", "HAS_CEO", "DEVELOPS"],
                         desc_emb=emb_google, cite_emb=cite_google),
        entity_b=_entity("Google DeepMind", "organization",
                         "AI research lab owned by Google",
                         _dt(2010), relation_types=["DEVELOPS", "RESEARCHES", "PUBLISHES"],
                         desc_emb=_similar_emb(emb_google, 0.18), cite_emb=_similar_emb(cite_google, 0.18)),
        label="DIFFERENT",
        description="[HARD] Parent vs AI lab — very similar emb + name prefix!",
    ))

    # 22. Samsung Electronics vs Samsung SDI — sibling subsidiaries
    pairs.append(LabelledPair(
        entity_a=_entity("Samsung Electronics", "organization",
                         "South Korean multinational electronics corporation",
                         _dt(1969), relation_types=["PRODUCES", "HAS_CEO", "OPERATES_IN"],
                         desc_emb=emb_samsung, cite_emb=cite_samsung),
        entity_b=_entity("Samsung SDI", "organization",
                         "South Korean energy solutions and battery manufacturer",
                         _dt(1970), relation_types=["PRODUCES", "MANUFACTURES", "SUPPLIES"],
                         desc_emb=_similar_emb(emb_samsung, 0.20), cite_emb=_similar_emb(cite_samsung, 0.20)),
        label="DIFFERENT",
        description="[HARD] Sibling subsidiaries — similar emb + name prefix!",
    ))

    # ── ADDITIONAL EASY SAME pairs ────────────────────────────────────────

    # 23. "Amazon" vs "Amazon.com"
    pairs.append(LabelledPair(
        entity_a=_entity("Amazon", "organization",
                         "E-commerce and cloud computing company",
                         _dt(1994), relation_types=["OPERATES", "HAS_CEO", "SELLS"],
                         desc_emb=emb_amazon, cite_emb=cite_amazon),
        entity_b=_entity("Amazon.com", "organization",
                         "Online retail and AWS cloud services",
                         _dt(1994), relation_types=["OPERATES", "SELLS", "HAS_CEO"],
                         desc_emb=_similar_emb(emb_amazon, 0.04), cite_emb=_similar_emb(cite_amazon, 0.04)),
        label="SAME",
        description="[EASY] Domain suffix variant",
    ))

    # 24. "Meta" vs "Meta Platforms"
    pairs.append(LabelledPair(
        entity_a=_entity("Meta", "organization",
                         "Social media company owning Facebook and Instagram",
                         _dt(2021), relation_types=["OWNS", "HAS_CEO", "OPERATES"],
                         desc_emb=emb_meta, cite_emb=cite_meta),
        entity_b=_entity("Meta Platforms", "organization",
                         "Technology company formerly known as Facebook",
                         _dt(2021), relation_types=["OWNS", "HAS_CEO"],
                         desc_emb=_similar_emb(emb_meta, 0.04), cite_emb=_similar_emb(cite_meta, 0.04)),
        label="SAME",
        description="[EASY] Abbreviated vs full company name",
    ))

    # 25. "Barack Obama" vs "Barack H. Obama"
    pairs.append(LabelledPair(
        entity_a=_entity("Barack Obama", "person",
                         "44th President of the United States",
                         _dt(1961), relation_types=["IS_PRESIDENT_OF", "REPRESENTED"],
                         desc_emb=emb_obama, cite_emb=cite_obama),
        entity_b=_entity("Barack H. Obama", "person",
                         "44th US President, Nobel Peace Prize laureate",
                         _dt(1961), relation_types=["IS_PRESIDENT_OF", "AWARDED"],
                         desc_emb=_similar_emb(emb_obama, 0.04), cite_emb=_similar_emb(cite_obama, 0.04)),
        label="SAME",
        description="[EASY] Middle initial variant",
    ))

    # 26. "WHO" vs "World Health Organization"
    pairs.append(LabelledPair(
        entity_a=_entity("WHO", "organization",
                         "International public health agency of the United Nations",
                         _dt(1948), relation_types=["MEMBER_OF", "PUBLISHES", "ADVISES"],
                         desc_emb=emb_who, cite_emb=cite_who),
        entity_b=_entity("World Health Organization", "organization",
                         "Specialized agency of the UN for global public health",
                         _dt(1948), relation_types=["MEMBER_OF", "PUBLISHES"],
                         desc_emb=_similar_emb(emb_who, 0.05), cite_emb=_similar_emb(cite_who, 0.05)),
        label="SAME",
        description="[EASY] Acronym vs full name",
    ))

    # 27. "Netflix" vs "Netflix, Inc."
    pairs.append(LabelledPair(
        entity_a=_entity("Netflix", "organization",
                         "Subscription streaming service for movies and TV shows",
                         _dt(1997), relation_types=["PRODUCES", "DISTRIBUTES", "HAS_CEO"],
                         desc_emb=emb_netflix, cite_emb=cite_netflix),
        entity_b=_entity("Netflix, Inc.", "organization",
                         "American over-the-top content platform and production company",
                         _dt(1997), relation_types=["PRODUCES", "DISTRIBUTES"],
                         desc_emb=_similar_emb(emb_netflix, 0.04), cite_emb=_similar_emb(cite_netflix, 0.04)),
        label="SAME",
        description="[EASY] Inc. suffix",
    ))

    # ── ADDITIONAL MEDIUM SAME pairs ─────────────────────────────────────

    # 28. "Twitter" vs "X (formerly Twitter)" — noise=0.30
    pairs.append(LabelledPair(
        entity_a=_entity("Twitter", "organization",
                         "Social media platform for short messages",
                         _dt(2006), relation_types=["OPERATES", "HAS_CEO"],
                         desc_emb=emb_twitter, cite_emb=cite_twitter),
        entity_b=_entity("X", "organization",
                         "Social media platform rebranded from Twitter by Elon Musk",
                         _dt(2006), relation_types=["OPERATES", "HAS_CEO"],
                         desc_emb=_similar_emb(emb_twitter, 0.30), cite_emb=_similar_emb(cite_twitter, 0.30)),
        label="SAME",
        description="[MEDIUM] Rebranded company (Twitter → X)",
    ))

    # 29. "Facebook" vs "Meta" — noise=0.28
    pairs.append(LabelledPair(
        entity_a=_entity("Facebook", "organization",
                         "Social networking site founded by Mark Zuckerberg",
                         _dt(2004), relation_types=["OPERATES", "HAS_CEO", "OWNS"],
                         desc_emb=emb_meta, cite_emb=cite_meta),
        entity_b=_entity("Meta Platforms, Inc.", "organization",
                         "Technology company that owns Facebook, Instagram and WhatsApp",
                         _dt(2004), relation_types=["OWNS", "HAS_CEO", "OPERATES"],
                         desc_emb=_similar_emb(emb_meta, 0.28), cite_emb=_similar_emb(cite_meta, 0.28)),
        label="SAME",
        description="[MEDIUM] Rebrand + legal name",
    ))

    # 30. "MIT" vs "Massachusetts Institute of Technology" — noise=0.25
    pairs.append(LabelledPair(
        entity_a=_entity("MIT", "organization",
                         "Top-ranked private research university in Cambridge, Massachusetts",
                         _dt(1861), relation_types=["AWARDS", "RESEARCHES", "EMPLOYS"],
                         desc_emb=emb_mit, cite_emb=cite_mit),
        entity_b=_entity("Massachusetts Institute of Technology", "organization",
                         "Leading American STEM university and research institution",
                         _dt(1861), relation_types=["AWARDS", "RESEARCHES"],
                         desc_emb=_similar_emb(emb_mit, 0.25), cite_emb=_similar_emb(cite_mit, 0.25)),
        label="SAME",
        description="[MEDIUM] Acronym vs full university name",
    ))

    # 31. "Angela Merkel" vs "Angela Dorothea Merkel" — noise=0.32
    pairs.append(LabelledPair(
        entity_a=_entity("Angela Merkel", "person",
                         "Chancellor of Germany from 2005 to 2021",
                         _dt(1954), relation_types=["IS_CHANCELLOR_OF", "MEMBER_OF"],
                         desc_emb=emb_merkel, cite_emb=cite_merkel),
        entity_b=_entity("Angela Dorothea Merkel", "person",
                         "German politician, served as chancellor for 16 years",
                         _dt(1954), relation_types=["IS_CHANCELLOR_OF", "LEADS"],
                         desc_emb=_similar_emb(emb_merkel, 0.32), cite_emb=_similar_emb(cite_merkel, 0.32)),
        label="SAME",
        description="[MEDIUM] Middle name variant",
    ))

    # 32. "UN" vs "United Nations" — noise=0.28
    pairs.append(LabelledPair(
        entity_a=_entity("UN", "organization",
                         "International intergovernmental organization",
                         _dt(1945), relation_types=["INCLUDES", "PUBLISHES", "ENFORCES"],
                         desc_emb=emb_un, cite_emb=cite_un),
        entity_b=_entity("United Nations", "organization",
                         "Global organization promoting international peace and cooperation",
                         _dt(1945), relation_types=["INCLUDES", "MEDIATES"],
                         desc_emb=_similar_emb(emb_un, 0.28), cite_emb=_similar_emb(cite_un, 0.28)),
        label="SAME",
        description="[MEDIUM] Acronym vs full name + embedding drift",
    ))

    # ── ADDITIONAL HARD SAME pairs ────────────────────────────────────────

    # 33. "Alibaba" vs "Alibaba Group Holding Limited" — noise=0.60
    pairs.append(LabelledPair(
        entity_a=_entity("Alibaba", "organization",
                         "Chinese e-commerce and technology conglomerate",
                         _dt(1999), relation_types=["OPERATES", "HAS_CEO", "OWNS"],
                         desc_emb=emb_alibaba, cite_emb=cite_alibaba),
        entity_b=_entity("Alibaba Group Holding Limited", "organization",
                         "Multinational technology company specializing in retail and cloud",
                         _dt(1999), relation_types=["OPERATES", "LISTED_ON"],
                         desc_emb=_similar_emb(emb_alibaba, 0.60), cite_emb=_similar_emb(cite_alibaba, 0.60)),
        label="SAME",
        description="[HARD] Short name vs full legal name + high noise",
    ))

    # 34. "Uber" vs "Uber Technologies, Inc." — noise=0.65
    pairs.append(LabelledPair(
        entity_a=_entity("Uber", "organization",
                         "Ride-hailing and food delivery technology company",
                         _dt(2009), relation_types=["OPERATES", "HAS_CEO", "PARTNERS"],
                         desc_emb=emb_uber, cite_emb=cite_uber),
        entity_b=_entity("Uber Technologies, Inc.", "organization",
                         "Transportation network company offering rideshare and delivery services",
                         _dt(2009), relation_types=["OPERATES", "LISTED_ON"],
                         desc_emb=_similar_emb(emb_uber, 0.65), cite_emb=_similar_emb(cite_uber, 0.65)),
        label="SAME",
        description="[HARD] Short name vs legal name + very high noise",
    ))

    # 35. "IBM" vs "International Business Machines" — noise=0.68
    pairs.append(LabelledPair(
        entity_a=_entity("IBM", "organization",
                         "Multinational technology and consulting corporation",
                         _dt(1911), relation_types=["PRODUCES", "HAS_CEO", "DEVELOPS"],
                         desc_emb=emb_ibm, cite_emb=cite_ibm),
        entity_b=_entity("International Business Machines", "organization",
                         "American technology firm specializing in hardware and AI",
                         _dt(1911), relation_types=["PRODUCES", "DEVELOPS", "RESEARCHES"],
                         desc_emb=_similar_emb(emb_ibm, 0.68), cite_emb=_similar_emb(cite_ibm, 0.68)),
        label="SAME",
        description="[HARD] Acronym vs full name + very high noise",
    ))

    # 36. "Oracle" vs "Oracle Corporation" — noise=0.72
    pairs.append(LabelledPair(
        entity_a=_entity("Oracle", "organization",
                         "Enterprise software and cloud computing company",
                         _dt(1977), relation_types=["PRODUCES", "HAS_CEO", "ACQUIRES"],
                         desc_emb=emb_oracle, cite_emb=cite_oracle),
        entity_b=_entity("Oracle Corporation", "organization",
                         "American technology corporation known for database software",
                         _dt(1977), relation_types=["PRODUCES", "HAS_CEO"],
                         desc_emb=_similar_emb(emb_oracle, 0.72), cite_emb=_similar_emb(cite_oracle, 0.72)),
        label="SAME",
        description="[HARD] Corp suffix + very high noise",
    ))

    # 37. Harvard abbreviations — noise=0.58
    pairs.append(LabelledPair(
        entity_a=_entity("Harvard University", "organization",
                         "Private Ivy League research university in Cambridge",
                         _dt(1636), relation_types=["AWARDS", "EMPLOYS", "RESEARCHES"],
                         desc_emb=emb_harvard, cite_emb=cite_harvard),
        entity_b=_entity("Harvard", "organization",
                         "Oldest US university, known for law, medicine and business schools",
                         _dt(1636), relation_types=["AWARDS", "ADMITS"],
                         desc_emb=_similar_emb(emb_harvard, 0.58), cite_emb=_similar_emb(cite_harvard, 0.58)),
        label="SAME",
        description="[HARD] Full name vs informal short name + high noise",
    ))

    # ── ADDITIONAL EASY DIFFERENT pairs ──────────────────────────────────

    # 38. Amazon (company) vs Amazon (river)
    pairs.append(LabelledPair(
        entity_a=_entity("Amazon", "organization",
                         "E-commerce and cloud computing company",
                         _dt(1994), relation_types=["OPERATES", "HAS_CEO"],
                         desc_emb=emb_amazon, cite_emb=cite_amazon),
        entity_b=_entity("Amazon River", "geo",
                         "Largest river in South America by water volume",
                         _dt(1500), relation_types=["FLOWS_THROUGH", "LOCATED_IN"],
                         desc_emb=_rand_emb(60), cite_emb=_rand_emb(61)),
        label="DIFFERENT",
        description="[EASY] Homonym: company vs river",
    ))

    # 39. Obama vs Biden
    pairs.append(LabelledPair(
        entity_a=_entity("Barack Obama", "person",
                         "44th President of the United States",
                         _dt(1961), relation_types=["IS_PRESIDENT_OF"],
                         desc_emb=emb_obama, cite_emb=cite_obama),
        entity_b=_entity("Joe Biden", "person",
                         "46th President of the United States",
                         _dt(1942), relation_types=["IS_PRESIDENT_OF"],
                         desc_emb=emb_biden, cite_emb=cite_biden),
        label="DIFFERENT",
        description="[EASY] Different presidents (shared role but different people)",
    ))

    # 40. Harvard vs MIT
    pairs.append(LabelledPair(
        entity_a=_entity("Harvard University", "organization",
                         "Private Ivy League research university",
                         _dt(1636), relation_types=["AWARDS", "RESEARCHES"],
                         desc_emb=emb_harvard, cite_emb=cite_harvard),
        entity_b=_entity("MIT", "organization",
                         "Top-ranked STEM research university",
                         _dt(1861), relation_types=["AWARDS", "RESEARCHES"],
                         desc_emb=emb_mit, cite_emb=cite_mit),
        label="DIFFERENT",
        description="[EASY] Different universities (same domain, different entity)",
    ))

    # 41. JP Morgan vs Bank of America
    pairs.append(LabelledPair(
        entity_a=_entity("JPMorgan Chase", "organization",
                         "Largest American bank by assets",
                         _dt(1799), relation_types=["UNDERWRITES", "ADVISES", "HAS_CEO"],
                         desc_emb=emb_jpmorgan, cite_emb=cite_jpmorgan),
        entity_b=_entity("Bank of America", "organization",
                         "American multinational investment bank and financial services",
                         _dt(1904), relation_types=["UNDERWRITES", "LENDS", "HAS_CEO"],
                         desc_emb=emb_boa, cite_emb=cite_boa),
        label="DIFFERENT",
        description="[EASY] Different banks (same industry, different entities)",
    ))

    # 42. Newton vs Einstein
    pairs.append(LabelledPair(
        entity_a=_entity("Isaac Newton", "person",
                         "English mathematician and physicist, discovered gravity",
                         _dt(1643), _dt(1727), relation_types=["INVENTED", "PUBLISHED"],
                         desc_emb=emb_newton, cite_emb=cite_newton),
        entity_b=_entity("Albert Einstein", "person",
                         "German-American physicist known for theory of relativity",
                         _dt(1879), _dt(1955), relation_types=["PUBLISHED", "AWARDED"],
                         desc_emb=emb_einstein, cite_emb=cite_einstein),
        label="DIFFERENT",
        description="[EASY] Different scientists (same domain, different people)",
    ))

    # ── ADDITIONAL MEDIUM DIFFERENT pairs ────────────────────────────────

    # 43. Stanford vs Harvard — similar academic context, semi-similar embs
    pairs.append(LabelledPair(
        entity_a=_entity("Stanford University", "organization",
                         "Private research university in Silicon Valley",
                         _dt(1885), relation_types=["AWARDS", "EMPLOYS", "RESEARCHES"],
                         desc_emb=emb_stanford, cite_emb=cite_stanford),
        entity_b=_entity("Harvard University", "organization",
                         "Private Ivy League research university in Cambridge",
                         _dt(1636), relation_types=["AWARDS", "EMPLOYS", "RESEARCHES"],
                         desc_emb=_similar_emb(emb_stanford, 0.45), cite_emb=_similar_emb(cite_stanford, 0.45)),
        label="DIFFERENT",
        description="[MEDIUM] Different universities + moderately similar embs",
    ))

    # 44. NATO vs UN — similar geopolitical orgs, semi-similar embs
    pairs.append(LabelledPair(
        entity_a=_entity("NATO", "organization",
                         "North Atlantic military alliance for collective defense",
                         _dt(1949), relation_types=["INCLUDES", "DEFENDS", "COORDINATES"],
                         desc_emb=emb_nato, cite_emb=cite_nato),
        entity_b=_entity("United Nations", "organization",
                         "Global intergovernmental organization for international peace",
                         _dt(1945), relation_types=["INCLUDES", "MEDIATES", "PUBLISHES"],
                         desc_emb=_similar_emb(emb_nato, 0.40), cite_emb=_similar_emb(cite_nato, 0.40)),
        label="DIFFERENT",
        description="[MEDIUM] Different international organizations + semi-similar embs",
    ))

    # 45. Airbnb vs Uber — same-era tech startups, semi-similar embs
    pairs.append(LabelledPair(
        entity_a=_entity("Airbnb", "organization",
                         "Online marketplace for short-term lodging and home rentals",
                         _dt(2008), relation_types=["OPERATES", "HAS_CEO", "LISTED_ON"],
                         desc_emb=emb_airbnb, cite_emb=cite_airbnb),
        entity_b=_entity("Uber", "organization",
                         "Ride-hailing and delivery technology platform",
                         _dt(2009), relation_types=["OPERATES", "HAS_CEO", "LISTED_ON"],
                         desc_emb=_similar_emb(emb_airbnb, 0.42), cite_emb=_similar_emb(cite_airbnb, 0.42)),
        label="DIFFERENT",
        description="[MEDIUM] Different startups + similar era + semi-similar embs",
    ))

    # 46. JPMorgan vs Goldman Sachs — similar finance context
    pairs.append(LabelledPair(
        entity_a=_entity("JPMorgan Chase", "organization",
                         "Largest American bank and investment firm",
                         _dt(1799), relation_types=["UNDERWRITES", "ADVISES"],
                         desc_emb=emb_jpmorgan, cite_emb=cite_jpmorgan),
        entity_b=_entity("Goldman Sachs", "organization",
                         "Global investment bank and financial services firm",
                         _dt(1869), relation_types=["UNDERWRITES", "ADVISES"],
                         desc_emb=_similar_emb(emb_jpmorgan, 0.45), cite_emb=_similar_emb(cite_jpmorgan, 0.45)),
        label="DIFFERENT",
        description="[MEDIUM] Different banks + very similar role profile",
    ))

    # 47. Obama vs Merkel — both heads of state, moderately similar embs
    pairs.append(LabelledPair(
        entity_a=_entity("Barack Obama", "person",
                         "44th President of the United States",
                         _dt(1961), relation_types=["IS_PRESIDENT_OF", "AWARDED"],
                         desc_emb=emb_obama, cite_emb=cite_obama),
        entity_b=_entity("Angela Merkel", "person",
                         "Chancellor of Germany 2005-2021",
                         _dt(1954), relation_types=["IS_CHANCELLOR_OF", "MEMBER_OF"],
                         desc_emb=_similar_emb(emb_obama, 0.48), cite_emb=_similar_emb(cite_obama, 0.48)),
        label="DIFFERENT",
        description="[MEDIUM] Different heads of state + semi-similar embs",
    ))

    # ── ADDITIONAL HARD DIFFERENT pairs ──────────────────────────────────

    # 48. "Oracle Database" vs "Oracle Corporation" — VERY similar embs
    pairs.append(LabelledPair(
        entity_a=_entity("Oracle Database", "product",
                         "Relational database management system by Oracle Corporation",
                         _dt(1977), relation_types=["PRODUCED_BY", "USED_BY", "COMPETES_WITH"],
                         desc_emb=emb_oracle, cite_emb=cite_oracle),
        entity_b=_entity("Oracle Corporation", "organization",
                         "Enterprise software and cloud computing company",
                         _dt(1977), relation_types=["PRODUCES", "HAS_CEO", "ACQUIRES"],
                         desc_emb=_similar_emb(emb_oracle, 0.12), cite_emb=_similar_emb(cite_oracle, 0.12)),
        label="DIFFERENT",
        description="[HARD] Product vs company — nearly identical embs!",
    ))

    # 49. "Amazon Web Services" vs "Amazon" — subsidiary vs parent
    pairs.append(LabelledPair(
        entity_a=_entity("Amazon Web Services", "organization",
                         "Cloud computing subsidiary of Amazon",
                         _dt(2006), relation_types=["OWNED_BY", "PROVIDES", "COMPETES_WITH"],
                         desc_emb=_similar_emb(emb_amazon, 0.15), cite_emb=_similar_emb(cite_amazon, 0.15)),
        entity_b=_entity("Amazon", "organization",
                         "E-commerce and cloud computing company",
                         _dt(1994), relation_types=["OPERATES", "HAS_CEO", "SELLS"],
                         desc_emb=emb_amazon, cite_emb=cite_amazon),
        label="DIFFERENT",
        description="[HARD] Subsidiary vs parent — very similar embs + name prefix!",
    ))

    # 50. "IBM Research" vs "IBM" — research division vs company
    pairs.append(LabelledPair(
        entity_a=_entity("IBM Research", "organization",
                         "Research and development division of IBM",
                         _dt(1945), relation_types=["PUBLISHES", "DEVELOPS", "OWNED_BY"],
                         desc_emb=_similar_emb(emb_ibm, 0.14), cite_emb=_similar_emb(cite_ibm, 0.14)),
        entity_b=_entity("IBM", "organization",
                         "Multinational technology and consulting corporation",
                         _dt(1911), relation_types=["PRODUCES", "HAS_CEO", "DEVELOPS"],
                         desc_emb=emb_ibm, cite_emb=cite_ibm),
        label="DIFFERENT",
        description="[HARD] Division vs company — very similar embs + name prefix!",
    ))

    # 51. "Facebook" vs "Facebook Messenger" — product vs parent
    pairs.append(LabelledPair(
        entity_a=_entity("Facebook", "organization",
                         "Social networking site with 3 billion users",
                         _dt(2004), relation_types=["OPERATES", "OWNED_BY", "HAS_CEO"],
                         desc_emb=emb_meta, cite_emb=cite_meta),
        entity_b=_entity("Facebook Messenger", "product",
                         "Instant messaging app by Meta Platforms",
                         _dt(2011), relation_types=["OWNED_BY", "USED_BY", "INTEGRATES"],
                         desc_emb=_similar_emb(emb_meta, 0.18), cite_emb=_similar_emb(cite_meta, 0.18)),
        label="DIFFERENT",
        description="[HARD] Platform vs messaging product — similar embs!",
    ))

    # 52. Biden vs Obama — both presidents, high emb similarity
    pairs.append(LabelledPair(
        entity_a=_entity("Joe Biden", "person",
                         "46th President of the United States, former Senator",
                         _dt(1942), relation_types=["IS_PRESIDENT_OF", "MEMBER_OF"],
                         desc_emb=emb_biden, cite_emb=cite_biden),
        entity_b=_entity("Barack Obama", "person",
                         "44th President of the United States, Nobel laureate",
                         _dt(1961), relation_types=["IS_PRESIDENT_OF", "AWARDED"],
                         desc_emb=_similar_emb(emb_biden, 0.22), cite_emb=_similar_emb(cite_biden, 0.22)),
        label="DIFFERENT",
        description="[HARD] Different presidents + similar role + semi-similar embs",
    ))

    # 53. "Twitter Blue" vs "Twitter" — product vs platform
    pairs.append(LabelledPair(
        entity_a=_entity("Twitter Blue", "product",
                         "Paid subscription tier of Twitter with extra features",
                         _dt(2021), relation_types=["OWNED_BY", "OFFERED_BY", "SUBSCRIBES"],
                         desc_emb=_similar_emb(emb_twitter, 0.16), cite_emb=_similar_emb(cite_twitter, 0.16)),
        entity_b=_entity("Twitter", "organization",
                         "Social media platform for short-form messages",
                         _dt(2006), relation_types=["OPERATES", "HAS_CEO"],
                         desc_emb=emb_twitter, cite_emb=cite_twitter),
        label="DIFFERENT",
        description="[HARD] Product vs platform — very similar embs!",
    ))

    # 54. Newton (physicist) vs Newton (unit) — homonym trap
    pairs.append(LabelledPair(
        entity_a=_entity("Isaac Newton", "person",
                         "English physicist who formulated laws of motion and gravity",
                         _dt(1643), _dt(1727), relation_types=["PUBLISHED", "INVENTED"],
                         desc_emb=emb_newton, cite_emb=cite_newton),
        entity_b=_entity("Newton", "concept",
                         "SI unit of force, named after Isaac Newton",
                         _dt(1948), relation_types=["NAMED_AFTER", "MEASURES", "DEFINED_BY"],
                         desc_emb=_similar_emb(emb_newton, 0.20), cite_emb=_similar_emb(cite_newton, 0.20)),
        label="DIFFERENT",
        description="[HARD] Person vs unit named after them — similar embs!",
    ))

    # 55. Stanford Research Institute vs Stanford University
    pairs.append(LabelledPair(
        entity_a=_entity("SRI International", "organization",
                         "Independent nonprofit research institute originally founded by Stanford",
                         _dt(1946), relation_types=["RESEARCHES", "DEVELOPS", "PARTNERS"],
                         desc_emb=_similar_emb(emb_stanford, 0.19), cite_emb=_similar_emb(cite_stanford, 0.19)),
        entity_b=_entity("Stanford University", "organization",
                         "Private research university in Silicon Valley",
                         _dt(1885), relation_types=["AWARDS", "EMPLOYS", "RESEARCHES"],
                         desc_emb=emb_stanford, cite_emb=cite_stanford),
        label="DIFFERENT",
        description="[HARD] Spin-off institute vs founding university — similar embs!",
    ))

    # 56. "NATO" vs "EU" — different international organizations, similar structure
    pairs.append(LabelledPair(
        entity_a=_entity("NATO", "organization",
                         "North Atlantic Treaty Organization military alliance",
                         _dt(1949), relation_types=["INCLUDES", "DEFENDS", "COORDINATES"],
                         desc_emb=emb_nato, cite_emb=cite_nato),
        entity_b=_entity("European Union", "organization",
                         "Political and economic union of European member states",
                         _dt(1993), relation_types=["INCLUDES", "LEGISLATES", "REGULATES"],
                         desc_emb=_similar_emb(emb_nato, 0.25), cite_emb=_similar_emb(cite_nato, 0.25)),
        label="DIFFERENT",
        description="[HARD] Different organizations + overlapping member states + semi-similar embs",
    ))

    # 57. "Netflix" vs "Disney+" — streaming competitors
    pairs.append(LabelledPair(
        entity_a=_entity("Netflix", "organization",
                         "Subscription streaming service for movies and TV shows",
                         _dt(1997), relation_types=["PRODUCES", "DISTRIBUTES", "HAS_CEO"],
                         desc_emb=emb_netflix, cite_emb=cite_netflix),
        entity_b=_entity("Disney+", "product",
                         "Streaming service operated by The Walt Disney Company",
                         _dt(2019), relation_types=["OWNED_BY", "DISTRIBUTES", "COMPETES_WITH"],
                         desc_emb=_similar_emb(emb_netflix, 0.30), cite_emb=_similar_emb(cite_netflix, 0.30)),
        label="DIFFERENT",
        description="[MEDIUM] Competing streaming services + similar embs",
    ))

    # 58. "Airbnb" vs "Booking.com" — competing platforms
    pairs.append(LabelledPair(
        entity_a=_entity("Airbnb", "organization",
                         "Online marketplace for short-term home rentals",
                         _dt(2008), relation_types=["OPERATES", "HAS_CEO", "LISTED_ON"],
                         desc_emb=emb_airbnb, cite_emb=cite_airbnb),
        entity_b=_entity("Booking.com", "organization",
                         "Online travel and accommodation reservation platform",
                         _dt(1996), relation_types=["OPERATES", "OWNED_BY", "PARTNERS"],
                         desc_emb=_similar_emb(emb_airbnb, 0.35), cite_emb=_similar_emb(cite_airbnb, 0.35)),
        label="DIFFERENT",
        description="[MEDIUM] Competing accommodation platforms + semi-similar embs",
    ))

    # 59. Amazon vs Alibaba — competing e-commerce giants
    pairs.append(LabelledPair(
        entity_a=_entity("Amazon", "organization",
                         "American e-commerce and cloud computing conglomerate",
                         _dt(1994), relation_types=["OPERATES", "HAS_CEO", "SELLS"],
                         desc_emb=emb_amazon, cite_emb=cite_amazon),
        entity_b=_entity("Alibaba", "organization",
                         "Chinese e-commerce and technology conglomerate",
                         _dt(1999), relation_types=["OPERATES", "HAS_CEO", "SELLS"],
                         desc_emb=_similar_emb(emb_amazon, 0.38), cite_emb=_similar_emb(cite_amazon, 0.38)),
        label="DIFFERENT",
        description="[MEDIUM] Competing e-commerce giants + similar role profile",
    ))

    # 60. Einstein vs Newton — both physicists, moderately similar embs
    pairs.append(LabelledPair(
        entity_a=_entity("Albert Einstein", "person",
                         "German-American physicist known for theory of relativity",
                         _dt(1879), _dt(1955), relation_types=["PUBLISHED", "AWARDED"],
                         desc_emb=emb_einstein, cite_emb=cite_einstein),
        entity_b=_entity("Isaac Newton", "person",
                         "English mathematician and physicist who formulated laws of motion",
                         _dt(1643), _dt(1727), relation_types=["PUBLISHED", "INVENTED"],
                         desc_emb=_similar_emb(emb_einstein, 0.40), cite_emb=_similar_emb(cite_einstein, 0.40)),
        label="DIFFERENT",
        description="[MEDIUM] Different physicists + similar domain + semi-similar embs",
    ))

    # 61. Biden vs Obama — president pair (harder version with higher emb similarity)
    pairs.append(LabelledPair(
        entity_a=_entity("Joseph R. Biden Jr.", "person",
                         "46th President of the United States, former VP",
                         _dt(1942), relation_types=["IS_PRESIDENT_OF", "WAS_VP_OF"],
                         desc_emb=emb_biden, cite_emb=cite_biden),
        entity_b=_entity("Obama", "person",
                         "44th President of the United States",
                         _dt(1961), relation_types=["IS_PRESIDENT_OF", "AWARDED"],
                         desc_emb=_similar_emb(emb_biden, 0.35), cite_emb=_similar_emb(cite_biden, 0.35)),
        label="DIFFERENT",
        description="[MEDIUM] Different presidents + similar context",
    ))

    # 62. UN vs NATO — acronym confusion trap
    pairs.append(LabelledPair(
        entity_a=_entity("UN", "organization",
                         "International intergovernmental organization for peace",
                         _dt(1945), relation_types=["INCLUDES", "PUBLISHES", "ENFORCES"],
                         desc_emb=emb_un, cite_emb=cite_un),
        entity_b=_entity("NATO", "organization",
                         "North Atlantic military alliance",
                         _dt(1949), relation_types=["INCLUDES", "DEFENDS", "COORDINATES"],
                         desc_emb=_similar_emb(emb_un, 0.30), cite_emb=_similar_emb(cite_un, 0.30)),
        label="DIFFERENT",
        description="[MEDIUM] Both are acronym international orgs + semi-similar embs",
    ))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    scorer_name: str
    config_label: str
    merge_threshold: float
    llm_threshold_low: float
    extra_params: dict = field(default_factory=dict)
    # Counts
    tp: int = 0           # correctly auto-merged (SAME & score >= merge_thresh)
    tn: int = 0           # correctly auto-separated (DIFFERENT & score < llm_thresh)
    fp: int = 0           # wrongly auto-merged (DIFFERENT & score >= merge_thresh)
    fn: int = 0           # wrongly auto-separated (SAME & score < llm_thresh)
    llm_same: int = 0     # LLM zone, label=SAME (would be caught by LLM)
    llm_diff: int = 0     # LLM zone, label=DIFFERENT (would be caught by LLM)
    n_total: int = 0

    @property
    def llm_calls(self) -> int:
        return self.llm_same + self.llm_diff

    @property
    def llm_call_rate(self) -> float:
        return self.llm_calls / max(self.n_total, 1)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        # Recall = TP / (TP + FN + llm_same)
        # llm_same would have been TP if LLM was called (they are SAME pairs in LLM zone)
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def recall_with_llm(self) -> float:
        """Recall if LLM zone calls are assumed correct."""
        return (self.tp + self.llm_same) / max(self.tp + self.fn + self.llm_same, 1)

    @property
    def accuracy(self) -> float:
        """Auto decisions only (excludes LLM zone)."""
        correct = self.tp + self.tn
        auto_total = self.tp + self.tn + self.fp + self.fn
        return correct / max(auto_total, 1)

    @property
    def accuracy_with_llm(self) -> float:
        """Accuracy assuming LLM zone calls always correct."""
        correct = self.tp + self.tn + self.llm_calls
        return correct / max(self.n_total, 1)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * p * r / max(p + r, 1e-9)

    @property
    def f1_with_llm(self) -> float:
        """F1 if LLM zone calls are assumed perfect."""
        p = (self.tp + self.llm_same) / max(self.tp + self.llm_same + self.fp, 1)
        r = self.recall_with_llm
        return 2 * p * r / max(p + r, 1e-9)

    def objective(self, lam: float = 0.2) -> float:
        """Combined objective: F1_with_llm - lambda * llm_call_rate."""
        return self.f1_with_llm - lam * self.llm_call_rate


def evaluate_scorer(
    scorer_fn,
    scorer_name: str,
    config: BTGraphRAGConfig,
    pairs: list[LabelledPair],
    merge_threshold: float,
    llm_threshold_low: float,
    config_label: str = "",
    extra_params: dict | None = None,
) -> EvalResult:
    """Score every pair and classify into auto-merge / LLM-zone / keep-separate."""
    result = EvalResult(
        scorer_name=scorer_name,
        config_label=config_label,
        merge_threshold=merge_threshold,
        llm_threshold_low=llm_threshold_low,
        extra_params=extra_params or {},
        n_total=len(pairs),
    )

    for pair in pairs:
        score, _ = scorer_fn(pair.entity_a, pair.entity_b, config)
        is_same = pair.label == "SAME"

        if score >= merge_threshold:
            # Auto-merge
            if is_same:
                result.tp += 1
            else:
                result.fp += 1
        elif score >= llm_threshold_low:
            # LLM zone
            if is_same:
                result.llm_same += 1
            else:
                result.llm_diff += 1
        else:
            # Auto-separate
            if is_same:
                result.fn += 1
            else:
                result.tn += 1

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Grid definitions
# ─────────────────────────────────────────────────────────────────────────────

# Fine-grained threshold sweep (0.025 steps)
_mt_vals = [round(0.40 + i * 0.025, 3) for i in range(25)]   # 0.40 → 0.98
_lt_vals = [round(0.10 + i * 0.025, 3) for i in range(25)]   # 0.10 → 0.70

THRESHOLD_GRID = {
    "merge_threshold":  _mt_vals,   # 25 values: 0.400 … 0.975
    "llm_threshold_low": _lt_vals,  # 25 values: 0.100 … 0.700
}

# Weight profiles for the 5-signal composite scorer
COMPOSITE_WEIGHT_PROFILES = {
    # Balanced baselines
    "balanced":            {"emb": 0.30, "bm25": 0.25, "jacc": 0.15, "temp": 0.15, "rel": 0.15},
    "uniform":             {"emb": 0.20, "bm25": 0.20, "jacc": 0.20, "temp": 0.20, "rel": 0.20},
    # Embedding-heavy
    "emb_dominant":        {"emb": 0.65, "bm25": 0.10, "jacc": 0.10, "temp": 0.08, "rel": 0.07},
    "emb_heavy":           {"emb": 0.55, "bm25": 0.15, "jacc": 0.10, "temp": 0.10, "rel": 0.10},
    "emb_moderate":        {"emb": 0.40, "bm25": 0.20, "jacc": 0.15, "temp": 0.13, "rel": 0.12},
    # Name-heavy (good for no-embedding fallback)
    "name_dominant":       {"emb": 0.00, "bm25": 0.50, "jacc": 0.35, "temp": 0.08, "rel": 0.07},
    "name_heavy":          {"emb": 0.10, "bm25": 0.40, "jacc": 0.30, "temp": 0.10, "rel": 0.10},
    "name_moderate":       {"emb": 0.20, "bm25": 0.32, "jacc": 0.22, "temp": 0.13, "rel": 0.13},
    "pure_name":           {"emb": 0.00, "bm25": 0.55, "jacc": 0.45, "temp": 0.00, "rel": 0.00},
    # No embedding
    "no_emb_balanced":     {"emb": 0.00, "bm25": 0.30, "jacc": 0.25, "temp": 0.25, "rel": 0.20},
    "no_emb_name_focus":   {"emb": 0.00, "bm25": 0.40, "jacc": 0.30, "temp": 0.15, "rel": 0.15},
    # Temporal-heavy
    "temporal_dominant":   {"emb": 0.15, "bm25": 0.15, "jacc": 0.10, "temp": 0.45, "rel": 0.15},
    "temporal_heavy":      {"emb": 0.20, "bm25": 0.15, "jacc": 0.15, "temp": 0.35, "rel": 0.15},
    "temporal_no_emb":     {"emb": 0.00, "bm25": 0.25, "jacc": 0.20, "temp": 0.35, "rel": 0.20},
    # Relation context-heavy
    "rel_dominant":        {"emb": 0.15, "bm25": 0.15, "jacc": 0.10, "temp": 0.15, "rel": 0.45},
    "rel_heavy":           {"emb": 0.20, "bm25": 0.15, "jacc": 0.15, "temp": 0.15, "rel": 0.35},
    # Combined signal focus
    "emb_name_only":       {"emb": 0.50, "bm25": 0.30, "jacc": 0.20, "temp": 0.00, "rel": 0.00},
    "name_rel_focus":      {"emb": 0.00, "bm25": 0.30, "jacc": 0.25, "temp": 0.10, "rel": 0.35},
    "emb_rel_focus":       {"emb": 0.40, "bm25": 0.10, "jacc": 0.10, "temp": 0.10, "rel": 0.30},
    "all_but_temporal":    {"emb": 0.30, "bm25": 0.28, "jacc": 0.17, "temp": 0.00, "rel": 0.25},
}

# Weight profiles for citation_and_description scorer
CITATION_WEIGHT_PROFILES = {
    "equal":          {"w_desc": 0.50, "w_cite": 0.50},
    "desc_dominant":  {"w_desc": 0.85, "w_cite": 0.15},
    "desc_heavy":     {"w_desc": 0.70, "w_cite": 0.30},
    "desc_moderate":  {"w_desc": 0.60, "w_cite": 0.40},
    "cite_moderate":  {"w_desc": 0.40, "w_cite": 0.60},
    "cite_heavy":     {"w_desc": 0.30, "w_cite": 0.70},
    "cite_dominant":  {"w_desc": 0.15, "w_cite": 0.85},
    "desc_only":      {"w_desc": 1.00, "w_cite": 0.00},
    "cite_only":      {"w_desc": 0.00, "w_cite": 1.00},
    "slight_desc":    {"w_desc": 0.55, "w_cite": 0.45},
}


def _make_composite_config(profile: dict[str, float]) -> BTGraphRAGConfig:
    return BTGraphRAGConfig(
        cger_embedding_weight=profile["emb"],
        cger_bm25_weight=profile["bm25"],
        cger_jaccard_weight=profile["jacc"],
        cger_temporal_overlap_weight=profile["temp"],
        cger_relation_context_weight=profile["rel"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(title: str) -> None:
    print("\n" + "═" * 80)
    print(f"  {title}")
    print("═" * 80)


def _print_results_table(results: list[EvalResult], top_n: int = 10) -> None:
    """Print the top N results ranked by objective(lambda=0.2)."""
    ranked = sorted(results, key=lambda r: r.objective(0.2), reverse=True)[:top_n]

    print(f"\n  {'Config':25s} {'MT':>5} {'LT':>5} {'TP':>3} {'FP':>3} "
          f"{'TN':>3} {'FN':>3} {'LLM':>4} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'F1+L':>6} {'LLM%':>6} {'Obj':>7}")
    print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*3} {'-'*3} "
          f"{'-'*3} {'-'*3} {'-'*4} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    for r in ranked:
        label = r.config_label[:25] if r.config_label else r.scorer_name[:25]
        print(
            f"  {label:25s} {r.merge_threshold:5.2f} {r.llm_threshold_low:5.2f} "
            f"{r.tp:3d} {r.fp:3d} {r.tn:3d} {r.fn:3d} {r.llm_calls:4d} "
            f"{r.precision:6.3f} {r.recall:6.3f} {r.f1:6.3f} "
            f"{r.f1_with_llm:6.3f} {r.llm_call_rate:5.1%} {r.objective(0.2):7.4f}"
        )


def _print_pareto(results: list[EvalResult]) -> None:
    """Show the best config per lambda."""
    print(f"\n  Pareto frontier (best config per lambda):")
    print(f"  {'Lambda':>7} {'Config':25s} {'MT':>5} {'LT':>5} {'F1+L':>6} {'LLM%':>6} {'Objective':>9}")
    print(f"  {'-'*7} {'-'*25} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*9}")
    for lam in LAMBDAS:
        best = max(results, key=lambda r: r.objective(lam))
        label = best.config_label[:25] if best.config_label else best.scorer_name[:25]
        print(
            f"  {lam:7.2f} {label:25s} {best.merge_threshold:5.2f} "
            f"{best.llm_threshold_low:5.2f} {best.f1_with_llm:6.3f} "
            f"{best.llm_call_rate:5.1%} {best.objective(lam):9.4f}"
        )


def _print_best_recommendation(results: list[EvalResult], scorer_name: str) -> None:
    """Print the recommended config (lambda=0.2) with full details."""
    best = max(results, key=lambda r: r.objective(0.2))
    print(f"\n  ★ RECOMMENDED for {scorer_name} (lambda=0.2):")
    print(f"    merge_threshold  = {best.merge_threshold}")
    print(f"    llm_threshold_low= {best.llm_threshold_low}")
    if best.extra_params:
        for k, v in best.extra_params.items():
            print(f"    {k:19s}= {v}")
    print(f"    F1 (auto only)   = {best.f1:.4f}")
    print(f"    F1 (with LLM)    = {best.f1_with_llm:.4f}")
    print(f"    Accuracy (auto)  = {best.accuracy:.4f}")
    print(f"    Accuracy (+LLM)  = {best.accuracy_with_llm:.4f}")
    print(f"    LLM call rate    = {best.llm_call_rate:.1%}  ({best.llm_calls}/{best.n_total} pairs)")
    print(f"    Objective        = {best.objective(0.2):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Scorer 1: embedding_only_entity_scorer
# ─────────────────────────────────────────────────────────────────────────────

def run_embedding_only(pairs: list[LabelledPair]) -> list[EvalResult]:
    _sep("SCORER 1 — embedding_only_entity_scorer")
    print("  Single signal: cosine(description_embedding)")
    print("  Hyperparameters: merge_threshold, llm_threshold_low only (no weights)")

    config = BTGraphRAGConfig()  # weights don't matter for this scorer
    results: list[EvalResult] = []

    for mt in THRESHOLD_GRID["merge_threshold"]:
        for lt in THRESHOLD_GRID["llm_threshold_low"]:
            if lt >= mt:
                continue  # llm_low must be < merge_threshold
            res = evaluate_scorer(
                embedding_only_entity_scorer, "emb_only",
                config, pairs, mt, lt,
                config_label=f"mt={mt:.2f} lt={lt:.2f}",
            )
            results.append(res)

    _print_results_table(results, top_n=15)
    _print_pareto(results)
    _print_best_recommendation(results, "embedding_only_entity_scorer")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Scorer 2: citation_and_description_entity_scorer
# ─────────────────────────────────────────────────────────────────────────────

def run_citation_description(pairs: list[LabelledPair]) -> list[EvalResult]:
    _sep("SCORER 2 — citation_and_description_entity_scorer")
    print("  Two signals: cosine(desc_emb) + cosine(cite_emb)")
    print("  Hyperparameters: w_desc, w_cite, merge_threshold, llm_threshold_low")

    config = BTGraphRAGConfig()  # base config
    results: list[EvalResult] = []

    for profile_name, weights in CITATION_WEIGHT_PROFILES.items():
        # Build a scorer closure with these weights
        w_d = weights["w_desc"]
        w_c = weights["w_cite"]

        def scorer_fn(ea, eb, cfg, _wd=w_d, _wc=w_c):
            return citation_and_description_entity_scorer(ea, eb, cfg, w_desc=_wd, w_cite=_wc)

        for mt in THRESHOLD_GRID["merge_threshold"]:
            for lt in THRESHOLD_GRID["llm_threshold_low"]:
                if lt >= mt:
                    continue
                res = evaluate_scorer(
                    scorer_fn, "cite_desc",
                    config, pairs, mt, lt,
                    config_label=f"{profile_name} mt={mt:.2f} lt={lt:.2f}",
                    extra_params={"w_desc": w_d, "w_cite": w_c},
                )
                results.append(res)

    _print_results_table(results, top_n=15)
    _print_pareto(results)
    _print_best_recommendation(results, "citation_and_description_entity_scorer")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Scorer 3: compute_entity_composite_score (5-signal)
# ─────────────────────────────────────────────────────────────────────────────

def run_composite(pairs: list[LabelledPair]) -> list[EvalResult]:
    _sep("SCORER 3 — compute_entity_composite_score (5-signal)")
    print("  Signals: cosine(desc) + BM25(name) + Jaccard(name) + TemporalOverlap + RelationCtx")
    print(f"  Weight profiles: {len(COMPOSITE_WEIGHT_PROFILES)}")
    print(f"  Threshold combos per profile: {sum(1 for mt in THRESHOLD_GRID['merge_threshold'] for lt in THRESHOLD_GRID['llm_threshold_low'] if lt < mt)}")

    results: list[EvalResult] = []

    for profile_name, weights in COMPOSITE_WEIGHT_PROFILES.items():
        cfg = _make_composite_config(weights)

        for mt in THRESHOLD_GRID["merge_threshold"]:
            for lt in THRESHOLD_GRID["llm_threshold_low"]:
                if lt >= mt:
                    continue
                res = evaluate_scorer(
                    compute_entity_composite_score, "composite",
                    cfg, pairs, mt, lt,
                    config_label=f"{profile_name} mt={mt:.2f} lt={lt:.2f}",
                    extra_params={
                        "profile": profile_name,
                        "emb": weights["emb"],
                        "bm25": weights["bm25"],
                        "jacc": weights["jacc"],
                        "temp": weights["temp"],
                        "rel":  weights["rel"],
                    },
                )
                results.append(res)

    _print_results_table(results, top_n=15)
    _print_pareto(results)
    _print_best_recommendation(results, "compute_entity_composite_score")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-scorer comparison
# ─────────────────────────────────────────────────────────────────────────────

def print_cross_comparison(
    emb_results: list[EvalResult],
    cite_results: list[EvalResult],
    comp_results: list[EvalResult],
) -> None:
    _sep("CROSS-SCORER COMPARISON")
    print("  Best configuration per scorer at lambda=0.2:")
    print()

    for name, results in [
        ("embedding_only",          emb_results),
        ("citation_and_description",cite_results),
        ("composite_5signal",       comp_results),
    ]:
        best = max(results, key=lambda r: r.objective(0.2))
        label = best.config_label
        print(f"  {name:30s}  F1+LLM={best.f1_with_llm:.3f}  "
              f"LLM%={best.llm_call_rate:5.1%}  "
              f"Acc+LLM={best.accuracy_with_llm:.3f}  "
              f"Obj={best.objective(0.2):.4f}")
        print(f"  {'':30s}  mt={best.merge_threshold:.2f}  lt={best.llm_threshold_low:.2f}")
        if best.extra_params:
            params_str = "  ".join(f"{k}={v}" for k, v in best.extra_params.items() if k != "profile")
            profile = best.extra_params.get("profile", "")
            if profile:
                print(f"  {'':30s}  profile={profile}  {params_str}")
            else:
                print(f"  {'':30s}  {params_str}")
        print()

    # Summary table across all lambdas
    print(f"  {'Lambda':>7} | {'embedding_only':^30s} | {'cite_desc':^30s} | {'composite':^30s}")
    print(f"  {'-'*7}-+-{'-'*30}-+-{'-'*30}-+-{'-'*30}")
    for lam in LAMBDAS:
        best_e = max(emb_results,  key=lambda r: r.objective(lam))
        best_c = max(cite_results, key=lambda r: r.objective(lam))
        best_p = max(comp_results, key=lambda r: r.objective(lam))
        print(
            f"  {lam:7.2f} | "
            f"F1={best_e.f1_with_llm:.3f} LLM={best_e.llm_call_rate:4.0%} obj={best_e.objective(lam):.3f} | "
            f"F1={best_c.f1_with_llm:.3f} LLM={best_c.llm_call_rate:4.0%} obj={best_c.objective(lam):.3f} | "
            f"F1={best_p.f1_with_llm:.3f} LLM={best_p.llm_call_rate:4.0%} obj={best_p.objective(lam):.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Score distribution per pair (diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def print_score_distributions(pairs: list[LabelledPair]) -> None:
    _sep("SCORE DISTRIBUTION PER PAIR")
    print(f"\n  {'#':>2} {'Label':8s} {'Emb':>6} {'Cite':>6} {'Comp':>6}  Description")
    print(f"  {'--':>2} {'--------':8s} {'------':>6} {'------':>6} {'------':>6}  -----------")

    cfg_comp = _make_composite_config(COMPOSITE_WEIGHT_PROFILES["balanced"])

    for i, pair in enumerate(pairs):
        s_emb, _ = embedding_only_entity_scorer(pair.entity_a, pair.entity_b, cfg_comp)
        s_cite, _ = citation_and_description_entity_scorer(pair.entity_a, pair.entity_b, cfg_comp)
        s_comp, _ = compute_entity_composite_score(pair.entity_a, pair.entity_b, cfg_comp)
        print(
            f"  {i+1:2d} {pair.label:8s} {s_emb:6.3f} {s_cite:6.3f} "
            f"{s_comp:6.3f}  {pair.description}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Real LLM verification
# ─────────────────────────────────────────────────────────────────────────────

async def run_llm_verification(pairs: list[LabelledPair]) -> None:
    """Call the real LLM on every ground-truth pair and measure its accuracy.

    This validates the assumption that LLM-zone pairs would be 100% correctly
    resolved.  The results show how accurate the LLM actually is, which informs
    how much to trust the 'F1_with_llm' metric from the grid search.
    """
    _sep("REAL LLM VERIFICATION — calling gpt-4.1-mini on all pairs")
    model = _real_llm()

    correct = 0
    total = len(pairs)
    print(f"\n  {'#':>2} {'Label':8s} {'LLM':10s} {'Match':>5}  Description")
    print(f"  {'--':>2} {'--------':8s} {'----------':10s} {'-----':>5}  -----------")

    for i, pair in enumerate(pairs):
        llm_answer = await llm_verify_entity_match(pair.entity_a, pair.entity_b, model)
        is_correct = (
            (llm_answer == "SAME" and pair.label == "SAME")
            or (llm_answer == "DIFFERENT" and pair.label == "DIFFERENT")
        )
        if is_correct:
            correct += 1
        mark = "✓" if is_correct else "✗"
        print(f"  {i+1:2d} {pair.label:8s} {llm_answer:10s} {mark:>5}  {pair.description}")

    accuracy = correct / max(total, 1)
    print(f"\n  LLM Accuracy: {correct}/{total} = {accuracy:.1%}")
    if accuracy < 1.0:
        print(f"  ⚠  LLM is not perfect — the F1_with_llm metric overestimates actual performance")
    else:
        print(f"  ✓  LLM is 100% accurate — F1_with_llm metric is trustworthy")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║   CGER Entity Scorer — Hyperparameter Optimisation (real LLM)               ║")
    print("║                                                                              ║")
    print("║   Objective = F1(with_llm) - lambda * LLM_call_rate                          ║")
    print("║   Goal: maximise accuracy while minimising LLM calls                         ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    embedding_model = _real_embedding_model()

    pairs = build_ground_truth()
    await _embed_pairs(pairs, embedding_model)

    n_same = sum(1 for p in pairs if p.label == "SAME")
    n_diff = sum(1 for p in pairs if p.label == "DIFFERENT")
    print(f"\n  Ground truth: {len(pairs)} pairs ({n_same} SAME, {n_diff} DIFFERENT)")

    # Score distributions first (helps interpret the grid search)
    print_score_distributions(pairs)

    # Real LLM verification (validates the "LLM is perfect" assumption)
    await run_llm_verification(pairs)

    # Run each scorer
    try:
        emb_results  = run_embedding_only(pairs)
        cite_results = run_citation_description(pairs)
        comp_results = run_composite(pairs)
        print_cross_comparison(emb_results, cite_results, comp_results)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise

    print("\n" + "═" * 80)
    print("  Hyperparameter optimisation complete.")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
