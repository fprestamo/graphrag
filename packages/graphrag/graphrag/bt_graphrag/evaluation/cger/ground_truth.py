"""CGER ground truth: canonical entities, alias surface forms, and test pairs.

Defines 20 canonical entities across 5 domains and 76 surface-form aliases.
Builds 94 positive merge pairs + 15 negative (must-not-merge) pairs = 109 total.
"""
from __future__ import annotations

import json
import os
import pathlib
from collections import defaultdict

# ---------------------------------------------------------------------------
# Canonical entities
# ---------------------------------------------------------------------------

CANONICAL_ENTITIES: list[dict] = [
    {"id": "CE01", "title": "OPENAI",       "type": "ORGANIZATION",
     "description": "Artificial intelligence research laboratory founded in 2015 that created ChatGPT and GPT-4."},
    {"id": "CE02", "title": "SAM ALTMAN",   "type": "PERSON",
     "description": "CEO of OpenAI who was briefly ousted in November 2023 before being reinstated."},
    {"id": "CE03", "title": "MICROSOFT",    "type": "ORGANIZATION",
     "description": "American multinational technology corporation headquartered in Redmond, Washington."},
    {"id": "CE04", "title": "CHATGPT",      "type": "PRODUCT",
     "description": "AI chatbot developed by OpenAI launched in November 2022."},
    {"id": "CE05", "title": "GOOGLE",       "type": "ORGANIZATION",
     "description": "American multinational corporation specializing in Internet services and artificial intelligence."},
    {"id": "CE06", "title": "DEEPMIND",     "type": "ORGANIZATION",
     "description": "AI research lab acquired by Google in 2014, known for AlphaGo and AlphaFold."},
    {"id": "CE07", "title": "EUROPEAN UNION", "type": "ORGANIZATION",
     "description": "Political and economic union of 27 European member states."},
    {"id": "CE08", "title": "CHINA",        "type": "GEO",
     "description": "East Asian country officially the People's Republic of China, the world's most populous nation."},
    {"id": "CE09", "title": "UNITED STATES", "type": "GEO",
     "description": "North American country and the world's largest economy."},
    {"id": "CE10", "title": "URSULA VON DER LEYEN", "type": "PERSON",
     "description": "President of the European Commission since 2019."},
    {"id": "CE11", "title": "NASA",         "type": "ORGANIZATION",
     "description": "U.S. government agency responsible for the nation's civilian space program."},
    {"id": "CE12", "title": "JAMES WEBB SPACE TELESCOPE", "type": "PRODUCT",
     "description": "Space telescope launched in 2021 designed to observe the infrared universe."},
    {"id": "CE13", "title": "CRISPR",       "type": "TECHNOLOGY",
     "description": "Gene-editing technology that allows precise modification of DNA sequences."},
    {"id": "CE14", "title": "WORLD HEALTH ORGANIZATION", "type": "ORGANIZATION",
     "description": "Specialized United Nations agency responsible for international public health."},
    {"id": "CE15", "title": "TESLA",        "type": "ORGANIZATION",
     "description": "American electric vehicle and clean energy company founded by Elon Musk."},
    {"id": "CE16", "title": "ELON MUSK",    "type": "PERSON",
     "description": "CEO of Tesla and SpaceX, owner of X (formerly Twitter)."},
    {"id": "CE17", "title": "FEDERAL RESERVE", "type": "ORGANIZATION",
     "description": "Central banking system of the United States."},
    {"id": "CE18", "title": "BITCOIN",      "type": "TECHNOLOGY",
     "description": "Decentralized digital cryptocurrency created in 2009."},
    {"id": "CE19", "title": "FIFA",         "type": "ORGANIZATION",
     "description": "International governing body of association football."},
    {"id": "CE20", "title": "LIONEL MESSI", "type": "PERSON",
     "description": "Argentine professional footballer widely regarded as one of the greatest of all time."},
]

# ---------------------------------------------------------------------------
# Alias map  surface_form_upper → canonical_id
# ---------------------------------------------------------------------------

ALIAS_MAP: dict[str, str] = {
    "OPENAI": "CE01", "OPEN AI": "CE01", "OPENAI INC": "CE01",
    "OPENAI INC.": "CE01", "THE OPENAI LAB": "CE01",
    "SAM ALTMAN": "CE02", "SAMUEL ALTMAN": "CE02", "ALTMAN": "CE02",
    "S. ALTMAN": "CE02", "SAM H. ALTMAN": "CE02",
    "MICROSOFT": "CE03", "MICROSOFT CORP": "CE03", "MICROSOFT CORPORATION": "CE03", "MSFT": "CE03",
    "CHATGPT": "CE04", "CHAT GPT": "CE04", "CHAT-GPT": "CE04",
    "OPENAI CHATGPT": "CE04", "GPT CHATBOT": "CE04",
    "GOOGLE": "CE05", "GOOGLE LLC": "CE05", "ALPHABET": "CE05", "GOOGLE INC": "CE05",
    "DEEPMIND": "CE06", "DEEP MIND": "CE06", "GOOGLE DEEPMIND": "CE06",
    "DEEPMIND TECHNOLOGIES": "CE06",
    "EUROPEAN UNION": "CE07", "EU": "CE07", "THE EU": "CE07", "E.U.": "CE07",
    "CHINA": "CE08", "PEOPLE'S REPUBLIC OF CHINA": "CE08", "PRC": "CE08",
    "MAINLAND CHINA": "CE08",
    "UNITED STATES": "CE09", "USA": "CE09", "U.S.": "CE09",
    "UNITED STATES OF AMERICA": "CE09", "AMERICA": "CE09", "THE US": "CE09",
    "URSULA VON DER LEYEN": "CE10", "VON DER LEYEN": "CE10",
    "NASA": "CE11", "N.A.S.A.": "CE11",
    "NATIONAL AERONAUTICS AND SPACE ADMINISTRATION": "CE11",
    "JAMES WEBB SPACE TELESCOPE": "CE12", "JWST": "CE12",
    "WEBB TELESCOPE": "CE12", "JAMES WEBB": "CE12",
    "CRISPR": "CE13", "CRISPR-CAS9": "CE13", "CRISPR TECHNOLOGY": "CE13",
    "WORLD HEALTH ORGANIZATION": "CE14", "WHO": "CE14", "THE WHO": "CE14", "W.H.O.": "CE14",
    "TESLA": "CE15", "TESLA INC": "CE15", "TESLA MOTORS": "CE15", "TESLA INC.": "CE15",
    "ELON MUSK": "CE16", "MUSK": "CE16", "E. MUSK": "CE16",
    "FEDERAL RESERVE": "CE17", "THE FED": "CE17", "FED": "CE17",
    "US FEDERAL RESERVE": "CE17", "FEDERAL RESERVE SYSTEM": "CE17",
    "BITCOIN": "CE18", "BTC": "CE18",
    "FIFA": "CE19", "FÉDÉRATION INTERNATIONALE DE FOOTBALL ASSOCIATION": "CE19",
    "LIONEL MESSI": "CE20", "MESSI": "CE20", "LEO MESSI": "CE20",
}

# Negative pairs that must NOT be merged
NEGATIVE_PAIRS: list[tuple[str, str]] = [
    ("OPENAI", "GOOGLE"), ("OPENAI", "DEEPMIND"), ("SAM ALTMAN", "ELON MUSK"),
    ("MICROSOFT", "GOOGLE"), ("CHATGPT", "DEEPMIND"), ("TESLA", "MICROSOFT"),
    ("NASA", "EUROPEAN UNION"), ("BITCOIN", "TESLA"), ("FIFA", "NASA"),
    ("LIONEL MESSI", "SAM ALTMAN"), ("FEDERAL RESERVE", "EUROPEAN UNION"),
    ("CHINA", "UNITED STATES"), ("CRISPR", "BITCOIN"),
    ("JAMES WEBB SPACE TELESCOPE", "CRISPR"), ("WORLD HEALTH ORGANIZATION", "FIFA"),
]


def build_ground_truth() -> dict:
    """Return the full CGER ground truth dict."""
    canonical_map = {ce["id"]: ce for ce in CANONICAL_ENTITIES}

    # Positive pairs: all alias pairs mapping to the same canonical id
    can_to_surfaces: dict[str, set[str]] = defaultdict(set)
    for sf, cid in ALIAS_MAP.items():
        can_to_surfaces[cid].add(sf)

    expected_merges: list[dict] = []
    for cid, surfaces in can_to_surfaces.items():
        slist = sorted(surfaces)
        for i in range(len(slist)):
            for j in range(i + 1, len(slist)):
                expected_merges.append({
                    "surface_a": slist[i], "surface_b": slist[j],
                    "canonical_id": cid, "should_merge": True,
                })

    for a, b in NEGATIVE_PAIRS:
        expected_merges.append({
            "surface_a": a, "surface_b": b,
            "canonical_id": None, "should_merge": False,
        })

    return {
        "component": "CGER",
        "canonical_entities": CANONICAL_ENTITIES,
        "alias_map": ALIAS_MAP,
        "expected_merges": expected_merges,
        "total_canonical_entities": len(CANONICAL_ENTITIES),
        "total_positive_pairs": sum(1 for m in expected_merges if m["should_merge"]),
        "total_negative_pairs": sum(1 for m in expected_merges if not m["should_merge"]),
    }


def write_ground_truth(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    gt = build_ground_truth()
    path = os.path.join(output_dir, "cger_ground_truth.json")
    with open(path, "w") as f:
        json.dump(gt, f, indent=2)
    pos = gt["total_positive_pairs"]
    neg = gt["total_negative_pairs"]
    print(f"  [CGER GT] Wrote {pos} positive + {neg} negative pairs → {path}")
