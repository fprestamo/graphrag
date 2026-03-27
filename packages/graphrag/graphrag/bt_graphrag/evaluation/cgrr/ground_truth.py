"""CGRR ground truth: canonical relation types, aliases, and test pairs.

Defines 18 canonical predicates across 5 domains with 55+ surface-form
aliases extracted from the same 32 evaluation documents used by CGER.
Builds ~110 positive normalization pairs + 20 negative (must-keep-separate)
pairs.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Canonical relation types
# ---------------------------------------------------------------------------

CANONICAL_RELATIONS: list[dict] = [
    # --- Leadership / employment ---
    {"id": "CR01", "type": "IS_CEO_OF",
     "description": "Subject is the chief executive officer of the object organization.",
     "example_source": "SAM ALTMAN", "example_target": "OPENAI"},
    {"id": "CR02", "type": "FOUNDED",
     "description": "Subject founded or co-founded the object organization.",
     "example_source": "SAM ALTMAN", "example_target": "OPENAI"},
    {"id": "CR03", "type": "INVESTED_IN",
     "description": "Subject has made a financial investment in the object.",
     "example_source": "MICROSOFT", "example_target": "OPENAI"},
    {"id": "CR04", "type": "ACQUIRED",
     "description": "Subject acquired or purchased the object organization.",
     "example_source": "GOOGLE", "example_target": "DEEPMIND"},
    {"id": "CR05", "type": "PARTNERED_WITH",
     "description": "Subject has formed a strategic partnership with the object.",
     "example_source": "MICROSOFT", "example_target": "OPENAI"},
    # --- Regulation / governance ---
    {"id": "CR06", "type": "REGULATES",
     "description": "Subject organization sets rules or regulations for the object.",
     "example_source": "EUROPEAN UNION", "example_target": "OPENAI"},
    {"id": "CR07", "type": "LEADS",
     "description": "Subject is the president or head of the object organization.",
     "example_source": "URSULA VON DER LEYEN", "example_target": "EUROPEAN UNION"},
    # --- Technology / product ---
    {"id": "CR08", "type": "DEVELOPED",
     "description": "Subject organization developed or created the object product or technology.",
     "example_source": "OPENAI", "example_target": "CHATGPT"},
    {"id": "CR09", "type": "USES",
     "description": "Subject uses the object as a tool, platform, or technology.",
     "example_source": "MICROSOFT", "example_target": "CHATGPT"},
    {"id": "CR10", "type": "COMPETES_WITH",
     "description": "Subject is a direct competitor of the object in the same market.",
     "example_source": "OPENAI", "example_target": "GOOGLE"},
    # --- Science / research ---
    {"id": "CR11", "type": "LAUNCHED",
     "description": "Subject organization launched or deployed the object vehicle or mission.",
     "example_source": "NASA", "example_target": "JAMES WEBB SPACE TELESCOPE"},
    {"id": "CR12", "type": "FUNDED",
     "description": "Subject provides financial funding for the object program or research.",
     "example_source": "EUROPEAN UNION", "example_target": "NASA"},
    {"id": "CR13", "type": "RESEARCHES",
     "description": "Subject organization conducts research on the object topic or technology.",
     "example_source": "DEEPMIND", "example_target": "CRISPR"},
    # --- Geopolitics ---
    {"id": "CR14", "type": "IMPOSES_SANCTIONS_ON",
     "description": "Subject country or bloc has imposed trade or financial sanctions on the object.",
     "example_source": "UNITED STATES", "example_target": "CHINA"},
    {"id": "CR15", "type": "COOPERATES_WITH",
     "description": "Subject country or organization cooperates with the object on shared goals.",
     "example_source": "UNITED STATES", "example_target": "EUROPEAN UNION"},
    # --- Finance ---
    {"id": "CR16", "type": "OWNS",
     "description": "Subject entity holds ownership of the object asset or company.",
     "example_source": "TESLA", "example_target": "BITCOIN"},
    {"id": "CR17", "type": "SETS_INTEREST_RATES_FOR",
     "description": "Subject central bank sets the benchmark interest rate for the object economy.",
     "example_source": "FEDERAL RESERVE", "example_target": "UNITED STATES"},
    # --- Sports ---
    {"id": "CR18", "type": "PLAYS_FOR",
     "description": "Subject athlete plays for or is affiliated with the object club or team.",
     "example_source": "LIONEL MESSI", "example_target": "FIFA"},
]

# ---------------------------------------------------------------------------
# Alias map  surface_type_upper → canonical_id
# Aliases represent how the same predicate appears with different surface forms
# across different documents.
# ---------------------------------------------------------------------------

ALIAS_MAP: dict[str, str] = {
    # CR01 — IS_CEO_OF
    "IS_CEO_OF": "CR01", "LEADS AS CEO": "CR01", "SERVES AS CEO OF": "CR01",
    "CEO OF": "CR01", "HEADS AS CHIEF EXECUTIVE": "CR01",
    # CR02 — FOUNDED
    "FOUNDED": "CR02", "CO-FOUNDED": "CR02", "ESTABLISHED": "CR02",
    "SET UP": "CR02", "CREATED": "CR02",
    # CR03 — INVESTED_IN
    "INVESTED_IN": "CR03", "HAS INVESTED IN": "CR03", "MADE INVESTMENT IN": "CR03",
    "BACKED": "CR03", "FUNDED BY": "CR03", "IS BACKER OF": "CR03",
    # CR04 — ACQUIRED
    "ACQUIRED": "CR04", "PURCHASED": "CR04", "BOUGHT": "CR04",
    "TOOK OVER": "CR04", "ABSORBED": "CR04",
    # CR05 — PARTNERED_WITH
    "PARTNERED_WITH": "CR05", "HAS PARTNERSHIP WITH": "CR05",
    "COLLABORATES WITH": "CR05", "WORKS WITH": "CR05",
    "FORMED ALLIANCE WITH": "CR05", "IN ALLIANCE WITH": "CR05",
    # CR06 — REGULATES
    "REGULATES": "CR06", "OVERSEES": "CR06", "GOVERNS": "CR06",
    "HAS AUTHORITY OVER": "CR06", "SETS RULES FOR": "CR06",
    # CR07 — LEADS
    "LEADS": "CR07", "IS PRESIDENT OF": "CR07", "HEADS": "CR07",
    "CHAIRS": "CR07", "IS LEADER OF": "CR07",
    # CR08 — DEVELOPED
    "DEVELOPED": "CR08", "BUILT": "CR08", "CREATED PRODUCT": "CR08",
    "MADE": "CR08", "PRODUCED": "CR08",
    # CR09 — USES
    "USES": "CR09", "INTEGRATES": "CR09", "DEPLOYS": "CR09",
    "INCORPORATES": "CR09", "ADOPTS": "CR09",
    # CR10 — COMPETES_WITH
    "COMPETES_WITH": "CR10", "RIVALS": "CR10", "IS COMPETITOR OF": "CR10",
    "COMPETES AGAINST": "CR10",
    # CR11 — LAUNCHED
    "LAUNCHED": "CR11", "DEPLOYED": "CR11", "SENT INTO ORBIT": "CR11",
    "PUT INTO SERVICE": "CR11",
    # CR12 — FUNDED
    "FUNDED": "CR12", "PROVIDED FUNDING FOR": "CR12", "FINANCED": "CR12",
    "GRANTED MONEY TO": "CR12",
    # CR13 — RESEARCHES
    "RESEARCHES": "CR13", "INVESTIGATES": "CR13", "STUDIES": "CR13",
    "CONDUCTS RESEARCH ON": "CR13",
    # CR14 — IMPOSES_SANCTIONS_ON
    "IMPOSES_SANCTIONS_ON": "CR14", "SANCTIONED": "CR14",
    "IMPOSED TRADE RESTRICTIONS ON": "CR14", "PLACED EXPORT CONTROLS ON": "CR14",
    # CR15 — COOPERATES_WITH
    "COOPERATES_WITH": "CR15", "COORDINATES WITH": "CR15",
    "WORKS JOINTLY WITH": "CR15", "COLLABORATES ON POLICY WITH": "CR15",
    # CR16 — OWNS
    "OWNS": "CR16", "HOLDS": "CR16", "POSSESSES": "CR16",
    "IS SHAREHOLDER IN": "CR16",
    # CR17 — SETS_INTEREST_RATES_FOR
    "SETS_INTEREST_RATES_FOR": "CR17", "CONTROLS MONETARY POLICY OF": "CR17",
    "DETERMINES RATES FOR": "CR17",
    # CR18 — PLAYS_FOR
    "PLAYS_FOR": "CR18", "IS AFFILIATED WITH": "CR18", "REPRESENTS": "CR18",
    "COMPETES UNDER": "CR18",
}

# Negative pairs that must NOT be normalized together
NEGATIVE_PAIRS: list[tuple[str, str]] = [
    ("IS_CEO_OF",            "INVESTED_IN"),
    ("FOUNDED",              "ACQUIRED"),
    ("REGULATES",            "PARTNERED_WITH"),
    ("DEVELOPED",            "LAUNCHED"),
    ("COMPETES_WITH",        "COOPERATES_WITH"),
    ("IMPOSES_SANCTIONS_ON", "COOPERATES_WITH"),
    ("OWNS",                 "RESEARCHES"),
    ("PLAYS_FOR",            "REGULATES"),
    ("SETS_INTEREST_RATES_FOR", "FOUNDED"),
    ("FUNDED",               "IMPOSES_SANCTIONS_ON"),
    ("IS_CEO_OF",            "PLAYS_FOR"),
    ("ACQUIRED",             "LAUNCHED"),
    ("USES",                 "OWNS"),
    ("LEADS",                "COMPETES_WITH"),
    ("FUNDED",               "COMPETES_WITH"),
    ("RESEARCHES",           "SETS_INTEREST_RATES_FOR"),
    ("PARTNERED_WITH",       "IMPOSES_SANCTIONS_ON"),
    ("CO-FOUNDED",           "REGULATES"),
    ("DEPLOYED",             "PLAYS_FOR"),
    ("RIVALS",               "COOPERATES_WITH"),
]

# Sample source/target endpoints for each alias
# Used to make endpoint_match signal realistic
ALIAS_ENDPOINTS: dict[str, tuple[str, str]] = {
    # CR01
    "IS_CEO_OF":               ("SAM ALTMAN",  "OPENAI"),
    "LEADS AS CEO":            ("SAM ALTMAN",  "OPENAI"),
    "SERVES AS CEO OF":        ("SAM ALTMAN",  "OPENAI"),
    "CEO OF":                  ("ELON MUSK",   "TESLA"),
    "HEADS AS CHIEF EXECUTIVE":("ELON MUSK",   "TESLA"),
    # CR02
    "FOUNDED":                 ("SAM ALTMAN",  "OPENAI"),
    "CO-FOUNDED":              ("ELON MUSK",   "OPENAI"),
    "ESTABLISHED":             ("SAM ALTMAN",  "OPENAI"),
    "SET UP":                  ("SAM ALTMAN",  "OPENAI"),
    "CREATED":                 ("SAM ALTMAN",  "OPENAI"),
    # CR03
    "INVESTED_IN":             ("MICROSOFT",   "OPENAI"),
    "HAS INVESTED IN":         ("MICROSOFT",   "OPENAI"),
    "MADE INVESTMENT IN":      ("MICROSOFT",   "OPENAI"),
    "BACKED":                  ("MICROSOFT",   "OPENAI"),
    "FUNDED BY":               ("OPENAI",      "MICROSOFT"),
    "IS BACKER OF":            ("MICROSOFT",   "OPENAI"),
    # CR04
    "ACQUIRED":                ("GOOGLE",      "DEEPMIND"),
    "PURCHASED":               ("GOOGLE",      "DEEPMIND"),
    "BOUGHT":                  ("GOOGLE",      "DEEPMIND"),
    "TOOK OVER":               ("GOOGLE",      "DEEPMIND"),
    "ABSORBED":                ("GOOGLE",      "DEEPMIND"),
    # CR05
    "PARTNERED_WITH":          ("MICROSOFT",   "OPENAI"),
    "HAS PARTNERSHIP WITH":    ("MICROSOFT",   "OPENAI"),
    "COLLABORATES WITH":       ("MICROSOFT",   "OPENAI"),
    "WORKS WITH":              ("MICROSOFT",   "OPENAI"),
    "FORMED ALLIANCE WITH":    ("MICROSOFT",   "OPENAI"),
    "IN ALLIANCE WITH":        ("MICROSOFT",   "OPENAI"),
    # CR06
    "REGULATES":               ("EUROPEAN UNION", "OPENAI"),
    "OVERSEES":                ("EUROPEAN UNION", "OPENAI"),
    "GOVERNS":                 ("EUROPEAN UNION", "OPENAI"),
    "HAS AUTHORITY OVER":      ("EUROPEAN UNION", "OPENAI"),
    "SETS RULES FOR":          ("EUROPEAN UNION", "OPENAI"),
    # CR07
    "LEADS":                   ("URSULA VON DER LEYEN", "EUROPEAN UNION"),
    "IS PRESIDENT OF":         ("URSULA VON DER LEYEN", "EUROPEAN UNION"),
    "HEADS":                   ("URSULA VON DER LEYEN", "EUROPEAN UNION"),
    "CHAIRS":                  ("URSULA VON DER LEYEN", "EUROPEAN UNION"),
    "IS LEADER OF":            ("URSULA VON DER LEYEN", "EUROPEAN UNION"),
    # CR08
    "DEVELOPED":               ("OPENAI",  "CHATGPT"),
    "BUILT":                   ("OPENAI",  "CHATGPT"),
    "CREATED PRODUCT":         ("OPENAI",  "CHATGPT"),
    "MADE":                    ("OPENAI",  "CHATGPT"),
    "PRODUCED":                ("OPENAI",  "CHATGPT"),
    # CR09
    "USES":                    ("MICROSOFT", "CHATGPT"),
    "INTEGRATES":              ("MICROSOFT", "CHATGPT"),
    "DEPLOYS":                 ("MICROSOFT", "CHATGPT"),
    "INCORPORATES":            ("MICROSOFT", "CHATGPT"),
    "ADOPTS":                  ("MICROSOFT", "CHATGPT"),
    # CR10
    "COMPETES_WITH":           ("OPENAI",  "GOOGLE"),
    "RIVALS":                  ("OPENAI",  "GOOGLE"),
    "IS COMPETITOR OF":        ("OPENAI",  "GOOGLE"),
    "COMPETES AGAINST":        ("OPENAI",  "GOOGLE"),
    # CR11
    "LAUNCHED":                ("NASA",    "JAMES WEBB SPACE TELESCOPE"),
    "DEPLOYED":                ("NASA",    "JAMES WEBB SPACE TELESCOPE"),
    "SENT INTO ORBIT":         ("NASA",    "JAMES WEBB SPACE TELESCOPE"),
    "PUT INTO SERVICE":        ("NASA",    "JAMES WEBB SPACE TELESCOPE"),
    # CR12
    "FUNDED":                  ("EUROPEAN UNION", "NASA"),
    "PROVIDED FUNDING FOR":    ("EUROPEAN UNION", "NASA"),
    "FINANCED":                ("EUROPEAN UNION", "NASA"),
    "GRANTED MONEY TO":        ("EUROPEAN UNION", "NASA"),
    # CR13
    "RESEARCHES":              ("DEEPMIND", "CRISPR"),
    "INVESTIGATES":            ("DEEPMIND", "CRISPR"),
    "STUDIES":                 ("DEEPMIND", "CRISPR"),
    "CONDUCTS RESEARCH ON":    ("DEEPMIND", "CRISPR"),
    # CR14
    "IMPOSES_SANCTIONS_ON":         ("UNITED STATES", "CHINA"),
    "SANCTIONED":                   ("UNITED STATES", "CHINA"),
    "IMPOSED TRADE RESTRICTIONS ON":("UNITED STATES", "CHINA"),
    "PLACED EXPORT CONTROLS ON":    ("UNITED STATES", "CHINA"),
    # CR15
    "COOPERATES_WITH":              ("UNITED STATES", "EUROPEAN UNION"),
    "COORDINATES WITH":             ("UNITED STATES", "EUROPEAN UNION"),
    "WORKS JOINTLY WITH":           ("UNITED STATES", "EUROPEAN UNION"),
    "COLLABORATES ON POLICY WITH":  ("UNITED STATES", "EUROPEAN UNION"),
    # CR16
    "OWNS":                    ("TESLA",           "BITCOIN"),
    "HOLDS":                   ("TESLA",           "BITCOIN"),
    "POSSESSES":               ("TESLA",           "BITCOIN"),
    "IS SHAREHOLDER IN":       ("TESLA",           "BITCOIN"),
    # CR17
    "SETS_INTEREST_RATES_FOR":     ("FEDERAL RESERVE", "UNITED STATES"),
    "CONTROLS MONETARY POLICY OF": ("FEDERAL RESERVE", "UNITED STATES"),
    "DETERMINES RATES FOR":        ("FEDERAL RESERVE", "UNITED STATES"),
    # CR18
    "PLAYS_FOR":               ("LIONEL MESSI", "FIFA"),
    "IS AFFILIATED WITH":      ("LIONEL MESSI", "FIFA"),
    "REPRESENTS":              ("LIONEL MESSI", "FIFA"),
    "COMPETES UNDER":          ("LIONEL MESSI", "FIFA"),
}

# Sample descriptions for each canonical type
CANONICAL_DESCRIPTIONS: dict[str, str] = {
    "CR01": "The person holds the chief executive officer position at the organization.",
    "CR02": "The subject person or group founded, created, or established the organization.",
    "CR03": "The subject entity has committed financial capital to the object company or project.",
    "CR04": "The subject organization acquired ownership of the object company through purchase.",
    "CR05": "The two parties have entered a formal strategic cooperation or partnership agreement.",
    "CR06": "The subject authority sets binding rules, standards, or laws governing the object.",
    "CR07": "The subject person holds the presidential or top leadership role at the organization.",
    "CR08": "The subject organization designed and built the object product or technology.",
    "CR09": "The subject entity employs or integrates the object product in its operations.",
    "CR10": "The two organizations compete directly against each other in the same market segment.",
    "CR11": "The subject organization launched or deployed the object spacecraft or mission.",
    "CR12": "The subject entity provides financial grants or subsidies to fund the object.",
    "CR13": "The subject organization conducts active scientific research on the object topic.",
    "CR14": "The subject country or bloc has imposed trade, financial, or export sanctions on the object.",
    "CR15": "The subject and object cooperate jointly on shared policy, scientific, or strategic goals.",
    "CR16": "The subject entity holds direct ownership or equity stakes in the object asset.",
    "CR17": "The subject central bank sets benchmark interest rates for the object economy.",
    "CR18": "The subject athlete plays for, represents, or is affiliated with the object organization.",
}


def build_ground_truth() -> dict:
    """Return the full CGRR ground truth dict."""
    canonical_map = {cr["id"]: cr for cr in CANONICAL_RELATIONS}

    can_to_aliases: dict[str, list[str]] = defaultdict(list)
    for alias, cid in ALIAS_MAP.items():
        can_to_aliases[cid].append(alias)

    expected: list[dict] = []

    # Positive: all alias pairs for the same canonical type
    for cid, aliases in can_to_aliases.items():
        aliases_sorted = sorted(aliases)
        for i in range(len(aliases_sorted)):
            for j in range(i + 1, len(aliases_sorted)):
                a, b = aliases_sorted[i], aliases_sorted[j]
                sa, ta = ALIAS_ENDPOINTS.get(a, ("", ""))
                sb, tb = ALIAS_ENDPOINTS.get(b, ("", ""))
                expected.append({
                    "alias_a": a, "alias_b": b,
                    "canonical_id": cid,
                    "canonical_type": canonical_map[cid]["type"],
                    "should_normalize": True,
                    "source_a": sa, "target_a": ta,
                    "source_b": sb, "target_b": tb,
                    "desc_a": CANONICAL_DESCRIPTIONS.get(cid, ""),
                    "desc_b": CANONICAL_DESCRIPTIONS.get(cid, ""),
                })

    # Negative: different canonical types
    for a, b in NEGATIVE_PAIRS:
        cid_a = ALIAS_MAP.get(a)
        cid_b = ALIAS_MAP.get(b)
        sa, ta = ALIAS_ENDPOINTS.get(a, ("", ""))
        sb, tb = ALIAS_ENDPOINTS.get(b, ("", ""))
        expected.append({
            "alias_a": a, "alias_b": b,
            "canonical_id": None,
            "canonical_type": None,
            "should_normalize": False,
            "source_a": sa, "target_a": ta,
            "source_b": sb, "target_b": tb,
            "desc_a": CANONICAL_DESCRIPTIONS.get(cid_a or "", ""),
            "desc_b": CANONICAL_DESCRIPTIONS.get(cid_b or "", ""),
        })

    return {
        "component": "CGRR",
        "canonical_relations": CANONICAL_RELATIONS,
        "alias_map": ALIAS_MAP,
        "alias_endpoints": ALIAS_ENDPOINTS,
        "canonical_descriptions": CANONICAL_DESCRIPTIONS,
        "expected_normalizations": expected,
        "total_canonical_relations": len(CANONICAL_RELATIONS),
        "total_positive_pairs": sum(1 for e in expected if e["should_normalize"]),
        "total_negative_pairs": sum(1 for e in expected if not e["should_normalize"]),
    }


def write_ground_truth(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    gt = build_ground_truth()
    path = os.path.join(output_dir, "cgrr_ground_truth.json")
    with open(path, "w") as f:
        json.dump(gt, f, indent=2)
    pos = gt["total_positive_pairs"]
    neg = gt["total_negative_pairs"]
    print(f"  [CGRR GT] Wrote {pos} positive + {neg} negative pairs → {path}")
