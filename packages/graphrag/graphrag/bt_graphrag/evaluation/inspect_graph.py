# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Inspect random entities and relationships from the btgraphrag Neo4j database.

Fetches a sample of Entity nodes and RELATIONSHIP edges (with all stored
properties) and prints them to stdout.

Usage:
    python packages/graphrag/graphrag/bt_graphrag/evaluation/inspect_graph.py

Optional env vars:
    NEO4J_URI        – default: neo4j://127.0.0.1:7687
    NEO4J_USER       – default: neo4j
    NEO4J_PASSWORD   – default: 12345678
    NEO4J_DATABASE   – default: btgraphrag
    SAMPLE_SIZE      – number of nodes/edges to fetch (default: 1)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installing the package
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[6]
_PACKAGES = _REPO_ROOT / "packages" / "graphrag"
if str(_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_PACKAGES))

from neo4j import AsyncGraphDatabase  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "12345678")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "btgraphrag")
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "1"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_props(props: dict, indent: int = 4) -> None:
    pad = " " * indent
    for k, v in sorted(props.items()):
        # Truncate very long strings for readability
        if isinstance(v, str) and len(v) > 120:
            v = v[:117] + "..."
        print(f"{pad}{k}: {json.dumps(v, default=str)}")


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

ENTITY_QUERY = """
MATCH (e:Entity)
WITH e, rand() AS r
ORDER BY r
LIMIT $limit
RETURN properties(e) AS props
"""

RELATIONSHIP_QUERY = """
MATCH (s:Entity)-[r:RELATIONSHIP]->(o:Entity)
WITH s, r, o, rand() AS rnd
ORDER BY rnd
LIMIT $limit
RETURN
    s.title        AS source,
    o.title        AS target,
    properties(r)  AS props
"""


async def main() -> None:
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    async with driver.session(database=NEO4J_DATABASE) as session:

        # ── Entities ────────────────────────────────────────────────────────
        _sep(f"ENTITIES  (random sample, n={SAMPLE_SIZE})")
        result = await session.run(ENTITY_QUERY, limit=SAMPLE_SIZE)
        records = await result.data()

        if not records:
            print("  [no Entity nodes found in the database]")
        else:
            for i, rec in enumerate(records, 1):
                props = rec["props"]
                title = props.get("title", props.get("id", f"entity-{i}"))
                print(f"\n  [{i}] {title}")
                _print_props(props)

        # ── Relationships ────────────────────────────────────────────────────
        _sep(f"RELATIONSHIPS  (random sample, n={SAMPLE_SIZE})")
        result = await session.run(RELATIONSHIP_QUERY, limit=SAMPLE_SIZE)
        records = await result.data()

        if not records:
            print("  [no RELATIONSHIP edges found in the database]")
        else:
            for i, rec in enumerate(records, 1):
                src = rec["source"]
                tgt = rec["target"]
                props = rec["props"]
                rel_type = props.get("relation_type", "?")
                print(f"\n  [{i}] ({src}) -[{rel_type}]-> ({tgt})")
                _print_props(props)

    await driver.close()
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
