"""Inspect random entities and relationships stored in Neo4j by BT-GraphRAG.

Usage
-----
    python inspect_neo4j.py [--uri neo4j://127.0.0.1:7687] [--user neo4j]
                            [--password 12345678] [--database btgraphrag]
                            [--entities 5] [--relationships 5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import textwrap


async def _fetch(uri: str, user: str, password: str, database: str,
                 n_entities: int, n_rels: int) -> None:
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async with driver.session(database=database) as session:

        # ------------------------------------------------------------------ #
        # Entities                                                             #
        # ------------------------------------------------------------------ #
        print("\n" + "=" * 80)
        print(f"  RANDOM ENTITIES  (sample: {n_entities})")
        print("=" * 80)

        result = await session.run(
            "MATCH (n:Entity) RETURN n ORDER BY rand() LIMIT $limit",
            limit=n_entities,
        )
        records = await result.data()

        if not records:
            print("  (no entities found)")
        else:
            for i, rec in enumerate(records, 1):
                node = rec["n"]
                print(f"\n  [{i}] ── Entity ─────────────────────────────────")
                _print_props(dict(node))

        # ------------------------------------------------------------------ #
        # Relationships                                                        #
        # ------------------------------------------------------------------ #
        print("\n" + "=" * 80)
        print(f"  RANDOM RELATIONSHIPS  (sample: {n_rels})")
        print("=" * 80)

        result = await session.run(
            """
            MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
            RETURN s.title AS source, t.title AS target, properties(r) AS props
            ORDER BY rand()
            LIMIT $limit
            """,
            limit=n_rels,
        )
        records = await result.data()

        if not records:
            print("  (no relationships found)")
        else:
            for i, rec in enumerate(records, 1):
                print(f"\n  [{i}] ({rec['source']}) ──► ({rec['target']})")
                _print_props(rec["props"])

        # ------------------------------------------------------------------ #
        # Summary counts                                                       #
        # ------------------------------------------------------------------ #
        print("\n" + "=" * 80)
        print("  SUMMARY")
        print("=" * 80)

        r = await session.run("MATCH (n:Entity) RETURN count(n) AS c")
        total_entities = (await r.single())["c"]

        r = await session.run("MATCH ()-[r:RELATIONSHIP]->() RETURN count(r) AS c")
        total_rels = (await r.single())["c"]

        r = await session.run(
            "MATCH ()-[r:RELATIONSHIP]->() "
            "RETURN r.relation_type AS t, count(*) AS c "
            "ORDER BY c DESC LIMIT 10"
        )
        top_types = await r.data()

        print(f"  Total entities      : {total_entities}")
        print(f"  Total relationships : {total_rels}")
        print(f"\n  Top relation types:")
        for row in top_types:
            print(f"    {row['t']:<40s}  {row['c']:>5d} edges")

        r = await session.run(
            "MATCH (n:Entity) "
            "RETURN n.description_embedding IS NOT NULL AS has_emb, count(*) AS c"
        )
        emb_counts = {rec["has_emb"]: rec["c"] for rec in await r.data()}
        print(f"\n  Entities with description_embedding : "
              f"{emb_counts.get(True, 0)} / {total_entities}")

        r = await session.run(
            "MATCH ()-[r:RELATIONSHIP]->() "
            "RETURN r.description_embedding IS NOT NULL AS has_d, "
            "       r.relation_type_embedding IS NOT NULL AS has_t, "
            "       count(*) AS c"
        )
        rows = await r.data()
        has_desc = sum(row["c"] for row in rows if row["has_d"])
        has_type = sum(row["c"] for row in rows if row["has_t"])
        print(f"  Rels with description_embedding     : {has_desc} / {total_rels}")
        print(f"  Rels with relation_type_embedding   : {has_type} / {total_rels}")

    await driver.close()


def _print_props(props: dict) -> None:
    """Pretty-print a property dict, truncating embedding vectors."""
    for key, val in sorted(props.items()):
        if key.endswith("_embedding"):
            vec = _parse_embedding(val)
            if vec:
                preview = ", ".join(f"{v:.4f}" for v in vec[:5])
                print(f"    {key:<35s} [{preview}, ...]  (dim={len(vec)})")
            else:
                print(f"    {key:<35s} (empty)")
        elif isinstance(val, str) and len(val) > 120:
            wrapped = textwrap.shorten(val, width=120, placeholder="…")
            print(f"    {key:<35s} {wrapped}")
        else:
            print(f"    {key:<35s} {val}")


def _parse_embedding(val: object) -> list[float]:
    """Parse an embedding stored as a JSON string or a native list."""
    if isinstance(val, list):
        return [float(v) for v in val]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect BT-GraphRAG Neo4j database.")
    p.add_argument("--uri",        default="neo4j://127.0.0.1:7687")
    p.add_argument("--user",       default="neo4j")
    p.add_argument("--password",   default="12345678")
    p.add_argument("--database",   default="btgraphrag")
    p.add_argument("--entities",   type=int, default=5, metavar="N",
                   help="Number of random entities to show (default: 5)")
    p.add_argument("--relationships", type=int, default=5, metavar="N",
                   help="Number of random relationships to show (default: 5)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(_fetch(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        n_entities=args.entities,
        n_rels=args.relationships,
    ))
