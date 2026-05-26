#!/usr/bin/env python3
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Wipe (DROP + CREATE) the Neo4j databases used by train / evaluate /
ragtest, restoring each to an empty state.

Target databases (mirrors what train.py, evaluate.py and
.ragtest/settings.yaml configure):

  - btgraphrag     # ragtest main graph
  - cgrreval       # CGER+CGRR evaluation harness
  - cgerbatch      # CGER Phase B scratch DB
  - cgrrbatch      # CGRR Phase B scratch DB
  - etcdreval      # ETCDR evaluation harness

Usage:
    python scripts/wipe_neo4j_dbs.py               # wipe all four
    python scripts/wipe_neo4j_dbs.py --dry-run     # list only, do not modify
    python scripts/wipe_neo4j_dbs.py --only btgraphrag,cgrreval

Requires:
    - Neo4j running on neo4j://127.0.0.1:7687 (override with --uri)
    - User `neo4j` / password `12345678` (override with --user / --password,
      or NEO4J_USER / NEO4J_PASSWORD env vars)
    - The user must have privileges to DROP / CREATE databases (Enterprise
      edition; the Community single-DB edition cannot run this script).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Iterable

from neo4j import AsyncGraphDatabase


DEFAULT_DBS = ["btgraphrag", "cgrreval", "cgerbatch", "cgrrbatch", "etcdreval"]
DEFAULT_URI = "neo4j://127.0.0.1:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "12345678"


async def _list_databases(session) -> dict[str, str]:
    """Return ``{name: currentStatus}`` for every database visible to the user."""
    result = await session.run("SHOW DATABASES YIELD name, currentStatus")
    out: dict[str, str] = {}
    async for rec in result:
        out[rec["name"]] = rec["currentStatus"]
    return out


async def _wait_until_status(
    drv, db_name: str, target: str, timeout_seconds: float = 30.0,
) -> str:
    """Poll SHOW DATABASES until *db_name* reports *target* (or timeout)."""
    waited = 0.0
    step = 0.4
    last = "(unknown)"
    while waited < timeout_seconds:
        async with drv.session(database="system") as session:
            status = (await _list_databases(session)).get(db_name, "(missing)")
        last = status
        if status == target:
            return status
        await asyncio.sleep(step)
        waited += step
    return last


async def _drop_create(drv, db_name: str) -> None:
    """DROP + CREATE one database, waiting for it to come back online."""
    async with drv.session(database="system") as session:
        await session.run(f"DROP DATABASE `{db_name}` IF EXISTS")
    print(f"  [{db_name}] dropped; recreating…")
    async with drv.session(database="system") as session:
        await session.run(f"CREATE DATABASE `{db_name}` IF NOT EXISTS")
    status = await _wait_until_status(drv, db_name, target="online")
    if status == "online":
        print(f"  [{db_name}] online")
    else:
        print(f"  [{db_name}] WARN: final status='{status}' (expected 'online')")


async def _verify_empty(drv, db_name: str) -> tuple[int, int, list[str]]:
    """Return (node_count, rel_count, user_index_names) for *db_name*.

    User indexes exclude Neo4j's two built-in token-lookup indexes that
    are auto-created on every new database.
    """
    async with drv.session(database=db_name) as session:
        nodes = (await (await session.run(
            "MATCH (n) RETURN count(n) AS c"
        )).single())["c"]
        rels = (await (await session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        )).single())["c"]
        idx_result = await session.run(
            "SHOW INDEXES YIELD name, type "
            "WHERE type <> 'LOOKUP' RETURN name"
        )
        user_indexes = [rec["name"] async for rec in idx_result]
    return nodes, rels, user_indexes


async def main(
    uri: str, user: str, password: str,
    dbs: Iterable[str], dry_run: bool,
) -> int:
    drv = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with drv.session(database="system") as session:
            existing = await _list_databases(session)

        print(f"Connected to {uri}")
        print(f"Visible databases: {sorted(existing)}")

        target = list(dbs)
        missing = [d for d in target if d not in existing]
        present = [d for d in target if d in existing]

        if missing:
            print(f"NOTE: the following targets do not exist yet "
                  f"(will be CREATEd): {missing}")

        print(f"\nWill DROP + CREATE: {target}")
        if dry_run:
            print("(--dry-run set; no changes performed)")
            return 0

        for db_name in target:
            if db_name in present:
                print(f"\n[wipe] {db_name} (status={existing[db_name]})")
                await _drop_create(drv, db_name)
            else:
                print(f"\n[create] {db_name} (was missing)")
                async with drv.session(database="system") as session:
                    await session.run(f"CREATE DATABASE `{db_name}` IF NOT EXISTS")
                status = await _wait_until_status(drv, db_name, target="online")
                print(f"  [{db_name}] {status}")

        print("\nVerification:")
        print(f"  {'database':14s}  {'nodes':>5s}  {'rels':>5s}  user_indexes")
        all_clean = True
        for db_name in target:
            try:
                n, r, idx = await _verify_empty(drv, db_name)
            except Exception as e:
                print(f"  {db_name:14s}  ERROR: {e}")
                all_clean = False
                continue
            ok = (n == 0 and r == 0 and not idx)
            marker = "" if ok else "  <- NOT CLEAN"
            print(f"  {db_name:14s}  {n:>5d}  {r:>5d}  {idx}{marker}")
            if not ok:
                all_clean = False

        print("\nDone." if all_clean else
              "\nDone (with warnings — see verification above).")
        return 0 if all_clean else 1
    finally:
        await drv.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DROP + CREATE the Neo4j databases used by train / "
                    "evaluate / ragtest.",
    )
    p.add_argument("--uri", default=os.environ.get("NEO4J_URI", DEFAULT_URI),
                   help=f"Neo4j Bolt URI (default: {DEFAULT_URI}, "
                        f"or $NEO4J_URI)")
    p.add_argument("--user", default=os.environ.get("NEO4J_USER", DEFAULT_USER),
                   help=f"Neo4j user (default: {DEFAULT_USER}, "
                        f"or $NEO4J_USER)")
    p.add_argument("--password",
                   default=os.environ.get("NEO4J_PASSWORD", DEFAULT_PASSWORD),
                   help=f"Neo4j password (default: hidden, "
                        f"or $NEO4J_PASSWORD)")
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of databases to wipe. "
                        f"Default: {','.join(DEFAULT_DBS)}")
    p.add_argument("--dry-run", action="store_true",
                   help="List the target databases without modifying anything.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.only:
        dbs = [d.strip() for d in args.only.split(",") if d.strip()]
        unknown = [d for d in dbs if d not in DEFAULT_DBS]
        if unknown:
            print(f"[WARN] --only includes names outside the known set "
                  f"({DEFAULT_DBS}): {unknown}", file=sys.stderr)
    else:
        dbs = DEFAULT_DBS

    exit(asyncio.run(main(
        uri=args.uri, user=args.user, password=args.password,
        dbs=dbs, dry_run=args.dry_run,
    )))
