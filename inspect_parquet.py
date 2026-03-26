"""Inspect random entities and relationships from GraphRAG Parquet storage.

Usage
-----
    python inspect_parquet.py [--output ragtest/output]
                              [--entities 5] [--relationships 5]
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _is_embedding_col(col: str) -> bool:
    return col.endswith("_embedding")


def _parse_embedding(val: object) -> list[float]:
    """Parse an embedding stored as a list, numpy array, or JSON string."""
    if isinstance(val, (list, np.ndarray)):
        return [float(v) for v in val]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _fmt_val(col: str, val: object) -> str:
    """Format a single cell value for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "(null)"
    if _is_embedding_col(col):
        vec = _parse_embedding(val)
        if vec:
            preview = ", ".join(f"{v:.4f}" for v in vec[:5])
            return f"[{preview}, ...]  (dim={len(vec)})"
        return "(empty)"
    if isinstance(val, (list, np.ndarray)):
        items = [str(v) for v in val]
        joined = ", ".join(items[:8])
        suffix = f", … +{len(items)-8}" if len(items) > 8 else ""
        return f"[{joined}{suffix}]"
    text = str(val)
    if len(text) > 120:
        return textwrap.shorten(text, width=120, placeholder="…")
    return text


def _print_row(row: pd.Series, skip_cols: set[str] | None = None) -> None:
    skip = skip_cols or set()
    for col in row.index:
        if col in skip:
            continue
        print(f"    {col:<35s} {_fmt_val(col, row[col])}")


def _separator(label: str, width: int = 80) -> None:
    print("\n" + "=" * width)
    print(f"  {label}")
    print("=" * width)


# --------------------------------------------------------------------------- #
# Main inspection                                                               #
# --------------------------------------------------------------------------- #

def inspect(output_dir: str, n_entities: int, n_rels: int) -> None:
    entities_path = os.path.join(output_dir, "entities.parquet")
    rels_path     = os.path.join(output_dir, "relationships.parquet")

    if not os.path.exists(entities_path):
        print(f"ERROR: {entities_path} not found.")
        return
    if not os.path.exists(rels_path):
        print(f"ERROR: {rels_path} not found.")
        return

    entities = pd.read_parquet(entities_path)
    rels     = pd.read_parquet(rels_path)

    # ------------------------------------------------------------------ #
    # Entities                                                             #
    # ------------------------------------------------------------------ #
    _separator(f"RANDOM ENTITIES  (sample: {n_entities} of {len(entities)})")
    print(f"  Columns: {list(entities.columns)}\n")

    sample_ent = entities.sample(min(n_entities, len(entities)), random_state=None)
    for i, (_, row) in enumerate(sample_ent.iterrows(), 1):
        title = row.get("title", row.get("id", f"row-{i}"))
        print(f"\n  [{i}] ── {title} ─────────────────────────────────")
        _print_row(row)

    # ------------------------------------------------------------------ #
    # Relationships                                                        #
    # ------------------------------------------------------------------ #
    _separator(f"RANDOM RELATIONSHIPS  (sample: {n_rels} of {len(rels)})")
    print(f"  Columns: {list(rels.columns)}\n")

    sample_rel = rels.sample(min(n_rels, len(rels)), random_state=None)
    for i, (_, row) in enumerate(sample_rel.iterrows(), 1):
        src = row.get("source", "?")
        tgt = row.get("target", "?")
        rtype = row.get("relation_type", "")
        label = f"({src}) -[{rtype}]→ ({tgt})" if rtype else f"({src}) → ({tgt})"
        print(f"\n  [{i}] {label}")
        _print_row(row, skip_cols={"source", "target", "relation_type"})

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    _separator("SUMMARY")

    print(f"  Entities      : {len(entities)}")
    print(f"  Relationships : {len(rels)}")

    # Embedding coverage
    for col in entities.columns:
        if _is_embedding_col(col):
            has = entities[col].apply(
                lambda v: bool(_parse_embedding(v)) if v is not None else False
            ).sum()
            print(f"\n  entities[{col}]  : {has} / {len(entities)} have vectors")

    for col in rels.columns:
        if _is_embedding_col(col):
            has = rels[col].apply(
                lambda v: bool(_parse_embedding(v)) if v is not None else False
            ).sum()
            print(f"  rels[{col}]  : {has} / {len(rels)} have vectors")

    # Top relation types
    if "relation_type" in rels.columns:
        print(f"\n  Top relation types:")
        for rtype, cnt in rels["relation_type"].value_counts().head(10).items():
            print(f"    {str(rtype):<40s}  {cnt:>5d} edges")

    # Temporal span
    for df, label in [(entities, "entities"), (rels, "rels")]:
        for tcol in ("active_start", "t_valid_start"):
            if tcol in df.columns:
                vals = df[tcol].dropna()
                if not vals.empty:
                    print(f"\n  {label}[{tcol}]  min={vals.min()}  max={vals.max()}")

    # Entity type distribution
    if "type" in entities.columns:
        print(f"\n  Entity type distribution:")
        for etype, cnt in entities["type"].value_counts().head(10).items():
            print(f"    {str(etype):<30s}  {cnt:>5d}")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect GraphRAG Parquet entities and relationships."
    )
    p.add_argument(
        "--output", default="ragtest/output",
        help="Path to the GraphRAG output directory (default: ragtest/output)",
    )
    p.add_argument("--entities", type=int, default=5, metavar="N",
                   help="Number of random entities to show (default: 5)")
    p.add_argument("--relationships", type=int, default=5, metavar="N",
                   help="Number of random relationships to show (default: 5)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    inspect(args.output, args.entities, args.relationships)
