# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing snapshot_graphml method definition."""

import json

import numpy as np
import networkx as nx
import pandas as pd
from graphrag_storage import Storage


def _is_list_like(val: object) -> bool:
    """Return True if *val* is a list, tuple, set, or numpy array."""
    return isinstance(val, (list, tuple, set, np.ndarray))


def _serialize_val(col: str, val: object) -> str | None:
    """Convert a column value to a GraphML-compatible string.

    Embedding columns (any name ending in ``_embedding``) are serialized
    as a compact JSON array so the full vector is preserved and can be
    parsed back.  Other list-like values are joined with ``"; "``.
    Null / NaN values return ``None`` (caller should skip them).
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if col.endswith("_embedding"):
        if _is_list_like(val):
            return json.dumps([float(v) for v in val])
        return str(val)
    if _is_list_like(val):
        return "; ".join(str(v) for v in val)
    return str(val)


async def snapshot_graphml(
    edges: pd.DataFrame,
    name: str,
    storage: Storage,
    entities: pd.DataFrame | None = None,
) -> None:
    """Take a entire snapshot of a graph to standard graphml format.

    When entities is provided, node attributes (type, description, temporal
    fields, and embeddings) are attached to every node in the graph.  Edge
    attributes include every column available in *edges*, with embedding
    vectors serialized as compact JSON arrays.
    """
    # Scalar edge columns go in directly; embedding / other list columns
    # are serialised to strings and added manually after graph construction.
    scalar_edge_attrs = [
        c for c in edges.columns
        if c not in ("source", "target")
        and not c.endswith("_embedding")
        and edges[c].apply(lambda v: not _is_list_like(v)).all()
    ]

    graph = nx.from_pandas_edgelist(
        edges, source="source", target="target",
        edge_attr=scalar_edge_attrs or ["weight"],
        create_using=nx.MultiDiGraph(),
    )

    # Add embedding and other list-like edge columns as serialised strings
    embedding_edge_cols = [
        c for c in edges.columns
        if c not in ("source", "target") and c not in scalar_edge_attrs
    ]
    if embedding_edge_cols:
        for _, row in edges.iterrows():
            src, tgt = row["source"], row["target"]
            if graph.has_edge(src, tgt):
                for col in embedding_edge_cols:
                    serialised = _serialize_val(col, row[col])
                    if serialised is not None:
                        # MultiDiGraph may have multiple edges; update all
                        for key in graph[src][tgt]:
                            graph[src][tgt][key][col] = serialised

    # Attach entity attributes (including embeddings) to nodes
    if entities is not None and not entities.empty:
        title_col = "title" if "title" in entities.columns else None
        if title_col:
            for _, row in entities.iterrows():
                node_id = row[title_col]
                if node_id in graph:
                    for col in entities.columns:
                        if col == title_col:
                            continue
                        serialised = _serialize_val(col, row[col])
                        if serialised is not None:
                            graph.nodes[node_id][col] = serialised

    graphml = "\n".join(nx.generate_graphml(graph))
    await storage.set(name + ".graphml", graphml)
