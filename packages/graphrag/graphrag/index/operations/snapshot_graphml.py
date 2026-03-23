# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing snapshot_graphml method definition."""

import numpy as np
import networkx as nx
import pandas as pd
from graphrag_storage import Storage


def _is_list_like(val: object) -> bool:
    """Return True if *val* is a list, tuple, set, or numpy array."""
    return isinstance(val, (list, tuple, set, np.ndarray))


async def snapshot_graphml(
    edges: pd.DataFrame,
    name: str,
    storage: Storage,
    entities: pd.DataFrame | None = None,
) -> None:
    """Take a entire snapshot of a graph to standard graphml format.

    When entities is provided, node attributes (type, description, temporal
    fields) are attached to every node in the graph.  Edge attributes always
    include every non-list column available in *edges*.
    """
    # Decide which edge columns to include (skip list-typed columns)
    edge_attrs = [
        c for c in edges.columns
        if c not in ("source", "target")
        and edges[c].apply(lambda v: not _is_list_like(v)).all()
    ]

    graph = nx.from_pandas_edgelist(
        edges, source="source", target="target", edge_attr=edge_attrs or ["weight"],
        create_using=nx.MultiDiGraph(),
    )

    # Attach entity attributes to nodes if available
    if entities is not None and not entities.empty:
        title_col = "title" if "title" in entities.columns else None
        if title_col:
            for _, row in entities.iterrows():
                node_id = row[title_col]
                if node_id in graph:
                    for col in entities.columns:
                        if col == title_col:
                            continue
                        val = row[col]
                        if _is_list_like(val):
                            val = "; ".join(str(v) for v in val)
                        elif val is None or (isinstance(val, float) and np.isnan(val)):
                            continue
                        graph.nodes[node_id][col] = str(val)

    graphml = "\n".join(nx.generate_graphml(graph))
    await storage.set(name + ".graphml", graphml)
