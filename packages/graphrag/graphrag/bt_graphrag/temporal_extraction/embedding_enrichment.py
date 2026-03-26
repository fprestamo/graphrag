# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Embedding enrichment for extracted entities and relationships.

Computes description_embedding and relation_type_embedding immediately
after the LLM extraction step, so every downstream stage (CGER, CGRR,
Neo4j store) has embeddings available without an extra pass.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from graphrag_llm.embedding import LLMEmbedding


def normalize_relation_type_text(relation_type: str) -> str:
    """Convert UPPER_SNAKE_CASE relation type to lowercase readable text.

    Used to produce a human-readable string for embedding, so the vector
    captures semantic meaning rather than surface syntax.

    Examples
    --------
    >>> normalize_relation_type_text("IS_CEO_OF")
    'is ceo of'
    >>> normalize_relation_type_text("ACQUIRED")
    'acquired'
    """
    return relation_type.replace("_", " ").lower()


async def embed_dataframes(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    embedding_model: "LLMEmbedding",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and attach embeddings to extracted entity and relationship DataFrames.

    Three embedding columns are produced in a single batched API call:

    * ``entities_df["description_embedding"]``  — from ``description``
    * ``relationships_df["description_embedding"]`` — from ``description``
    * ``relationships_df["relation_type_embedding"]`` — from
      :func:`normalize_relation_type_text` applied to ``relation_type``
      (only when the column is present, i.e. new extraction format)

    Empty DataFrames and rows with empty text are handled gracefully;
    empty texts receive an empty list ``[]`` as their embedding.

    Parameters
    ----------
    entities_df:
        DataFrame produced by the temporal extraction parser.
    relationships_df:
        DataFrame produced by the temporal extraction parser.
    embedding_model:
        A ``graphrag_llm.embedding.LLMEmbedding`` instance.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Updated (entities_df, relationships_df) with embedding columns added.
    """
    entity_descs: list[str] = (
        entities_df["description"].fillna("").tolist()
        if not entities_df.empty
        else []
    )
    rel_descs: list[str] = (
        relationships_df["description"].fillna("").tolist()
        if not relationships_df.empty
        else []
    )

    has_rel_types = (
        not relationships_df.empty
        and "relation_type" in relationships_df.columns
    )
    rel_type_texts: list[str] = (
        [
            normalize_relation_type_text(rt)
            for rt in relationships_df["relation_type"].fillna("").tolist()
        ]
        if has_rel_types
        else []
    )

    all_texts = entity_descs + rel_descs + rel_type_texts

    # Nothing to embed — return DataFrames unchanged
    if not all_texts or all(not t for t in all_texts):
        return entities_df, relationships_df

    # Single batched embedding call — minimises API round-trips
    response = await embedding_model.embedding_async(input=all_texts)
    all_embeddings: list[list[float]] = response.embeddings

    n_ent = len(entity_descs)
    n_rel = len(rel_descs)

    ent_embeddings = all_embeddings[:n_ent]
    rel_desc_embeddings = all_embeddings[n_ent : n_ent + n_rel]
    rel_type_embeddings = all_embeddings[n_ent + n_rel :] if has_rel_types else []

    if n_ent > 0:
        entities_df = entities_df.copy()
        entities_df["description_embedding"] = ent_embeddings

    if n_rel > 0:
        relationships_df = relationships_df.copy()
        relationships_df["description_embedding"] = rel_desc_embeddings
        if has_rel_types:
            relationships_df["relation_type_embedding"] = rel_type_embeddings

    return entities_df, relationships_df


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Return the element-wise mean of a list of equal-length vectors.

    Returns an empty list if *vectors* is empty or all entries are empty.
    """
    valid = [v for v in vectors if v]
    if not valid:
        return []
    dim = len(valid[0])
    total = [0.0] * dim
    for v in valid:
        for i, x in enumerate(v):
            total[i] += x
    n = len(valid)
    return [x / n for x in total]


async def enrich_entities_with_text_unit_embeddings(
    entities_df: pd.DataFrame,
    text_units_df: pd.DataFrame,
    embedding_model: "LLMEmbedding",
) -> pd.DataFrame:
    """Add ``text_unit_embedding`` to each entity row.

    For every entity the function:

    1. Collects the ``text`` of all text units whose ID appears in the
       entity's ``text_unit_ids``.
    2. Embeds those texts in a single batched API call.
    3. Averages the resulting vectors into one representative vector and
       stores it as ``entities_df["text_unit_embedding"]``.

    If an entity cites no text units, or the text-units DataFrame has no
    ``text`` column, the field is set to ``None`` for that entity.

    Parameters
    ----------
    entities_df:
        Entity DataFrame — must have a ``text_unit_ids`` column.
    text_units_df:
        Text-units DataFrame — must have ``id`` and ``text`` columns.
    embedding_model:
        A ``graphrag_llm.embedding.LLMEmbedding`` instance.

    Returns
    -------
    pd.DataFrame
        *entities_df* with a new ``text_unit_embedding`` column added.
    """
    if entities_df.empty:
        return entities_df

    if "text_unit_ids" not in entities_df.columns or "text" not in text_units_df.columns:
        entities_df = entities_df.copy()
        entities_df["text_unit_embedding"] = None
        return entities_df

    # Build a fast lookup: text_unit_id -> text
    tu_text: dict[str, str] = dict(
        zip(text_units_df["id"].astype(str), text_units_df["text"].fillna(""))
    )

    # For each entity collect its cited texts (deduplicated, non-empty)
    entity_cited_texts: list[list[str]] = []
    for _, row in entities_df.iterrows():
        ids = row.get("text_unit_ids") or []
        if not isinstance(ids, list):
            try:
                import ast
                ids = ast.literal_eval(str(ids))
            except Exception:
                ids = []
        texts = list(dict.fromkeys(
            tu_text[str(tid)] for tid in ids
            if str(tid) in tu_text and tu_text[str(tid)]
        ))
        entity_cited_texts.append(texts)

    # Flatten all unique texts to embed in one batch
    all_unique_texts: list[str] = list(dict.fromkeys(
        t for texts in entity_cited_texts for t in texts
    ))

    if not all_unique_texts:
        entities_df = entities_df.copy()
        entities_df["text_unit_embedding"] = None
        return entities_df

    response = await embedding_model.embedding_async(input=all_unique_texts)
    text_to_emb: dict[str, list[float]] = dict(
        zip(all_unique_texts, response.embeddings)
    )

    # Compute per-entity mean vector
    text_unit_embeddings: list[list[float] | None] = []
    for texts in entity_cited_texts:
        vecs = [text_to_emb[t] for t in texts if t in text_to_emb]
        mean = _mean_vector(vecs)
        text_unit_embeddings.append(mean if mean else None)

    entities_df = entities_df.copy()
    entities_df["text_unit_embedding"] = text_unit_embeddings
    return entities_df
