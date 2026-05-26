# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Stage 4: Bitemporal Graph Store backed by Neo4j.

Every edge carries a Temporal State Quad: [t_valid_start, t_valid_end,
t_tx_start, t_tx_end]. SCD2 non-destructive operations ensure no
information is ever physically deleted.

Composite indexes on both (subject, relation, t_valid_start, t_tx_end) and
(object, relation, t_valid_start, t_tx_end) support efficient bidirectional
conflict queries.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from graphrag.bt_graphrag.models.temporal_types import (
    INFINITY,
    INFINITY_ISO,
    TemporalEntity,
    TemporalRelationship,
    TemporalStateQuad,
    utcnow,
)

if TYPE_CHECKING:
    from neo4j import AsyncDriver, AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

_INIT_CYPHER = [
    # Indexes for entity lookup
    "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.title)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.id)",
    # Composite indexes for efficient subject-side conflict queries
    "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.t_valid_start)",
    "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.t_tx_end)",
    "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.relation_type)",
    "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.id)",
]

# Template — dimensions filled at runtime from config
_VECTOR_INDEX_CYPHER = """
CREATE VECTOR INDEX entity_description_embedding IF NOT EXISTS
FOR (n:Entity) ON (n.description_embedding)
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {dimensions},
    `vector.similarity_function`: 'cosine'
  }}
}}
"""

_RELATIONSHIP_VECTOR_INDEX_CYPHER = """
CREATE VECTOR INDEX relationship_description_embedding IF NOT EXISTS
FOR ()-[r:RELATIONSHIP]-() ON (r.description_embedding)
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {dimensions},
    `vector.similarity_function`: 'cosine'
  }}
}}
"""


async def init_schema(
    driver: "AsyncDriver",
    database: str = "neo4j",
    vector_dimensions: int = 3072,
) -> None:
    """Create required indexes and constraints in Neo4j.

    Includes vector indexes on Entity.description_embedding (for CGER)
    and RELATIONSHIP.description_embedding (for CGRR) for efficient
    ANN similarity search.
    """
    async with driver.session(database=database) as session:
        for cypher in _INIT_CYPHER:
            try:
                await session.run(cypher)
            except Exception:
                logger.debug("Index may already exist: %s", cypher)
        # Vector index for CGER top-K retrieval (Entity nodes)
        try:
            vec_cypher = _VECTOR_INDEX_CYPHER.format(dimensions=vector_dimensions)
            await session.run(vec_cypher)
            logger.info("BT-GraphRAG: Created entity vector index (dim=%d)", vector_dimensions)
        except Exception:
            logger.debug("Entity vector index may already exist or Neo4j version does not support it")
        # Vector index for CGRR top-K retrieval (Relationship edges)
        try:
            rel_vec_cypher = _RELATIONSHIP_VECTOR_INDEX_CYPHER.format(dimensions=vector_dimensions)
            await session.run(rel_vec_cypher)
            logger.info("BT-GraphRAG: Created relationship vector index (dim=%d)", vector_dimensions)
        except Exception:
            logger.debug("Relationship vector index may already exist or Neo4j version does not support it")
    logger.info("BT-GraphRAG: Neo4j schema initialized")


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------


async def upsert_entity(
    session: "AsyncSession",
    entity: TemporalEntity,
    verbose: bool = True,
) -> str:
    """Insert or update an entity node in Neo4j.

    If the node exists (by title), merge and update temporal fields.
    Otherwise create a new node. Returns "CREATED" or "MATCHED".
    """
    props = entity.to_neo4j_properties()
    result = await session.run(
        """
        MERGE (n:Entity {title: $title})
        ON CREATE SET
            n.id = $id,
            n.type = $type,
            n.description = $description,
            n.first_seen = $first_seen,
            n.last_seen = $last_seen,
            n.active_start = $active_start,
            n.active_end = $active_end,
            n.description_embedding = $description_embedding,
            n._action = 'CREATED'
        ON MATCH SET
            n.description = CASE
                WHEN n.description IS NULL OR size(n.description) < size($description) THEN $description
                ELSE n.description
            END,
            n.last_seen = $last_seen,
            n.active_end = CASE WHEN $active_end = $infinity THEN n.active_end ELSE $active_end END,
            n.description_embedding = CASE
                WHEN $description_embedding IS NOT NULL THEN $description_embedding
                ELSE n.description_embedding
            END,
            n._action = 'MATCHED'
        RETURN n._action AS action
        """,
        title=props["title"],
        id=props["id"],
        type=props["type"],
        description=props["description"],
        first_seen=props.get("first_seen"),
        last_seen=props.get("last_seen"),
        active_start=props.get("active_start"),
        active_end=props.get("active_end"),
        description_embedding=props.get("description_embedding"),
        infinity=INFINITY_ISO,
    )
    record = await result.single()
    action = record["action"] if record else "UNKNOWN"

    # Clean up temporary property
    await session.run(
        "MATCH (n:Entity {title: $title}) REMOVE n._action",
        title=props["title"],
    )

    return action


async def upsert_entities_batch(
    session: "AsyncSession",
    entities: list[TemporalEntity],
) -> None:
    """Batch-upsert a list of entities."""
    for entity in entities:
        await upsert_entity(session, entity)


# ---------------------------------------------------------------------------
# Relationship operations (SCD2 non-destructive)
# ---------------------------------------------------------------------------


async def insert_relationship(
    session: "AsyncSession",
    rel: TemporalRelationship,
) -> str:
    """Insert a new relationship edge with full Temporal State Quad.

    Never overwrites existing edges — creates a new edge record.
    Returns the edge ID.
    """
    if not rel.id:
        rel.id = str(uuid4())

    props = rel.to_neo4j_properties()

    # Verify source and target entities exist before inserting
    check_result = await session.run(
        """
        OPTIONAL MATCH (s:Entity {title: $source})
        OPTIONAL MATCH (t:Entity {title: $target})
        RETURN s IS NOT NULL AS source_exists, t IS NOT NULL AS target_exists
        """,
        source=rel.source,
        target=rel.target,
    )
    check_rec = await check_result.single()
    source_ok = check_rec["source_exists"] if check_rec else False
    target_ok = check_rec["target_exists"] if check_rec else False

    if not source_ok or not target_ok:
        missing = []
        if not source_ok:
            missing.append(f"source='{rel.source[:30]}'")
        if not target_ok:
            missing.append(f"target='{rel.target[:30]}'")
        logger.warning(
            "insert_relationship: Missing node(s) %s — edge %s will be skipped",
            ", ".join(missing), rel.id,
        )
        print(f"      WARNING: Skipping edge — missing node(s): {', '.join(missing)}")
        return rel.id

    await session.run(
        """
        MATCH (s:Entity {title: $source})
        MATCH (t:Entity {title: $target})
        CREATE (s)-[r:RELATIONSHIP]->(t)
        SET r = $props
        """,
        source=rel.source,
        target=rel.target,
        props=props,
    )

    # Derive epistemic state for display
    quad = rel.temporal_quad
    if quad:
        state = quad.epistemic_state.value
    else:
        state = "UNKNOWN"

    logger.debug(
        "Inserted edge %s: (%s)-[%s]->(%s) [%s]",
        rel.id, rel.source, rel.relation_type, rel.target, state,
    )
    return rel.id


async def insert_relationships_batch(
    session: "AsyncSession",
    relationships: list[TemporalRelationship],
) -> list[str]:
    """Batch-insert relationships."""
    ids = []
    for rel in relationships:
        edge_id = await insert_relationship(session, rel)
        ids.append(edge_id)
    return ids


# ---------------------------------------------------------------------------
# Temporal queries
# ---------------------------------------------------------------------------


async def find_entities_by_titles(
    session: "AsyncSession",
    titles: list[str],
) -> list[dict[str, Any]]:
    """Resolve a list of free-form entity names to graph entities.

    Strategy (cheap → expensive):
      1. Exact (case-sensitive) title match.
      2. Case-insensitive normalized match.
      3. Substring containment (either direction).

    Returns one record per resolved entity with: title, type, description,
    active_start, active_end, and ``_query_term`` (the original input name).
    Duplicates are removed by title; first hit wins.
    """
    if not titles:
        return []

    # Build normalized variants once
    norm_pairs = [(t, t.strip()) for t in titles if t and t.strip()]
    if not norm_pairs:
        return []

    raw_titles = [t for t, _ in norm_pairs]
    norm_titles = [n for _, n in norm_pairs]
    lower_titles = [n.lower() for n in norm_titles]

    query = """
    UNWIND range(0, size($raw) - 1) AS i
    WITH i, $raw[i] AS query_term, $norm[i] AS norm_term, $lower[i] AS lower_term
    OPTIONAL MATCH (e:Entity)
    WHERE e.title = norm_term
       OR toLower(e.title) = lower_term
       OR toLower(e.title) CONTAINS lower_term
       OR lower_term CONTAINS toLower(e.title)
    WITH query_term, e
    WHERE e IS NOT NULL
    RETURN query_term, properties(e) AS props
    """
    result = await session.run(
        query, raw=raw_titles, norm=norm_titles, lower=lower_titles
    )
    seen_titles: set[str] = set()
    records: list[dict[str, Any]] = []
    async for record in result:
        entity = dict(record["props"])
        title = entity.get("title")
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        entity["_query_term"] = record["query_term"]
        records.append(entity)
    return records


async def get_subgraph_around_entities(
    session: "AsyncSession",
    entity_titles: list[str],
    k_hop: int = 1,
    valid_at: datetime | None = None,
    valid_range: tuple[datetime, datetime] | None = None,
    tx_at: datetime | None = None,
    include_disputed: bool = True,
    relation_types: list[str] | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Retrieve a temporally-filtered k-hop subgraph around seed entities.

    Bitemporal filtering:
      - ``valid_at``: edge must satisfy ``t_valid_start <= valid_at < t_valid_end``.
      - ``valid_range = (start, end)``: edge valid period must OVERLAP ``[start, end)``.
      - ``tx_at``: edge must satisfy ``t_tx_start <= tx_at < t_tx_end``.
        Defaults to "currently believed" (``t_tx_end = INFINITY``) when not given.

    Returns ``{"edges": [...], "entities": [...], "seed_titles": [...]}``.
    """
    if not entity_titles:
        return {"edges": [], "entities": [], "seed_titles": []}

    # K-hop neighbourhood via the variable-length pattern.  We expand both
    # directions because RAG-style retrieval needs neighbours regardless of
    # edge direction.
    k_hop = max(1, min(k_hop, 3))

    params: dict[str, Any] = {
        "seeds": entity_titles,
        "infinity": INFINITY_ISO,
        "limit": int(limit),
    }
    conditions: list[str] = []

    if tx_at is not None:
        conditions.append("r.t_tx_start <= $tx_at AND r.t_tx_end > $tx_at")
        params["tx_at"] = tx_at.isoformat()
    else:
        conditions.append("r.t_tx_end = $infinity")

    if valid_at is not None:
        conditions.append(
            "r.t_valid_start <= $valid_at AND r.t_valid_end > $valid_at"
        )
        params["valid_at"] = valid_at.isoformat()
    elif valid_range is not None:
        # Edge valid period overlaps [range_start, range_end)
        range_start, range_end = valid_range
        conditions.append(
            "r.t_valid_start < $valid_end AND r.t_valid_end > $valid_start"
        )
        params["valid_start"] = range_start.isoformat()
        params["valid_end"] = range_end.isoformat()

    if not include_disputed:
        conditions.append("r.status <> 'disputed'")

    if relation_types:
        conditions.append("r.relation_type IN $relation_types")
        params["relation_types"] = relation_types

    where_clause = " AND ".join(conditions) if conditions else "true"

    # Two-step query: first find candidate edges within k_hop of seeds, then
    # return them along with the participating entities.  We use APOC-style
    # variable-length patterns and rely on the seed list to anchor the path.
    cypher = f"""
    MATCH (seed:Entity)
    WHERE seed.title IN $seeds
    CALL {{
        WITH seed
        MATCH path = (seed)-[:RELATIONSHIP*1..{k_hop}]-(:Entity)
        UNWIND relationships(path) AS r
        WITH DISTINCT r
        RETURN r
    }}
    WITH r
    MATCH (s:Entity)-[r]->(t:Entity)
    WHERE {where_clause}
    WITH DISTINCT r, s, t
    RETURN s.title AS source, t.title AS target,
           properties(r) AS edge_props,
           properties(s) AS source_props,
           properties(t) AS target_props
    LIMIT $limit
    """

    result = await session.run(cypher, **params)

    edges: list[dict[str, Any]] = []
    entities_seen: dict[str, dict[str, Any]] = {}
    async for record in result:
        edge = dict(record["edge_props"])
        edge["source"] = record["source"]
        edge["target"] = record["target"]
        edges.append(edge)
        for ent_props_key, title in (
            ("source_props", record["source"]),
            ("target_props", record["target"]),
        ):
            if title and title not in entities_seen:
                entities_seen[title] = dict(record[ent_props_key])

    return {
        "edges": edges,
        "entities": list(entities_seen.values()),
        "seed_titles": list(entity_titles),
    }


async def get_active_edges(
    session: "AsyncSession",
    subject: str | None = None,
    obj: str | None = None,
    relation_type: str | None = None,
    at_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """Query edges that are active (believed and valid) at a given time.

    If at_time is None, queries the current state (t_valid_end = INFINITY AND t_tx_end = INFINITY).
    """
    conditions = []
    params: dict[str, Any] = {}

    if at_time:
        conditions.extend([
            "r.t_valid_start <= $at_time",
            "r.t_valid_end > $at_time",
            "r.t_tx_start <= $at_time",
            "r.t_tx_end > $at_time",
        ])
        params["at_time"] = at_time.isoformat()
    else:
        conditions.extend([
            "r.t_valid_end = $infinity",
            "r.t_tx_end = $infinity",
        ])
        params["infinity"] = INFINITY_ISO

    if subject:
        conditions.append("s.title = $subject")
        params["subject"] = subject
    if obj:
        conditions.append("t.title = $obj")
        params["obj"] = obj
    if relation_type:
        conditions.append("r.relation_type = $relation_type")
        params["relation_type"] = relation_type

    where_clause = " AND ".join(conditions) if conditions else "true"

    query = f"""
    MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
    WHERE {where_clause}
    RETURN s.title AS source, t.title AS target,
           properties(r) AS edge_props
    """

    result = await session.run(query, **params)
    records = []
    async for record in result:
        edge = dict(record["edge_props"])
        edge["source"] = record["source"]
        edge["target"] = record["target"]
        records.append(edge)
    return records


async def get_entity_history(
    session: "AsyncSession",
    entity_title: str,
) -> list[dict[str, Any]]:
    """Get all edges (including closed/retracted) involving an entity."""
    query = """
    MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
    WHERE s.title = $title OR t.title = $title
    RETURN s.title AS source, t.title AS target,
           properties(r) AS edge_props
    ORDER BY r.t_valid_start
    """
    result = await session.run(query, title=entity_title)
    records = []
    async for record in result:
        edge = dict(record["edge_props"])
        edge["source"] = record["source"]
        edge["target"] = record["target"]
        records.append(edge)
    return records


async def get_system_state_at(
    session: "AsyncSession",
    tx_time: datetime,
) -> list[dict[str, Any]]:
    """Temporal Audit: what did the system believe at tx_time?

    Returns all edges that were believed at the given transaction time,
    regardless of their valid-time status.
    """
    query = """
    MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
    WHERE r.t_tx_start <= $tx_time
      AND r.t_tx_end > $tx_time
    RETURN s.title AS source, t.title AS target,
           properties(r) AS edge_props
    ORDER BY r.t_valid_start
    """
    result = await session.run(query, tx_time=tx_time.isoformat())
    records = []
    async for record in result:
        edge = dict(record["edge_props"])
        edge["source"] = record["source"]
        edge["target"] = record["target"]
        records.append(edge)
    return records


async def get_disputed_edges(
    session: "AsyncSession",
    entity_title: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve edges with disputed status for conflict resolution queries."""
    conditions = ["r.status = 'disputed'"]
    params: dict[str, Any] = {}

    if entity_title:
        conditions.append("(s.title = $title OR t.title = $title)")
        params["title"] = entity_title

    where_clause = " AND ".join(conditions)

    query = f"""
    MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
    WHERE {where_clause}
    RETURN s.title AS source, t.title AS target,
           properties(r) AS edge_props
    """
    result = await session.run(query, **params)
    records = []
    async for record in result:
        edge = dict(record["edge_props"])
        edge["source"] = record["source"]
        edge["target"] = record["target"]
        records.append(edge)
    return records


# ---------------------------------------------------------------------------
# Knowledge Maintenance (Section 2.10)
# ---------------------------------------------------------------------------


async def privacy_compliant_removal(
    session: "AsyncSession",
    entity_title: str,
    t_now: datetime | None = None,
) -> int:
    """GDPR-compliant removal: close t_tx_end on all edges for an entity.

    Does NOT physically delete — sets t_tx_end to mark as no longer believed.
    Returns the number of edges invalidated.
    """
    t_now = t_now or utcnow()
    result = await session.run(
        """
        MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
        WHERE (s.title = $title OR t.title = $title)
          AND r.t_tx_end = $infinity
        SET r.t_tx_end = $t_tx_end
        RETURN count(r) AS affected
        """,
        title=entity_title,
        t_tx_end=t_now.isoformat(),
        infinity=INFINITY_ISO,
    )
    record = await result.single()
    count = record["affected"] if record else 0
    logger.info("GDPR removal: invalidated %d edges for entity '%s'", count, entity_title)
    return count


async def bulk_source_retraction(
    session: "AsyncSession",
    source_document_id: str,
    t_now: datetime | None = None,
) -> int:
    """Batch-invalidate all edges from a retracted source document.

    Sets t_tx_end on all edges whose provenance traces to the specified document.
    Returns the number of edges invalidated.
    """
    t_now = t_now or utcnow()
    result = await session.run(
        """
        MATCH ()-[r:RELATIONSHIP]->()
        WHERE r.source_document_id = $doc_id
          AND r.t_tx_end = $infinity
        SET r.t_tx_end = $t_tx_end
        RETURN count(r) AS affected
        """,
        doc_id=source_document_id,
        t_tx_end=t_now.isoformat(),
        infinity=INFINITY_ISO,
    )
    record = await result.single()
    count = record["affected"] if record else 0
    logger.info(
        "Bulk retraction: invalidated %d edges from document '%s'",
        count, source_document_id,
    )
    return count


async def close_valid_time_for_edge(
    session: "AsyncSession",
    edge_id: str,
    t_valid_end: datetime,
) -> None:
    """Close an edge's valid-time (Evolution resolution)."""
    await session.run(
        """
        MATCH ()-[r:RELATIONSHIP]->()
        WHERE r.id = $edge_id
        SET r.t_valid_end = $t_valid_end
        """,
        edge_id=edge_id,
        t_valid_end=t_valid_end.isoformat(),
    )


async def retract_edge(
    session: "AsyncSession",
    edge_id: str,
    t_tx_end: datetime | None = None,
) -> None:
    """Retract an edge (Correction resolution)."""
    t_tx_end = t_tx_end or utcnow()
    await session.run(
        """
        MATCH ()-[r:RELATIONSHIP]->()
        WHERE r.id = $edge_id
        SET r.t_tx_end = $t_tx_end
        """,
        edge_id=edge_id,
        t_tx_end=t_tx_end.isoformat(),
    )


# ---------------------------------------------------------------------------
# Driver management helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Relationship type queries for CGRR
# ---------------------------------------------------------------------------


async def get_existing_relation_types_with_embeddings(
    session: "AsyncSession",
    entity_titles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch distinct relation types with a sample description_embedding.

    If *entity_titles* is given, only returns relation types that share at
    least one endpoint with those entities (same pre-filter CGRR already
    uses).  Otherwise returns all active relation types.

    Each returned dict has: relation_type, description, source, target,
    all_sources, all_targets, edge_count, description_embedding.
    """
    from graphrag.bt_graphrag.models.temporal_types import INFINITY_ISO

    if entity_titles:
        query = """
        MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
        WHERE r.t_tx_end = $infinity AND r.t_valid_end = $infinity
          AND (s.title IN $entity_titles OR t.title IN $entity_titles)
        WITH r.relation_type AS rel_type,
             collect(DISTINCT s.title) AS all_sources,
             collect(DISTINCT t.title) AS all_targets,
             collect({
               desc: r.description,
               source: s.title,
               target: t.title,
               emb: r.description_embedding,
               rt_emb: r.relation_type_embedding
             })[0] AS sample,
             count(*) AS edge_count
        RETURN rel_type, sample.desc AS description,
               sample.source AS source, sample.target AS target,
               all_sources, all_targets, edge_count,
               sample.emb AS description_embedding,
               sample.rt_emb AS relation_type_embedding
        ORDER BY edge_count DESC
        """
        result = await session.run(
            query, entity_titles=entity_titles, infinity=INFINITY_ISO,
        )
    else:
        query = """
        MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
        WHERE r.t_tx_end = $infinity AND r.t_valid_end = $infinity
        WITH r.relation_type AS rel_type,
             collect(DISTINCT s.title) AS all_sources,
             collect(DISTINCT t.title) AS all_targets,
             collect({
               desc: r.description,
               source: s.title,
               target: t.title,
               emb: r.description_embedding,
               rt_emb: r.relation_type_embedding
             })[0] AS sample,
             count(*) AS edge_count
        RETURN rel_type, sample.desc AS description,
               sample.source AS source, sample.target AS target,
               all_sources, all_targets, edge_count,
               sample.emb AS description_embedding,
               sample.rt_emb AS relation_type_embedding
        ORDER BY edge_count DESC
        """
        result = await session.run(query, infinity=INFINITY_ISO)

    records: list[dict[str, Any]] = []
    async for record in result:
        records.append({
            "relation_type": record["rel_type"] or "",
            "description": record["description"] or "",
            "source": record["source"] or "",
            "target": record["target"] or "",
            "all_sources": record["all_sources"] or [],
            "all_targets": record["all_targets"] or [],
            "edge_count": record["edge_count"],
            "description_embedding": record["description_embedding"],
            "relation_type_embedding": record["relation_type_embedding"],
        })
    return records


# ---------------------------------------------------------------------------
# Vector search for CGER
# ---------------------------------------------------------------------------


async def vector_search_entities(
    session: "AsyncSession",
    query_embedding: list[float],
    top_k: int = 20,
    index_name: str = "entity_description_embedding",
) -> list[dict[str, Any]]:
    """Return the top-K most similar entities using the Neo4j vector index.

    Each returned dict contains all Entity node properties (title, type,
    description, description_embedding, active_start, active_end, etc.)
    plus a ``score`` field with the cosine similarity.
    """
    query = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node, score
    RETURN properties(node) AS props, score
    """
    result = await session.run(
        query,
        index_name=index_name,
        top_k=top_k,
        embedding=query_embedding,
    )
    records: list[dict[str, Any]] = []
    async for record in result:
        entity = dict(record["props"])
        entity["_vector_score"] = record["score"]
        records.append(entity)
    return records


async def get_all_entities(
    session: "AsyncSession",
) -> list[dict[str, Any]]:
    """Return all Entity nodes with their full properties.

    Used as a fallback when the vector index is unavailable.
    """
    result = await session.run(
        "MATCH (n:Entity) RETURN properties(n) AS props"
    )
    records: list[dict[str, Any]] = []
    async for record in result:
        records.append(dict(record["props"]))
    return records


# ---------------------------------------------------------------------------
# Vector search for CGRR (Relationship edges)
# ---------------------------------------------------------------------------


async def vector_search_relationships(
    session: "AsyncSession",
    query_embedding: list[float],
    top_k: int = 10,
    index_name: str = "relationship_description_embedding",
) -> list[dict[str, Any]]:
    """Return the top-K most similar relationships using the Neo4j vector index.

    Uses db.index.vector.queryRelationships() (Neo4j 5.18+).
    Each returned dict contains: relation_type, description, source, target,
    description_embedding, and _vector_score.

    Falls back to empty list if the index does not exist.
    """
    query = """
    CALL db.index.vector.queryRelationships($index_name, $top_k, $embedding)
    YIELD relationship, score
    MATCH (s:Entity)-[relationship]->(t:Entity)
    RETURN properties(relationship) AS props,
           s.title AS source, t.title AS target,
           score
    """
    try:
        result = await session.run(
            query,
            index_name=index_name,
            top_k=top_k,
            embedding=query_embedding,
        )
        records: list[dict[str, Any]] = []
        async for record in result:
            edge = dict(record["props"])
            edge["source"] = record["source"]
            edge["target"] = record["target"]
            edge["_vector_score"] = record["score"]
            records.append(edge)
        return records
    except Exception:
        logger.debug(
            "vector_search_relationships failed (index '%s' may not exist); "
            "falling back to in-memory cosine",
            index_name,
        )
        return []


# ---------------------------------------------------------------------------
# Driver management helper
# ---------------------------------------------------------------------------


async def create_neo4j_driver(
    uri: str,
    user: str,
    password: str,
) -> "AsyncDriver":
    """Create and return a Neo4j async driver.

    Requires the neo4j package: pip install neo4j
    """
    from neo4j import AsyncGraphDatabase

    return AsyncGraphDatabase.driver(uri, auth=(user, password))


# ---------------------------------------------------------------------------
# CGER Phase B: temporary database for intra-batch candidate retrieval
# ---------------------------------------------------------------------------


class CGERBatchDB:
    """Scratch Neo4j database scoping CGER Phase B candidate retrieval.

    Uses a pre-existing, user-created Neo4j database as an isolated
    workspace.  The database must already exist
    (``CREATE DATABASE <name>`` run manually); this helper only wipes its
    contents at entry and exit and (re)creates the indexes it needs.

    For each incoming entity, ``find_candidates`` returns the union of
    (a) entities sharing the same case-insensitive name and
    (b) the top-K most similar entities by description-embedding cosine.

    Use as an async context manager — the workspace is cleaned on exit
    even if an exception fires.
    """

    def __init__(
        self,
        driver: "AsyncDriver",
        db_name: str,
        vector_dimensions: int,
        label: str = "CGERBatchEntity",
    ) -> None:
        self.driver = driver
        self.db_name = db_name
        self.dimensions = vector_dimensions
        self.label = label
        self.index_name = "cger_batch_desc_embedding"

    async def __aenter__(self) -> "CGERBatchDB":
        await self._wipe_workspace()
        await self._create_indexes()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            await self._wipe_workspace()
        except Exception as wipe_err:
            logger.warning(
                "CGERBatchDB: failed to wipe scratch database '%s': %s",
                self.db_name, wipe_err,
            )

    async def _wipe_workspace(self) -> None:
        """Delete all nodes/indexes the helper owns in the scratch DB.

        Drops both the vector index and any nodes carrying our label so a
        re-run starts from a clean state.  Other databases are untouched.
        """
        async with self.driver.session(database=self.db_name) as session:
            # Drop vector index first so node deletion does not churn it.
            try:
                await session.run(f"DROP INDEX {self.index_name} IF EXISTS")
            except Exception:
                logger.debug("CGERBatchDB: drop index '%s' no-op", self.index_name)
            # Detach-delete all nodes carrying our label (in batches to
            # avoid heap pressure on very large prior runs).
            await session.run(f"MATCH (n:{self.label}) DETACH DELETE n")
        logger.info("CGERBatchDB: wiped scratch workspace in '%s'", self.db_name)

    async def _create_indexes(self) -> None:
        vec_cypher = f"""
        CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
        FOR (n:{self.label}) ON (n.description_embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {self.dimensions},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        async with self.driver.session(database=self.db_name) as session:
            await session.run(
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self.label}) ON (n.title)"
            )
            await session.run(
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self.label}) ON (n.name_lower)"
            )
            await session.run(vec_cypher)

    async def bulk_load(self, entities: list[dict[str, Any]]) -> None:
        """Insert all entities into the temp DB in a single batched query."""
        rows: list[dict[str, Any]] = []
        for ent in entities:
            title = ent.get("title")
            emb = ent.get("description_embedding")
            if not title or not emb:
                continue
            rows.append({
                "title": str(title),
                "name_lower": str(title).strip().lower(),
                "type": str(ent.get("type") or ""),
                "description_embedding": list(emb),
            })
        if not rows:
            return
        cypher = f"""
        UNWIND $rows AS row
        MERGE (n:{self.label} {{title: row.title}})
        SET n.name_lower = row.name_lower,
            n.type = row.type,
            n.description_embedding = row.description_embedding
        """
        async with self.driver.session(database=self.db_name) as session:
            await session.run(cypher, rows=rows)
            # Wait for the vector index to come online before any query.
            try:
                await session.run(
                    "CALL db.awaitIndex($name, $timeout)",
                    name=self.index_name, timeout=60,
                )
            except Exception:
                # awaitIndex unavailable; vector queries may briefly return
                # stale results. Index population is typically instant for
                # batch sizes <10k.
                logger.debug("CGERBatchDB: db.awaitIndex not available")

    async def find_candidates(
        self,
        embedding: list[float],
        name_lower: str,
        entity_type: str | None,
        top_k: int,
        exclude_title: str,
        allowed_titles: set[str],
    ) -> list[dict[str, Any]]:
        """Return candidates for one Phase B entity.

        Result is the union of (a) entities in *allowed_titles* with the
        same ``name_lower`` and (b) the top-K most similar by description
        embedding (also restricted to *allowed_titles*).  The new entity
        itself is excluded via ``exclude_title``.
        """
        if not allowed_titles:
            return []

        allowed = list(allowed_titles)
        # Over-fetch from the vector index so post-filter survivors >= top_k
        # in the common case.  Bounded above to keep query cheap.
        index_fetch = min(max(top_k * 4, top_k + 5), 200)

        vector_cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $fetch, $embedding)
        YIELD node, score
        WHERE node.title IN $allowed AND node.title <> $exclude
        RETURN node.title AS title, node.type AS type, score
        LIMIT $top_k
        """
        name_cypher = f"""
        MATCH (n:{self.label})
        WHERE n.name_lower = $name_lower
          AND n.title IN $allowed
          AND n.title <> $exclude
        RETURN n.title AS title, n.type AS type
        """
        titles: dict[str, dict[str, Any]] = {}
        async with self.driver.session(database=self.db_name) as session:
            vec_result = await session.run(
                vector_cypher,
                index_name=self.index_name,
                fetch=index_fetch,
                embedding=embedding,
                allowed=allowed,
                exclude=exclude_title,
                top_k=top_k,
            )
            async for rec in vec_result:
                titles[rec["title"]] = {
                    "title": rec["title"],
                    "type": rec["type"],
                    "_vector_score": rec["score"],
                    "_match_reason": "embedding_top_k",
                }
            name_result = await session.run(
                name_cypher,
                name_lower=name_lower,
                allowed=allowed,
                exclude=exclude_title,
            )
            async for rec in name_result:
                t = rec["title"]
                if t in titles:
                    titles[t]["_match_reason"] = "embedding_top_k+same_name"
                else:
                    titles[t] = {
                        "title": t,
                        "type": rec["type"],
                        "_vector_score": None,
                        "_match_reason": "same_name",
                    }

        if entity_type:
            etype_norm = str(entity_type).strip().lower()
            titles = {
                t: meta for t, meta in titles.items()
                if str(meta.get("type") or "").strip().lower() == etype_norm
            }

        return list(titles.values())


class CGRRBatchDB:
    """Scratch Neo4j database scoping CGRR Phase B candidate retrieval.

    Mirrors :class:`CGERBatchDB` but indexes ``relation_type_embedding``
    on nodes that represent distinct relation types from the incoming
    batch. Phase B turns N^2 pairwise scoring into N * top-K vector
    lookups against the established canonicals.

    The backing database must already exist (``CREATE DATABASE <name>``
    run manually); this helper only wipes its contents at entry and exit
    and (re)creates the indexes it needs.

    Use as an async context manager — the workspace is cleaned on exit
    even if an exception fires.
    """

    def __init__(
        self,
        driver: "AsyncDriver",
        db_name: str,
        vector_dimensions: int,
        label: str = "CGRRBatchRelType",
    ) -> None:
        self.driver = driver
        self.db_name = db_name
        self.dimensions = vector_dimensions
        self.label = label
        self.index_name = "cgrr_batch_rt_embedding"

    async def __aenter__(self) -> "CGRRBatchDB":
        await self._wipe_workspace()
        await self._create_indexes()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            await self._wipe_workspace()
        except Exception as wipe_err:
            logger.warning(
                "CGRRBatchDB: failed to wipe scratch database '%s': %s",
                self.db_name, wipe_err,
            )

    async def _wipe_workspace(self) -> None:
        async with self.driver.session(database=self.db_name) as session:
            try:
                await session.run(f"DROP INDEX {self.index_name} IF EXISTS")
            except Exception:
                logger.debug("CGRRBatchDB: drop index '%s' no-op", self.index_name)
            await session.run(f"MATCH (n:{self.label}) DETACH DELETE n")
        logger.info("CGRRBatchDB: wiped scratch workspace in '%s'", self.db_name)

    async def _create_indexes(self) -> None:
        vec_cypher = f"""
        CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
        FOR (n:{self.label}) ON (n.relation_type_embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {self.dimensions},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        async with self.driver.session(database=self.db_name) as session:
            await session.run(
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self.label}) ON (n.relation_type)"
            )
            await session.run(vec_cypher)

    async def bulk_load(self, rel_types: list[dict[str, Any]]) -> None:
        """Insert one node per distinct relation type into the temp DB.

        Each item in ``rel_types`` must expose ``relation_type`` and
        ``relation_type_embedding``; ``description``, ``source`` and
        ``target`` are stored for downstream prompt context (not used
        by the vector query).
        """
        rows: list[dict[str, Any]] = []
        for rt in rel_types:
            label = rt.get("relation_type")
            emb = rt.get("relation_type_embedding")
            if not label or not emb:
                continue
            rows.append({
                "relation_type": str(label),
                "description": str(rt.get("description") or ""),
                "source": str(rt.get("source") or ""),
                "target": str(rt.get("target") or ""),
                "relation_type_embedding": list(emb),
            })
        if not rows:
            return
        cypher = f"""
        UNWIND $rows AS row
        MERGE (n:{self.label} {{relation_type: row.relation_type}})
        SET n.description = row.description,
            n.source = row.source,
            n.target = row.target,
            n.relation_type_embedding = row.relation_type_embedding
        """
        async with self.driver.session(database=self.db_name) as session:
            await session.run(cypher, rows=rows)
            try:
                await session.run(
                    "CALL db.awaitIndex($name, $timeout)",
                    name=self.index_name, timeout=60,
                )
            except Exception:
                logger.debug("CGRRBatchDB: db.awaitIndex not available")

    async def find_candidates(
        self,
        embedding: list[float],
        top_k: int,
        exclude_relation_type: str,
        allowed_relation_types: set[str],
    ) -> list[dict[str, Any]]:
        """Return the top-K canonical relation types by cosine, restricted
        to *allowed_relation_types* and excluding ``exclude_relation_type``.

        Returns dicts with ``relation_type``, ``description``, ``source``,
        ``target`` and ``_vector_score``.
        """
        if not allowed_relation_types:
            return []

        allowed = list(allowed_relation_types)
        index_fetch = min(max(top_k * 4, top_k + 5), 200)

        vector_cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $fetch, $embedding)
        YIELD node, score
        WHERE node.relation_type IN $allowed
          AND node.relation_type <> $exclude
        RETURN node.relation_type AS relation_type,
               node.description AS description,
               node.source AS source,
               node.target AS target,
               score
        LIMIT $top_k
        """
        records: list[dict[str, Any]] = []
        async with self.driver.session(database=self.db_name) as session:
            result = await session.run(
                vector_cypher,
                index_name=self.index_name,
                fetch=index_fetch,
                embedding=embedding,
                allowed=allowed,
                exclude=exclude_relation_type,
                top_k=top_k,
            )
            async for rec in result:
                records.append({
                    "relation_type": rec["relation_type"],
                    "description": rec["description"] or "",
                    "source": rec["source"] or "",
                    "target": rec["target"] or "",
                    "_vector_score": rec["score"],
                })
        return records


# ---------------------------------------------------------------------------
# ETCDR Phase B: temporary database for intra-batch candidate retrieval
# ---------------------------------------------------------------------------


class ETCDRBatchDB:
    """Scratch Neo4j database scoping ETCDR Phase B candidate retrieval.

    Holds every relationship already accepted into the current batch as a
    real ``(:Entity)-[:RELATIONSHIP]->(:Entity)`` graph mirroring the main
    BT-GraphRAG schema. ETCDR reuses the exact Cypher of
    :func:`run_subject_side_query`, :func:`run_object_side_query`, etc. by
    opening a session against this scratch DB instead of the main one.

    A vector index on ``r.description_embedding`` is created so the top-K
    cosine pass over a structurally-filtered candidate pool stays cheap.

    Mutations triggered by ETCDR (EVOLUTION / CORRECTION / CORROBORATION)
    apply through the same :func:`apply_evolution` / :func:`apply_correction`
    / :func:`apply_corroboration` functions used for the main DB, by
    binding their ``session`` argument to one of this scratch DB.

    The backing database must already exist (``CREATE DATABASE <name>``
    run manually); this helper only wipes its contents on entry/exit and
    (re)creates the indexes it needs.

    Use as an async context manager — the workspace is cleaned on exit
    even if an exception fires.
    """

    def __init__(
        self,
        driver: "AsyncDriver",
        db_name: str,
        vector_dimensions: int,
    ) -> None:
        self.driver = driver
        self.db_name = db_name
        self.dimensions = vector_dimensions
        self.rel_vector_index = "etcdr_batch_rel_description_embedding"

    async def __aenter__(self) -> "ETCDRBatchDB":
        await self._wipe_workspace()
        await self._create_indexes()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            await self._wipe_workspace()
        except Exception as wipe_err:
            logger.warning(
                "ETCDRBatchDB: failed to wipe scratch database '%s': %s",
                self.db_name, wipe_err,
            )

    async def _wipe_workspace(self) -> None:
        """Drop our vector index and delete every node/edge in the scratch DB."""
        async with self.driver.session(database=self.db_name) as session:
            try:
                await session.run(f"DROP INDEX {self.rel_vector_index} IF EXISTS")
            except Exception:
                logger.debug(
                    "ETCDRBatchDB: drop index '%s' no-op", self.rel_vector_index
                )
            await session.run("MATCH (n) DETACH DELETE n")
        logger.info("ETCDRBatchDB: wiped scratch workspace in '%s'", self.db_name)

    async def _create_indexes(self) -> None:
        """Create the indexes ETCDR's structural + vector Cypher needs here."""
        vec_cypher = f"""
        CREATE VECTOR INDEX {self.rel_vector_index} IF NOT EXISTS
        FOR ()-[r:RELATIONSHIP]-() ON (r.description_embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {self.dimensions},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        async with self.driver.session(database=self.db_name) as session:
            await session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.title)"
            )
            await session.run(
                "CREATE INDEX IF NOT EXISTS "
                "FOR ()-[r:RELATIONSHIP]-() ON (r.relation_type)"
            )
            await session.run(
                "CREATE INDEX IF NOT EXISTS "
                "FOR ()-[r:RELATIONSHIP]-() ON (r.t_tx_end)"
            )
            await session.run(
                "CREATE INDEX IF NOT EXISTS "
                "FOR ()-[r:RELATIONSHIP]-() ON (r.id)"
            )
            try:
                await session.run(vec_cypher)
            except Exception:
                logger.debug(
                    "ETCDRBatchDB: relationship vector index may already exist "
                    "or Neo4j version does not support it"
                )

    async def add_relationship(self, rel: TemporalRelationship) -> None:
        """Insert one accepted batch edge into the scratch DB.

        MERGEs the endpoint ``:Entity {title}`` nodes (so the structural
        Cypher in ETCDR can MATCH them) and CREATEs the ``:RELATIONSHIP``
        edge with the full property bag from ``rel.to_neo4j_properties()``,
        including ``description_embedding`` for the vector index.
        """
        if not rel.id:
            return
        props = rel.to_neo4j_properties()
        cypher = """
        MERGE (s:Entity {title: $source})
        MERGE (t:Entity {title: $target})
        CREATE (s)-[r:RELATIONSHIP]->(t)
        SET r = $props
        """
        async with self.driver.session(database=self.db_name) as session:
            await session.run(
                cypher,
                source=rel.source,
                target=rel.target,
                props=props,
            )

    def session(self):
        """Return an async session bound to this scratch database.

        Callers use it with the same ``run_*_query`` / ``apply_*`` functions
        that target the main DB.
        """
        return self.driver.session(database=self.db_name)
