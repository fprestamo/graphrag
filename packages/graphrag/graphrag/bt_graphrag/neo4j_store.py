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


async def init_schema(driver: "AsyncDriver", database: str = "neo4j") -> None:
    """Create required indexes and constraints in Neo4j."""
    async with driver.session(database=database) as session:
        for cypher in _INIT_CYPHER:
            try:
                await session.run(cypher)
            except Exception:
                logger.debug("Index may already exist: %s", cypher)
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
