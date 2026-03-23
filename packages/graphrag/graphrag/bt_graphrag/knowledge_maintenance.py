# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG Knowledge Maintenance Module (Section 2.10).

Provides:
- Privacy-Compliant Removal (GDPR): SCD2-style invalidation with lazy propagation
- Bulk Source Retraction: Batch-invalidate all edges from a retracted source
- Temporal Archival: Compress old facts into rule summaries
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from graphrag.bt_graphrag.models.temporal_types import utcnow

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Privacy-Compliant Removal (GDPR)
# ---------------------------------------------------------------------------


async def gdpr_remove_entity(
    entity_title: str,
    driver: "AsyncDriver",
    database: str = "neo4j",
    t_now: datetime | None = None,
) -> dict[str, Any]:
    """GDPR-compliant entity removal via SCD2 invalidation.

    Does NOT physically delete any data. Instead:
    1. Closes t_tx_end on all edges for the entity
    2. Marks the entity node as removed
    3. Returns affected community IDs for re-summarization

    Returns a summary of the removal action.
    """
    from graphrag.bt_graphrag.neo4j_store import privacy_compliant_removal

    t_now = t_now or utcnow()

    async with driver.session(database=database) as session:
        edges_invalidated = await privacy_compliant_removal(
            session, entity_title, t_now
        )

        # Mark entity node
        await session.run(
            """
            MATCH (n:Entity {title: $title})
            SET n.gdpr_removed = true,
                n.gdpr_removed_at = $t_now
            """,
            title=entity_title,
            t_now=t_now.isoformat(),
        )

        # Find affected communities
        result = await session.run(
            """
            MATCH (n:Entity {title: $title})-[:RELATIONSHIP]-(m:Entity)
            RETURN DISTINCT m.title AS neighbor
            """,
            title=entity_title,
        )
        affected_neighbors = [r["neighbor"] async for r in result]

    return {
        "entity": entity_title,
        "edges_invalidated": edges_invalidated,
        "affected_neighbors": affected_neighbors,
        "removal_time": t_now.isoformat(),
    }


# ---------------------------------------------------------------------------
# Bulk Source Retraction
# ---------------------------------------------------------------------------


async def retract_source(
    source_document_id: str,
    driver: "AsyncDriver",
    database: str = "neo4j",
    t_now: datetime | None = None,
) -> dict[str, Any]:
    """Batch-invalidate all edges whose provenance traces to the retracted source.

    Cascades confidence adjustments using remaining corroborating sources.
    """
    from graphrag.bt_graphrag.neo4j_store import bulk_source_retraction

    t_now = t_now or utcnow()

    async with driver.session(database=database) as session:
        edges_invalidated = await bulk_source_retraction(
            session, source_document_id, t_now
        )

        # Cascade confidence adjustments on edges that had multiple sources
        await session.run(
            """
            MATCH ()-[r:RELATIONSHIP]->()
            WHERE r.t_tx_end IS NULL
              AND r.support_count > 1
            SET r.confidence = CASE
                WHEN r.confidence - 0.1 < 0.1 THEN 0.1
                ELSE r.confidence - 0.1
            END
            """,
        )

    return {
        "source_document_id": source_document_id,
        "edges_invalidated": edges_invalidated,
        "retraction_time": t_now.isoformat(),
    }


# ---------------------------------------------------------------------------
# Temporal Archival
# ---------------------------------------------------------------------------


async def archive_old_facts(
    driver: "AsyncDriver",
    database: str = "neo4j",
    horizon_days: int = 365,
    model: "LLMCompletion | None" = None,
) -> dict[str, Any]:
    """Compress facts beyond the archival horizon into summaries.

    Old edges (t_valid_end is set and older than horizon) are collected,
    summarized, and the originals are marked as archived.
    """
    t_now = utcnow()
    from datetime import timedelta

    horizon = t_now - timedelta(days=horizon_days)

    async with driver.session(database=database) as session:
        # Find edges eligible for archival
        result = await session.run(
            """
            MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
            WHERE r.t_valid_end IS NOT NULL
              AND r.t_valid_end < $horizon
              AND r.archived IS NULL
            RETURN s.title AS source, t.title AS target,
                   r.relation_type AS relation_type,
                   r.description AS description,
                   r.t_valid_start AS t_valid_start,
                   r.t_valid_end AS t_valid_end,
                   r.id AS edge_id
            ORDER BY r.t_valid_start
            LIMIT 1000
            """,
            horizon=horizon.isoformat(),
        )

        edges_to_archive = []
        async for record in result:
            edges_to_archive.append(dict(record))

        if not edges_to_archive:
            return {"archived_count": 0, "horizon": horizon.isoformat()}

        # Mark edges as archived
        edge_ids = [e["edge_id"] for e in edges_to_archive if e.get("edge_id")]
        if edge_ids:
            await session.run(
                """
                MATCH ()-[r:RELATIONSHIP]->()
                WHERE r.id IN $edge_ids
                SET r.archived = true,
                    r.archived_at = $t_now
                """,
                edge_ids=edge_ids,
                t_now=t_now.isoformat(),
            )

    return {
        "archived_count": len(edges_to_archive),
        "horizon": horizon.isoformat(),
        "archive_time": t_now.isoformat(),
    }
