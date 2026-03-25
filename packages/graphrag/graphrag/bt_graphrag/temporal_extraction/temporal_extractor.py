# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Temporal-aware graph extraction.

Extends GraphRAG's extraction to produce relationships with valid-time
intervals and entities with active periods.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pandas as pd

from graphrag.bt_graphrag.models.temporal_types import (
    INFINITY,
    ProvenanceRecord,
    TemporalRelationship,
    TemporalStateQuad,
    utcnow,
)
from graphrag.bt_graphrag.temporal_extraction.temporal_normalization import (
    extract_temporal_anchors,
    parse_date_from_string,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


# Updated extraction prompt that requests temporal information
TEMPORAL_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

For each entity, extract:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities

For each relationship, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation of why the entities are related
- relationship_strength: a numeric score indicating strength of the relationship (1-10)
- valid_time_start: when this relationship started being true (ISO date or description, or UNKNOWN)
- valid_time_end: when this relationship stopped being true (ISO date or ONGOING if still true, or UNKNOWN)

Pay special attention to temporal expressions in the text. When the text mentions specific dates, time periods, or temporal markers (e.g., "As of Q3 2023", "from 2018 through 2021", "until last Tuesday", "since 2020"), capture these as the valid_time_start and valid_time_end of the relevant relationships.

Format each output as:
("entity"<|>ENTITY_NAME<|>ENTITY_TYPE<|>ENTITY_DESCRIPTION)
("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>STRENGTH<|>VALID_TIME_START<|>VALID_TIME_END)

Use ## as the record delimiter and <|COMPLETE|> when done.

######################
-Real Data-
######################
Entity_types: [{entity_types}]
Text: {{input_text}}
######################
Output:
"""


def parse_temporal_extraction_result(
    result: str,
    source_id: str,
    document_t_valid: datetime,
    document_t_tx: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse LLM output into entity and relationship DataFrames with temporal data.

    Extends the standard GraphRAG parser to handle valid_time_start/end fields.
    """
    from graphrag.index.utils.string import clean_str

    TUPLE_DELIMITER = "<|>"
    RECORD_DELIMITER = "##"
    COMPLETION_DELIMITER = "<|COMPLETE|>"

    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    records = [r.strip() for r in result.split(RECORD_DELIMITER)]

    for raw_record in records:
        record = re.sub(r"^\(|\)$", "", raw_record.strip())
        if not record or record == COMPLETION_DELIMITER:
            continue

        record_attributes = record.split(TUPLE_DELIMITER)
        record_type = record_attributes[0].strip().strip('"')

        if record_type == "entity" and len(record_attributes) >= 4:
            entity_name = clean_str(record_attributes[1].upper())
            entity_type = clean_str(record_attributes[2].upper())
            entity_description = clean_str(record_attributes[3])
            entities.append({
                "title": entity_name,
                "type": entity_type,
                "description": entity_description,
                "source_id": source_id,
            })

        if record_type == "relationship" and len(record_attributes) >= 5:
            source = clean_str(record_attributes[1].upper())
            target = clean_str(record_attributes[2].upper())
            edge_description = clean_str(record_attributes[3])

            # Detect whether the LLM included relation_type (8 fields)
            # or used the legacy format (7 fields: no relation_type).
            # New format: desc | relation_type | strength | start | end
            # Old format: desc | strength | start | end
            relation_type: str | None = None
            if len(record_attributes) >= 8:
                # New format — field 4 is relation_type
                relation_type = clean_str(record_attributes[4]).upper().replace(" ", "_")
                try:
                    weight = float(record_attributes[5])
                except (ValueError, IndexError):
                    weight = 1.0
                temporal_offset = 6
            else:
                # Legacy format — field 4 is strength
                try:
                    weight = float(record_attributes[4])
                except (ValueError, IndexError):
                    weight = 1.0
                temporal_offset = 5

            # Parse temporal fields
            t_valid_start = document_t_valid
            t_valid_end: float | datetime = INFINITY

            if len(record_attributes) >= temporal_offset + 1:
                start_str = record_attributes[temporal_offset].strip()
                if start_str and start_str.upper() not in ("UNKNOWN", "N/A", ""):
                    parsed = parse_date_from_string(start_str, document_t_valid)
                    if parsed:
                        t_valid_start = parsed

            if len(record_attributes) >= temporal_offset + 2:
                end_str = record_attributes[temporal_offset + 1].strip()
                if end_str and end_str.upper() not in ("UNKNOWN", "ONGOING", "N/A", "PRESENT", ""):
                    parsed = parse_date_from_string(end_str, document_t_valid)
                    if parsed:
                        t_valid_end = parsed

            rel_data = {
                "source": source,
                "target": target,
                "description": edge_description,
                "source_id": source_id,
                "weight": weight,
                "t_valid_start": t_valid_start.isoformat(),
                "t_valid_end": t_valid_end if t_valid_end == INFINITY else t_valid_end.isoformat() if isinstance(t_valid_end, datetime) else str(t_valid_end),
                "t_tx_start": document_t_tx.isoformat(),
                "t_tx_end": None,  # Open-ended: currently believed
                "confidence": weight / 10.0,  # Normalize strength to 0-1
            }

            if relation_type:
                rel_data["relation_type"] = relation_type

            relationships.append(rel_data)

    entities_df = pd.DataFrame(entities) if entities else pd.DataFrame(
        columns=["title", "type", "description", "source_id"]
    )
    relationships_df = pd.DataFrame(relationships) if relationships else pd.DataFrame(
        columns=[
            "source", "target", "description", "source_id", "weight",
            "t_valid_start", "t_valid_end", "t_tx_start", "t_tx_end", "confidence",
        ]
    )

    return entities_df, relationships_df


async def temporal_extract_graph(
    text: str,
    source_id: str,
    entity_types: list[str],
    model: "LLMCompletion",
    prompt: str | None,
    max_gleanings: int,
    document_t_valid: datetime,
    document_t_tx: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract entities and relationships with temporal annotations.

    Uses a temporal-aware prompt to get valid-time intervals from the LLM.
    Falls back to document-level timestamps when LLM doesn't provide them.
    """
    from graphrag_llm.utils import CompletionMessagesBuilder
    from graphrag.prompts.index.extract_graph import CONTINUE_PROMPT, LOOP_PROMPT

    extraction_prompt = prompt or TEMPORAL_EXTRACTION_PROMPT

    messages_builder = CompletionMessagesBuilder().add_user_message(
        extraction_prompt.format(
            input_text=text,
            entity_types=",".join(entity_types),
        )
    )

    response = await model.completion_async(messages=messages_builder.build())
    results = response.content
    messages_builder.add_assistant_message(results)

    if max_gleanings > 0:
        for i in range(max_gleanings):
            messages_builder.add_user_message(CONTINUE_PROMPT)
            response = await model.completion_async(messages=messages_builder.build())
            response_text = response.content
            messages_builder.add_assistant_message(response_text)
            results += response_text

            if i >= max_gleanings - 1:
                break

            messages_builder.add_user_message(LOOP_PROMPT)
            response = await model.completion_async(messages=messages_builder.build())
            if response.content != "Y":
                break

    return parse_temporal_extraction_result(
        results, source_id, document_t_valid, document_t_tx,
    )


def enrich_text_units_with_temporal(
    text_units_df: pd.DataFrame,
    document_t_valid: datetime,
    document_t_tx: datetime,
) -> pd.DataFrame:
    """Add temporal metadata columns to text units DataFrame.

    Each text unit gets the document-level timestamps plus any
    fine-grained temporal anchors found in its text.
    """
    text_units_df = text_units_df.copy()
    text_units_df["t_valid"] = document_t_valid.isoformat()
    text_units_df["t_tx"] = document_t_tx.isoformat()

    # Extract per-text-unit temporal anchors
    anchors_list = []
    for _, row in text_units_df.iterrows():
        text = row.get("text", "")
        anchors = extract_temporal_anchors(text, document_t_valid)
        if anchors:
            # Use the earliest anchor as the text unit's fine-grained valid time
            earliest = min(anchors, key=lambda a: a["start"])
            anchors_list.append(earliest["start"].isoformat())
        else:
            anchors_list.append(document_t_valid.isoformat())

    text_units_df["t_valid_fine"] = anchors_list

    return text_units_df
