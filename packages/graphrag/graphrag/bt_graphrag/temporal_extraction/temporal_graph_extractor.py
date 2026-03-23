# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Temporal-aware graph extractor.

Extends the standard GraphExtractor to parse temporal fields
(valid_time_start, valid_time_end) from the LLM extraction output.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from graphrag.bt_graphrag.temporal_extraction.temporal_normalization import (
    parse_date_from_string,
)
from graphrag.index.operations.extract_graph.graph_extractor import (
    COMPLETION_DELIMITER,
    RECORD_DELIMITER,
    TUPLE_DELIMITER,
    GraphExtractor,
)
from graphrag.index.utils.string import clean_str

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


class TemporalGraphExtractor(GraphExtractor):
    """Graph extractor that also parses temporal fields from LLM output.

    The temporal-aware prompt instructs the LLM to include valid_time_start
    and valid_time_end in relationship tuples. This extractor parses those
    additional fields.
    """

    _document_date: datetime | None

    def __init__(
        self,
        model: "LLMCompletion",
        prompt: str,
        max_gleanings: int,
        document_date: datetime | None = None,
        on_error=None,
    ):
        super().__init__(model, prompt, max_gleanings, on_error)
        self._document_date = document_date

    def _process_result(
        self,
        result: str,
        source_id: str,
        tuple_delimiter: str,
        record_delimiter: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Parse results including temporal fields."""
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        records = [r.strip() for r in result.split(record_delimiter)]

        for raw_record in records:
            record = re.sub(r"^\(|\)$", "", raw_record.strip())
            if not record or record == COMPLETION_DELIMITER:
                continue

            record_attributes = record.split(tuple_delimiter)
            record_type = record_attributes[0]

            if record_type == '"entity"' and len(record_attributes) >= 4:
                entity_name = clean_str(record_attributes[1].upper())
                entity_type = clean_str(record_attributes[2].upper())
                entity_description = clean_str(record_attributes[3])
                entities.append({
                    "title": entity_name,
                    "type": entity_type,
                    "description": entity_description,
                    "source_id": source_id,
                })

            if record_type == '"relationship"' and len(record_attributes) >= 5:
                source = clean_str(record_attributes[1].upper())
                target = clean_str(record_attributes[2].upper())
                edge_description = clean_str(record_attributes[3])
                try:
                    weight = float(record_attributes[4])
                except ValueError:
                    weight = 1.0

                # Parse temporal fields (fields 5 and 6)
                t_valid_start = None
                t_valid_end = None
                ref_date = self._document_date or datetime.now(timezone.utc)

                if len(record_attributes) >= 6:
                    start_str = clean_str(record_attributes[5])
                    if start_str and start_str.upper() not in (
                        "UNKNOWN", "N/A", "", "NONE"
                    ):
                        t_valid_start = parse_date_from_string(
                            start_str, ref_date
                        )

                if len(record_attributes) >= 7:
                    end_str = clean_str(record_attributes[6])
                    if end_str and end_str.upper() not in (
                        "UNKNOWN", "ONGOING", "N/A", "PRESENT", "", "NONE"
                    ):
                        t_valid_end = parse_date_from_string(
                            end_str, ref_date
                        )

                rel_data: dict[str, Any] = {
                    "source": source,
                    "target": target,
                    "description": edge_description,
                    "source_id": source_id,
                    "weight": weight,
                }

                if t_valid_start:
                    rel_data["t_valid_start"] = t_valid_start.isoformat()
                if t_valid_end:
                    rel_data["t_valid_end"] = t_valid_end.isoformat()

                relationships.append(rel_data)

        entities_df = (
            pd.DataFrame(entities) if entities
            else pd.DataFrame(columns=["title", "type", "description", "source_id"])
        )
        relationships_df = (
            pd.DataFrame(relationships) if relationships
            else pd.DataFrame(
                columns=["source", "target", "weight", "description", "source_id"]
            )
        )

        return entities_df, relationships_df
