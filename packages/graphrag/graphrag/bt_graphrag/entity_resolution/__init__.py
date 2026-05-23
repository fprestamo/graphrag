# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Stage 2: Cross-Graph Entity Resolution (CGER) and Relationship Resolution (CGRR)."""

from graphrag.bt_graphrag.entity_resolution.cger import (
    apply_merge_map_to_relationships,
    resolve_entities,
)
from graphrag.bt_graphrag.entity_resolution.cgrr import (
    apply_normalize_map_to_cardinality,
    get_existing_relations_for_entities,
    resolve_relationships,
)
from graphrag.bt_graphrag.entity_resolution.scorers import (
    EntityScorer,
    RelationshipScorer,
    description_cosine_entity_scorer,
    description_cosine_relationship_scorer,
)

__all__ = [
    "apply_merge_map_to_relationships",
    "apply_normalize_map_to_cardinality",
    "description_cosine_entity_scorer",
    "description_cosine_relationship_scorer",
    "EntityScorer",
    "get_existing_relations_for_entities",
    "RelationshipScorer",
    "resolve_entities",
    "resolve_relationships",
]
