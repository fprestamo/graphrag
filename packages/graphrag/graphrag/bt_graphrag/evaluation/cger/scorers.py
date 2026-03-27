"""Registry of all entity scorers available for CGER evaluation."""
from graphrag.bt_graphrag.entity_resolution.scorers import (
    EntityScorer,
    compute_entity_composite_score,
    embedding_only_entity_scorer,
    citation_and_description_entity_scorer,
)

SCORER_REGISTRY: dict[str, EntityScorer] = {
    "composite_5signal":        compute_entity_composite_score,
    "embedding_only":           embedding_only_entity_scorer,
    "citation_and_description": citation_and_description_entity_scorer,
}
