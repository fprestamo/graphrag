"""Registry of all relationship scorers available for CGRR evaluation."""
from graphrag.bt_graphrag.entity_resolution.scorers import (
    RelationshipScorer,
    compute_relationship_composite_score,
    bm25_only_relationship_scorer,
    semantic_only_relationship_scorer,
    type_and_endpoint_relationship_scorer,
)

SCORER_REGISTRY: dict[str, RelationshipScorer] = {
    "composite_3signal":    compute_relationship_composite_score,
    "bm25_only":            bm25_only_relationship_scorer,
    "semantic_only":        semantic_only_relationship_scorer,
    "type_and_endpoint":    type_and_endpoint_relationship_scorer,
}
