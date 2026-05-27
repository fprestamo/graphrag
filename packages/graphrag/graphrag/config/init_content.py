# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Content for the init CLI command to generate a default configuration."""

import graphrag.config.defaults as defs
from graphrag.config.defaults import (
    graphrag_config_defaults,
    vector_store_defaults,
)

INIT_YAML = f"""\
### This config file contains required core defaults that must be set, along with a handful of common optional settings.
### For a full list of available settings, see https://microsoft.github.io/graphrag/config/yaml/

### LLM settings ###
## There are a number of settings to tune the threading and token limits for LLM calls - check the docs.

completion_models:
  {defs.DEFAULT_COMPLETION_MODEL_ID}:
    model_provider: {defs.DEFAULT_MODEL_PROVIDER}
    model: <DEFAULT_COMPLETION_MODEL>
    auth_method: {defs.DEFAULT_COMPLETION_MODEL_AUTH_TYPE} # or azure_managed_identity
    api_key: ${{GRAPHRAG_API_KEY}} # set this in the generated .env file, or remove if managed identity
    retry:
      type: exponential_backoff

embedding_models:
  {defs.DEFAULT_EMBEDDING_MODEL_ID}:
    model_provider: {defs.DEFAULT_MODEL_PROVIDER}
    model: <DEFAULT_EMBEDDING_MODEL>
    auth_method: {defs.DEFAULT_EMBEDDING_MODEL_AUTH_TYPE}
    api_key: ${{GRAPHRAG_API_KEY}}
    retry:
      type: exponential_backoff

### Document processing settings ###

input:
  type: {graphrag_config_defaults.input.type.value} # [csv, text, json, jsonl]

chunking:
  type: {graphrag_config_defaults.chunking.type}
  size: {graphrag_config_defaults.chunking.size}
  overlap: {graphrag_config_defaults.chunking.overlap}
  encoding_model: {graphrag_config_defaults.chunking.encoding_model}

### Storage settings ###
## If blob storage is specified in the following four sections,
## connection_string and container_name must be provided

input_storage:
  type: {graphrag_config_defaults.input_storage.type} # [file, blob, cosmosdb]
  base_dir: "{graphrag_config_defaults.input_storage.base_dir}"

output_storage:
  type: {graphrag_config_defaults.output_storage.type} # [file, blob, cosmosdb]
  base_dir: "{graphrag_config_defaults.output_storage.base_dir}"

reporting:
  type: {graphrag_config_defaults.reporting.type.value} # [file, blob]
  base_dir: "{graphrag_config_defaults.reporting.base_dir}"

cache:
  type: {graphrag_config_defaults.cache.type} # [json, memory, none]
  storage:
    type: {graphrag_config_defaults.cache.storage.type} # [file, blob, cosmosdb]
    base_dir: "{graphrag_config_defaults.cache.storage.base_dir}"
    
vector_store:
  type: {vector_store_defaults.type}
  db_uri: {vector_store_defaults.db_uri}

### Workflow settings ###

embed_text:
  embedding_model_id: {graphrag_config_defaults.embed_text.embedding_model_id}

extract_graph:
  completion_model_id: {graphrag_config_defaults.extract_graph.completion_model_id}
  prompt: "prompts/extract_graph.txt"
  entity_types: [{",".join(graphrag_config_defaults.extract_graph.entity_types)}]
  max_gleanings: {graphrag_config_defaults.extract_graph.max_gleanings}

summarize_descriptions:
  completion_model_id: {graphrag_config_defaults.summarize_descriptions.completion_model_id}
  prompt: "prompts/summarize_descriptions.txt"
  max_length: {graphrag_config_defaults.summarize_descriptions.max_length}

extract_graph_nlp:
  text_analyzer:
    extractor_type: {graphrag_config_defaults.extract_graph_nlp.text_analyzer.extractor_type.value} # [regex_english, syntactic_parser, cfg]

cluster_graph:
  max_cluster_size: {graphrag_config_defaults.cluster_graph.max_cluster_size}

extract_claims:
  enabled: false
  completion_model_id: {graphrag_config_defaults.extract_claims.completion_model_id}
  prompt: "prompts/extract_claims.txt"
  description: "{graphrag_config_defaults.extract_claims.description}"
  max_gleanings: {graphrag_config_defaults.extract_claims.max_gleanings}

community_reports:
  completion_model_id: {graphrag_config_defaults.community_reports.completion_model_id}
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"
  max_length: {graphrag_config_defaults.community_reports.max_length}
  max_input_length: {graphrag_config_defaults.community_reports.max_input_length}

snapshots:
  graphml: true
  embeddings: false

### Query settings ###
## The prompt locations are required here, but each search method has a number of optional knobs that can be tuned.
## See the config docs: https://microsoft.github.io/graphrag/config/yaml/#query

local_search:
  completion_model_id: {graphrag_config_defaults.local_search.completion_model_id}
  embedding_model_id: {graphrag_config_defaults.local_search.embedding_model_id}
  prompt: "prompts/local_search_system_prompt.txt"

global_search:
  completion_model_id: {graphrag_config_defaults.global_search.completion_model_id}
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"

drift_search:
  completion_model_id: {graphrag_config_defaults.drift_search.completion_model_id}
  embedding_model_id: {graphrag_config_defaults.drift_search.embedding_model_id}
  prompt: "prompts/drift_search_system_prompt.txt"
  reduce_prompt: "prompts/drift_search_reduce_prompt.txt"

basic_search:
  completion_model_id: {graphrag_config_defaults.basic_search.completion_model_id}
  embedding_model_id: {graphrag_config_defaults.basic_search.embedding_model_id}
  prompt: "prompts/basic_search_system_prompt.txt"

### BT-GraphRAG settings ###
## Bitemporal extension for temporal-aware knowledge graphs.
## To use: graphrag index --method bitemporal
## Requires Neo4j running for Stages 2-4 (CGER, ETCDR, graph store).

bt_graphrag:
  enabled: true
  # Prompt files (customizable)
  extraction_prompt: "prompts/bt_extract_graph.txt"
  community_report_prompt: "prompts/bt_community_report.txt"
  cardinality_prompt: "prompts/bt_cardinality_classification.txt"
  # Neo4j connection
  neo4j_uri: "neo4j://127.0.0.1:7687"
  neo4j_user: "neo4j"
  neo4j_password: "12345678"
  neo4j_database: "btgraphrag"
  # Stage 1: Temporal Extraction
  late_arrival_threshold_days: 30
  # Stage 2: CGER (Cross-Graph Entity Resolution)
  cger_enabled: true
  cger_cosine_threshold: 0.686       # description-cosine at/above which LLM decides the merge
  cger_candidate_top_k: 3            # top-K candidates via Neo4j vector search
  neo4j_vector_dimensions: 3072      # embedding dimensions (must match model)
  # Stage 2b: CGRR (Cross-Graph Relationship Resolution)
  cgrr_enabled: true
  cgrr_cosine_threshold: 1.01        # set above cosine ceiling (1.0) to disable CGRR — ETCDR cross-type queries already cover synonym conflicts
  cgrr_candidate_top_k: 3            # top-K by description embedding similarity
  # Stage 3: ETCDR (Conflict Detection)
  etcdr_enabled: true
  etcdr_confidence_threshold: 0.7
  etcdr_topk: 5                      # top-K existing edges routed to Decision Router per phase
  # Debug: save per-stage CSV/JSON files for CGER, CGRR, ETCDR
  debug_output_dir: "output/bt_debug"
  # Stages 5-6: Incremental Community Update
  community_update_k_hop: 2
  # Stage 7: Query Pipeline
  temporal_decay_alpha: 0.1
"""

INIT_DOTENV = """\
GRAPHRAG_API_KEY=<API_KEY>
"""
