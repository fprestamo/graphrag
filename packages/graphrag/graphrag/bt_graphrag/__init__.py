# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG: Bitemporal extension for Microsoft GraphRAG.

Implements a seven-stage pipeline that extends GraphRAG with:
- Temporal extraction and document dating (Stage 1)
- Cross-Graph Entity Resolution (Stage 2)
- Edge-Level Temporal Conflict Detection and Resolution (Stage 3)
- Bitemporal Graph Store via Neo4j (Stage 4)
- Incremental Community Update (Stage 5)
- Selective LLM Summarization (Stage 6)
- Temporal Query Pipeline (Stage 7)
"""
