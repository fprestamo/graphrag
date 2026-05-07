# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CRAG (Comprehensive RAG) benchmark for BT-GraphRAG.

CRAG is a public Q&A benchmark released by Meta in 2024 covering 5 domains
(finance, sports, music, movie, open) and 8 question types (simple, simple
with condition, set, comparison, aggregation, multi-hop, post-processing,
false-premise).  Evaluation follows the CRAG truthfulness score:

    S = (#correct - #incorrect) / #total

Scoring uses the LLM-as-judge defined in :mod:`evaluation.core.llm_judge`.
"""
