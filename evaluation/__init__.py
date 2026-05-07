# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BT-GraphRAG evaluation pipeline.

Top-level package for benchmarking BT-GraphRAG against public Q&A and
RAG datasets (CRAG, HotpotQA, MuSiQue, MultiHop-RAG, NarrativeQA, ...).

Layout::

    evaluation/
        core/                    # shared utilities (runner, metrics, judge, CLI)
        benchmarks/
            crag/                # one self-contained subfolder per benchmark
            hotpotqa/
            musique/
            multihop_rag/
            narrativeqa/
        run_all.py               # orchestrator that runs every benchmark

Each benchmark exposes the same two entry points::

    python -m evaluation.benchmarks.<name>.load_dataset --out <path>
    python -m evaluation.benchmarks.<name>.evaluate     --dataset <path> --out <path>
"""
