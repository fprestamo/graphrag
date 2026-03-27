"""Evaluation framework for BT-GraphRAG pipeline components.

Each pipeline component (CGER, CGRR, …) has its own completely independent
sub-package with:
  • its own ground truth
  • its own documents / data folder
  • its own scorer registry
  • its own metaheuristic optimizer  (DE, SA, PSO)
  • its own results folder

All components share the utilities in ``shared/``:
  • mock_embeddings   – deterministic SHA-256 embeddings (no API key needed)
  • hypothesis_testing – McNemar, bootstrap, permutation, Wilcoxon, Friedman

Sub-packages
------------
cger
    Cross-Graph Entity Resolution evaluation.
cgrr
    Cross-Graph Relationship Resolution evaluation.
shared
    Component-agnostic statistical helpers.

Entry point
-----------
run_evaluation
    CLI orchestrator: runs any subset of components, supports
    ``--mock``, ``--skip-optimize``, ``--component``, ``--methods``, etc.

Usage
-----
# Quick smoke-test (no API key, skip optimisation)
python -m graphrag.bt_graphrag.evaluation.run_evaluation --mock --skip-optimize

# CGER only, real embeddings
python -m graphrag.bt_graphrag.evaluation.run_evaluation --component cger --model text-embedding-3-small

# Full run, all components and optimizers
python -m graphrag.bt_graphrag.evaluation.run_evaluation --mock
"""
