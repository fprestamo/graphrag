"""Deterministic mock embedding function — no API key required.

Any text maps to the same vector every time (SHA-256 hash → unit sphere),
so entities that share a canonical description get very similar vectors while
unrelated entities diverge.  Useful for CI, offline testing, and structural
validation of the evaluation framework.

Usage
-----
from graphrag.bt_graphrag.evaluation.shared.mock_embeddings import mock_embedding_fn
results = await mock_embedding_fn(["text A", "text B"])
"""
from __future__ import annotations

import hashlib
import math


def _mock_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic unit-norm vector derived from the text's SHA-256 hash."""
    h = hashlib.sha256(text.encode()).hexdigest()
    raw: list[float] = []
    seed = text
    while len(raw) < dim:
        h2 = hashlib.sha256(seed.encode()).hexdigest()
        raw.extend(int(h2[i : i + 2], 16) / 255.0 for i in range(0, len(h2), 2))
        seed = h2
    raw = raw[:dim]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm > 0 else raw


async def mock_embedding_fn(texts: list[str]) -> list[list[float]]:
    """Async batch mock embedding — drop-in replacement for real embed functions."""
    return [_mock_embedding(t) for t in texts]
