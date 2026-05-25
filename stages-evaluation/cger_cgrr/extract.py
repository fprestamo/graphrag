# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Stage 1 of the CGER/CGRR evaluation: corpus -> text units -> entities + relationships.

Reads every *.txt file in ``data/<split>-corpus/`` (split = train|test),
splits each document into token-based text units, runs the temporal
graph extractor (with embeddings) on each unit, and writes the
consolidated output to ``data/<split>-extracted/`` as JSON:

    data/<split>-extracted/text_units.json
    data/<split>-extracted/entities.json
    data/<split>-extracted/relationships.json

The downstream ``evaluate.py`` script consumes the ``test-extracted``
JSON files together with the manually authored ground truth in
``data/test-ground-true/`` to score CGER and CGRR.  ``train.py``
consumes the ``train-extracted`` / ``train-ground-true`` siblings to
search for the best cosine thresholds.

Usage:
    python stages-evaluation/cger_cgrr/extract.py --split test
    python stages-evaluation/cger_cgrr/extract.py --split train

Documents are processed concurrently (default 10 at a time). Chunks within
a document still run sequentially, so the worst-case in-flight LLM calls
is roughly ``concurrency * max_gleanings``. Override with ``--concurrency``.

Requirements:
    - ``GRAPHRAG_API_KEY`` (or ``OPENAI_API_KEY``) defined either in the
      current shell or in the project-root ``.env`` file — the script
      auto-loads ``.env`` before reading the variable.
    - The graphrag, graphrag_llm and graphrag_chunking packages installed
      (already part of this monorepo).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from graphrag.bt_graphrag.temporal_extraction.embedding_enrichment import embed_dataframes
from graphrag.bt_graphrag.temporal_extraction.temporal_graph_extractor import (
    TemporalGraphExtractor,
)
from graphrag.bt_graphrag.prompts import TEMPORAL_GRAPH_EXTRACTION_PROMPT
from graphrag_chunking.token_chunker import TokenChunker
from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig, RetryConfig
from graphrag_llm.config.types import LLMProviderType, RetryType
from graphrag_llm.embedding import create_embedding


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
DATA_DIR = HERE / "data"
ENV_PATH = PROJECT_ROOT / ".env"
VALID_SPLITS = ("train", "test")

# Same defaults the .ragtest settings.yaml uses, so the eval mirrors prod.
COMPLETION_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
MAX_GLEANINGS = 1

ENTITY_TYPES = ["organization","person","geo","event","other"]

DEFAULT_CONCURRENCY = 15

# Transient LLM errors (e.g. "Server disconnected", connection resets) are
# common during long extraction runs.  Without retries, a single failed
# chunk kills the whole asyncio.gather and we lose in-flight progress.
_RETRY_CONFIG = RetryConfig(
    type=RetryType.ExponentialBackoff,
    max_retries=6,
    base_delay=2.0,
    max_delay=60.0,
    jitter=True,
)


def _load_env_file(path: Path) -> None:
    """Populate ``os.environ`` from a ``.env`` file without clobbering existing vars."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _api_key() -> str:
    _load_env_file(ENV_PATH)
    key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print(f"[ERROR] GRAPHRAG_API_KEY not found in env or {ENV_PATH}.",
              file=sys.stderr)
        sys.exit(1)
    return key


def _completion():
    return create_completion(
        ModelConfig(
            type=LLMProviderType.LiteLLM,
            model_provider="openai",
            model=COMPLETION_MODEL,
            api_key=_api_key(),
            retry=_RETRY_CONFIG,
        )
    )


def _embedding():
    return create_embedding(
        ModelConfig(
            type=LLMProviderType.LiteLLM,
            model_provider="openai",
            model=EMBEDDING_MODEL,
            api_key=_api_key(),
            retry=_RETRY_CONFIG,
        )
    )


def _chunker(embedding_model) -> TokenChunker:
    tokenizer = embedding_model.tokenizer
    return TokenChunker(
        size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        encode=tokenizer.encode,
        decode=tokenizer.decode,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def _process_document(
    doc_path: Path,
    chunker: TokenChunker,
    extractor: TemporalGraphExtractor,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    """Extract one document end-to-end.

    The ``semaphore`` caps how many documents are in-flight at once. Per-doc
    log lines are buffered and flushed in a single ``print`` call so output
    from concurrent docs doesn't interleave line-by-line.
    """
    async with semaphore:
        text = doc_path.read_text(encoding="utf-8")
        chunks = chunker.chunk(text)

        doc_units: list[dict] = []
        doc_entities: list[pd.DataFrame] = []
        doc_relationships: list[pd.DataFrame] = []
        log_lines: list[str] = [f"  [{doc_path.name}] {len(chunks)} chunk(s)"]

        for chunk_idx, chunk in enumerate(chunks):
            unit_id = str(uuid4())
            unit_text = chunk.text

            doc_units.append({
                "id": unit_id,
                "document": doc_path.name,
                "chunk_index": chunk_idx,
                "text": unit_text,
            })

            ents_df, rels_df = await extractor(
                text=unit_text,
                entity_types=ENTITY_TYPES,
                source_id=unit_id,
            )

            if not ents_df.empty:
                ents_df = ents_df.copy()
                ents_df["document"] = doc_path.name
                doc_entities.append(ents_df)
            if not rels_df.empty:
                rels_df = rels_df.copy()
                rels_df["document"] = doc_path.name
                doc_relationships.append(rels_df)

            log_lines.append(
                f"    chunk {chunk_idx}: "
                f"{0 if ents_df.empty else len(ents_df)} entities, "
                f"{0 if rels_df.empty else len(rels_df)} relationships"
            )

        print("\n".join(log_lines))
        return doc_path.name, doc_units, doc_entities, doc_relationships


async def extract(split: str, concurrency: int = DEFAULT_CONCURRENCY) -> None:
    corpus_dir = DATA_DIR / f"{split}-corpus"
    out_dir = DATA_DIR / f"{split}-extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = sorted(p for p in corpus_dir.glob("*.txt") if p.name != ".gitkeep")
    if not docs:
        print(f"[ERROR] No .txt files found in {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[EXTRACT] split={split}: {len(docs)} document(s) in {corpus_dir} "
          f"(concurrency={concurrency})")

    completion = _completion()
    embedding = _embedding()
    chunker = _chunker(embedding)

    # The temporal extraction prompt embeds {document_date} in multiple
    # places.  The parent GraphExtractor only substitutes {input_text} and
    # {entity_types}, so we pre-fill {document_date} here before handing
    # the prompt off.
    document_date = datetime.now(timezone.utc)
    prompt = TEMPORAL_GRAPH_EXTRACTION_PROMPT.replace(
        "{document_date}", document_date.date().isoformat(),
    )

    extractor = TemporalGraphExtractor(
        model=completion,
        prompt=prompt,
        max_gleanings=MAX_GLEANINGS,
        document_date=document_date,
        embedding_model=embedding,
    )

    # Fan out: each document is its own task; the semaphore caps how many
    # run concurrently. gather() preserves the input order so text_units /
    # entities / relationships end up in the same order as before.
    semaphore = asyncio.Semaphore(max(1, concurrency))
    doc_results = await asyncio.gather(*(
        _process_document(d, chunker, extractor, semaphore) for d in docs
    ))

    text_units: list[dict] = []
    all_entities: list[pd.DataFrame] = []
    all_relationships: list[pd.DataFrame] = []
    for _name, units, ents, rels in doc_results:
        text_units.extend(units)
        all_entities.extend(ents)
        all_relationships.extend(rels)

    entities_df = (
        pd.concat(all_entities, ignore_index=True)
        if all_entities else pd.DataFrame()
    )
    relationships_df = (
        pd.concat(all_relationships, ignore_index=True)
        if all_relationships else pd.DataFrame()
    )

    # Re-embed once more in case anything was missed by per-chunk enrichment.
    entities_df, relationships_df = await embed_dataframes(
        entities_df, relationships_df, embedding,
    )

    (out_dir / "text_units.json").write_text(
        json.dumps(text_units, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    entities_df.to_json(
        out_dir / "entities.json", orient="records", indent=2, force_ascii=False,
    )
    relationships_df.to_json(
        out_dir / "relationships.json", orient="records", indent=2, force_ascii=False,
    )

    print()
    print(f"[EXTRACT] Wrote {len(text_units)} text units, "
          f"{len(entities_df)} entities, {len(relationships_df)} relationships to {out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract entities and relationships from a corpus split.")
    p.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="test",
        help="Which split to process: read data/<split>-corpus/, write data/<split>-extracted/ (default: test).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"How many documents to extract in parallel (default: {DEFAULT_CONCURRENCY}). "
             f"Each in-flight document holds the LLM busy on at most one "
             f"chunk at a time, so peak concurrent LLM calls is roughly "
             f"this value (times max_gleanings).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.concurrency < 1:
        print(f"[ERROR] --concurrency must be >= 1 (got {args.concurrency})",
              file=sys.stderr)
        sys.exit(1)
    asyncio.run(extract(args.split, concurrency=args.concurrency))
