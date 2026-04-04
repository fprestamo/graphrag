# GraphRAG Pipeline Instructions for `ragtest`

This document provides step-by-step instructions to run the full GraphRAG pipeline in the `ragtest` workspace. The pipeline is divided into two phases:

- **Phase 1 — Fresh Indexing from Scratch:** Set up the workspace from zero and index the initial documents.
- **Phase 2 — Re-indexing with New Documents:** Add new documents to an existing workspace and re-index without losing prior data.

---

## Prerequisites

- Python 3.10–3.12 installed
- An OpenAI API key (or Azure OpenAI credentials)
- `graphrag` installed in your Python environment

### Install GraphRAG

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Unix/macOS
# .venv\Scripts\activate        # Windows

# Install GraphRAG
pip install graphrag
```

---

## Folder Structure Overview

```
ragtest/
├── PIPELINE_INSTRUCTIONS.md   ← this file
├── ragtestdocs/
│   ├── phase1/                ← initial documents (copy before Phase 1)
│   │   ├── doc1_artificial_intelligence.txt
│   │   ├── doc2_knowledge_graphs.txt
│   │   └── doc3_rag_overview.txt
│   └── phase2/                ← new documents to add (copy before Phase 2)
│       ├── doc4_large_language_models.txt
│       └── doc5_vector_databases.txt
└── workspace/                 ← created automatically by graphrag init
    ├── input/                 ← drop your .txt documents here
    ├── output/                ← pipeline outputs (parquet files, etc.)
    ├── .env                   ← your API key goes here
    └── settings.yaml          ← pipeline configuration
```

---

## Phase 1 — Fresh Indexing from Scratch

### Step 1: Navigate to the ragtest folder

```bash
cd ragtest
```

### Step 2: Create the workspace and initialize GraphRAG

```bash
mkdir workspace
cd workspace
graphrag init
```

When prompted, enter your preferred chat and embedding model names (e.g., `gpt-4.1` and `text-embedding-3-small` for OpenAI).

This creates:
- `input/` — folder where your documents go
- `.env` — environment variables file
- `settings.yaml` — pipeline configuration file

### Step 3: Set your API key

Open `.env` and replace the placeholder with your real key:

```env
GRAPHRAG_API_KEY=your-openai-api-key-here
```

> **Azure OpenAI users:** Also update `settings.yaml` with your deployment name, endpoint, and API version. Search for the `models:` section and set `model_provider: azure`, `deployment_name`, `api_base`, and `api_version`.

### Step 4: Copy Phase 1 documents into the input folder

From the `ragtest/` root:

```bash
# Unix/macOS
cp ragtestdocs/phase1/*.txt workspace/input/

# Windows
copy ragtestdocs\phase1\*.txt workspace\input\
```

After copying, your `workspace/input/` folder should contain:
- `doc1_artificial_intelligence.txt`
- `doc2_knowledge_graphs.txt`
- `doc3_rag_overview.txt`

### Step 5: Run the indexing pipeline

```bash
cd workspace
graphrag index
```

> ⚠️ This operation calls your LLM and will consume API credits. For a first run, keep the document set small. The process may take several minutes depending on document size and API rate limits.

Wait for the pipeline to finish. You will see progress messages in the terminal. When done, the `output/` folder will be populated with Parquet files representing the knowledge graph.

### Step 6: Verify the output

Check that the `output/` folder contains Parquet files:

```bash
ls workspace/output/
```

You should see files such as `entities.parquet`, `relationships.parquet`, `communities.parquet`, etc.

### Step 7: Query the indexed data

Run a **global search** (high-level, across the whole corpus):

```bash
graphrag query "What are the main topics covered in these documents?"
```

Run a **local search** (specific, entity-level):

```bash
graphrag query "How does Retrieval-Augmented Generation work?" --method local
```

---

## Phase 2 — Re-indexing with New Documents (Incremental Update)

> **When to use this phase:** You have already completed Phase 1 and want to add new documents to the existing knowledge graph without starting over.

### Step 1: Confirm your workspace is ready

Make sure the `workspace/output/` folder from Phase 1 exists and is not empty. If it is empty, go back to Phase 1.

### Step 2: Copy Phase 2 documents into the input folder

From the `ragtest/` root:

```bash
# Unix/macOS
cp ragtestdocs/phase2/*.txt workspace/input/

# Windows
copy ragtestdocs\phase2\*.txt workspace\input\
```

After copying, your `workspace/input/` folder should now contain **all five documents**:
- `doc1_artificial_intelligence.txt` (from Phase 1)
- `doc2_knowledge_graphs.txt` (from Phase 1)
- `doc3_rag_overview.txt` (from Phase 1)
- `doc4_large_language_models.txt` ← **new**
- `doc5_vector_databases.txt` ← **new**

> **Important:** Keep the Phase 1 documents in the `input/` folder. GraphRAG will only process documents that have not been indexed yet, but having all documents present ensures the graph remains consistent.

### Step 3: Re-run the indexing pipeline

```bash
cd workspace
graphrag index
```

GraphRAG will detect the new documents and add them to the existing graph. Documents already indexed in Phase 1 will not be re-processed (they are tracked in the pipeline's cache/state).

> ⚠️ Again, this will consume API credits proportional to the amount of **new** content being indexed.

### Step 4: Verify the updated output

After the pipeline completes, confirm that the output reflects the new documents:

```bash
ls workspace/output/
```

The Parquet files will be updated/regenerated to include the new entities and relationships.

### Step 5: Query using the expanded knowledge graph

Test that the new documents are queryable:

```bash
# Global search across all documents
graphrag query "Summarize the key concepts across all the indexed documents."

# Local search targeting new content
graphrag query "What are the main vector database solutions available?" --method local

# Cross-document query (linking old and new content)
graphrag query "How do knowledge graphs and vector databases complement each other in RAG systems?" --method local
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `GRAPHRAG_API_KEY` not set | Check your `.env` file and ensure the key is correct |
| Pipeline fails with rate limit errors | Reduce batch size in `settings.yaml` or wait and retry |
| Output folder is empty after indexing | Check the terminal logs for errors; ensure `input/` is not empty |
| New documents not being indexed in Phase 2 | Ensure the files were copied to `workspace/input/` and that the file names are unique |
| Out of memory errors | Reduce document chunk size in `settings.yaml` |

---

## Additional Resources

- [Official GraphRAG Documentation](https://microsoft.github.io/graphrag)
- [GraphRAG Getting Started Guide](https://microsoft.github.io/graphrag/get_started/)
- [GraphRAG CLI Reference](https://microsoft.github.io/graphrag/cli/)
- [Configuration Overview](https://microsoft.github.io/graphrag/config/overview/)
- [Prompt Tuning Guide](https://microsoft.github.io/graphrag/prompt_tuning/overview/)
