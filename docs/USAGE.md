# Usage guide

## Install

```bash
pip install rag-python
```

Optional extras: see [Configuration](CONFIGURATION.md#environment-variables) and [Providers](PROVIDERS.md).

---

## Basic Python API

```python
from rag_python import RAG

rag = RAG(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)

# Ingest files or directories
rag.ingest(["./data", "./policy.pdf"], reindex=True)

# Query
answer = rag.query("How many days of annual leave?")
print(answer.text)         # Generated answer
print(answer.sources)      # Retrieved chunks with scores
print(answer.evaluation)   # faithfulness / relevance
print(answer.retried)      # Whether self-correction ran
```

---

## Ingest

`rag.ingest()` accepts files and folders. Supported extensions:

`.txt` `.md` `.pdf` `.docx` `.csv` `.json` `.html`

```python
# Multiple paths
rag.ingest(["./docs", "handbook.pdf", "notes.md"], reindex=True)

# Custom extensions
rag = RAG(document_extensions=(".txt", ".md", ".csv"))
```

Use `reindex=True` to clear the vector store before ingesting.

---

## Retrieval modes

| Mode | `retriever=` | Description |
|------|--------------|-------------|
| Multi-query | `multi_query` (default) | Query rewriting + multiple vector searches |
| Vector | `vector` | Single embedding search |
| Hybrid | `hybrid` | BM25 + vector fused with RRF (`pip install rag-python[hybrid]`) |

```python
from rag_python import RAG, SearchConfig

rag = RAG(retriever="hybrid")
rag.ingest(["./data"], reindex=True)

answer = rag.query(
    "annual leave policy",
    search=SearchConfig(
        retriever="hybrid",
        metadata_filter={"filename": "hr-policy.pdf"},
    ),
)
```

---

## Streaming

Stream tokens for responsive UIs:

```python
import rag_python

rag_python.configure_logging()

stream = rag.query_stream("How many days of annual leave?")
for token in stream:
    print(token, end="", flush=True)

# After iteration — full result with sources and evaluation
result = stream.result
print(result.sources)
```

CLI: `rag-python query "annual leave" --stream`

---

## Multi-provider setups

See [Providers](PROVIDERS.md) for full details.

```python
# Claude + OpenAI embeddings
rag = RAG(
    llm_provider="anthropic",
    llm_model="claude-opus-4-6",
    embedding_provider="openai",
)

# Offline embeddings
rag = RAG(
    embedding_provider="local",
    embedding_model="all-MiniLM-L6-v2",
)
```

---

## Low-level API

For advanced use without the `RAG` client:

```python
from rag_python import ingest, query, query_stream

ingest(data_path="./data", reindex=True)
response = query("What is the leave policy?")
```

---

## CLI

See the full [CLI reference](CLI.md).

```bash
rag-python ingest ./data --reindex
rag-python query "How many days of annual leave?" -v
rag-python docs quickstart
```

---

## Next steps

- [Configuration](CONFIGURATION.md) — env vars and `RAGConfig`
- [CLI reference](CLI.md) — all terminal flags
- [Providers](PROVIDERS.md) — LLM and embedding backends
