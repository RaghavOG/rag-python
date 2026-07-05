# Configuration

Configure rag-python via environment variables, `.env` files, or Python `RAGConfig`.

## Environment variables

Copy [`.env.example`](../.env.example) to `.env` and set your keys.

### API keys & providers

| Variable | Required when | Description |
|----------|---------------|-------------|
| `OPENAI_API_KEY` | `openai` provider | Default LLM and embeddings |
| `ANTHROPIC_API_KEY` | `anthropic` LLM | Claude models |
| `GEMINI_API_KEY` | `gemini` LLM | Gemini models |
| `AZURE_OPENAI_ENDPOINT` | Azure | Resource endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure | API key |
| `OPENAI_API_VERSION` | Azure | Default `2023-09-01-preview` |
| `OLLAMA_BASE_URL` | Ollama | Default `http://localhost:11434` |
| `LOCAL_EMBEDDING_MODEL` | `local` embeddings | Default `all-MiniLM-L6-v2` |

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_PYTHON_DATA_DIR` | `./data` | Default ingest directory |
| `RAG_PYTHON_CHROMA_DIR` | `./chroma_db` | ChromaDB persistence path |

### Pipeline tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o-mini` | Default chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Default embedding model |
| `CHUNK_STRATEGY` | `recursive` | `recursive`, `structure_aware`, `semantic` |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RETRIEVE` | `20` | Chunks retrieved before rerank |
| `TOP_K_RERANK` | `5` | Chunks kept after rerank |
| `MULTI_QUERY_N` | `3` | Rewritten queries for multi-query |
| `GUARDRAILS_ENABLED` | `true` | Prompt injection + hallucination checks |
| `MAX_RETRIES` | `2` | Self-correction retries |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder model |

---

## Python configuration

### `RAGConfig`

```python
from rag_python import RAG, RAGConfig, ChunkingConfig, SearchConfig, DocumentConfig, QueryConfig

rag = RAG(
    config=RAGConfig(
        chunking=ChunkingConfig(
            strategy="recursive",
            chunk_size=512,
            chunk_overlap=64,
        ),
        search=SearchConfig(
            retriever="hybrid",
            top_k_retrieve=20,
            top_k_rerank=5,
            multi_query_n=3,
            rerank_enabled=True,
            metadata_filter={"filename": "policy.pdf"},
        ),
        documents=DocumentConfig(
            extensions=(".txt", ".md", ".pdf", ".csv"),
            clean=True,
            copy_to_data_dir=True,
        ),
        query=QueryConfig(
            use_guardrails=True,
            use_retry=True,
            max_retries=2,
            eval_threshold=0.6,
        ),
    ),
)
```

### Shorthand on `RAG()`

Pass these directly to `RAG(...)` without building `RAGConfig`:

| Parameter | Maps to |
|-----------|---------|
| `chunk_strategy` | `ChunkingConfig.strategy` |
| `chunk_size` | `ChunkingConfig.chunk_size` |
| `chunk_overlap` | `ChunkingConfig.chunk_overlap` |
| `retriever` | `SearchConfig.retriever` |
| `metadata_filter` | `SearchConfig.metadata_filter` |
| `top_k_retrieve` | `SearchConfig.top_k_retrieve` |
| `top_k_rerank` | `SearchConfig.top_k_rerank` |
| `multi_query_n` | `SearchConfig.multi_query_n` |
| `rerank_enabled` | `SearchConfig.rerank_enabled` |
| `document_extensions` | `DocumentConfig.extensions` |
| `data_dir` | Custom data directory |
| `chroma_dir` | Custom Chroma persistence path |

### Per-query overrides

```python
from rag_python import SearchConfig, QueryConfig

answer = rag.query(
    "annual leave",
    search=SearchConfig(retriever="vector", metadata_filter={"filename": "hr.txt"}),
    query_config=QueryConfig(use_retry=False),
)
```

---

## Logging

```python
import rag_python

rag_python.configure_logging()  # INFO to console
# or
import logging
logging.getLogger("rag_python").setLevel(logging.DEBUG)
```

Log events include ingest counts, retrieval mode, guardrail blocks, and evaluation scores.
