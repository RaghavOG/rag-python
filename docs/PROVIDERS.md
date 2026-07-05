# Providers

rag-python separates **LLM providers** (generation, query rewriting, guardrails, evaluation) from **embedding providers** (vector retrieval).

## LLM providers

| Provider | `llm_provider=` | Install / env |
|----------|-----------------|---------------|
| OpenAI | `openai` (default) | `OPENAI_API_KEY` |
| Azure OpenAI | `azure_openai` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` |
| Anthropic (Claude) | `anthropic` | `ANTHROPIC_API_KEY` + `pip install rag-python[anthropic]` |
| Gemini | `gemini` | `GEMINI_API_KEY` + `pip install rag-python[gemini]` |
| Ollama | `ollama` | Ollama running locally; set `--llm-model` |

### Streaming support

Token streaming (`query_stream`) is supported on OpenAI, Azure OpenAI, Ollama, and Anthropic. Other providers yield the full answer in one chunk.

---

## Embedding providers

| Provider | `embedding_provider=` | Notes |
|----------|----------------------|-------|
| OpenAI | `openai` (default) | `OPENAI_API_KEY` |
| Azure OpenAI | `azure_openai` | Use your embedding deployment name |
| Ollama | `ollama` | Requires embedding model in Ollama |
| Local (offline) | `local` | `pip install rag-python[local]` |

Anthropic and Gemini are **LLM-only** — pair them with OpenAI, Ollama, or local embeddings.

---

## Optional features

### Reranking

```bash
pip install rag-python[rerank]
```

Uses `sentence-transformers` cross-encoder (`BAAI/bge-reranker-base` by default). Without `[rerank]`, retrieval order is used.

### Hybrid search

```bash
pip install rag-python[hybrid]
```

Combines BM25 keyword search with vector similarity via reciprocal rank fusion.

---

## Examples

### OpenAI end-to-end

```python
from rag_python import RAG

rag = RAG(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)
```

### Claude + OpenAI embeddings

```python
rag = RAG(
    llm_provider="anthropic",
    llm_model="claude-opus-4-6",
    embedding_provider="openai",
)
```

### Fully local embeddings

```python
rag = RAG(
    llm_provider="ollama",
    llm_model="llama3.1",
    embedding_provider="local",
    embedding_model="all-MiniLM-L6-v2",
)
```

### Azure OpenAI

```python
rag = RAG(
    llm_provider="azure_openai",
    llm_model="your-chat-deployment",
    embedding_provider="azure_openai",
    embedding_model="your-embedding-deployment",
    azure_endpoint="https://<resource>.openai.azure.com",
    azure_api_key="...",
)
```

---

## CLI provider flags

```bash
rag-python query "question" --llm-provider anthropic --llm-model claude-opus-4-6
rag-python ingest ./data --embedding-provider local
rag-python query "question" --llm-provider ollama --llm-model llama3.1 --ollama-base-url http://localhost:11434
```

See [CLI reference](CLI.md) for all flags.
