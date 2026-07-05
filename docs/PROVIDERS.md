## Providers

MiniRAG separates **LLM provider** (generation, rewriting, guardrails) from **embedding provider** (retrieval).

### LLM providers

| Provider | `llm_provider=` | Env |
|----------|-----------------|-----|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Azure OpenAI | `azure_openai` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` + `pip install minirag[anthropic]` |
| Gemini | `gemini` | `GEMINI_API_KEY` + `pip install minirag[gemini]` |
| Ollama | `ollama` | Ollama running locally |

### Embedding providers

| Provider | `embedding_provider=` |
|----------|----------------------|
| OpenAI | `openai` |
| Azure OpenAI | `azure_openai` |
| Ollama | `ollama` |

Anthropic and Gemini are **LLM-only** in this version — pair them with OpenAI or Ollama embeddings.

### Reranking (optional)

```bash
pip install minirag[rerank]
```

Uses `sentence-transformers` cross-encoder (`BAAI/bge-reranker-base` by default).  
Without `[rerank]`, retrieval order is used as fallback.

### Examples

```python
# OpenAI end-to-end
rag = RAG(llm_provider="openai", embedding_provider="openai")

# Claude + OpenAI embeddings
rag = RAG(llm_provider="anthropic", llm_model="claude-opus-4-6", embedding_provider="openai")

# Fully local with Ollama
rag = RAG(llm_provider="ollama", llm_model="llama3.1", embedding_provider="ollama", embedding_model="mxbai-embed-large")
```
