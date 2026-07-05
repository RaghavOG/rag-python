# rag-python documentation

Welcome to the **rag-python** user guide. This library gives you a production-ready RAG pipeline in a few lines of Python or from the terminal.

| Guide | Description |
|-------|-------------|
| [Usage](USAGE.md) | Python API, streaming, hybrid search, logging |
| [CLI reference](CLI.md) | `rag-python ingest`, `query`, `docs` |
| [Configuration](CONFIGURATION.md) | `RAGConfig`, environment variables |
| [Providers](PROVIDERS.md) | OpenAI, Azure, Anthropic, Gemini, Ollama, local |
| [Changelog](../CHANGELOG.md) | Release history |

## Terminal quick reference

Install and run without writing code:

```bash
pip install rag-python
export OPENAI_API_KEY=sk-...

rag-python ingest ./my-docs --reindex
rag-python query "What is our leave policy?"
rag-python query "annual leave" --stream -v
rag-python docs quickstart
rag-python docs --list
```

## Python quick reference

```python
from rag_python import RAG

rag = RAG(llm_model="gpt-4o-mini")
rag.ingest(["./my-docs"], reindex=True)

answer = rag.query("What is our leave policy?")
print(answer.text)
print(answer.sources)
```

## Optional extras

```bash
pip install rag-python[local]    # offline embeddings
pip install rag-python[hybrid]   # BM25 + vector search
pip install rag-python[rerank]   # cross-encoder reranking
pip install rag-python[all]      # everything
```

## Links

- PyPI: https://pypi.org/project/rag-python/
- GitHub: https://github.com/RaghavOG/rag-python
- Issues: https://github.com/RaghavOG/rag-python/issues
