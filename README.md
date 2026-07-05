# RAGKit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-RaghavOG%2Frag--python-blue)](https://github.com/RaghavOG/rag-python)

**RAGKit** is a production-oriented Python library for **Retrieval-Augmented Generation (RAG)**.

Ingest your documents, ask questions, get grounded answers — with query rewriting, multi-query retrieval, reranking, guardrails, and multi-LLM support.

**Author:** [Raghav Singla](https://github.com/RaghavOG)  
**Repository:** [github.com/RaghavOG/rag-python](https://github.com/RaghavOG/rag-python)

---

## Features

- Document pipeline: loaders → cleaning → chunking → embeddings → ChromaDB
- Query pipeline: rewriting → multi-query retrieval → reranking
- Generation with guardrails (prompt injection + hallucination checks)
- Evaluation scores + self-correction retry loop
- **LLM providers:** OpenAI, Azure OpenAI, Anthropic, Gemini, Ollama

---

## Install

```bash
pip install ragkit
# or from source
pip install -e .
# with reranking + extra providers
pip install -e ".[rerank,anthropic,gemini,all]"
```

---

## Quickstart

```python
from ragkit import RAG

rag = RAG(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)

rag.ingest(["./data"], reindex=True)
answer = rag.query("How many days of annual leave?")
print(answer.text)
```

### CLI

```bash
export OPENAI_API_KEY=sk-...
ragkit ingest ./data --reindex
ragkit query "How many days of annual leave?" -v
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For OpenAI | Default LLM + embeddings |
| `ANTHROPIC_API_KEY` | For Claude | LLM only |
| `GEMINI_API_KEY` | For Gemini | LLM only |
| `AZURE_OPENAI_ENDPOINT` | For Azure | Azure OpenAI |
| `AZURE_OPENAI_API_KEY` | For Azure | Azure OpenAI |
| `OPENAI_API_VERSION` | Azure | Default `2023-09-01-preview` |
| `OLLAMA_BASE_URL` | Ollama | Default `http://localhost:11434` |
| `RAGKIT_DATA_DIR` | Optional | Default `./data` |
| `RAGKIT_CHROMA_DIR` | Optional | Default `./chroma_db` |

See [`.env.example`](.env.example) for all tuning options.

---

## Project structure

```text
.
├── src/ragkit/          # Installable package (PyPI)
│   ├── client.py        # High-level RAG API
│   ├── rag_pipeline.py  # Full pipeline
│   └── providers/       # OpenAI, Azure, Anthropic, Gemini, Ollama
├── tests/
├── examples/
├── docs/
├── data/                # Sample documents
├── pyproject.toml
└── main.py              # Local dev CLI wrapper
```

---

## Publish to PyPI (maintainers)

```bash
pip install build twine
python -m build
twine upload --repository testpypi dist/*   # test first
twine upload dist/*
```

---

## Docs

- [Usage](docs/USAGE.md)
- [Providers](docs/PROVIDERS.md)
- [Changelog](CHANGELOG.md)

---

## License

MIT © [Raghav Singla](https://github.com/RaghavOG)
