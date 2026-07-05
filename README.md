# rag-python

[![PyPI version](https://img.shields.io/pypi/v/rag-python.svg)](https://pypi.org/project/rag-python/)
[![PyPI downloads](https://img.shields.io/pypi/dm/rag-python.svg)](https://pypi.org/project/rag-python/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-GitHub-blue)](https://github.com/RaghavOG/rag-python/tree/main/docs)

**Production-grade Retrieval-Augmented Generation (RAG) for Python** — ingest documents, ask questions, get grounded answers with multi-LLM support, hybrid search, streaming, and guardrails.

```bash
pip install rag-python
export OPENAI_API_KEY=sk-...
rag-python ingest ./docs --reindex
rag-python query "What is our leave policy?"
```

**Author:** [Raghav Singla](https://github.com/RaghavOG) · **Repo:** [github.com/RaghavOG/rag-python](https://github.com/RaghavOG/rag-python)

---

## Why rag-python?

| Capability | What you get |
|------------|--------------|
| **Ingest** | TXT, MD, PDF, DOCX, CSV, JSON, HTML → chunk → embed → ChromaDB |
| **Retrieve** | Multi-query rewriting, **hybrid BM25+vector**, reranking, metadata filters |
| **Generate** | Multi-LLM answers with guardrails, evaluation, and retry loop |
| **Stream** | `rag.query_stream()` and `--stream` CLI for responsive UX |
| **Offline** | Local embeddings via sentence-transformers |
| **CLI** | `rag-python ingest`, `query`, `docs` — no code required |

---

## Install

```bash
pip install rag-python
```

| Extra | Install | Enables |
|-------|---------|---------|
| `local` | `pip install rag-python[local]` | Offline embeddings (sentence-transformers) |
| `hybrid` | `pip install rag-python[hybrid]` | BM25 + vector hybrid retrieval |
| `rerank` | `pip install rag-python[rerank]` | Cross-encoder reranking |
| `anthropic` | `pip install rag-python[anthropic]` | Claude LLM |
| `gemini` | `pip install rag-python[gemini]` | Gemini LLM |
| `all` | `pip install rag-python[all]` | All optional features |

---

## Quickstart (Python)

```python
from rag_python import RAG

rag = RAG(llm_model="gpt-4o-mini")
rag.ingest(["./data"], reindex=True)

answer = rag.query("How many days of annual leave?")
print(answer.text)
print(answer.sources)
```

### Streaming

```python
stream = rag.query_stream("How many days of annual leave?")
for token in stream:
    print(token, end="", flush=True)
print(stream.result.evaluation)
```

### Hybrid search + metadata filter

```python
rag = RAG(
    retriever="hybrid",  # pip install rag-python[hybrid]
    metadata_filter={"filename": "leave-policy.pdf"},
)
rag.ingest(["./policies/"])
print(rag.query("annual leave policy").text)
```

---

## Quickstart (CLI)

```bash
export OPENAI_API_KEY=sk-...

rag-python ingest ./data --reindex
rag-python query "How many days of annual leave?"
rag-python query "PTO policy" --stream -v
rag-python query "benefits" --retriever hybrid

# Built-in terminal docs
rag-python docs quickstart
rag-python docs --list
rag-python --help
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| [**Docs index**](docs/README.md) | Start here |
| [Usage](docs/USAGE.md) | Python API, streaming, retrieval |
| [CLI reference](docs/CLI.md) | All `rag-python` commands and flags |
| [Configuration](docs/CONFIGURATION.md) | Env vars and `RAGConfig` |
| [Providers](docs/PROVIDERS.md) | OpenAI, Azure, Anthropic, Gemini, Ollama, local |
| [Changelog](CHANGELOG.md) | Release notes |

**In the terminal:** `rag-python docs [topic]` — topics: `quickstart`, `install`, `cli`, `config`, `providers`, `features`

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Default LLM + embeddings |
| `ANTHROPIC_API_KEY` | Claude |
| `GEMINI_API_KEY` | Gemini |
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` | Azure OpenAI |
| `OLLAMA_BASE_URL` | Local Ollama (default `http://localhost:11434`) |
| `RAG_PYTHON_DATA_DIR` | Document dir (default `./data`) |
| `RAG_PYTHON_CHROMA_DIR` | Vector store (default `./chroma_db`) |

See [Configuration](docs/CONFIGURATION.md) and [`.env.example`](.env.example).

---

## Project layout

```text
src/rag_python/     # pip install rag-python → import rag_python
  client.py         # RAG, RAGAnswer, query_stream
  rag_pipeline.py   # ingest / query pipeline
  providers/        # OpenAI, Azure, Anthropic, Gemini, Ollama, local
docs/               # User documentation (linked from PyPI README)
tests/
examples/
```

---

## License

MIT © [Raghav Singla](https://github.com/RaghavOG)
