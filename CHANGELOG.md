# Changelog

All notable changes to **rag-python** are documented here.

## [0.3.1] - 2026-07-05

### Added
- `rag-python docs` — built-in terminal documentation (`quickstart`, `cli`, `config`, `providers`, …)
- Full docs site: `docs/README.md`, `USAGE.md`, `CLI.md`, `CONFIGURATION.md`, `PROVIDERS.md`
- PyPI README overhaul with install extras table and documentation links

## [0.3.0] - 2026-07-05

### Added
- **Hybrid retrieval** (`retriever="hybrid"`): BM25 + vector search fused with reciprocal rank fusion — `pip install rag-python[hybrid]`
- **Metadata filtering** at query time via `SearchConfig.metadata_filter` or `RAG(metadata_filter={...})`
- Document loaders for **CSV**, **JSON**, and **HTML**
- CLI flags: `--retriever`, `--metadata-filter`, `--stream`
- **`rag.query_stream()`** — stream answer tokens for responsive UX
- Structured **logging** via `rag_python.get_logger()` / `configure_logging()`
- Pipeline logging for ingest, retrieval, guardrails, and evaluation
- **Documentation:** `rag-python docs` terminal guide, `docs/` folder, PyPI README overhaul

### Changed
- Default supported ingest extensions include `.csv`, `.json`, `.html`

## [0.2.0] - 2026-07-05

### Added
- **Local embeddings** provider (`embedding_provider="local"`) via sentence-transformers — `pip install rag-python[local]`
- Unit tests for chunking, document loaders, and mocked pipeline
- CLI `--version` flag
- PyPI badges on README

### Fixed
- CLI `--no-multi-query` now correctly switches retriever to `vector` mode

### Changed
- Removed legacy `backend/`, `ragkit/`, `minirag/` directories from workspace

## [0.1.0] - 2026-07-05

### Added
- Production `src/rag_python/` package layout (PyPI: `rag-python`)
- High-level `RAG` client with multi-provider LLM support (OpenAI, Azure, Anthropic, Gemini, Ollama)
- Full RAG pipeline: ingest → query rewrite → multi-query retrieval → rerank → generate → guardrails → eval/retry
- CLI: `rag-python ingest` / `rag-python query`
- Configurable chunking, retrieval, and multi-document ingest via `RAGConfig`
- Optional extras: `[rerank]`, `[anthropic]`, `[gemini]`, `[all]`

### Changed
- PyPI package name: **`rag-python`** (import: `rag_python`)
- Data/chroma paths default to cwd (`RAG_PYTHON_DATA_DIR`, `RAG_PYTHON_CHROMA_DIR`)
- Reranking moved to optional dependency (`pip install rag-python[rerank]`)

[0.3.1]: https://github.com/RaghavOG/rag-python/releases/tag/v0.3.1
[0.3.0]: https://github.com/RaghavOG/rag-python/releases/tag/v0.3.0
[0.2.0]: https://github.com/RaghavOG/rag-python/releases/tag/v0.2.0
[0.1.0]: https://github.com/RaghavOG/rag-python/releases/tag/v0.1.0
