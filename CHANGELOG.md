# Changelog

All notable changes to **rag-python** are documented here.

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

[0.2.0]: https://github.com/RaghavOG/rag-python/releases/tag/v0.2.0
[0.1.0]: https://github.com/RaghavOG/rag-python/releases/tag/v0.1.0
