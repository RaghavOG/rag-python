# Changelog

All notable changes to **RAGKit** (`ragkit` on PyPI) are documented here.

## [0.1.0] - 2026-07-05

### Added
- Production `src/ragkit/` package layout (PyPI-ready)
- High-level `RAG` client with multi-provider LLM support (OpenAI, Azure, Anthropic, Gemini, Ollama)
- Full RAG pipeline: ingest → query rewrite → multi-query retrieval → rerank → generate → guardrails → eval/retry
- CLI: `ragkit ingest` / `ragkit query`
- Optional extras: `[rerank]`, `[anthropic]`, `[gemini]`, `[all]`

### Changed
- Renamed public package from `complete-basic-rag` → **`ragkit`**
- Moved all implementation under installable `src/ragkit/`
- Data/chroma paths default to cwd (`RAGKIT_DATA_DIR`, `RAGKIT_CHROMA_DIR`)
- Reranking moved to optional dependency (`pip install ragkit[rerank]`)

[0.1.0]: https://github.com/RaghavOG/rag-python/releases/tag/v0.1.0
