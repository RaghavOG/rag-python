# Changelog

All notable changes to **MiniRAG** (`minirag` on PyPI) are documented here.

## [0.1.0] - 2026-07-05

### Added
- Production `src/minirag/` package layout (PyPI-ready)
- High-level `RAG` client with multi-provider LLM support (OpenAI, Azure, Anthropic, Gemini, Ollama)
- Full RAG pipeline: ingest → query rewrite → multi-query retrieval → rerank → generate → guardrails → eval/retry
- CLI: `minirag ingest` / `minirag query`
- Optional extras: `[rerank]`, `[anthropic]`, `[gemini]`, `[all]`

### Changed
- Renamed public package from `complete-basic-rag` → **`minirag`**
- Moved all implementation under installable `src/minirag/`
- Data/chroma paths default to cwd (`MINIRAG_DATA_DIR`, `MINIRAG_CHROMA_DIR`)
- Reranking moved to optional dependency (`pip install minirag[rerank]`)

[0.1.0]: https://github.com/RaghavOG/rag-python/releases/tag/v0.1.0
