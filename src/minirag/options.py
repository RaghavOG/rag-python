"""User-configurable options for RAG ingest, retrieval, and generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GUARDRAILS_ENABLED,
    MAX_RETRIES,
    MULTI_QUERY_N,
    RERANK_ENABLED,
    TOP_K_RERANK,
    TOP_K_RETRIEVE,
)

ChunkStrategy = Literal["recursive", "structure_aware", "semantic"]
RetrieverStrategy = Literal["vector", "multi_query"]


@dataclass
class ChunkingConfig:
    """How documents are split before embedding."""

    strategy: ChunkStrategy = "recursive"
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP


@dataclass
class SearchConfig:
    """Retrieval and reranking behaviour at query time."""

    retriever: RetrieverStrategy = "multi_query"
    top_k_retrieve: int = TOP_K_RETRIEVE
    top_k_rerank: int = TOP_K_RERANK
    multi_query_n: int = MULTI_QUERY_N
    rerank_enabled: bool = RERANK_ENABLED


@dataclass
class DocumentConfig:
    """Which files to load and how to preprocess them."""

    extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx")
    clean: bool = True
    copy_to_data_dir: bool = True


@dataclass
class QueryConfig:
    """Guardrails, evaluation, and retry settings."""

    use_guardrails: bool = GUARDRAILS_ENABLED
    use_retry: bool = True
    max_retries: int = MAX_RETRIES
    eval_threshold: float = 0.6


@dataclass
class RAGConfig:
    """All tunable pipeline options (pass to ``RAG(config=...)``)."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    documents: DocumentConfig = field(default_factory=DocumentConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
