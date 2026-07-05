"""High-level RAG client API.

This wraps the full RAG pipeline behind a simple interface:

    from rag_python import RAG, RAGConfig, ChunkingConfig, SearchConfig

    rag = RAG(
        llm_model="gpt-4o-mini",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        config=RAGConfig(
            chunking=ChunkingConfig(strategy="recursive", chunk_size=512),
            search=SearchConfig(retriever="multi_query", top_k_retrieve=20),
        ),
    )
    rag.ingest(["./docs", "./policies.pdf", "README.md"])
    answer = rag.query("What is our leave policy?")
    print(answer.text)
"""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .config import CHROMA_PERSIST_DIR, DATA_DIR, EMBEDDING_MODEL, LLM_MODEL
from .options import (
    ChunkingConfig,
    DocumentConfig,
    QueryConfig,
    RAGConfig,
    SearchConfig,
)
from .providers import make_llm_provider, make_embedding_provider
from .rag_pipeline import ingest as _ingest, query as _query, RAGResponse
from .vector_store import set_persist_dir


@dataclass
class RAGAnswer:
    text: str
    sources: list[dict]
    evaluation: dict
    retried: bool


class RAG:
    """User-facing RAG client with configurable chunking, retrieval, and embeddings."""

    def __init__(
        self,
        *,
        llm_provider: str = "openai",
        llm_model: str | None = None,
        embedding_provider: str = "openai",
        embedding_model: str | None = None,
        data_dir: str | Path | None = None,
        chroma_dir: str | Path | None = None,
        config: RAGConfig | None = None,
        # Shorthand overrides (merged into ``config`` when provided)
        chunk_strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        retriever: str | None = None,
        top_k_retrieve: int | None = None,
        top_k_rerank: int | None = None,
        multi_query_n: int | None = None,
        rerank_enabled: bool | None = None,
        document_extensions: tuple[str, ...] | None = None,
        # Provider kwargs (optional)
        openai_api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_api_key: str | None = None,
        azure_api_version: str | None = None,
        anthropic_api_key: str | None = None,
        gemini_api_key: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        self.llm_provider_name = llm_provider
        self.embedding_provider_name = embedding_provider
        self.llm_model = llm_model or LLM_MODEL
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.data_dir = Path(data_dir) if data_dir else Path(DATA_DIR)

        if chroma_dir:
            set_persist_dir(chroma_dir)
        elif CHROMA_PERSIST_DIR:
            set_persist_dir(CHROMA_PERSIST_DIR)

        self.config = config or RAGConfig()
        if chunk_strategy is not None:
            self.config.chunking = replace(self.config.chunking, strategy=chunk_strategy)  # type: ignore[arg-type]
        if chunk_size is not None:
            self.config.chunking = replace(self.config.chunking, chunk_size=chunk_size)
        if chunk_overlap is not None:
            self.config.chunking = replace(self.config.chunking, chunk_overlap=chunk_overlap)
        if retriever is not None:
            self.config.search = replace(self.config.search, retriever=retriever)  # type: ignore[arg-type]
        if top_k_retrieve is not None:
            self.config.search = replace(self.config.search, top_k_retrieve=top_k_retrieve)
        if top_k_rerank is not None:
            self.config.search = replace(self.config.search, top_k_rerank=top_k_rerank)
        if multi_query_n is not None:
            self.config.search = replace(self.config.search, multi_query_n=multi_query_n)
        if rerank_enabled is not None:
            self.config.search = replace(self.config.search, rerank_enabled=rerank_enabled)
        if document_extensions is not None:
            self.config.documents = replace(self.config.documents, extensions=document_extensions)

        self.llm = make_llm_provider(
            llm_provider,  # type: ignore[arg-type]
            api_key=openai_api_key or anthropic_api_key or gemini_api_key or azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            base_url=ollama_base_url,
        )
        self.embedder = make_embedding_provider(
            embedding_provider,  # type: ignore[arg-type]
            api_key=openai_api_key or azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            base_url=ollama_base_url,
        )

    def ingest(self, paths: Iterable[str | Path], *, reindex: bool = False) -> int:
        """Ingest one or more files/directories into the vector store."""
        path_list = [Path(p) for p in paths]
        doc_cfg: DocumentConfig = self.config.documents
        chunk_cfg: ChunkingConfig = self.config.chunking

        if doc_cfg.copy_to_data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            for p in path_list:
                if p.is_file():
                    target = self.data_dir / p.name
                    if str(p.resolve()) != str(target.resolve()):
                        target.write_bytes(p.read_bytes())
                elif p.is_dir():
                    for f in p.rglob("*"):
                        if f.is_file():
                            rel = f.relative_to(p)
                            target = self.data_dir / rel
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if str(f.resolve()) != str(target.resolve()):
                                target.write_bytes(f.read_bytes())
            return _ingest(
                data_path=self.data_dir,
                clean=doc_cfg.clean,
                chunk_strategy=chunk_cfg.strategy,
                chunk_size=chunk_cfg.chunk_size,
                chunk_overlap=chunk_cfg.chunk_overlap,
                extensions=doc_cfg.extensions,
                reindex=reindex,
                embedding_model=self.embedding_model,
                embedder=self.embedder,
            )

        return _ingest(
            paths=path_list,
            clean=doc_cfg.clean,
            chunk_strategy=chunk_cfg.strategy,
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
            extensions=doc_cfg.extensions,
            reindex=reindex,
            embedding_model=self.embedding_model,
            embedder=self.embedder,
        )

    def query(
        self,
        question: str,
        *,
        search: SearchConfig | None = None,
        query_config: QueryConfig | None = None,
    ) -> RAGAnswer:
        """Run a full RAG query and return a friendly answer object."""
        resp: RAGResponse = _query(
            question,
            search=search or self.config.search,
            query_config=query_config or self.config.query,
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            llm=self.llm,
            embedder=self.embedder,
        )
        return RAGAnswer(
            text=resp.answer,
            sources=resp.sources,
            evaluation=resp.evaluation,
            retried=resp.retried,
        )
