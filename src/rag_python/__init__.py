"""rag-python — production-grade RAG for Python.

Quick start::

    from rag_python import RAG

    rag = RAG(llm_model="gpt-4o-mini")
    rag.ingest(["./docs"], reindex=True)
    print(rag.query("What is our leave policy?").text)
"""

__version__ = "0.3.0"

from .client import RAG, RAGAnswer
from .rag_pipeline import ingest, query, query_stream, RAGResponse, RAGStream
from .providers import make_llm_provider, make_embedding_provider
from .log import configure_logging, get_logger
from .options import (
    ChunkingConfig,
    DocumentConfig,
    QueryConfig,
    RAGConfig,
    SearchConfig,
)

__all__ = [
    "__version__",
    "RAG",
    "RAGAnswer",
    "RAGConfig",
    "ChunkingConfig",
    "SearchConfig",
    "DocumentConfig",
    "QueryConfig",
    "ingest",
    "query",
    "query_stream",
    "RAGStream",
    "RAGResponse",
    "configure_logging",
    "get_logger",
    "make_llm_provider",
    "make_embedding_provider",
]
