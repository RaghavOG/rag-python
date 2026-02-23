"""Backend package for the RAG system."""

from .rag_pipeline import ingest, query, RAGResponse

__all__ = ["ingest", "query", "RAGResponse"]

