"""Backend package for the RAG system."""

from .rag_pipeline import ingest, query, RAGResponse
from .client import RAG, RAGAnswer

__all__ = ["ingest", "query", "RAGResponse", "RAG", "RAGAnswer"]

