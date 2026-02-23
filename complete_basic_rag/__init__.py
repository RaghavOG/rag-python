"""Public package API.

Preferred imports:

    from complete_basic_rag import RAG
"""

from backend.client import RAG, RAGAnswer
from backend.rag_pipeline import ingest, query, RAGResponse
from backend.providers import make_llm_provider, make_embedding_provider

__all__ = [
    "RAG",
    "RAGAnswer",
    "ingest",
    "query",
    "RAGResponse",
    "make_llm_provider",
    "make_embedding_provider",
]

