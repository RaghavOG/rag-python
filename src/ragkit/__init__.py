"""RAGKit — production-grade modular RAG for Python.

Quick start::

    from ragkit import RAG

    rag = RAG(llm_model="gpt-4o-mini")
    rag.ingest(["./docs"], reindex=True)
    print(rag.query("What is our leave policy?").text)
"""

__version__ = "0.1.0"

from .client import RAG, RAGAnswer
from .rag_pipeline import ingest, query, RAGResponse
from .providers import make_llm_provider, make_embedding_provider

__all__ = [
    "__version__",
    "RAG",
    "RAGAnswer",
    "ingest",
    "query",
    "RAGResponse",
    "make_llm_provider",
    "make_embedding_provider",
]
