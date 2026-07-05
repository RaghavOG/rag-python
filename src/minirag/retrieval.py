"""Retrieval: multi-query retrieval + reranking."""
from typing import Any

from .vector_store import retrieve as chroma_retrieve
from .query_rewriting import rewrite_for_retrieval
from .reranker import rerank_with_metadata
from .providers import EmbeddingProvider, LLMProvider
from .options import RetrieverStrategy
from .config import TOP_K_RETRIEVE, TOP_K_RERANK, MULTI_QUERY_N


def retrieve(
    query: str,
    *,
    embedder: EmbeddingProvider,
    embedding_model: str | None = None,
    retriever: RetrieverStrategy = "multi_query",
    multi_query: bool | None = None,
    n_queries: int | None = None,
    top_k_retrieve: int | None = None,
    top_k_rerank: int | None = None,
    rerank_enabled: bool | None = None,
    llm: LLMProvider | None = None,
    llm_model: str | None = None,
) -> list[tuple[str, dict[str, Any], float]]:
    """
    Retrieve relevant chunks using vector or multi-query search, then rerank.
    Returns list of (document_text, metadata, rerank_score).
    """
    top_k_retrieve = top_k_retrieve or TOP_K_RETRIEVE
    top_k_rerank = top_k_rerank or TOP_K_RERANK
    n_queries = n_queries or MULTI_QUERY_N
    use_multi_query = retriever == "multi_query" if multi_query is None else multi_query

    queries = [query]
    if use_multi_query and n_queries > 1:
        rewritten = rewrite_for_retrieval(query, n_queries=n_queries, llm=llm, llm_model=llm_model)
        if rewritten:
            queries = rewritten

    seen_docs: set[str] = set()
    all_candidates: list[tuple[str, dict, float]] = []
    for q in queries:
        emb = embedder.embed([q], model=embedding_model)[0]
        hits = chroma_retrieve(emb, top_k=top_k_retrieve)
        for doc, meta, dist in hits:
            key = (doc[:200], meta.get("source", ""))
            if key in seen_docs:
                continue
            seen_docs.add(key)
            all_candidates.append((doc, meta, -dist))

    if not all_candidates:
        return []
    docs = [c[0] for c in all_candidates]
    metas = [c[1] for c in all_candidates]
    reranked = rerank_with_metadata(
        query, list(zip(docs, metas)), top_k=top_k_rerank, rerank_enabled=rerank_enabled
    )
    return reranked

