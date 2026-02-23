"""Retrieval: multi-query retrieval + reranking."""
from typing import Any

from .embeddings import embed_single
from .vector_store import retrieve as chroma_retrieve
from .query_rewriting import rewrite_for_retrieval
from .reranker import rerank_with_metadata
from config import TOP_K_RETRIEVE, TOP_K_RERANK, MULTI_QUERY_N


def retrieve(
    query: str,
    *,
    multi_query: bool = True,
    n_queries: int | None = None,
    top_k_retrieve: int | None = None,
    top_k_rerank: int | None = None,
) -> list[tuple[str, dict[str, Any], float]]:
    """
    Multi-query retrieval + reranking.
    Returns list of (document_text, metadata, rerank_score).
    """
    top_k_retrieve = top_k_retrieve or TOP_K_RETRIEVE
    top_k_rerank = top_k_rerank or TOP_K_RERANK
    n_queries = n_queries or MULTI_QUERY_N

    queries = [query]
    if multi_query and n_queries > 1:
        rewritten = rewrite_for_retrieval(query, n_queries=n_queries)
        if rewritten:
            queries = rewritten

    seen_docs: set[str] = set()
    all_candidates: list[tuple[str, dict, float]] = []
    for q in queries:
        emb = embed_single(q)
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
    reranked = rerank_with_metadata(query, list(zip(docs, metas)), top_k=top_k_rerank)
    return reranked

