"""Retrieval: vector, multi-query, hybrid (BM25+vector), and reranking."""
from typing import Any

from .vector_store import retrieve as chroma_retrieve, list_documents
from .query_rewriting import rewrite_for_retrieval
from .reranker import rerank_with_metadata
from .hybrid_search import bm25_retrieve, reciprocal_rank_fusion
from .providers import EmbeddingProvider, LLMProvider
from .options import RetrieverStrategy
from .config import TOP_K_RETRIEVE, TOP_K_RERANK, MULTI_QUERY_N


def _dedupe_candidates(candidates: list[tuple[str, dict, float]]) -> list[tuple[str, dict, float]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, dict, float]] = []
    for doc, meta, score in candidates:
        key = (doc[:200], str(meta.get("source", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append((doc, meta, score))
    return out


def _vector_candidates(
    queries: list[str],
    *,
    embedder: EmbeddingProvider,
    embedding_model: str | None,
    top_k_retrieve: int,
    where: dict | None,
) -> list[tuple[str, dict, float]]:
    seen_docs: set[tuple[str, str]] = set()
    all_candidates: list[tuple[str, dict, float]] = []
    for q in queries:
        emb = embedder.embed([q], model=embedding_model)[0]
        hits = chroma_retrieve(emb, top_k=top_k_retrieve, where=where)
        for doc, meta, dist in hits:
            key = (doc[:200], str(meta.get("source", "")))
            if key in seen_docs:
                continue
            seen_docs.add(key)
            all_candidates.append((doc, meta, -dist))
    return all_candidates


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
    metadata_filter: dict | None = None,
    llm: LLMProvider | None = None,
    llm_model: str | None = None,
) -> list[tuple[str, dict[str, Any], float]]:
    """
    Retrieve relevant chunks using vector, multi-query, or hybrid search, then rerank.
    Returns list of (document_text, metadata, rerank_score).
    """
    top_k_retrieve = top_k_retrieve or TOP_K_RETRIEVE
    top_k_rerank = top_k_rerank or TOP_K_RERANK
    n_queries = n_queries or MULTI_QUERY_N

    if retriever == "hybrid":
        emb = embedder.embed([query], model=embedding_model)[0]
        vector_hits = chroma_retrieve(emb, top_k=top_k_retrieve, where=metadata_filter)
        vector_ranked = [(d, m, -dist) for d, m, dist in vector_hits]

        docs, metas = list_documents(where=metadata_filter)
        bm25_ranked = bm25_retrieve(query, docs, metas, top_k=top_k_retrieve)
        fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked])[:top_k_retrieve]
        all_candidates = _dedupe_candidates(fused)
    else:
        use_multi_query = retriever == "multi_query" if multi_query is None else multi_query
        queries = [query]
        if use_multi_query and n_queries > 1:
            rewritten = rewrite_for_retrieval(query, n_queries=n_queries, llm=llm, llm_model=llm_model)
            if rewritten:
                queries = rewritten
        all_candidates = _vector_candidates(
            queries,
            embedder=embedder,
            embedding_model=embedding_model,
            top_k_retrieve=top_k_retrieve,
            where=metadata_filter,
        )

    if not all_candidates:
        return []

    docs = [c[0] for c in all_candidates]
    metas = [c[1] for c in all_candidates]
    return rerank_with_metadata(
        query, list(zip(docs, metas)), top_k=top_k_rerank, rerank_enabled=rerank_enabled
    )
