"""Reranking: cross-encoder to score query–document relevance. Essential for quality."""
from typing import Any

from config import RERANKER_MODEL

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(
    query: str,
    documents: list[str],
    top_k: int = 5,
) -> list[tuple[str, float, int]]:
    """Return (document, score, original_index) sorted by relevance. Higher score = more relevant."""
    if not documents:
        return []
    pairs = [(query, d) for d in documents]
    model = _get_reranker()
    scores = model.predict(pairs)
    indexed = [(documents[i], float(scores[i]), i) for i in range(len(documents))]
    indexed.sort(key=lambda x: -x[1])
    return indexed[: top_k if top_k < len(indexed) else len(indexed)]


def rerank_with_metadata(
    query: str,
    doc_meta_list: list[tuple[str, dict[str, Any]]],
    top_k: int = 5,
) -> list[tuple[str, dict[str, Any], float]]:
    """Rerank (document, metadata) pairs; return (doc, meta, score)."""
    if not doc_meta_list:
        return []
    docs = [d for d, _ in doc_meta_list]
    metas = [m for _, m in doc_meta_list]
    results = rerank(query, docs, top_k=min(top_k, len(docs)))
    out = []
    for doc, score, idx in results:
        out.append((doc, metas[idx], score))
    return out

