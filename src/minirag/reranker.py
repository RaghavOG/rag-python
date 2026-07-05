"""Reranking: cross-encoder to score query–document relevance."""
from typing import Any

from .config import RERANKER_MODEL, RERANK_ENABLED

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "Reranking requires optional dependencies. Install with: pip install minirag[rerank]"
            ) from e
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(
    query: str,
    documents: list[str],
    top_k: int = 5,
    *,
    rerank_enabled: bool | None = None,
) -> list[tuple[str, float, int]]:
    """Return (document, score, original_index) sorted by relevance."""
    if not documents:
        return []
    use_rerank = RERANK_ENABLED if rerank_enabled is None else rerank_enabled
    if not use_rerank:
        return [(documents[i], 0.0, i) for i in range(min(top_k, len(documents)))]

    try:
        pairs = [(query, d) for d in documents]
        model = _get_reranker()
        scores = model.predict(pairs)
        indexed = [(documents[i], float(scores[i]), i) for i in range(len(documents))]
        indexed.sort(key=lambda x: -x[1])
        return indexed[: top_k if top_k < len(indexed) else len(indexed)]
    except ImportError:
        # Graceful fallback: keep retrieval order
        return [(documents[i], 0.0, i) for i in range(min(top_k, len(documents)))]


def rerank_with_metadata(
    query: str,
    doc_meta_list: list[tuple[str, dict[str, Any]]],
    top_k: int = 5,
    *,
    rerank_enabled: bool | None = None,
) -> list[tuple[str, dict[str, Any], float]]:
    """Rerank (document, metadata) pairs; return (doc, meta, score)."""
    if not doc_meta_list:
        return []
    docs = [d for d, _ in doc_meta_list]
    metas = [m for _, m in doc_meta_list]
    results = rerank(query, docs, top_k=min(top_k, len(docs)), rerank_enabled=rerank_enabled)
    out = []
    for doc, score, idx in results:
        out.append((doc, metas[idx], score))
    return out
