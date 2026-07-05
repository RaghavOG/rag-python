"""BM25 + vector fusion via reciprocal rank fusion (RRF)."""
from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, dict[str, Any], float]]],
    *,
    rrf_k: int = 60,
) -> list[tuple[str, dict[str, Any], float]]:
    """Merge ranked lists with RRF. Higher score is better."""
    scores: dict[tuple[str, str], float] = {}
    doc_map: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}

    for ranking in rankings:
        for rank, (doc, meta, _score) in enumerate(ranking):
            key = (doc[:200], str(meta.get("source", "")))
            doc_map[key] = (doc, meta)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    merged = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(doc_map[key][0], doc_map[key][1], score) for key, score in merged]


def bm25_retrieve(
    query: str,
    documents: list[str],
    metadatas: list[dict[str, Any]],
    *,
    top_k: int = 20,
) -> list[tuple[str, dict[str, Any], float]]:
    """Keyword retrieval with BM25. Requires ``pip install rag-python[hybrid]``."""
    if not documents:
        return []
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise ImportError(
            "Hybrid search requires optional dependencies. Install with: pip install rag-python[hybrid]"
        ) from e

    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(
        ((documents[i], metadatas[i], float(scores[i])) for i in range(len(documents))),
        key=lambda item: item[2],
        reverse=True,
    )
    return ranked[:top_k]
