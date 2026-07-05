import pytest

from rag_python.hybrid_search import bm25_retrieve, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_prefers_shared_docs():
    vector = [
        ("doc a", {"source": "a"}, 0.9),
        ("doc b", {"source": "b"}, 0.8),
    ]
    bm25 = [
        ("doc b", {"source": "b"}, 0.95),
        ("doc c", {"source": "c"}, 0.7),
    ]
    merged = reciprocal_rank_fusion([vector, bm25])
    assert len(merged) == 3
    assert merged[0][0] == "doc b"


def test_bm25_retrieve_ranks_relevant_doc():
    docs = [
        "annual leave policy grants twenty days per year",
        "office cafeteria menu and lunch hours",
    ]
    metas = [{"source": "policy.txt"}, {"source": "cafe.txt"}]
    try:
        hits = bm25_retrieve("annual leave days", docs, metas, top_k=1)
    except ImportError:
        pytest.skip("rank_bm25 not installed")
    assert hits[0][0] == docs[0]
    assert hits[0][1]["source"] == "policy.txt"


def test_bm25_retrieve_empty_corpus():
    assert bm25_retrieve("query", [], [], top_k=5) == []
