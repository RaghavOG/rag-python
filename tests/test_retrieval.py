from unittest.mock import MagicMock, patch

from rag_python.retrieval import retrieve


def test_hybrid_retriever_fuses_vector_and_bm25():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2]]

    vector_hits = [("vector doc", {"source": "v.txt"}, 0.1)]
    bm25_hits = [("bm25 doc", {"source": "b.txt"}, 1.5)]
    fused = [("vector doc", {"source": "v.txt"}, 0.5), ("bm25 doc", {"source": "b.txt"}, 0.4)]

    with (
        patch("rag_python.retrieval.chroma_retrieve", return_value=vector_hits) as mock_chroma,
        patch("rag_python.retrieval.list_documents", return_value=(["bm25 doc"], [{"source": "b.txt"}])),
        patch("rag_python.retrieval.bm25_retrieve", return_value=bm25_hits) as mock_bm25,
        patch("rag_python.retrieval.reciprocal_rank_fusion", return_value=fused) as mock_rrf,
        patch("rag_python.retrieval.rerank_with_metadata", side_effect=lambda q, pairs, **kw: pairs),
    ):
        hits = retrieve(
            "leave policy",
            embedder=embedder,
            retriever="hybrid",
            rerank_enabled=False,
            metadata_filter={"filename": "policy.txt"},
        )

    mock_chroma.assert_called_once()
    assert mock_chroma.call_args.kwargs["where"] == {"filename": "policy.txt"}
    mock_bm25.assert_called_once()
    mock_rrf.assert_called_once()
    assert len(hits) == 2


def test_vector_retriever_passes_metadata_filter():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.5, 0.5]]

    with (
        patch("rag_python.retrieval.chroma_retrieve", return_value=[]) as mock_chroma,
        patch("rag_python.retrieval.rerank_with_metadata", return_value=[]),
    ):
        retrieve(
            "question",
            embedder=embedder,
            retriever="vector",
            metadata_filter={"source": "/data/policy.txt"},
        )

    mock_chroma.assert_called_once()
    assert mock_chroma.call_args.kwargs["where"] == {"source": "/data/policy.txt"}
