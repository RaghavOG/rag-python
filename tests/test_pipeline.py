from unittest.mock import MagicMock, patch

from rag_python.rag_pipeline import ingest, query


def test_ingest_with_mocked_embedder(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Employees receive twenty days of annual leave per year.", encoding="utf-8")

    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with patch("rag_python.rag_pipeline.ingest_chunks") as mock_ingest:
        n = ingest(
            data_path=f,
            clean=False,
            chunk_strategy="recursive",
            chunk_size=128,
            chunk_overlap=0,
            reindex=False,
            embedder=embedder,
        )
    assert n >= 1
    mock_ingest.assert_called_once()
    assert mock_ingest.call_args.kwargs["embedder"] is embedder


def test_query_with_mocked_providers():
    llm = MagicMock()
    llm.generate.return_value = "Twenty days of annual leave."

    embedder = MagicMock()
    embedder.embed.return_value = [[0.5, 0.5, 0.5]]

    with (
        patch("rag_python.rag_pipeline.check_prompt_injection", return_value=(True, "")),
        patch("rag_python.rag_pipeline.rag_retrieve") as mock_retrieve,
        patch("rag_python.rag_pipeline.check_hallucination", return_value=(True, "")),
        patch("rag_python.rag_pipeline.evaluate_rag", return_value={"faithfulness": 0.9, "relevance": 0.9}),
        patch("rag_python.rag_pipeline.should_retry", return_value=False),
    ):
        mock_retrieve.return_value = [
            ("Employees receive twenty days of annual leave.", {"source": "doc.txt"}, 0.95),
        ]
        resp = query(
            "How many days of annual leave?",
            use_guardrails=True,
            use_retry=False,
            llm=llm,
            embedder=embedder,
        )

    assert "twenty" in resp.answer.lower() or "Twenty" in resp.answer
    assert resp.sources
    mock_retrieve.assert_called_once()
