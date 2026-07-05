from unittest.mock import MagicMock, patch

from rag_python.options import QueryConfig
from rag_python.rag_pipeline import query_stream


def test_query_stream_yields_tokens_and_result():
    llm = MagicMock()
    llm.generate_stream.return_value = iter(["Twenty ", "days ", "of leave."])

    embedder = MagicMock()
    embedder.embed.return_value = [[0.5, 0.5, 0.5]]

    with (
        patch("rag_python.rag_pipeline.check_prompt_injection", return_value=(True, "")),
        patch("rag_python.rag_pipeline.rag_retrieve") as mock_retrieve,
        patch("rag_python.rag_pipeline.generate_stream") as mock_stream,
        patch("rag_python.rag_pipeline.check_hallucination", return_value=(True, "")),
        patch("rag_python.rag_pipeline.evaluate_rag", return_value={"faithfulness": 0.9, "relevance": 0.9}),
        patch("rag_python.rag_pipeline.should_retry", return_value=False),
    ):
        mock_retrieve.return_value = [
            ("Employees receive twenty days of annual leave.", {"source": "doc.txt"}, 0.95),
        ]
        mock_stream.return_value = iter(["Twenty ", "days."])

        stream = query_stream(
            "How many days of annual leave?",
            query_config=QueryConfig(use_guardrails=True, use_retry=False),
            llm=llm,
            embedder=embedder,
        )
        tokens = list(stream)

    assert tokens == ["Twenty ", "days."]
    assert stream.result.answer == "Twenty days."
    assert stream.result.sources
    mock_stream.assert_called_once()
