def test_import_rag_python():
    import rag_python
    from rag_python import RAG, RAGAnswer, ingest, query

    assert rag_python.__version__ == "0.1.0"
    assert RAG is not None
    assert RAGAnswer is not None
    assert callable(ingest)
    assert callable(query)
