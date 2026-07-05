def test_import_minirag():
    import minirag
    from minirag import RAG, RAGAnswer, ingest, query

    assert minirag.__version__ == "0.1.0"
    assert RAG is not None
    assert RAGAnswer is not None
    assert callable(ingest)
    assert callable(query)
