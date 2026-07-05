def test_import_ragkit():
    import ragkit
    from ragkit import RAG, RAGAnswer, ingest, query

    assert ragkit.__version__ == "0.1.0"
    assert RAG is not None
    assert RAGAnswer is not None
    assert callable(ingest)
    assert callable(query)
