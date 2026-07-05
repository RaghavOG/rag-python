from rag_python.providers import make_embedding_provider


def test_make_local_embedding_provider():
    provider = make_embedding_provider("local", model="all-MiniLM-L6-v2")
    assert provider.default_model == "all-MiniLM-L6-v2"
