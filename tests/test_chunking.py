from rag_python.chunking import chunk_recursive, chunk_structure_aware, chunk_text


def test_chunk_recursive_splits_long_text():
    text = "word " * 500
    chunks = chunk_recursive(text, chunk_size=64, overlap=8, metadata={"source": "t.txt"})
    assert len(chunks) > 1
    assert all(c.metadata["chunk_strategy"] == "recursive" for c in chunks)
    assert all(c.metadata["source"] == "t.txt" for c in chunks)


def test_chunk_structure_aware_respects_headings():
    text = "# Policy\n\nAnnual leave is 20 days.\n\n# Benefits\n\nHealth insurance is provided."
    chunks = chunk_structure_aware(text, chunk_size=512, overlap=0)
    assert len(chunks) >= 2
    sections = {c.metadata.get("section") for c in chunks}
    assert "Policy" in sections
    assert "Benefits" in sections


def test_chunk_text_unified_entry():
    text = "Simple paragraph for testing."
    chunks = chunk_text(text, strategy="recursive", metadata={"source": "x"})
    assert len(chunks) >= 1
    assert chunks[0].text
