from pathlib import Path

from rag_python.document_loaders import load_file, load_directory


def test_load_txt_file(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("Hello from rag-python.", encoding="utf-8")
    doc = load_file(f)
    assert doc is not None
    assert "Hello" in doc.content
    assert doc.metadata["filename"] == "note.txt"


def test_load_markdown_file(tmp_path: Path):
    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nBody text.", encoding="utf-8")
    doc = load_file(f)
    assert doc is not None
    assert "Title" in doc.content


def test_load_directory_skips_empty_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("content a", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    docs = list(load_directory(tmp_path))
    assert len(docs) == 1
    assert docs[0].metadata["filename"] == "a.txt"
