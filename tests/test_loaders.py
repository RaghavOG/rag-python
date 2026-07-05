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


def test_load_csv_file(tmp_path: Path):
    f = tmp_path / "data.csv"
    f.write_text("name,days\nAlice,20\nBob,15\n", encoding="utf-8")
    doc = load_file(f)
    assert doc is not None
    assert "Alice" in doc.content
    assert doc.metadata.get("rows") == 2


def test_load_json_file(tmp_path: Path):
    f = tmp_path / "data.json"
    f.write_text('[{"text": "Annual leave is twenty days."}]', encoding="utf-8")
    doc = load_file(f)
    assert doc is not None
    assert "twenty days" in doc.content


def test_load_html_file(tmp_path: Path):
    f = tmp_path / "page.html"
    f.write_text("<html><body><h1>Policy</h1><p>Twenty days leave.</p></body></html>", encoding="utf-8")
    doc = load_file(f)
    assert doc is not None
    assert "Policy" in doc.content
    assert "Twenty days" in doc.content


def test_load_directory_skips_empty_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("content a", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    docs = list(load_directory(tmp_path))
    assert len(docs) == 1
    assert docs[0].metadata["filename"] == "a.txt"
