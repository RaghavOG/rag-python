import subprocess
import sys

import rag_python.help_text as help_text


def test_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "rag_python.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "rag-python ingest" in result.stdout or "ingest" in result.stdout
    assert "docs" in result.stdout


def test_cli_docs_quickstart():
    result = subprocess.run(
        [sys.executable, "-m", "rag_python.cli", "docs", "quickstart"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Quick start" in result.stdout
    assert "pip install rag-python" in result.stdout


def test_cli_docs_list():
    result = subprocess.run(
        [sys.executable, "-m", "rag_python.cli", "docs", "--list"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for topic in help_text.list_topics():
        assert topic in result.stdout


def test_cli_ingest_help():
    result = subprocess.run(
        [sys.executable, "-m", "rag_python.cli", "ingest", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--reindex" in result.stdout
    assert ".pdf" in result.stdout or "Supported" in result.stdout
