import importlib.metadata


def test_package_metadata():
    dist = importlib.metadata.metadata("rag-python")
    assert dist["Name"] == "rag-python"
    assert dist["Version"] == "0.3.1"
    author = dist.get("Author") or dist.get("Author-email") or ""
    assert "Raghav Singla" in author or "RaghavOG" in author


def test_cli_entry_point():
    eps = importlib.metadata.entry_points(group="console_scripts")
    names = {ep.name for ep in eps}
    assert "rag-python" in names
