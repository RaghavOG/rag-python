import importlib.metadata


def test_package_metadata():
    dist = importlib.metadata.metadata("ragkit")
    assert dist["Name"] == "ragkit"
    assert dist["Version"] == "0.1.0"
    assert "Raghav Singla" in dist["Author"]


def test_cli_entry_point():
    eps = importlib.metadata.entry_points(group="console_scripts")
    names = {ep.name for ep in eps}
    assert "ragkit" in names
