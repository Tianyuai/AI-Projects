from __future__ import annotations

import tomllib
from pathlib import Path


def test_torch_profiles_are_explicit_and_mutually_exclusive() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    optional = project["project"]["optional-dependencies"]
    conflicts = project["tool"]["uv"]["conflicts"]
    sources = project["tool"]["uv"]["sources"]["torch"]
    indexes = {item["name"]: item["url"] for item in project["tool"]["uv"]["index"]}

    assert not any(item.startswith("torch") for item in dependencies)
    assert not any(item.startswith("sentence-transformers") for item in dependencies)
    embedding_profile = ["sentence-transformers>=3.3,<6", "torch==2.5.1"]
    assert optional["cpu"] == embedding_profile
    assert optional["cuda"] == embedding_profile
    assert conflicts == [[{"extra": "cpu"}, {"extra": "cuda"}]]
    assert sources == [
        {"index": "pytorch-cu121", "extra": "cuda"},
        {"index": "pytorch-cpu", "extra": "cpu"},
    ]
    assert indexes == {
        "pytorch-cpu": "https://download.pytorch.org/whl/cpu",
        "pytorch-cu121": "https://download.pytorch.org/whl/cu121",
    }


def test_hardware_tests_are_explicitly_marked() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "hardware: requires an explicitly selected accelerator environment" in (
        project["tool"]["pytest"]["ini_options"]["markers"]
    )
