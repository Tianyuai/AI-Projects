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

def test_cpu_first_validation_docs_are_copyable_and_consistent() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    onboarding = Path("docs/TEAMMATE_ONBOARDING.md").read_text(encoding="utf-8")
    data_readme = Path("data/README.md").read_text(encoding="utf-8")
    combined = "\n".join((readme, onboarding, data_readme))

    assert "uv sync --locked --extra cpu" in readme
    assert "uv sync --locked --extra cpu" in onboarding
    assert "uv sync --locked --extra cpu" in data_readme
    assert "uv sync --all-groups" not in combined
    assert "uv sync --locked --extra cuda" in readme
    assert "python -m paper_search.health" in readme
    assert "--require-accelerator cuda" in readme
    assert "--no-env-file" in combined
    assert "fresh clone" in readme.lower()


def test_human_annotation_docs_publish_the_exact_private_workflow() -> None:
    onboarding = Path("docs/TEAMMATE_ONBOARDING.md").read_text(encoding="utf-8")
    data_readme = Path("data/README.md").read_text(encoding="utf-8")

    for value in (
        "--kind type-domain",
        "type_domain_labels.jsonl",
        "type_domain_annotation.ids.json",
        "constraint_labels.jsonl",
        "constraint_annotation.ids.json",
        "overlap_labels.jsonl",
        "overlap_annotation.ids.json",
        "domain-labels-v1",
        "member-a",
        "member-b",
        "90",
        "40",
        "20",
        "0.80",
        "SHA-256",
    ):
        assert value in onboarding

    assert "仅标注输入" in onboarding
    assert "waiting_for_human_label_freeze" in onboarding
    assert "等待主负责人发出" not in data_readme
    assert "只固定标注输入" in data_readme
    assert "waiting_for_human_label_freeze" in data_readme
