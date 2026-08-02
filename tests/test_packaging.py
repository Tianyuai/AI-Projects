from __future__ import annotations

import tomllib
import subprocess
import tarfile
import zipfile
from pathlib import Path


def test_console_entry_point_is_stable() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"] == {"paper-search": "paper_search.cli:main"}


def test_ui_assets_are_in_built_wheel_and_source_distributions(tmp_path: Path) -> None:
    expected = {
        "paper_search/ui/static/index.html",
        "paper_search/ui/static/app.js",
        "paper_search/ui/static/styles.css",
    }
    subprocess.run(
        ["uv", "build", "--wheel", "--sdist", "--out-dir", str(tmp_path)],
        check=True,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    wheel = next(tmp_path.glob("*.whl"))
    source_distribution = next(tmp_path.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        wheel_entries = set(archive.namelist())
    with tarfile.open(source_distribution) as archive:
        source_entries = {member.name for member in archive.getmembers()}

    assert expected.issubset(wheel_entries)
    assert all(any(entry.endswith(f"/{asset}") for entry in source_entries) for asset in expected)


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


def test_embedding_focused_verification_command_is_portable() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    focused_command = (
        "uv run --no-sync --no-env-file pytest tests/unit/test_embedding.py "
        "tests/unit/test_sentence_transformer.py tests/evaluation/test_embedding_benchmark.py "
        "tests/integration/test_orchestrator.py -q"
    )

    assert focused_command in readme
    assert r"D:\AI Projects\Projects\.venv" not in readme
    assert r"D:\Dev\uv\uv.exe" not in readme


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
