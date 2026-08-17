from __future__ import annotations

import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path


def test_project_declares_only_supported_runtime_dependencies() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = project["dependencies"]

    assert project["scripts"] == {"paper-search": "paper_search.cli:main"}
    assert any(item.startswith("rank-bm25") for item in dependencies)
    assert not any(
        item.startswith(prefix)
        for item in dependencies
        for prefix in ("torch", "sentence-transformers", "transformers", "faiss-cpu")
    )
    assert "optional-dependencies" not in project


def test_ui_assets_are_in_built_distributions(tmp_path: Path) -> None:
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
