from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest



def _load_preparation_module() -> ModuleType:
    module_path = Path("scripts/prepare_task2_data.py")
    spec = importlib.util.spec_from_file_location("prepare_task2_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preparation = _load_preparation_module()


def _source_bytes(prefix: str) -> bytes:
    rows = [
        {
            "qid": f"{prefix}-q1",
            "question": f"Synthetic question one for {prefix}",
            "answer": ["Synthetic Paper One"],
            "answer_arxiv_id": ["2501.10120"],
            "source_meta": {"published_time": "2025-01-01"},
        },
        {
            "qid": f"{prefix}-q2",
            "question": f"Synthetic question two for {prefix}",
            "answer": ["Synthetic Paper Two", "Synthetic Paper Three"],
            "answer_arxiv_id": ["1706.03762", "2305.10601"],
            "source_meta": {"published_time": "2024-01-01"},
        },
    ]
    return (
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    ).encode()


FIXTURE_FILES = {
    "AutoScholarQuery/dev.jsonl": _source_bytes("auto-dev"),
    "AutoScholarQuery/test.jsonl": _source_bytes("auto-test"),
    "RealScholarQuery/test.jsonl": _source_bytes("real-test"),
}


def _prepare_with_fixtures(output_root: Path) -> dict[str, object]:
    def downloader(repo_id: str, revision: str, path: str, token: str) -> bytes:
        assert repo_id == preparation.PASA_REPO_ID
        assert revision == preparation.PASA_REVISION
        assert token == "test-token"
        return FIXTURE_FILES[path]

    return preparation.prepare(
        output_root=output_root,
        token="test-token",
        downloader=downloader,
        expected_counts={path: 2 for path in FIXTURE_FILES},
        dev_size=1,
        validation_size=1,
        simulated_test_size=2,
        constraint_annotation_size=1,
        overlap_annotation_size=1,
    )


def test_prepare_requires_hf_token_without_calling_downloader(tmp_path: Path) -> None:
    def forbidden_downloader(
        repo_id: str,
        revision: str,
        path: str,
        token: str,
    ) -> bytes:
        del repo_id, revision, path, token
        raise AssertionError("downloader must not be called")

    with pytest.raises(RuntimeError, match="HF_TOKEN is required") as error:
        preparation.prepare(
            output_root=tmp_path,
            token=None,
            downloader=forbidden_downloader,
        )

    assert "hf_" not in str(error.value)


def test_prepare_requests_only_fixed_files_and_revision(tmp_path: Path) -> None:
    calls: list[tuple[str, str, str]] = []

    def downloader(repo_id: str, revision: str, path: str, token: str) -> bytes:
        assert token == "secret"
        calls.append((repo_id, revision, path))
        return FIXTURE_FILES[path]

    preparation.prepare(
        output_root=tmp_path,
        token="secret",
        downloader=downloader,
        expected_counts={path: 2 for path in FIXTURE_FILES},
        dev_size=1,
        validation_size=1,
        simulated_test_size=2,
        constraint_annotation_size=1,
        overlap_annotation_size=1,
    )

    assert calls == [
        (preparation.PASA_REPO_ID, preparation.PASA_REVISION, path)
        for path in preparation.PASA_FILES
    ]


def test_prepare_manifest_is_reproducible_and_idempotent(tmp_path: Path) -> None:
    first = _prepare_with_fixtures(tmp_path)
    second = _prepare_with_fixtures(tmp_path)

    assert first == second
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == first
    assert manifest["repo_id"] == preparation.PASA_REPO_ID
    assert manifest["revision"] == preparation.PASA_REVISION
    assert manifest["random_seed"] == 20260714
    assert manifest["sampling_algorithm"] == "answer-count-largest-remainder-v1"
    assert manifest["status"] == "waiting_for_human_label_freeze"
    assert all("sha256" in item for item in manifest["source_files"])
    assert (tmp_path / "splits" / "dev.ids.json").is_file()
    assert (tmp_path / "splits" / "validation.ids.json").is_file()
    assert (tmp_path / "splits" / "simulated_test.ids.json").is_file()


def test_prepare_generates_private_annotation_sources_and_safe_id_lists(
    tmp_path: Path,
) -> None:
    manifest = _prepare_with_fixtures(tmp_path)

    assert (tmp_path / "annotation_work" / "type_domain_source.jsonl").is_file()
    assert (tmp_path / "annotation_work" / "constraints_source.jsonl").is_file()
    constraint_ids = json.loads(
        (tmp_path / "splits" / "constraint_annotation.ids.json").read_text(
            encoding="utf-8"
        )
    )
    overlap_ids = json.loads(
        (tmp_path / "splits" / "overlap_annotation.ids.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(constraint_ids) == 1
    assert overlap_ids == constraint_ids
    assert manifest["work_packages"]["type_domain"]["count"] == 2
    assert manifest["work_packages"]["constraints"]["count"] == 1
    assert manifest["work_packages"]["overlap"]["count"] == 1


def test_prepare_rejects_inconsistent_frozen_rerun(tmp_path: Path) -> None:
    _prepare_with_fixtures(tmp_path)

    def changed_downloader(
        repo_id: str,
        revision: str,
        path: str,
        token: str,
    ) -> bytes:
        del repo_id, revision, token
        content = FIXTURE_FILES[path]
        return content.replace(b"Synthetic Paper One", b"Changed Paper One")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preparation.prepare(
            output_root=tmp_path,
            token="test-token",
            downloader=changed_downloader,
            expected_counts={path: 2 for path in FIXTURE_FILES},
            dev_size=1,
            validation_size=1,
            simulated_test_size=2,
            constraint_annotation_size=1,
            overlap_annotation_size=1,
        )


def test_repository_pins_hashed_data_checkout_bytes_to_lf() -> None:
    paths = (
        "data/manifest.json",
        "data/splits/dev.ids.json",
        "data/freeze_reports/example.json",
    )
    result = subprocess.run(
        ["git", "check-attr", "text", "eol", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "data/manifest.json: text: set",
        "data/manifest.json: eol: lf",
        "data/splits/dev.ids.json: text: set",
        "data/splits/dev.ids.json: eol: lf",
        "data/freeze_reports/example.json: text: set",
        "data/freeze_reports/example.json: eol: lf",
    ]


def test_onboarding_requires_fresh_checkout_for_hashed_metadata() -> None:
    onboarding = Path("docs/TEAMMATE_ONBOARDING.md").read_text(encoding="utf-8")

    assert "不要在旧 checkout 中只执行 `git pull --ff-only`" in onboarding
    assert "创建全新 clone 或全新 worktree" in onboarding


def test_fresh_windows_checkout_preserves_manifest_id_hashes(
    tmp_path: Path,
) -> None:
    repository = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    checkout = tmp_path / "checkout"
    subprocess.run(
        [
            "git",
            "clone",
            "--no-hardlinks",
            "--quiet",
            "--no-checkout",
            str(repository),
            str(checkout),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "core.autocrlf", "true"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--quiet", "HEAD"],
        check=True,
    )

    data_root = checkout / "data"
    manifest_bytes = (data_root / "manifest.json").read_bytes()
    assert "sha256:" + hashlib.sha256(manifest_bytes).hexdigest() == (
        "sha256:1426653afe56a0771a034b28cf8b15e26c164026b32e6abe6aeac929c3043aa3"
    )
