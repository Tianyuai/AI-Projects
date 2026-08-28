from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from paper_search.evaluation.release_package import (
    ReleaseVerificationError,
    build_release,
    collect_release_files,
    verify_release,
)


PROJECT_ROOT = Path(__file__).parents[2]


def test_release_collects_every_hash_bound_runtime_artifact() -> None:
    paths = set(collect_release_files(PROJECT_ROOT))

    assert {
        "artifacts/identity/conservative-pasa-identity-alias-v1.json",
        "artifacts/models/gated-feature-fusion-18314-unified-context-v3-v1/manifest.json",
        "artifacts/models/gated-feature-fusion-18314-unified-context-v3-v1/weights.bundle",
        "artifacts/models/production-document-ranker-selection.json",
        "artifacts/models/reliability-fusion-18314-unified-context-v3-v1/manifest.json",
        "artifacts/models/reliability-fusion-18314-unified-context-v3-v1/weights.bundle",
        "artifacts/models/supervised-lexical-bridge-openalex-v2/manifest.json",
        "artifacts/models/supervised-lexical-bridge-openalex-v2/model.joblib",
        "configs/prompts/query_analyze_protected_actions_v3.yaml",
        "data/annotation_work/pricing_v1.yaml",
        "data/identifier-map.json",
        "deliverables/evaluator/live-evaluator.lock.yaml",
        "examples/safe-replay/queries.jsonl",
        "scripts/run_evaluator_batch.py",
        "scripts/run_evaluator_package.py",
        "scripts/verify_evaluator_release.py",
    }.issubset(paths)
    assert not any(
        component in {".env", ".ruff_cache", ".pytest_cache", "__pycache__", "runs"}
        for path in paths
        for component in Path(path).parts
    )


def test_built_release_is_exactly_manifested_and_detects_tampering(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "vivaai-paper-search-evaluator-runtime"
    archive_path = tmp_path / "vivaai-paper-search-evaluator-runtime.zip"

    summary = build_release(PROJECT_ROOT, release_root, archive_path=archive_path)
    verified = verify_release(release_root)

    assert verified["valid"] is True
    assert summary["file_count"] == verified["file_count"]
    release = json.loads((release_root / "RELEASE.json").read_text(encoding="utf-8"))
    assert release["production_default"] == "F5-gated-fusion"
    assert release["runtime_failover_order"] == [
        "F5-gated-fusion",
        "F4-reliability",
        "B0",
    ]
    manifest_lines = (release_root / "MANIFEST.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    assert manifest_lines == sorted(manifest_lines, key=lambda line: line[66:])
    assert not (release_root / ".env").exists()
    assert summary["archive_sha256"].startswith("sha256:")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert names
    assert all(name.startswith(f"{release_root.name}/") for name in names)
    assert not any("/.ruff_cache/" in name or name.endswith("/.env") for name in names)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    assert checksum_path.read_text(encoding="utf-8") == (
        summary["archive_sha256"].removeprefix("sha256:")
        + f"  {archive_path.name}\n"
    )

    runtime_cache = release_root / "src/paper_search/__pycache__/runtime.pyc"
    runtime_cache.parent.mkdir(parents=True)
    runtime_cache.write_bytes(b"generated after verification")
    assert verify_release(release_root)["valid"] is True
    (release_root / ".env").write_text("DO_NOT_ACCEPT=1\n", encoding="utf-8")
    with pytest.raises(ReleaseVerificationError, match="unexpected release file"):
        verify_release(release_root)
    (release_root / ".env").unlink()

    (release_root / "README.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ReleaseVerificationError, match="hash mismatch"):
        verify_release(release_root)
