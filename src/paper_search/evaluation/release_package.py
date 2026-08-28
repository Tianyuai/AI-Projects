"""Build and verify the self-contained evaluator runtime release."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


RELEASE_NAME = "vivaai-paper-search-evaluator-runtime"
RELEASE_SCHEMA_VERSION = "vivaai-evaluator-release-v1"
MANIFEST_NAME = "MANIFEST.sha256"
RELEASE_METADATA_NAME = "RELEASE.json"

_PRODUCTION_LOCK = "deliverables/evaluator/live-evaluator.lock.yaml"
_MODEL_SELECTION = "artifacts/models/production-document-ranker-selection.json"
_REQUIRED_FILES = (
    ".env.example",
    ".gitattributes",
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "docs/evaluator-submission-quickstart.md",
    "docs/judge-guide.md",
    "deliverables/evaluator/README.md",
    _PRODUCTION_LOCK,
    _MODEL_SELECTION,
    "scripts/bind_production_document_ranker.py",
    "scripts/run_delivery_rehearsal.py",
    "scripts/run_evaluator_batch.py",
    "scripts/run_evaluator_package.py",
    "scripts/validate_evaluator_submission.py",
    "scripts/verify_evaluator_release.py",
)
_REQUIRED_DIRECTORIES = (
    "examples/evaluator",
    "examples/safe-replay",
    "src/paper_search",
)
_PROHIBITED_COMPONENTS = frozenset(
    {
        ".env",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tmp",
        ".venv",
        "__pycache__",
        "outputs",
        "runs",
    }
)
_PROHIBITED_SUFFIXES = frozenset({".db", ".pyc", ".pyo", ".sqlite3", ".tmp"})
_IGNORABLE_RUNTIME_COMPONENTS = frozenset(
    {".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__", "outputs", "runs"}
)
_IGNORABLE_RUNTIME_SUFFIXES = frozenset({".pyc", ".pyo"})
_SELECTION_ARTIFACTS = (
    ("default_manifest", "default_manifest_sha256"),
    ("default_weights", "default_weights_sha256"),
    ("fallback_manifest", "fallback_manifest_sha256"),
    ("fallback_weights", "fallback_weights_sha256"),
    ("emergency_manifest", "emergency_manifest_sha256"),
    ("emergency_weights", "emergency_weights_sha256"),
)


class ReleaseVerificationError(ValueError):
    """The release is incomplete, unsafe, or no longer hash-identical."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filesystem_path(path: Path) -> Path:
    """Use the extended Windows namespace so deep release paths stay readable."""

    resolved = path.resolve()
    raw = str(resolved)
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return resolved
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw.removeprefix("\\\\"))
    return Path("\\\\?\\" + raw)


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseVerificationError("artifact path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ReleaseVerificationError("artifact path escapes the release root")
    return value


def _is_prohibited(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        any(component in _PROHIBITED_COMPONENTS for component in path.parts)
        or path.suffix.casefold() in _PROHIBITED_SUFFIXES
    )


def _is_ignorable_runtime_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        any(component in _IGNORABLE_RUNTIME_COMPONENTS for component in path.parts)
        or path.suffix.casefold() in _IGNORABLE_RUNTIME_SUFFIXES
    )


def _require_file(root: Path, relative: str) -> Path:
    safe = _safe_relative(relative)
    if _is_prohibited(safe):
        raise ReleaseVerificationError(f"prohibited release path: {safe}")
    filesystem_root = _filesystem_path(root)
    path = filesystem_root / PurePosixPath(safe)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReleaseVerificationError(f"required release file is missing: {safe}") from error
    if not resolved.is_file() or not resolved.is_relative_to(filesystem_root):
        raise ReleaseVerificationError(f"required release file is invalid: {safe}")
    return resolved


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".json":
            value = json.loads(path.read_bytes())
        else:
            value = yaml.safe_load(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ReleaseVerificationError(f"invalid release metadata: {path.name}") from error
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"invalid release metadata: {path.name}")
    return value


def _lock_artifacts(lock: Mapping[str, Any]) -> dict[str, str]:
    artifacts: dict[str, str] = {}

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            path = value.get("path")
            digest = value.get("sha256")
            if path is not None or digest is not None:
                relative = _safe_relative(path)
                if not isinstance(digest, str) or not digest.startswith("sha256:"):
                    raise ReleaseVerificationError(
                        f"lock artifact hash is invalid: {relative}"
                    )
                artifacts[relative] = digest.removeprefix("sha256:")
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(lock)
    return artifacts


def _selection_artifacts(selection: Mapping[str, Any]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path_field, hash_field in _SELECTION_ARTIFACTS:
        relative = _safe_relative(selection.get(path_field))
        digest = selection.get(hash_field)
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ReleaseVerificationError(
                f"model selection hash is invalid: {path_field}"
            )
        artifacts[f"artifacts/models/{relative}"] = digest.removeprefix("sha256:")
    return artifacts


def _verify_bound_artifacts(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = _require_file(root, _PRODUCTION_LOCK)
    selection_path = _require_file(root, _MODEL_SELECTION)
    lock = _load_mapping(lock_path)
    selection = _load_mapping(selection_path)
    for relative, expected in {
        **_lock_artifacts(lock),
        **_selection_artifacts(selection),
    }.items():
        actual = _sha256(_require_file(root, relative))
        if actual != expected:
            raise ReleaseVerificationError(f"hash mismatch: {relative}")
    return lock, selection


def collect_release_files(repository_root: Path) -> tuple[str, ...]:
    """Return the exact allowlisted source paths for the evaluator release."""

    root = repository_root.resolve(strict=True)
    filesystem_root = _filesystem_path(root)
    files = set(_REQUIRED_FILES)
    for relative in _REQUIRED_DIRECTORIES:
        directory = filesystem_root / PurePosixPath(_safe_relative(relative))
        if not directory.is_dir():
            raise ReleaseVerificationError(
                f"required release directory is missing: {relative}"
            )
        for path in directory.rglob("*"):
            if path.is_file():
                candidate = path.relative_to(filesystem_root).as_posix()
                if not _is_prohibited(candidate):
                    files.add(candidate)
    lock, selection = _verify_bound_artifacts(filesystem_root)
    files.update(_lock_artifacts(lock))
    files.update(_selection_artifacts(selection))
    for relative in sorted(files):
        _require_file(root, relative)
    return tuple(sorted(files))


def _release_metadata(
    repository_root: Path,
    paths: tuple[str, ...],
) -> dict[str, object]:
    lock_path = _require_file(repository_root, _PRODUCTION_LOCK)
    selection_path = _require_file(repository_root, _MODEL_SELECTION)
    lock = _load_mapping(lock_path)
    selection = _load_mapping(selection_path)
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_name": RELEASE_NAME,
        "file_count": len(paths) + 1,
        "production_lock": _PRODUCTION_LOCK,
        "production_lock_sha256": "sha256:" + _sha256(lock_path),
        "production_lock_source_git_sha": lock.get("source_git_sha"),
        "model_selection": _MODEL_SELECTION,
        "model_selection_sha256": "sha256:" + _sha256(selection_path),
        "production_default": selection.get("production_default"),
        "production_fallback": selection.get("production_fallback"),
        "runtime_failover_order": selection.get("runtime_failover_order"),
        "training_query_count": selection.get("training_query_count"),
        "test_partition_touched": selection.get("test_partition_touched"),
    }


def _write_manifest(release_root: Path) -> int:
    filesystem_root = _filesystem_path(release_root)
    paths = sorted(
        path.relative_to(filesystem_root).as_posix()
        for path in filesystem_root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    )
    lines = [f"{_sha256(filesystem_root / PurePosixPath(path))}  {path}" for path in paths]
    (filesystem_root / MANIFEST_NAME).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return len(paths)


def _write_deterministic_zip(release_root: Path, archive_path: Path) -> str:
    filesystem_root = _filesystem_path(release_root)
    release_name = release_root.name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".release-",
        suffix=".tmp",
        dir=archive_path.parent,
    )
    os.close(descriptor)
    Path(temporary_name).unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary_name,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(item for item in filesystem_root.rglob("*") if item.is_file()):
                relative = path.relative_to(filesystem_root).as_posix()
                info = zipfile.ZipInfo(
                    f"{release_name}/{relative}",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compresslevel=9)
        Path(temporary_name).replace(archive_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return _sha256(archive_path)


def _write_archive_checksum(archive_path: Path, digest: str) -> Path:
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".checksum-",
        suffix=".tmp",
        dir=checksum_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            f"{digest}  {archive_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(checksum_path)
    finally:
        temporary.unlink(missing_ok=True)
    return checksum_path


def build_release(
    repository_root: Path,
    release_root: Path,
    *,
    archive_path: Path | None = None,
) -> dict[str, object]:
    """Build one exact release directory and optionally a deterministic ZIP."""

    source = repository_root.resolve(strict=True)
    destination = release_root.resolve()
    if destination == source or source.is_relative_to(destination):
        raise ReleaseVerificationError("release destination is too broad")
    paths = collect_release_files(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(
        tempfile.mkdtemp(prefix=".release-", dir=destination.parent)
    )
    # Keep the staging component deliberately short. Repeating the release name
    # here can cross the legacy Windows MAX_PATH limit for versioned model files.
    staged = temporary_parent / "payload"
    staged.mkdir()
    try:
        for relative in paths:
            target = staged / PurePosixPath(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_require_file(source, relative), target)
        (staged / RELEASE_METADATA_NAME).write_text(
            json.dumps(
                _release_metadata(source, paths),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        file_count = _write_manifest(staged)
        verify_release(staged)
        if destination.exists():
            if not destination.is_dir():
                raise ReleaseVerificationError("release destination is not a directory")
            shutil.rmtree(destination)
        staged.replace(destination)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)

    summary: dict[str, object] = {
        "release_root": str(destination),
        "file_count": file_count,
        "manifest_sha256": "sha256:" + _sha256(destination / MANIFEST_NAME),
    }
    if archive_path is not None:
        archive = archive_path.resolve()
        if archive == source or source.is_relative_to(archive):
            raise ReleaseVerificationError("release archive destination is too broad")
        archive_digest = _write_deterministic_zip(destination, archive)
        checksum_path = _write_archive_checksum(archive, archive_digest)
        summary.update(
            {
                "archive": str(archive),
                "archive_sha256": "sha256:" + archive_digest,
                "archive_checksum": str(checksum_path),
            }
        )
    return summary


def verify_release(release_root: Path) -> dict[str, object]:
    """Verify the exact file set, manifest, lock bindings, and release metadata."""

    display_root = release_root.resolve(strict=True)
    root = _filesystem_path(display_root)
    manifest_path = root / MANIFEST_NAME
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ReleaseVerificationError("release manifest is unavailable") from error
    expected: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ReleaseVerificationError("release manifest line is invalid")
        digest, relative = line[:64], _safe_relative(line[66:])
        if any(character not in "0123456789abcdef" for character in digest):
            raise ReleaseVerificationError("release manifest digest is invalid")
        if relative == MANIFEST_NAME or relative in expected or _is_prohibited(relative):
            raise ReleaseVerificationError("release manifest path is invalid")
        expected[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    missing = set(expected) - actual
    if missing:
        raise ReleaseVerificationError("release file set does not match manifest")
    unexpected = actual - set(expected)
    if any(not _is_ignorable_runtime_path(relative) for relative in unexpected):
        raise ReleaseVerificationError("unexpected release file is not in manifest")
    for relative, digest in expected.items():
        if _sha256(_require_file(root, relative)) != digest:
            raise ReleaseVerificationError(f"hash mismatch: {relative}")

    _, selection = _verify_bound_artifacts(root)
    metadata = _load_mapping(_require_file(root, RELEASE_METADATA_NAME))
    if (
        metadata.get("schema_version") != RELEASE_SCHEMA_VERSION
        or metadata.get("release_name") != RELEASE_NAME
        or metadata.get("file_count") != len(expected)
        or metadata.get("production_lock_sha256")
        != "sha256:" + _sha256(root / _PRODUCTION_LOCK)
        or metadata.get("model_selection_sha256")
        != "sha256:" + _sha256(root / _MODEL_SELECTION)
        or metadata.get("production_default") != selection.get("production_default")
        or metadata.get("runtime_failover_order")
        != selection.get("runtime_failover_order")
    ):
        raise ReleaseVerificationError("release metadata is inconsistent")
    return {
        "valid": True,
        "release_root": str(display_root),
        "file_count": len(expected),
        "manifest_sha256": "sha256:" + _sha256(manifest_path),
        "production_lock_sha256": metadata["production_lock_sha256"],
        "model_selection_sha256": metadata["model_selection_sha256"],
    }


__all__ = [
    "MANIFEST_NAME",
    "RELEASE_NAME",
    "ReleaseVerificationError",
    "build_release",
    "collect_release_files",
    "verify_release",
]
