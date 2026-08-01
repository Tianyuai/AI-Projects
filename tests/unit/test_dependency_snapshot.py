from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencyRequestIdentity,
    DependencySnapshotManifestV2,
    DependencySnapshotReader,
)


CAPTURED_AT = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _identity(dependency: str = "openalex") -> DependencyRequestIdentity:
    return DependencyRequestIdentity.from_canonical_request(
        dependency=dependency,
        operation="search",
        method="POST" if dependency == "llm" else "GET",
        endpoint="chat/completions" if dependency == "llm" else "/works",
        model_or_adapter="model-v1" if dependency == "llm" else "adapter-v1",
        canonical_request={"query": "retrieval augmented generation", "limit": 2},
    )


@pytest.mark.parametrize("dependency", ["llm", "openalex", "semantic_scholar"])
def test_store_round_trips_exact_bytes_for_every_dependency(
    tmp_path: Path, dependency: str
) -> None:
    store = DependencyCaptureStore(tmp_path, clock=lambda: CAPTURED_AT)
    identity = _identity(dependency)
    response = b'{"answer":"\xe4\xbd\xa0\xe5\xa5\xbd"}\n'

    staged = store.stage_success(
        identity,
        response_bytes=response,
        safe_headers={"content-type": "application/json"},
        captured_at=CAPTURED_AT,
    )
    manifest = store.seal()
    reader = DependencySnapshotReader(
        store.manifest_path,
        snapshot_manifest_sha256=store.manifest_sha256,
        snapshot_set_id=manifest.snapshot_set_id,
    )

    replayed = reader.read(identity)
    assert replayed.response_bytes == response
    assert replayed.ref == staged
    assert reader.snapshot_set_id == manifest.snapshot_set_id


def test_identity_and_cache_key_are_deterministic(tmp_path: Path) -> None:
    first = _identity()
    second = DependencyRequestIdentity.from_canonical_request(
        dependency="openalex",
        operation="search",
        method="GET",
        endpoint="/works",
        model_or_adapter="adapter-v1",
        canonical_request={"limit": 2, "query": "retrieval augmented generation"},
    )
    assert first == second

    one = DependencyCaptureStore(tmp_path / "one", clock=lambda: CAPTURED_AT)
    two = DependencyCaptureStore(tmp_path / "two", clock=lambda: CAPTURED_AT)
    first_ref = one.stage_success(
        first, response_bytes=b"{}", safe_headers={}, captured_at=CAPTURED_AT
    )
    second_ref = two.stage_success(
        second, response_bytes=b"{}", safe_headers={}, captured_at=CAPTURED_AT
    )
    assert first_ref.cache_key == second_ref.cache_key
    assert one.seal().snapshot_set_id == two.seal().snapshot_set_id


def test_duplicate_cache_key_is_rejected(tmp_path: Path) -> None:
    store = DependencyCaptureStore(tmp_path)
    identity = _identity()
    store.stage_success(
        identity, response_bytes=b"{}", safe_headers={}, captured_at=CAPTURED_AT
    )

    with pytest.raises(ValueError, match="duplicate cache key"):
        store.stage_success(
            identity, response_bytes=b"{}", safe_headers={}, captured_at=CAPTURED_AT
        )


def test_unsealed_store_cannot_be_read_and_sealed_store_cannot_be_written(
    tmp_path: Path,
) -> None:
    store = DependencyCaptureStore(tmp_path)
    identity = _identity()
    store.stage_success(
        identity, response_bytes=b"{}", safe_headers={}, captured_at=CAPTURED_AT
    )
    with pytest.raises(FileNotFoundError, match="sealed manifest"):
        DependencySnapshotReader(
            store.manifest_path,
            snapshot_manifest_sha256=_sha256(b"not-a-manifest"),
        )

    store.seal()
    with pytest.raises(RuntimeError, match="sealed"):
        store.stage_success(
            _identity("semantic_scholar"),
            response_bytes=b"{}",
            safe_headers={},
            captured_at=CAPTURED_AT,
        )


def test_snapshot_miss_fails_closed_without_live_client(tmp_path: Path) -> None:
    store = DependencyCaptureStore(tmp_path, clock=lambda: CAPTURED_AT)
    store.stage_success(
        _identity(), response_bytes=b"{}", safe_headers={}, captured_at=CAPTURED_AT
    )
    manifest = store.seal()

    reader = DependencySnapshotReader(
        store.manifest_path,
        snapshot_manifest_sha256=store.manifest_sha256,
        snapshot_set_id=manifest.snapshot_set_id,
    )
    with pytest.raises(KeyError, match="snapshot unavailable"):
        reader.read(_identity("semantic_scholar"))


def test_manifest_tamper_is_rejected_against_lock(tmp_path: Path) -> None:
    store = DependencyCaptureStore(tmp_path, clock=lambda: CAPTURED_AT)
    store.stage_success(
        _identity(), response_bytes=b"{}", safe_headers={}, captured_at=CAPTURED_AT
    )
    store.seal()
    locked_hash = store.manifest_sha256
    payload = json.loads(store.manifest_path.read_bytes())
    payload["entries"][0]["safe_headers"] = {"x-request-id": "tampered"}
    store.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest hash"):
        DependencySnapshotReader(
            store.manifest_path, snapshot_manifest_sha256=locked_hash
        )


def test_manifest_is_read_once_but_payload_is_verified_on_every_read(
    tmp_path: Path,
) -> None:
    store = DependencyCaptureStore(tmp_path, clock=lambda: CAPTURED_AT)
    identity = _identity()
    ref = store.stage_success(
        identity, response_bytes=b"original", safe_headers={}, captured_at=CAPTURED_AT
    )
    manifest = store.seal()
    reader = DependencySnapshotReader(
        store.manifest_path,
        snapshot_manifest_sha256=store.manifest_sha256,
        snapshot_set_id=manifest.snapshot_set_id,
    )

    store.manifest_path.write_text("tampered after construction", encoding="utf-8")
    assert reader.read(identity).response_bytes == b"original"
    (tmp_path / ref.snapshot_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="response hash"):
        reader.read(identity)


@pytest.mark.parametrize("response_path", ["C:/outside.bin", "../outside.bin"])
def test_manifest_rejects_absolute_or_escaping_paths(
    tmp_path: Path, response_path: str
) -> None:
    entry = {
        "entry_id": "entry-1",
        "request": _identity().model_dump(mode="json"),
        "cache_key": _sha256(b"cache"),
        "response_sha256": _sha256(b"{}"),
        "captured_at": CAPTURED_AT.isoformat(),
        "response_path": response_path,
        "safe_headers": {},
    }
    with pytest.raises(ValidationError, match="safe relative path"):
        DependencySnapshotManifestV2.model_validate(
            {
                "schema_version": "dependency-snapshot-v2",
                "snapshot_set_id": _sha256(b"set"),
                "sealed_at": CAPTURED_AT.isoformat(),
                "entries": [entry],
            }
        )


def test_reader_rejects_symlink_response_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DependencyCaptureStore(tmp_path / "store", clock=lambda: CAPTURED_AT)
    identity = _identity()
    ref = store.stage_success(
        identity, response_bytes=b"{}", safe_headers={}, captured_at=CAPTURED_AT
    )
    manifest = store.seal()
    response_path = store.root / ref.snapshot_path
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == response_path or original_is_symlink(path),
    )

    reader = DependencySnapshotReader(
        store.manifest_path,
        snapshot_manifest_sha256=store.manifest_sha256,
        snapshot_set_id=manifest.snapshot_set_id,
    )
    with pytest.raises(ValueError, match="symlink"):
        reader.read(identity)


@pytest.mark.parametrize(
    ("name", "value"),
    [("authorization", "redacted"), ("x-request-id", "Bearer secret-token")],
)
def test_secret_shaped_safe_headers_are_rejected(
    tmp_path: Path, name: str, value: str
) -> None:
    store = DependencyCaptureStore(tmp_path)
    with pytest.raises(ValueError, match="safe header"):
        store.stage_success(
            _identity(),
            response_bytes=b"{}",
            safe_headers={name: value},
            captured_at=CAPTURED_AT,
        )
