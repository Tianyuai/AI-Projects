from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

import paper_search.application.locks as locks_module
from paper_search.application.locks import (
    CandidateLock,
    InputLock,
    ReplayLock,
    ValidationLock,
    canonical_lock_bytes,
    load_input_lock,
    lock_sha256,
)


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "application"
PROJECT_ROOT = Path(__file__).parents[2]
ZERO_SHA256 = "sha256:" + "0" * 64


def fixture_data(name: str) -> dict[str, Any]:
    raw = yaml.safe_load((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def artifact_payloads() -> dict[str, bytes]:
    return {
        "data/manifest.json": b'{"schema_version":"fixture-v1"}\n',
        "data/identifier-map.json": b'{"fixture-query":"fixture-paper"}\n',
        "configs/prompts/query_analyze.yaml": b"version: query-analyze-v1\n",
        "configs/budget_balanced.yaml": b"max_total_tokens: 24000\n",
        "configs/pricing_v1.yaml": b"schema_version: pricing-policy-v1\n",
        "configs/quality_gates_v1.yaml": b"schema_version: quality-gates-v1\n",
    }


def sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_lock_fixtures_pin_the_real_quality_gate_policy_hash() -> None:
    expected = sha256((PROJECT_ROOT / "configs" / "quality_gates_v1.yaml").read_bytes())

    for fixture_name in ("candidate.lock.yaml", "validation.lock.yaml", "replay.lock.yaml"):
        raw = fixture_data(fixture_name)
        quality_gates = raw["quality_gates"]
        assert isinstance(quality_gates, dict)
        assert quality_gates["sha256"] == expected


def write_artifact_root(root: Path) -> dict[str, bytes]:
    payloads = artifact_payloads()
    for relative_path, payload in payloads.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return payloads


def bind_hashes(raw: dict[str, Any], payloads: dict[str, bytes]) -> dict[str, Any]:
    bound = deepcopy(raw)

    def bind_artifact(value: dict[str, Any]) -> None:
        value["sha256"] = sha256(payloads[value["path"]])

    frozen_data = bound["frozen_data"]
    bind_artifact(frozen_data["manifest"])
    bind_artifact(frozen_data["identifier_map"])
    frozen_data["partition_sha256"] = sha256(b"fixture-partition\n")
    bind_artifact(bound["baseline"]["planner"]["prompt_config"])
    for name in ("budget_config", "pricing_policy", "quality_gates"):
        bind_artifact(bound[name])
    bound["capture_policy"]["capture_policy_sha256"] = sha256(b"capture-policy-v1\n")
    if "promoted_from_dev_run_sha256" in bound:
        bound["promoted_from_dev_run_sha256"] = sha256(b"dev-run-001\n")
    if "snapshot_manifest_sha256" in bound:
        bound["snapshot_manifest_sha256"] = sha256(b"snapshot-manifest\n")
    return bound


def write_lock(tmp_path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "artifacts"
    payloads = write_artifact_root(root)
    raw = bind_hashes(fixture_data(name), payloads)
    lock_path = tmp_path / name
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return lock_path, raw


def test_live_lock_models_have_exact_serialized_fields() -> None:
    expected_live = {
        "schema_version",
        "lock_kind",
        "created_at",
        "source_git_sha",
        "runtime_allow_live",
        "frozen_data",
        "baseline",
        "budget_config",
        "pricing_policy",
        "quality_gates",
        "capture_policy",
        "project_ledger",
        "approval_ref",
    }

    assert set(CandidateLock.model_fields) == expected_live
    assert set(ValidationLock.model_fields) == expected_live | {
        "promoted_from_dev_run_id",
        "promoted_from_dev_run_sha256",
    }
    assert CandidateLock.__bases__ == ValidationLock.__bases__


def test_replay_lock_has_exact_serialized_fields_and_no_approval_ref() -> None:
    assert set(ReplayLock.model_fields) == {
        "schema_version",
        "lock_kind",
        "created_at",
        "source_capture_run_id",
        "source_git_sha",
        "runtime_allow_live",
        "frozen_data",
        "baseline",
        "budget_config",
        "pricing_policy",
        "quality_gates",
        "capture_policy",
        "project_ledger",
        "snapshot_set_id",
        "snapshot_manifest_sha256",
    }


@pytest.mark.parametrize(
    ("fixture_name", "expected_type"),
    [
        ("candidate.lock.yaml", CandidateLock),
        ("validation.lock.yaml", ValidationLock),
        ("replay.lock.yaml", ReplayLock),
    ],
)
def test_loader_discriminates_lock_kind_and_hashes_canonical_bytes(
    tmp_path: Path,
    fixture_name: str,
    expected_type: type[CandidateLock | ValidationLock | ReplayLock],
) -> None:
    lock_path, _ = write_lock(tmp_path, fixture_name)

    lock = load_input_lock(lock_path, artifact_root=tmp_path / "artifacts")

    assert isinstance(lock, expected_type)
    assert lock_sha256(lock) == lock_sha256(lock)
    assert lock_sha256(lock) == sha256(canonical_lock_bytes(lock))
    assert canonical_lock_bytes(lock) == canonical_lock_bytes(lock)


def test_candidate_and_validation_reject_snapshot_fields() -> None:
    adapter: TypeAdapter[CandidateLock | ValidationLock | ReplayLock] = TypeAdapter(InputLock)
    for fixture_name in ("candidate.lock.yaml", "validation.lock.yaml"):
        raw = fixture_data(fixture_name)
        raw["snapshot_set_id"] = "forbidden"
        raw["snapshot_manifest_sha256"] = ZERO_SHA256

        with pytest.raises(ValidationError):
            adapter.validate_python(raw)


def test_replay_requires_snapshot_manifest_fields() -> None:
    adapter: TypeAdapter[CandidateLock | ValidationLock | ReplayLock] = TypeAdapter(InputLock)
    raw = fixture_data("replay.lock.yaml")
    raw.pop("snapshot_set_id")
    raw.pop("snapshot_manifest_sha256")

    with pytest.raises(ValidationError):
        adapter.validate_python(raw)


def test_replay_preserves_live_capability_bit(tmp_path: Path) -> None:
    lock_path, raw = write_lock(tmp_path, "replay.lock.yaml")
    raw["runtime_allow_live"] = True
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    lock = load_input_lock(lock_path, artifact_root=tmp_path / "artifacts")

    assert isinstance(lock, ReplayLock)
    assert lock.runtime_allow_live is True


@pytest.mark.parametrize(
    ("fixture_name", "split"),
    [
        ("candidate.lock.yaml", "validation"),
        ("validation.lock.yaml", "dev"),
    ],
)
def test_live_lock_split_restrictions(
    tmp_path: Path, fixture_name: str, split: str
) -> None:
    lock_path, raw = write_lock(tmp_path, fixture_name)
    raw["frozen_data"]["split"] = split
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="split"):
        load_input_lock(lock_path, artifact_root=tmp_path / "artifacts")


def test_loader_rejects_artifact_path_escape(tmp_path: Path) -> None:
    lock_path, raw = write_lock(tmp_path, "candidate.lock.yaml")
    raw["budget_config"]["path"] = "../outside.yaml"
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="safe relative path"):
        load_input_lock(lock_path, artifact_root=tmp_path / "artifacts")


def test_loader_rejects_symlink_escape_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    lock_path, raw = write_lock(tmp_path, "candidate.lock.yaml")
    outside = tmp_path / "outside.yaml"
    outside.write_text("outside\n", encoding="utf-8")
    link = root / "configs" / "budget_balanced.yaml"
    try:
        link.unlink()
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    raw["budget_config"]["sha256"] = sha256(outside.read_bytes())
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact root"):
        load_input_lock(lock_path, artifact_root=root)


def test_loader_rejects_swap_after_preflight_before_artifact_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    lock_path, raw = write_lock(tmp_path, "candidate.lock.yaml")
    target = root / "configs" / "budget_balanced.yaml"
    outside = tmp_path / "outside.yaml"
    outside.write_text("outside\n", encoding="utf-8")
    raw["budget_config"]["sha256"] = sha256(outside.read_bytes())
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    original_resolve = Path.resolve
    swapped = False

    def swap_after_preflight(path: Path, strict: bool = False) -> Path:
        nonlocal swapped
        resolved = original_resolve(path, strict=strict)
        if path == target and not swapped:
            try:
                target.unlink()
                target.symlink_to(outside)
            except OSError:
                pytest.skip("symlink creation is unavailable on this platform")
            swapped = True
        return resolved

    monkeypatch.setattr(Path, "resolve", swap_after_preflight)

    with pytest.raises(ValueError, match="artifact root|symlink|escape"):
        load_input_lock(lock_path, artifact_root=root)

    assert swapped


def test_loader_reads_a_referenced_artifact_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_path, raw = write_lock(tmp_path, "candidate.lock.yaml")
    raw["pricing_policy"] = deepcopy(raw["budget_config"])
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    original_read = locks_module._read_confined_bytes
    reads = 0

    def counted_read(root: Path, relative_path: str) -> bytes:
        nonlocal reads
        if relative_path == "configs/budget_balanced.yaml":
            reads += 1
        return original_read(root, relative_path)

    monkeypatch.setattr(locks_module, "_read_confined_bytes", counted_read)

    load_input_lock(lock_path, artifact_root=tmp_path / "artifacts")

    assert reads == 1


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("frozen_data", "query_count"), "2"),
        (("runtime_allow_live",), "true"),
        (("baseline", "timeout", "connect_seconds"), "5"),
        (("baseline", "retry", "retry_timeouts"), "true"),
    ],
)
def test_lock_models_reject_coercible_scalar_values(
    path: tuple[str, ...], value: object
) -> None:
    raw = fixture_data("candidate.lock.yaml")
    target: dict[str, Any] = raw
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        TypeAdapter(InputLock).validate_python(raw)


def test_lock_hash_changes_when_each_top_level_lock_field_changes(tmp_path: Path) -> None:
    lock_path, _ = write_lock(tmp_path, "validation.lock.yaml")
    lock = load_input_lock(lock_path, artifact_root=tmp_path / "artifacts")
    original_hash = lock_sha256(lock)
    mutations: dict[str, Any] = {
        "schema_version": "changed-schema-version",
        "lock_kind": "changed-lock-kind",
        "created_at": lock.created_at.replace(day=31),
        "source_git_sha": "changed-commit",
        "runtime_allow_live": False,
        "frozen_data": lock.frozen_data.model_copy(update={"query_count": 3}),
        "baseline": lock.baseline.model_copy(
            update={
                "planner": lock.baseline.planner.model_copy(
                    update={
                        "prompt_config": lock.baseline.planner.prompt_config.model_copy(
                            update={"path": "configs/prompts/changed.yaml"}
                        )
                    }
                )
            }
        ),
        "budget_config": lock.budget_config.model_copy(
            update={"path": "configs/changed-budget.yaml"}
        ),
        "pricing_policy": lock.pricing_policy.model_copy(
            update={"path": "configs/changed-pricing.yaml"}
        ),
        "quality_gates": lock.quality_gates.model_copy(
            update={"path": "configs/changed-gates.yaml"}
        ),
        "capture_policy": lock.capture_policy.model_copy(
            update={"capture_policy_sha256": sha256(b"changed-capture-policy\n")}
        ),
        "approval_ref": "changed-approval",
        "promoted_from_dev_run_id": "changed-dev-run",
        "promoted_from_dev_run_sha256": sha256(b"changed-dev-run\n"),
    }

    for field_name, value in mutations.items():
        changed = lock.model_copy(update={field_name: value})
        assert lock_sha256(changed) != original_hash, field_name
