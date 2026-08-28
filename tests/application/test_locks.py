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
    load_verified_input_lock,
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


def test_candidate_lock_accepts_openalex_only_routing() -> None:
    raw = deepcopy(fixture_data("candidate.lock.yaml"))
    raw["baseline"]["retrieval"]["semantic_scholar_calls_max"] = 0

    lock = CandidateLock.model_validate(raw)

    assert lock.baseline.retrieval.semantic_scholar_calls_max == 0


def test_candidate_lock_accepts_semantic_action_prompt_v2() -> None:
    raw = deepcopy(fixture_data("candidate.lock.yaml"))
    raw["baseline"]["prompt_version"] = "query-analyze-semantic-actions-v2"

    lock = CandidateLock.model_validate(raw)

    assert lock.baseline.prompt_version == "query-analyze-semantic-actions-v2"


def test_candidate_lock_accepts_protected_action_prompt_v3() -> None:
    raw = deepcopy(fixture_data("candidate.lock.yaml"))
    raw["baseline"]["prompt_version"] = "query-analyze-protected-actions-v3"

    lock = CandidateLock.model_validate(raw)

    assert lock.baseline.prompt_version == "query-analyze-protected-actions-v3"


def test_candidate_lock_accepts_bounded_unconstrained_supplement() -> None:
    raw = deepcopy(fixture_data("candidate.lock.yaml"))
    raw["baseline"]["strategy"] = "bounded-two-stage-unconstrained"
    raw["baseline"]["cross_vocabulary_supplement"] = {
        "enabled": True,
        "policy_version": "contrastive-bridge-anchor-conditioned-v2",
        "eligible_profile": "unconstrained",
        "strict_negation_abstention": True,
        "max_actions": 1,
        "max_total_openalex_actions": 7,
        "max_additional_raw_candidates": 50,
        "max_total_raw_candidates": 350,
    }

    lock = CandidateLock.model_validate(raw)

    supplement = lock.baseline.cross_vocabulary_supplement
    assert supplement is not None
    assert supplement.max_total_openalex_actions == 7
    assert supplement.strict_negation_abstention is True


def test_lock_verifies_low_confidence_llm_supplement_prompt_artifact(
    tmp_path: Path,
) -> None:
    lock_path, raw = write_lock(tmp_path, "candidate.lock.yaml")
    root = tmp_path / "artifacts"
    prompt_path = "configs/prompts/query_analyze_protected_actions_v3.yaml"
    prompt_bytes = b"version: query-analyze-protected-actions-v3\n"
    prompt_file = root / prompt_path
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_bytes(prompt_bytes)
    raw["baseline"]["low_confidence_llm_supplement"] = {
        "enabled": True,
        "policy_version": "low-confidence-llm-lexical-supplement-v1",
        "confidence_policy_version": "openalex-runtime-confidence-v2",
        "prompt_version": "query-analyze-protected-actions-v3",
        "prompt_config": {
            "path": prompt_path,
            "sha256": sha256(prompt_bytes),
        },
        "strict_negation_abstention": True,
        "max_actions": 1,
        "max_provider_calls": 2,
        "max_additional_raw_candidates": 50,
        "max_additional_deduplicated_candidates": 50,
        "max_total_deduplicated_candidates": 250,
        "max_input_tokens": 4000,
        "max_output_tokens": 2000,
        "max_llm_attempts": 3,
    }
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    verified = load_verified_input_lock(lock_path, artifact_root=root)

    binding = verified.lock.baseline.low_confidence_llm_supplement
    assert binding is not None
    assert verified.artifact_bytes[binding.prompt_config.path] == prompt_bytes


def test_lock_verifies_supervised_bridge_and_pasa_alias_artifacts(
    tmp_path: Path,
) -> None:
    lock_path, raw = write_lock(tmp_path, "candidate.lock.yaml")
    root = tmp_path / "artifacts"
    bridge_manifest = b'{"model_id":"supervised-lexical-bridge-openalex-v2"}\n'
    bridge_model = b"hash-bound-joblib"
    alias_map = b'{"openalex:W123":"arxiv:2401.00001"}\n'
    payloads = {
        "models/lexical-bridge/manifest.json": bridge_manifest,
        "models/lexical-bridge/model.joblib": bridge_model,
        "identity/conservative-pasa-aliases.json": alias_map,
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    raw["baseline"]["supervised_lexical_bridge"] = {
        "enabled": True,
        "policy_version": "supervised-lexical-bridge-openalex-v2",
        "manifest": {
            "path": "models/lexical-bridge/manifest.json",
            "sha256": sha256(bridge_manifest),
        },
        "model": {
            "path": "models/lexical-bridge/model.joblib",
            "sha256": sha256(bridge_model),
        },
        "max_actions": 1,
    }
    raw["baseline"]["pasa_identity_aliases"] = {
        "enabled": True,
        "policy_version": "conservative-pasa-identity-alias-v1",
        "alias_map": {
            "path": "identity/conservative-pasa-aliases.json",
            "sha256": sha256(alias_map),
        },
        "alias_count": 1,
    }
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    verified = load_verified_input_lock(lock_path, artifact_root=root)

    bridge = verified.lock.baseline.supervised_lexical_bridge
    aliases = verified.lock.baseline.pasa_identity_aliases
    assert bridge is not None
    assert aliases is not None
    assert verified.artifact_bytes[bridge.model.path] == bridge_model
    assert verified.artifact_bytes[aliases.alias_map.path] == alias_map


def test_legacy_qwen_model_is_replay_only() -> None:
    replay = deepcopy(fixture_data("replay.lock.yaml"))
    replay["baseline"]["primary_model"] = "qwen3.7-plus"
    replay["baseline"]["fallback_model"] = "qwen3.6-flash"
    candidate = deepcopy(fixture_data("candidate.lock.yaml"))
    candidate["baseline"]["primary_model"] = "qwen3.7-plus"
    candidate["baseline"]["fallback_model"] = "qwen3.6-flash"

    assert ReplayLock.model_validate(replay).baseline.primary_model == "qwen3.7-plus"
    with pytest.raises(ValidationError, match="live locks require"):
        CandidateLock.model_validate(candidate)


def test_legacy_lock_canonical_identity_omits_absent_document_ranker() -> None:
    lock = CandidateLock.model_validate(fixture_data("candidate.lock.yaml"))

    canonical = canonical_lock_bytes(lock)

    assert b"document_ranker" not in canonical


def test_enabled_document_ranker_is_verified_as_part_of_input_lock(
    tmp_path: Path,
) -> None:
    lock_path, raw = write_lock(tmp_path, "candidate.lock.yaml")
    root = tmp_path / "artifacts"
    manifest_path = "models/document-ranker.json"
    weights_path = "models/document-ranker.f64"
    manifest_bytes = b'{"schema_version":"fixture-ranker-v1"}\n'
    weights_bytes = b"locked-ranker-weights"
    (root / "models").mkdir()
    (root / manifest_path).write_bytes(manifest_bytes)
    (root / weights_path).write_bytes(weights_bytes)
    raw["baseline"]["document_ranker"] = {
        "enabled": True,
        "manifest": {"path": manifest_path, "sha256": sha256(manifest_bytes)},
        "weights": {"path": weights_path, "sha256": sha256(weights_bytes)},
    }
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    verified = load_verified_input_lock(lock_path, artifact_root=root)

    binding = verified.lock.baseline.document_ranker
    assert binding is not None
    assert verified.artifact_bytes[binding.manifest.path] == manifest_bytes
    assert verified.artifact_bytes[binding.weights.path] == weights_bytes


def test_enabled_document_ranker_rejects_weight_hash_mismatch(tmp_path: Path) -> None:
    lock_path, raw = write_lock(tmp_path, "candidate.lock.yaml")
    root = tmp_path / "artifacts"
    manifest_path = "models/document-ranker.json"
    weights_path = "models/document-ranker.f64"
    (root / "models").mkdir()
    (root / manifest_path).write_bytes(b"manifest")
    (root / weights_path).write_bytes(b"weights")
    raw["baseline"]["document_ranker"] = {
        "enabled": True,
        "manifest": {"path": manifest_path, "sha256": sha256(b"manifest")},
        "weights": {"path": weights_path, "sha256": ZERO_SHA256},
    }
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_input_lock(lock_path, artifact_root=root)


def test_document_ranker_chain_verifies_f5_f4_and_b0_artifacts(tmp_path: Path) -> None:
    lock_path, raw = write_lock(tmp_path, "candidate.lock.yaml")
    root = tmp_path / "artifacts"
    (root / "models").mkdir()
    payloads = {
        "models/f5.json": b"f5-manifest",
        "models/f5.bundle": b"f5-weights",
        "models/f4.json": b"f4-manifest",
        "models/f4.bundle": b"f4-weights",
        "models/b0.json": b"b0-manifest",
        "models/b0.f64": b"b0-weights",
    }
    for relative, payload in payloads.items():
        (root / relative).write_bytes(payload)
    def binding(relative: str) -> dict[str, str]:
        return {"path": relative, "sha256": sha256(payloads[relative])}
    raw["baseline"]["document_ranker"] = {
        "enabled": True,
        "manifest": binding("models/f5.json"),
        "weights": binding("models/f5.bundle"),
        "fallback_manifest": binding("models/f4.json"),
        "fallback_weights": binding("models/f4.bundle"),
        "emergency_manifest": binding("models/b0.json"),
        "emergency_weights": binding("models/b0.f64"),
    }
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    verified = load_verified_input_lock(lock_path, artifact_root=root)

    ranker = verified.lock.baseline.document_ranker
    assert ranker is not None
    assert verified.artifact_bytes[ranker.fallback_weights.path] == b"f4-weights"
    assert verified.artifact_bytes[ranker.emergency_weights.path] == b"b0-weights"


def test_document_ranker_primary_artifact_failure_keeps_verified_fallback_chain(
    tmp_path: Path,
) -> None:
    lock_path, raw = write_lock(tmp_path, "candidate.lock.yaml")
    root = tmp_path / "artifacts"
    (root / "models").mkdir()
    payloads = {
        "models/f5.json": b"f5-manifest",
        "models/f5.bundle": b"f5-weights",
        "models/f4.json": b"f4-manifest",
        "models/f4.bundle": b"f4-weights",
        "models/b0.json": b"b0-manifest",
        "models/b0.f64": b"b0-weights",
    }
    for relative, payload in payloads.items():
        (root / relative).write_bytes(payload)

    def binding(relative: str) -> dict[str, str]:
        return {"path": relative, "sha256": sha256(payloads[relative])}

    raw["baseline"]["document_ranker"] = {
        "enabled": True,
        "manifest": binding("models/f5.json"),
        "weights": {
            "path": "models/f5.bundle",
            "sha256": ZERO_SHA256,
        },
        "fallback_manifest": binding("models/f4.json"),
        "fallback_weights": binding("models/f4.bundle"),
        "emergency_manifest": binding("models/b0.json"),
        "emergency_weights": binding("models/b0.f64"),
    }
    lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    verified = load_verified_input_lock(lock_path, artifact_root=root)

    ranker = verified.lock.baseline.document_ranker
    assert ranker is not None
    assert ranker.weights.path not in verified.artifact_bytes
    assert verified.ranker_artifact_failures == {
        ranker.weights.path: "hash_mismatch",
    }
    assert verified.artifact_bytes[ranker.fallback_manifest.path] == b"f4-manifest"
    assert verified.artifact_bytes[ranker.fallback_weights.path] == b"f4-weights"


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
