from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

import paper_search.evaluation.freeze_schema as freeze_schema
from paper_search.evaluation.freeze_schema import canonical_gold_set_sha256


QUALITY_POLICY_PATH = Path("configs/quality_gates_v1.yaml")
TEST_PRICING_POLICY_PATH = Path(
    "tests/fixtures/pricing/pricing-policy-test-v1.yaml"
)
FIXED_NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
PUBLIC_STATUS_PATHS = (
    Path("data/manifest.json"),
    Path("README.md"),
    Path("data/README.md"),
    Path("PRD.md"),
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _gate0() -> Any:
    try:
        return importlib.import_module("paper_search.evaluation.gate0")
    except ModuleNotFoundError:
        pytest.fail("paper_search.evaluation.gate0 must be implemented")


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _canonical_document(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _production_pricing_bytes() -> bytes:
    raw = yaml.safe_load(TEST_PRICING_POLICY_PATH.read_bytes())
    assert isinstance(raw, dict)
    raw["source_identity"] = "operator-verified-production-2026-07-30"
    rates = raw["rates"]
    assert isinstance(rates, list)
    for rate in rates:
        assert isinstance(rate, dict)
        if rate["dependency"] == "llm":
            rate["model_or_adapter"] = "deepseek-production-v1"
    return yaml.safe_dump(raw, sort_keys=False).encode("utf-8")


def _readiness_bytes() -> bytes:
    return _canonical_document(
        {
            "schema_version": "gate0-readiness-v1",
            "generated_at": "2026-07-30T11:59:00Z",
            "capabilities": [
                {
                    "name": "llm",
                    "state": "ready",
                    "observed_at": "2026-07-30T11:58:00Z",
                },
                {
                    "name": "openalex",
                    "state": "ready",
                    "observed_at": "2026-07-30T11:58:10Z",
                },
                {
                    "name": "semantic_scholar",
                    "state": "ready",
                    "observed_at": "2026-07-30T11:58:20Z",
                },
            ],
        }
    )


@contextmanager
def _directory_redirect(link: Path, target: Path) -> object:
    if os.name == "nt":
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            result = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                pytest.skip("Windows directory reparse creation is unavailable")
    else:
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"POSIX directory symlink creation is unavailable: {error}")
    try:
        yield
    finally:
        if link.is_symlink():
            link.unlink()
        elif os.name == "nt":
            try:
                os.lstat(link)
            except FileNotFoundError:
                pass
            else:
                link.rmdir()


def _run_gate0_process(
    args: list[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "paper_search.evaluation.gate0", *args],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        cwd=cwd,
    )


@dataclass
class Gate0Fixture:
    root: Path
    data_root: Path
    manifest_path: Path
    pricing_path: Path
    quality_path: Path
    readiness_path: Path
    report_path: Path
    manifest: dict[str, object]
    approval: dict[str, object]

    def verify(self, *, clock: Callable[[], datetime] = lambda: FIXED_NOW) -> Any:
        return _gate0().verify_gate0(
            data_root=self.data_root,
            manifest_path=self.manifest_path,
            pricing_policy_path=self.pricing_path,
            quality_gates_path=self.quality_path,
            readiness_report_path=self.readiness_path,
            clock=clock,
        )

    def write_manifest(self) -> None:
        self.manifest_path.write_bytes(_canonical_document(self.manifest))

    def bind_approval(self) -> None:
        approval_bytes = _canonical_document(self.approval)
        approval_binding = self.manifest["approval"]
        assert isinstance(approval_binding, dict)
        approval_path = self.data_root / str(approval_binding["report_path"])
        approval_path.write_bytes(approval_bytes)
        approval_binding["report_sha256"] = _sha256(approval_bytes)

    def bind_identifier_map(self, content: bytes) -> None:
        identifier_binding = self.manifest["identifier_map"]
        assert isinstance(identifier_binding, dict)
        payload = json.loads(content)
        assert isinstance(payload, dict)
        (self.data_root / str(identifier_binding["path"])).write_bytes(content)
        identifier_binding["sha256"] = _sha256(content)
        identifier_binding["entry_count"] = len(payload)
        self.approval["identifier_map_sha256"] = _sha256(content)
        self.bind_approval()
        self.write_manifest()

    def cli_args(self) -> list[str]:
        return [
            "--data-root",
            str(self.data_root),
            "--manifest",
            str(self.manifest_path),
            "--pricing-policy",
            str(self.pricing_path),
            "--quality-gates",
            str(self.quality_path),
            "--readiness",
            str(self.readiness_path),
            "--report",
            str(self.report_path),
        ]


@pytest.fixture
def passing_gate0(tmp_path: Path) -> Gate0Fixture:
    root = tmp_path / "private gate0"
    data_root = root / "frozen"
    data_root.mkdir(parents=True)
    dev_bytes = (
        b'{"query_id":"dev-1","query":"synthetic dev",'
        b'"relevant_paper_ids":["doi:10.1000/dev"]}\n'
    )
    validation_bytes = (
        b'{"query_id":"validation-1","query":"synthetic validation",'
        b'"relevant_paper_ids":["arxiv:2501.10120"]}\n'
    )
    identifier_map_bytes = _canonical_document(
        {
            "doi:10.1000/dev": "openalex:W100",
            "arxiv:2501.10120": "openalex:W200",
        }
    )
    artifact_bytes = {
        "dev/gold.jsonl": dev_bytes,
        "validation/gold.jsonl": validation_bytes,
        "identifier-map.json": identifier_map_bytes,
    }
    for relative_path, content in artifact_bytes.items():
        path = data_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    partition_hashes = {
        "dev": _sha256(dev_bytes),
        "validation": _sha256(validation_bytes),
    }
    approval: dict[str, object] = {
        "schema_version": "freeze-approval-v2",
        "approval_requested": True,
        "approved_at": "2026-07-30T00:00:00Z",
        "approver_ref": "operator-approval-2026-07-30",
        "audit_sha256": _sha256(b"synthetic-safe-audit"),
        "partition_hashes": partition_hashes,
        "identifier_map_sha256": _sha256(identifier_map_bytes),
    }
    approval_bytes = _canonical_document(approval)
    approval_path = data_root / "freeze_reports" / "approval.json"
    approval_path.parent.mkdir()
    approval_path.write_bytes(approval_bytes)
    manifest: dict[str, object] = {
        "schema_version": "paper-search-freeze-v2",
        "dataset_revision": "synthetic-v2-revision",
        "created_at": "2026-07-30T00:00:00Z",
        "annotation_status": "frozen",
        "freeze_status": "approved",
        "partitions": [
            {
                "name": "dev",
                "path": "dev/gold.jsonl",
                "query_count": 1,
                "sha256": partition_hashes["dev"],
                "zero_answer_policy": "forbid",
            },
            {
                "name": "validation",
                "path": "validation/gold.jsonl",
                "query_count": 1,
                "sha256": partition_hashes["validation"],
                "zero_answer_policy": "allow",
            },
        ],
        "gold_sha256": canonical_gold_set_sha256(
            partition_hashes["dev"],
            partition_hashes["validation"],
        ),
        "identifier_map": {
            "path": "identifier-map.json",
            "sha256": _sha256(identifier_map_bytes),
            "entry_count": 2,
        },
        "partition_immutability": "content_addressed",
        "approval": {
            "report_path": "freeze_reports/approval.json",
            "report_sha256": _sha256(approval_bytes),
            "approved_at": "2026-07-30T00:00:00Z",
            "approver_ref": "operator-approval-2026-07-30",
        },
    }
    manifest_path = data_root / "manifest.json"
    manifest_path.write_bytes(_canonical_document(manifest))
    pricing_path = root / "pricing-policy-v1.yaml"
    pricing_path.write_bytes(_production_pricing_bytes())
    quality_path = root / "quality-gates-v1.yaml"
    quality_path.write_bytes(QUALITY_POLICY_PATH.read_bytes())
    readiness_path = root / "provider-readiness.json"
    readiness_path.write_bytes(_readiness_bytes())
    return Gate0Fixture(
        root=root,
        data_root=data_root,
        manifest_path=manifest_path,
        pricing_path=pricing_path,
        quality_path=quality_path,
        readiness_path=readiness_path,
        report_path=root / "gate0-report.json",
        manifest=manifest,
        approval=approval,
    )


def test_complete_synthetic_v2_evidence_passes_with_safe_deterministic_report(
    passing_gate0: Gate0Fixture,
) -> None:
    report = passing_gate0.verify()

    assert report.schema_version == "gate0-report-v1"
    assert report.generated_at == FIXED_NOW
    assert report.passed is True
    assert report.blocking_reasons == []
    assert report.manifest is not None
    assert report.manifest.identity == "paper-search-freeze-v2"
    assert report.manifest.sha256 == _sha256(passing_gate0.manifest_path.read_bytes())
    assert [(item.identity, item.count) for item in report.partitions] == [
        ("dev", 1),
        ("validation", 1),
    ]
    assert report.identifier_map is not None
    assert report.identifier_map.identity == "identifier-map-v1"
    assert report.identifier_map.count == 2
    assert report.pricing_policy_sha256 == _sha256(
        passing_gate0.pricing_path.read_bytes()
    )
    assert report.quality_gates_sha256 == _sha256(
        passing_gate0.quality_path.read_bytes()
    )
    assert report.readiness_report_sha256 == _sha256(
        passing_gate0.readiness_path.read_bytes()
    )
    serialized = report.model_dump_json()
    assert str(passing_gate0.root) not in serialized
    assert "synthetic dev" not in serialized
    assert "doi:10.1000/dev" not in serialized


def test_manifest_private_revision_text_never_enters_report(
    passing_gate0: Gate0Fixture,
) -> None:
    secrets = (
        "sk-credential-shaped-sentinel",
        "GATED_QUERY_SENTINEL",
    )
    passing_gate0.manifest["dataset_revision"] = "-".join(secrets)
    passing_gate0.write_manifest()

    report = passing_gate0.verify()
    serialized = report.model_dump_json()

    assert report.passed
    assert report.manifest is not None
    assert report.manifest.identity == "paper-search-freeze-v2"
    assert all(secret not in serialized for secret in secrets)


def test_identifier_coverage_uses_relevant_paper_ids_never_query_id(
    passing_gate0: Gate0Fixture,
) -> None:
    dev_path = passing_gate0.data_root / "dev" / "gold.jsonl"
    dev_bytes = (
        b'{"query_id":"doi:10.9999/query-id-is-not-a-paper",'
        b'"query":"synthetic dev",'
        b'"relevant_paper_ids":["doi:10.1000/dev"]}\n'
    )
    dev_path.write_bytes(dev_bytes)
    partitions = passing_gate0.manifest["partitions"]
    assert isinstance(partitions, list)
    dev = next(
        item
        for item in partitions
        if isinstance(item, dict) and item.get("name") == "dev"
    )
    dev["sha256"] = _sha256(dev_bytes)
    approval_hashes = passing_gate0.approval["partition_hashes"]
    assert isinstance(approval_hashes, dict)
    approval_hashes["dev"] = _sha256(dev_bytes)
    validation = next(
        item
        for item in partitions
        if isinstance(item, dict) and item.get("name") == "validation"
    )
    passing_gate0.manifest["gold_sha256"] = canonical_gold_set_sha256(
        str(dev["sha256"]), str(validation["sha256"])
    )
    passing_gate0.bind_approval()
    passing_gate0.write_manifest()

    report = passing_gate0.verify()

    assert report.passed is True
    assert report.blocking_reasons == []


@pytest.mark.parametrize(
    ("reason", "mutate"),
    [
        ("manifest_missing", lambda fixture: fixture.manifest_path.unlink()),
        (
            "manifest_invalid",
            lambda fixture: fixture.manifest_path.write_bytes(b"{}"),
        ),
        (
            "approval_invalid",
            lambda fixture: (
                fixture.data_root / "freeze_reports" / "approval.json"
            ).write_bytes(b"{}"),
        ),
        (
            "partition_hash_mismatch",
            lambda fixture: (fixture.data_root / "dev" / "gold.jsonl").write_bytes(
                b'{"query_id":"dev-2","query":"changed","relevant_paper_ids":[]}\n'
            ),
        ),
        (
            "partition_count_mismatch",
            lambda fixture: _set_dev_query_count(fixture, 2),
        ),
        (
            "identifier_map_missing",
            lambda fixture: (fixture.data_root / "identifier-map.json").unlink(),
        ),
        (
            "identifier_map_hash_mismatch",
            lambda fixture: (fixture.data_root / "identifier-map.json").write_bytes(
                b'{"doi:10.1000/changed":"openalex:W999"}\n'
            ),
        ),
        (
            "identifier_map_coverage_failed",
            lambda fixture: fixture.bind_identifier_map(
                _canonical_document(
                    {"doi:10.1000/other": "openalex:W999"}
                )
            ),
        ),
        ("pricing_policy_missing", lambda fixture: fixture.pricing_path.unlink()),
        (
            "pricing_policy_invalid",
            lambda fixture: fixture.pricing_path.write_bytes(
                TEST_PRICING_POLICY_PATH.read_bytes()
            ),
        ),
        (
            "quality_policy_invalid",
            lambda fixture: fixture.quality_path.write_bytes(
                b"schema_version: quality-gates-v1\nrules: []\n"
            ),
        ),
        (
            "readiness_evidence_invalid",
            lambda fixture: fixture.readiness_path.write_bytes(
                b'{"schema_version":"gate0-readiness-v1","Authorization":'
                b'"Bearer SECRET_SENTINEL"}'
            ),
        ),
    ],
)
def test_each_gate0_reason_isolated(
    passing_gate0: Gate0Fixture,
    reason: str,
    mutate: Callable[[Gate0Fixture], object],
) -> None:
    mutate(passing_gate0)

    report = passing_gate0.verify()

    assert report.passed is False
    assert report.blocking_reasons == [reason]


def _set_dev_query_count(fixture: Gate0Fixture, count: int) -> None:
    partitions = fixture.manifest["partitions"]
    assert isinstance(partitions, list)
    dev = next(
        item
        for item in partitions
        if isinstance(item, dict) and item.get("name") == "dev"
    )
    dev["query_count"] = count
    fixture.write_manifest()


def _bind_model_invalid_approval(fixture: Gate0Fixture) -> None:
    invalid_approval = _canonical_document({})
    approval_binding = fixture.manifest["approval"]
    assert isinstance(approval_binding, dict)
    approval_path = fixture.data_root / str(approval_binding["report_path"])
    approval_path.write_bytes(invalid_approval)
    approval_binding["report_sha256"] = _sha256(invalid_approval)
    fixture.write_manifest()


def _rewrite_pricing_identity(
    fixture: Gate0Fixture,
    *,
    source_identity: str | None = None,
    adapter_identity: str | None = None,
) -> None:
    raw = yaml.safe_load(fixture.pricing_path.read_bytes())
    assert isinstance(raw, dict)
    if source_identity is not None:
        raw["source_identity"] = source_identity
    if adapter_identity is not None:
        rates = raw["rates"]
        assert isinstance(rates, list)
        first = rates[0]
        assert isinstance(first, dict)
        first["model_or_adapter"] = adapter_identity
    fixture.pricing_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("field", "identity"),
    [
        ("source", "operator-verified-safe"),
        ("source", "operator-verified-production-unknown"),
        ("source", "operator-verified-production-mock"),
        ("source", "operator-verified-production-synthetic"),
        ("source", "operator-verified-production-test"),
        ("source", "operator-verified-production-fixture"),
        ("adapter", "unknown-adapter"),
        ("adapter", "mock-adapter"),
        ("adapter", "synthetic-adapter"),
        ("adapter", "deepseek-test-v1"),
        ("adapter", "fixture-adapter"),
    ],
)
def test_production_pricing_requires_positive_source_and_safe_adapters(
    passing_gate0: Gate0Fixture,
    field: str,
    identity: str,
) -> None:
    _rewrite_pricing_identity(
        passing_gate0,
        source_identity=identity if field == "source" else None,
        adapter_identity=identity if field == "adapter" else None,
    )

    report = passing_gate0.verify()

    assert report.passed is False
    assert report.blocking_reasons == ["pricing_policy_invalid"]


def test_gate0_reads_then_rehashes_each_bound_source_on_the_same_descriptor(
    passing_gate0: Gate0Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = freeze_schema._read_descriptor
    descriptors: list[int] = []

    def record_read(descriptor: int) -> bytes:
        descriptors.append(descriptor)
        return original(descriptor)

    monkeypatch.setattr(freeze_schema, "_read_descriptor", record_read)

    report = passing_gate0.verify()

    assert report.passed
    assert len(descriptors) == 16
    assert set(Counter(descriptors).values()) == {2}


def test_gate0_forms_provisional_decision_while_all_artifacts_are_open(
    passing_gate0: Gate0Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _gate0()
    original_read = freeze_schema._read_descriptor
    original_report = module.Gate0Report
    descriptors: list[int] = []
    decisions = 0

    def record_read(descriptor: int) -> bytes:
        descriptors.append(descriptor)
        return original_read(descriptor)

    def assert_open_report(**kwargs: object) -> object:
        nonlocal decisions
        decisions += 1
        assert len(descriptors) == 8
        for descriptor in descriptors:
            os.fstat(descriptor)
        return original_report(**kwargs)

    monkeypatch.setattr(freeze_schema, "_read_descriptor", record_read)
    monkeypatch.setattr(module, "Gate0Report", assert_open_report)

    report = passing_gate0.verify()

    assert report.passed
    assert decisions == 1


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("approval_model_invalid", "approval_invalid"),
        ("partition", "partition_count_mismatch"),
        ("identifier_map", "identifier_map_coverage_failed"),
    ],
)
def test_invalid_freeze_evidence_stays_open_through_decision_and_is_rehashed(
    passing_gate0: Gate0Fixture,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_reason: str,
) -> None:
    if mutation == "approval_model_invalid":
        _bind_model_invalid_approval(passing_gate0)
    elif mutation == "partition":
        _set_dev_query_count(passing_gate0, 2)
    else:
        passing_gate0.bind_identifier_map(
            _canonical_document({"": "openalex:W999"})
        )

    module = _gate0()
    original_read = freeze_schema._read_descriptor
    original_report = module.Gate0Report
    read_counts: Counter[tuple[int, int]] = Counter()
    descriptor_by_identity: dict[tuple[int, int], int] = {}
    decisions = 0

    def record_read(descriptor: int) -> bytes:
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        read_counts[identity] += 1
        descriptor_by_identity.setdefault(identity, descriptor)
        return original_read(descriptor)

    def assert_invalid_evidence_open(**kwargs: object) -> object:
        nonlocal decisions
        decisions += 1
        assert len(descriptor_by_identity) == 8
        assert set(read_counts.values()) == {1}
        for identity, descriptor in descriptor_by_identity.items():
            metadata = os.fstat(descriptor)
            assert (metadata.st_dev, metadata.st_ino) == identity
        return original_report(**kwargs)

    monkeypatch.setattr(freeze_schema, "_read_descriptor", record_read)
    monkeypatch.setattr(module, "Gate0Report", assert_invalid_evidence_open)

    report = passing_gate0.verify()

    assert report.passed is False
    assert report.blocking_reasons == [expected_reason]
    assert decisions == 1
    assert len(read_counts) == 8
    assert set(read_counts.values()) == {2}


def test_parse_invalid_manifest_stays_open_until_decision_rehash_then_closes(
    passing_gate0: Gate0Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passing_gate0.manifest_path.write_bytes(b'{"schema_version":')
    manifest_metadata = passing_gate0.manifest_path.stat()
    manifest_identity = (manifest_metadata.st_dev, manifest_metadata.st_ino)
    module = _gate0()
    original_read = freeze_schema._read_descriptor
    original_report = module.Gate0Report
    manifest_reads: list[int] = []
    decisions = 0

    def record_manifest_read(descriptor: int) -> bytes:
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == manifest_identity:
            manifest_reads.append(descriptor)
        return original_read(descriptor)

    def assert_manifest_open_at_decision(**kwargs: object) -> object:
        nonlocal decisions
        decisions += 1
        assert len(manifest_reads) == 1
        metadata = os.fstat(manifest_reads[0])
        assert (metadata.st_dev, metadata.st_ino) == manifest_identity
        return original_report(**kwargs)

    monkeypatch.setattr(freeze_schema, "_read_descriptor", record_manifest_read)
    monkeypatch.setattr(module, "Gate0Report", assert_manifest_open_at_decision)

    report = passing_gate0.verify()

    assert report.passed is False
    assert report.blocking_reasons == ["manifest_invalid"]
    assert decisions == 1
    assert len(manifest_reads) == 2
    assert len(set(manifest_reads)) == 1
    with pytest.raises(OSError):
        os.fstat(manifest_reads[0])


def test_parse_invalid_manifest_rehashes_and_closes_when_clock_fails(
    passing_gate0: Gate0Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passing_gate0.manifest_path.write_bytes(b'{"schema_version":')
    manifest_metadata = passing_gate0.manifest_path.stat()
    manifest_identity = (manifest_metadata.st_dev, manifest_metadata.st_ino)
    original_read = freeze_schema._read_descriptor
    manifest_reads: list[int] = []

    def record_manifest_read(descriptor: int) -> bytes:
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == manifest_identity:
            manifest_reads.append(descriptor)
        return original_read(descriptor)

    def fail_clock() -> datetime:
        raise LookupError("PRIVATE_CLOCK_SENTINEL")

    monkeypatch.setattr(freeze_schema, "_read_descriptor", record_manifest_read)

    with pytest.raises(LookupError, match="PRIVATE_CLOCK_SENTINEL"):
        passing_gate0.verify(clock=fail_clock)

    assert len(manifest_reads) == 2
    assert len(set(manifest_reads)) == 1
    with pytest.raises(OSError):
        os.fstat(manifest_reads[0])


@pytest.mark.parametrize("failure_stage", ["clock", "report"])
def test_gate0_closes_every_bound_descriptor_when_decision_construction_fails(
    passing_gate0: Gate0Fixture,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    module = _gate0()
    original_stack = module.ExitStack
    original_read = freeze_schema._read_descriptor
    retained_stacks: list[Any] = []
    descriptors: set[int] = set()

    def retain_stack() -> Any:
        stack = original_stack()
        retained_stacks.append(stack)
        return stack

    def record_read(descriptor: int) -> bytes:
        descriptors.add(descriptor)
        return original_read(descriptor)

    def fail_clock() -> datetime:
        raise LookupError("PRIVATE_CLOCK_SENTINEL")

    def fail_report(**_: object) -> object:
        raise LookupError("PRIVATE_REPORT_SENTINEL")

    monkeypatch.setattr(module, "ExitStack", retain_stack)
    monkeypatch.setattr(freeze_schema, "_read_descriptor", record_read)
    if failure_stage == "report":
        monkeypatch.setattr(module, "Gate0Report", fail_report)

    with pytest.raises(LookupError):
        passing_gate0.verify(
            clock=fail_clock if failure_stage == "clock" else lambda: FIXED_NOW
        )

    assert len(descriptors) == 8
    still_open: list[int] = []
    for descriptor in descriptors:
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        still_open.append(descriptor)
    for stack in retained_stacks:
        stack.close()
    retained_stacks.clear()
    assert still_open == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX same-inode mutation contract")
@pytest.mark.parametrize(
    ("relative_path", "expected_reason"),
    [
        ("manifest.json", "manifest_invalid"),
        ("freeze_reports/approval.json", "approval_invalid"),
        ("dev/gold.jsonl", "partition_hash_mismatch"),
        ("identifier-map.json", "identifier_map_hash_mismatch"),
        ("pricing-policy-v1.yaml", "pricing_policy_invalid"),
        ("quality-gates-v1.yaml", "quality_policy_invalid"),
        ("provider-readiness.json", "readiness_evidence_invalid"),
    ],
)
def test_gate0_detects_same_inode_truncate_write_with_per_artifact_reason(
    passing_gate0: Gate0Fixture,
    relative_path: str,
    expected_reason: str,
) -> None:
    path = (
        passing_gate0.data_root / relative_path
        if relative_path
        in {
            "manifest.json",
            "freeze_reports/approval.json",
            "dev/gold.jsonl",
            "identifier-map.json",
        }
        else passing_gate0.root / relative_path
    )
    original_identity = (path.stat().st_dev, path.stat().st_ino)

    def mutate_after_all_evidence_is_open() -> datetime:
        path.write_bytes(b"same inode, different exact bytes")
        assert (path.stat().st_dev, path.stat().st_ino) == original_identity
        return FIXED_NOW

    report = passing_gate0.verify(clock=mutate_after_all_evidence_is_open)

    assert report.passed is False
    assert report.blocking_reasons == [expected_reason]


@pytest.mark.parametrize(
    ("path_field", "expected_reason"),
    [
        ("pricing_path", "pricing_policy_invalid"),
        ("quality_path", "quality_policy_invalid"),
        ("readiness_path", "readiness_evidence_invalid"),
    ],
)
def test_external_lexical_parent_redirect_is_rejected_with_artifact_reason(
    passing_gate0: Gate0Fixture,
    tmp_path: Path,
    path_field: str,
    expected_reason: str,
) -> None:
    source = getattr(passing_gate0, path_field)
    redirected_parent = tmp_path / f"redirected-{path_field}"
    redirected_parent.mkdir()
    (redirected_parent / source.name).write_bytes(source.read_bytes())
    lexical_parent = passing_gate0.root / f"lexical-{path_field}"

    with _directory_redirect(lexical_parent, redirected_parent):
        setattr(passing_gate0, path_field, lexical_parent / source.name)
        report = passing_gate0.verify()

    assert report.passed is False
    assert report.blocking_reasons == [expected_reason]


@pytest.mark.parametrize(
    ("path_field", "expected_reason"),
    [
        ("manifest_path", "manifest_invalid"),
        ("pricing_path", "pricing_policy_invalid"),
        ("quality_path", "quality_policy_invalid"),
        ("readiness_path", "readiness_evidence_invalid"),
    ],
)
def test_missing_file_below_dangling_lexical_parent_is_not_classified_missing(
    passing_gate0: Gate0Fixture,
    tmp_path: Path,
    path_field: str,
    expected_reason: str,
) -> None:
    missing_target = tmp_path / f"missing-target-{path_field}"
    lexical_parent = passing_gate0.root / f"dangling-{path_field}"

    with _directory_redirect(lexical_parent, missing_target):
        setattr(passing_gate0, path_field, lexical_parent / "missing-evidence")
        report = passing_gate0.verify()

    assert report.passed is False
    assert report.blocking_reasons == [expected_reason]


@pytest.mark.parametrize(
    ("relative_path", "expected_reason"),
    [
        ("manifest.json", "manifest_invalid"),
        ("freeze_reports/approval.json", "approval_invalid"),
        ("dev/gold.jsonl", "partition_hash_mismatch"),
        ("identifier-map.json", "identifier_map_hash_mismatch"),
        ("pricing-policy-v1.yaml", "pricing_policy_invalid"),
        ("quality-gates-v1.yaml", "quality_policy_invalid"),
        ("provider-readiness.json", "readiness_evidence_invalid"),
    ],
)
@pytest.mark.parametrize("error_type", [ValueError, OSError, RuntimeError])
def test_gate0_exit_identity_failure_downgrades_with_artifact_reason(
    passing_gate0: Gate0Fixture,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    expected_reason: str,
    error_type: type[Exception],
) -> None:
    original = freeze_schema.BoundArtifact.verify_path_identity

    def fail_selected(artifact: freeze_schema.BoundArtifact) -> None:
        if artifact.relative_path == relative_path:
            raise error_type("PRIVATE_PATH_SENTINEL")
        original(artifact)

    monkeypatch.setattr(
        freeze_schema.BoundArtifact,
        "verify_path_identity",
        fail_selected,
    )

    report = passing_gate0.verify()

    assert report.passed is False
    assert report.blocking_reasons == [expected_reason]
    assert "PRIVATE_PATH_SENTINEL" not in report.model_dump_json()
    invalidated_fields: dict[str, tuple[str, object]] = {
        "manifest_invalid": ("manifest", None),
        "partition_hash_mismatch": ("partitions", []),
        "identifier_map_hash_mismatch": ("identifier_map", None),
        "pricing_policy_invalid": ("pricing_policy_sha256", None),
        "quality_policy_invalid": ("quality_gates_sha256", None),
        "readiness_evidence_invalid": ("readiness_report_sha256", None),
    }
    if expected_reason in invalidated_fields:
        field, empty_value = invalidated_fields[expected_reason]
        assert getattr(report, field) == empty_value


@pytest.mark.parametrize(
    ("path_field", "expected_reason"),
    [
        ("pricing_path", "pricing_policy_invalid"),
        ("quality_path", "quality_policy_invalid"),
        ("readiness_path", "readiness_evidence_invalid"),
    ],
)
@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_external_path_resolution_failures_are_isolated_and_sanitized(
    passing_gate0: Gate0Fixture,
    monkeypatch: pytest.MonkeyPatch,
    path_field: str,
    expected_reason: str,
    error_type: type[Exception],
) -> None:
    source = getattr(passing_gate0, path_field)
    isolated_parent = passing_gate0.root / f"isolated-{path_field}"
    isolated_parent.mkdir()
    isolated_path = isolated_parent / source.name
    isolated_path.write_bytes(source.read_bytes())
    setattr(passing_gate0, path_field, isolated_path)
    original_resolve = Path.resolve

    def fail_target(
        candidate: Path,
        *args: object,
        **kwargs: object,
    ) -> Path:
        if candidate == isolated_parent:
            raise error_type("PRIVATE_PATH_SENTINEL")
        return original_resolve(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_target)

    report = passing_gate0.verify()

    assert report.passed is False
    assert report.blocking_reasons == [expected_reason]
    assert "PRIVATE_PATH_SENTINEL" not in report.model_dump_json()


def test_cli_sanitizes_report_parent_symlink_loop(
    passing_gate0: Gate0Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_parent = tmp_path / "isolated-report-parent"
    report_parent.mkdir()
    passing_gate0.report_path = report_parent / "gate0-report.json"
    original_resolve = Path.resolve

    def fail_report_parent(
        candidate: Path,
        *args: object,
        **kwargs: object,
    ) -> Path:
        if candidate == report_parent:
            raise RuntimeError("PRIVATE_PATH_SENTINEL")
        return original_resolve(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_report_parent)

    result = _gate0().main(passing_gate0.cli_args())

    assert result == 2
    assert capsys.readouterr().out == (
        "gate0 status=error reasons=report_write_failed\n"
    )
    assert not passing_gate0.report_path.exists()


def test_cli_rejects_real_report_parent_redirect_without_writing_target(
    passing_gate0: Gate0Fixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    redirected_parent = tmp_path / "redirected-report-parent"
    redirected_parent.mkdir()
    lexical_parent = passing_gate0.root / "lexical-report-parent"

    with _directory_redirect(lexical_parent, redirected_parent):
        passing_gate0.report_path = lexical_parent / "gate0-report.json"
        result = _gate0().main(passing_gate0.cli_args())

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == "gate0 status=error reasons=report_write_failed\n"
    assert captured.err == ""
    assert not (redirected_parent / "gate0-report.json").exists()


@pytest.mark.parametrize(
    "error_type", [ValueError, OSError, RuntimeError, LookupError]
)
def test_cli_sanitizes_unexpected_verification_failure(
    passing_gate0: Gate0Fixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
) -> None:
    def fail_verification(**_: object) -> object:
        raise error_type("PRIVATE_PATH_SENTINEL")

    monkeypatch.setattr(_gate0(), "verify_gate0", fail_verification)

    result = _gate0().main(passing_gate0.cli_args())

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == "gate0 status=error reasons=verification_failed\n"
    assert captured.err == ""
    assert "PRIVATE_PATH_SENTINEL" not in captured.out
    assert not passing_gate0.report_path.exists()


def test_cli_sanitizes_unexpected_report_write_failure(
    passing_gate0: Gate0Fixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_write(*_: object, **__: object) -> None:
        raise LookupError("PRIVATE_REPORT_WRITE_SENTINEL")

    monkeypatch.setattr(_gate0(), "write_gate0_report", fail_write)

    result = _gate0().main(passing_gate0.cli_args())

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == "gate0 status=error reasons=report_write_failed\n"
    assert captured.err == ""
    assert "PRIVATE_REPORT_WRITE_SENTINEL" not in captured.out


def test_invalid_cli_arguments_are_sanitized_at_process_boundary() -> None:
    sentinel = r"D:\private\GATED_QUERY_AUTH_SECRET_SENTINEL"
    completed = _run_gate0_process(
        ["--unknown-private-argument", sentinel]
    )

    assert completed.returncode == 2
    assert completed.stdout == "gate0 status=error reasons=invalid_arguments\n"
    assert completed.stderr == ""
    assert sentinel not in completed.stdout
    assert sentinel not in completed.stderr


@pytest.mark.parametrize("missing_input", ["manifest_path", "pricing_path"])
def test_cli_rejects_missing_input_report_alias_before_creating_it(
    passing_gate0: Gate0Fixture,
    missing_input: str,
) -> None:
    alias = passing_gate0.root / f"missing-{missing_input}.json"
    setattr(passing_gate0, missing_input, alias)
    passing_gate0.report_path = alias

    completed = _run_gate0_process(passing_gate0.cli_args())

    assert completed.returncode == 2
    assert completed.stdout == "gate0 status=error reasons=path_alias\n"
    assert completed.stderr == ""
    assert not alias.exists()


def test_cli_rejects_missing_targets_below_aliased_directory_object(
    passing_gate0: Gate0Fixture,
) -> None:
    input_parent = passing_gate0.root / "real-input-parent"
    input_parent.mkdir()
    report_parent = passing_gate0.root / "aliased-report-parent"
    missing_name = "missing-manifest.json"
    input_path = input_parent / missing_name
    report_path = report_parent / missing_name
    args = passing_gate0.cli_args()
    args[args.index("--manifest") + 1] = str(
        Path(input_parent.name) / missing_name
    )
    args[args.index("--report") + 1] = str(
        Path(report_parent.name) / missing_name
    )

    with _directory_redirect(report_parent, input_parent):
        assert input_parent.samefile(report_parent)
        completed = _run_gate0_process(args, cwd=passing_gate0.root)

    assert completed.returncode == 2
    assert completed.stdout == "gate0 status=error reasons=path_alias\n"
    assert completed.stderr == ""
    assert not input_path.exists()
    assert not report_path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows namespace behavior")
@pytest.mark.parametrize("suffix", [".", " "])
def test_cli_rejects_missing_input_alias_via_windows_terminal_stripping(
    passing_gate0: Gate0Fixture,
    suffix: str,
) -> None:
    missing_input = passing_gate0.root / "missing-manifest.json"
    supplied_report = Path(f"{missing_input}{suffix}")
    passing_gate0.manifest_path = missing_input
    passing_gate0.report_path = supplied_report

    completed = _run_gate0_process(passing_gate0.cli_args())

    assert completed.returncode == 2
    assert completed.stdout == "gate0 status=error reasons=path_alias\n"
    assert completed.stderr == ""
    assert not missing_input.exists()
    assert not supplied_report.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows namespace behavior")
@pytest.mark.parametrize(
    "flag",
    [
        "--data-root",
        "--manifest",
        "--pricing-policy",
        "--quality-gates",
        "--readiness",
        "--report",
    ],
)
def test_cli_preflight_rejects_unstable_component_on_every_operator_path(
    passing_gate0: Gate0Fixture,
    flag: str,
) -> None:
    args = passing_gate0.cli_args()
    value_index = args.index(flag) + 1
    args[value_index] = f"{args[value_index]}."

    completed = _run_gate0_process(args)

    assert completed.returncode == 2
    assert completed.stdout == "gate0 status=error reasons=path_alias\n"
    assert completed.stderr == ""
    assert not passing_gate0.report_path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows namespace behavior")
@pytest.mark.parametrize("case", ["reserved", "extended_namespace", "ads"])
def test_cli_rejects_windows_device_namespace_without_leaking_spelling(
    passing_gate0: Gate0Fixture,
    case: str,
) -> None:
    sentinel = "GATED_WINDOWS_NAMESPACE_SENTINEL"
    actual_target = passing_gate0.root / f"{sentinel}.json"
    if case == "reserved":
        supplied_report = passing_gate0.root / f"CON.{sentinel}"
    elif case == "extended_namespace":
        supplied_report = Path("\\\\?\\" + str(actual_target.resolve()))
    else:
        supplied_report = Path(f"{actual_target}:{sentinel}")
    passing_gate0.report_path = supplied_report
    before_entries = {path.name for path in passing_gate0.root.iterdir()}

    completed = _run_gate0_process(passing_gate0.cli_args())

    assert completed.returncode == 2
    assert completed.stdout == "gate0 status=error reasons=path_alias\n"
    assert completed.stderr == ""
    assert sentinel not in completed.stdout
    assert sentinel not in completed.stderr
    assert {path.name for path in passing_gate0.root.iterdir()} == before_entries
    assert not actual_target.exists()


def test_cli_allows_same_missing_basename_below_unrelated_real_parents(
    passing_gate0: Gate0Fixture,
) -> None:
    input_parent = passing_gate0.root / "unrelated-input-parent"
    report_parent = passing_gate0.root / "unrelated-report-parent"
    input_parent.mkdir()
    report_parent.mkdir()
    missing_name = "missing-manifest.json"
    input_path = input_parent / missing_name
    report_path = report_parent / missing_name
    passing_gate0.manifest_path = input_path
    passing_gate0.report_path = report_path

    completed = _run_gate0_process(passing_gate0.cli_args())

    assert completed.returncode == 1
    assert completed.stdout == "gate0 status=blocked reasons=manifest_missing\n"
    assert completed.stderr == ""
    assert not input_path.exists()
    payload = json.loads(report_path.read_bytes())
    assert payload["passed"] is False
    assert payload["blocking_reasons"] == ["manifest_missing"]


def test_cli_rejects_equivalent_relative_input_and_absolute_report_path(
    passing_gate0: Gate0Fixture,
) -> None:
    args = passing_gate0.cli_args()
    args[args.index("--data-root") + 1] = "."
    args[args.index("--manifest") + 1] = "manifest.json"
    args[args.index("--report") + 1] = str(passing_gate0.manifest_path)
    before = passing_gate0.manifest_path.read_bytes()

    completed = _run_gate0_process(args, cwd=passing_gate0.data_root)

    assert completed.returncode == 2
    assert completed.stdout == "gate0 status=error reasons=path_alias\n"
    assert completed.stderr == ""
    assert passing_gate0.manifest_path.read_bytes() == before


def test_cli_rejects_existing_report_hardlink_to_input_object(
    passing_gate0: Gate0Fixture,
) -> None:
    report_alias = passing_gate0.root / "pricing-report-hardlink.yaml"
    os.link(passing_gate0.pricing_path, report_alias)
    passing_gate0.report_path = report_alias
    before = passing_gate0.pricing_path.read_bytes()

    completed = _run_gate0_process(passing_gate0.cli_args())

    assert completed.returncode == 2
    assert completed.stdout == "gate0 status=error reasons=path_alias\n"
    assert completed.stderr == ""
    assert passing_gate0.pricing_path.read_bytes() == before
    assert report_alias.read_bytes() == before


@pytest.mark.parametrize("protected_path", PUBLIC_STATUS_PATHS)
def test_cli_rejects_every_protected_public_report_path(
    passing_gate0: Gate0Fixture,
    protected_path: Path,
) -> None:
    before = protected_path.read_bytes()
    passing_gate0.report_path = protected_path.resolve()

    completed = _run_gate0_process(passing_gate0.cli_args(), cwd=PROJECT_ROOT)

    assert completed.returncode == 2
    assert completed.stdout == "gate0 status=error reasons=path_alias\n"
    assert completed.stderr == ""
    assert protected_path.read_bytes() == before


def test_cli_protects_public_report_path_from_different_working_directory(
    passing_gate0: Gate0Fixture,
) -> None:
    protected_path = PROJECT_ROOT / "README.md"
    before = protected_path.read_bytes()
    passing_gate0.report_path = protected_path

    completed = _run_gate0_process(
        passing_gate0.cli_args(),
        cwd=passing_gate0.root,
    )

    assert completed.returncode == 2
    assert completed.stdout == "gate0 status=error reasons=path_alias\n"
    assert completed.stderr == ""
    assert protected_path.read_bytes() == before


def test_report_writer_sanitizes_parent_resolution_failure(
    passing_gate0: Gate0Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_parent = tmp_path / "isolated-writer-parent"
    report_parent.mkdir()
    report_path = report_parent / "gate0-report.json"
    original_resolve = Path.resolve

    def fail_report_parent(
        candidate: Path,
        *args: object,
        **kwargs: object,
    ) -> Path:
        if candidate == report_parent:
            raise RuntimeError("PRIVATE_PATH_SENTINEL")
        return original_resolve(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_report_parent)

    with pytest.raises(ValueError) as error:
        _gate0().write_gate0_report(report_path, passing_gate0.verify())

    assert str(error.value) == "gate0 report is invalid"
    assert "PRIVATE_PATH_SENTINEL" not in str(error.value)
    assert not report_path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows namespace behavior")
@pytest.mark.parametrize("suffix", [".", " "])
def test_report_writer_rejects_windows_unstable_final_name(
    passing_gate0: Gate0Fixture,
    suffix: str,
) -> None:
    actual_report = passing_gate0.report_path
    supplied_report = Path(f"{actual_report}{suffix}")

    with pytest.raises(ValueError, match="gate0 report is invalid"):
        _gate0().write_gate0_report(supplied_report, passing_gate0.verify())

    assert not actual_report.exists()
    assert not supplied_report.exists()


def test_invalid_evidence_never_mutates_sources_or_public_status(
    passing_gate0: Gate0Fixture,
) -> None:
    sources = [
        path
        for path in passing_gate0.root.rglob("*")
        if path.is_file()
    ]
    before_sources = {path: path.read_bytes() for path in sources}
    before_public = {path: path.read_bytes() for path in PUBLIC_STATUS_PATHS}
    passing_gate0.pricing_path.unlink()
    expected_sources = {
        path: content
        for path, content in before_sources.items()
        if path != passing_gate0.pricing_path
    }

    report = passing_gate0.verify()

    assert report.blocking_reasons == ["pricing_policy_missing"]
    assert {
        path: path.read_bytes()
        for path in expected_sources
    } == expected_sources
    assert {path: path.read_bytes() for path in PUBLIC_STATUS_PATHS} == before_public


def test_report_writer_is_atomic_exact_match_and_no_overwrite(
    passing_gate0: Gate0Fixture,
) -> None:
    module = _gate0()
    report = passing_gate0.verify()

    module.write_gate0_report(passing_gate0.report_path, report)
    first = passing_gate0.report_path.read_bytes()
    module.write_gate0_report(passing_gate0.report_path, report)

    later = passing_gate0.verify(clock=lambda: FIXED_NOW + timedelta(seconds=1))
    with pytest.raises(FileExistsError):
        module.write_gate0_report(passing_gate0.report_path, later)

    assert passing_gate0.report_path.read_bytes() == first
    assert first.endswith(b"\n")
    assert json.loads(first)["passed"] is True
    assert list(passing_gate0.root.glob(".gate0-report.json.*.tmp")) == []


def test_report_model_rejects_pass_with_blocking_reasons() -> None:
    module = _gate0()

    with pytest.raises(ValueError):
        module.Gate0Report(
            schema_version="gate0-report-v1",
            generated_at=FIXED_NOW,
            passed=True,
            blocking_reasons=["manifest_invalid"],
            manifest=None,
            partitions=[],
            identifier_map=None,
            pricing_policy_sha256=None,
            quality_gates_sha256=None,
            readiness_report_sha256=None,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("manifest", None),
        ("partitions", []),
        ("identifier_map", None),
        ("pricing_policy_sha256", None),
        ("quality_gates_sha256", None),
        ("readiness_report_sha256", None),
    ],
)
def test_report_model_rejects_incomplete_passing_evidence(
    passing_gate0: Gate0Fixture,
    field: str,
    value: object,
) -> None:
    module = _gate0()
    payload = passing_gate0.verify().model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValueError):
        module.Gate0Report.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "identity",
    [
        "D:\\private\\freeze\\manifest.json",
        "/private/freeze/manifest.json",
        "synthetic-v2-revision",
    ],
)
def test_artifact_evidence_accepts_only_fixed_public_identities(identity: str) -> None:
    module = _gate0()

    with pytest.raises(ValueError):
        module.Gate0ArtifactEvidence(
            identity=identity,
            sha256=_sha256(b"evidence"),
            count=None,
        )


def test_report_writer_revalidates_constructed_report_before_serializing(
    tmp_path: Path,
) -> None:
    module = _gate0()
    unsafe_artifact = module.Gate0ArtifactEvidence.model_construct(
        identity="D:\\private\\freeze\\manifest.json",
        sha256=_sha256(b"manifest"),
        count=None,
    )
    fake_pass = module.Gate0Report.model_construct(
        schema_version="gate0-report-v1",
        generated_at=FIXED_NOW,
        passed=True,
        blocking_reasons=[],
        manifest=unsafe_artifact,
        partitions=[],
        identifier_map=None,
        pricing_policy_sha256=None,
        quality_gates_sha256=None,
        readiness_report_sha256=None,
    )
    path = tmp_path / "gate0-report.json"

    with pytest.raises(ValueError, match="gate0 report is invalid"):
        module.write_gate0_report(path, fake_pass)

    assert not path.exists()


def test_cli_prints_only_sanitized_deterministic_summary(
    passing_gate0: Gate0Fixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    unsafe = (
        '{"schema_version":"gate0-readiness-v1",'
        '"Authorization":"Bearer AUTH_SECRET_SENTINEL",'
        '"api_key":"sk-credential-shaped-sentinel",'
        '"query":"GATED_QUERY_SENTINEL",'
        '"path":"D:\\\\private\\\\freeze\\\\labels.json"}'
    ).encode()
    passing_gate0.readiness_path.write_bytes(unsafe)

    result = _gate0().main(passing_gate0.cli_args())

    assert result == 1
    assert capsys.readouterr().out == (
        "gate0 status=blocked reasons=readiness_evidence_invalid\n"
    )
    serialized = passing_gate0.report_path.read_text(encoding="utf-8")
    for secret in (
        "Authorization",
        "Bearer",
        "AUTH_SECRET_SENTINEL",
        "api_key",
        "sk-credential-shaped-sentinel",
        "GATED_QUERY_SENTINEL",
        "D:\\private",
        str(passing_gate0.root),
    ):
        assert secret not in serialized


def test_cli_returns_zero_only_for_passing_report(
    passing_gate0: Gate0Fixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _gate0().main(passing_gate0.cli_args())

    assert result == 0
    assert capsys.readouterr().out == "gate0 status=passed reasons=none\n"
    assert json.loads(passing_gate0.report_path.read_bytes())["passed"] is True


def test_private_gate0_and_runtime_roots_are_ignored() -> None:
    entries = {
        line.strip()
        for line in Path(".gitignore").read_text(encoding="utf-8").splitlines()
    }

    assert {"/runs/", "/validation-attempts/", "/private-gate0/"} <= entries
