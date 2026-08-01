"""Read-only, deterministic Gate 0 evidence reconciliation."""

from __future__ import annotations

import argparse
import re
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal, NoReturn, Sequence, TypeAlias

from pydantic import ConfigDict, ValidationError, field_validator, model_validator

from paper_search.control.pricing import (
    PricingPolicy,
    parse_pricing_policy_bytes,
    parse_quality_gate_policy_bytes,
)
from paper_search.domain.models import (
    DomainModel,
    NonNegativeInt,
    Sha256,
)
from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap
from paper_search.evaluation.freeze_schema import (
    BoundArtifact,
    FreezeEvidenceError,
    FreezeManifestV2,
    ValidatedFreezeEvidence,
    canonical_document_bytes,
    open_confined_artifact,
    open_validated_freeze_evidence,
    parse_json_object_bytes,
    publish_confined_bytes_no_overwrite,
)


Gate0ReasonCode = Literal[
    "manifest_missing",
    "manifest_invalid",
    "approval_invalid",
    "partition_hash_mismatch",
    "partition_count_mismatch",
    "identifier_map_missing",
    "identifier_map_hash_mismatch",
    "identifier_map_coverage_failed",
    "pricing_policy_missing",
    "pricing_policy_invalid",
    "quality_policy_invalid",
    "readiness_evidence_invalid",
]


class Gate0ArtifactEvidence(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    identity: Literal[
        "paper-search-freeze-v2",
        "dev",
        "validation",
        "identifier-map-v1",
    ]
    sha256: Sha256
    count: NonNegativeInt | None


class Gate0Report(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["gate0-report-v1"]
    generated_at: datetime
    passed: bool
    blocking_reasons: list[Gate0ReasonCode]
    manifest: Gate0ArtifactEvidence | None
    partitions: list[Gate0ArtifactEvidence]
    identifier_map: Gate0ArtifactEvidence | None
    pricing_policy_sha256: Sha256 | None
    quality_gates_sha256: Sha256 | None
    readiness_report_sha256: Sha256 | None

    @field_validator("generated_at")
    @classmethod
    def normalize_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("blocking_reasons")
    @classmethod
    def sort_blocking_reasons(
        cls,
        values: list[Gate0ReasonCode],
    ) -> list[Gate0ReasonCode]:
        return sorted(set(values))

    @field_validator("partitions")
    @classmethod
    def sort_partitions(
        cls,
        values: list[Gate0ArtifactEvidence],
    ) -> list[Gate0ArtifactEvidence]:
        identities = [item.identity for item in values]
        if len(identities) != len(set(identities)):
            raise ValueError("partition evidence identities must be unique")
        return sorted(values, key=lambda item: item.identity)

    @model_validator(mode="after")
    def validate_pass_consistency(self) -> Gate0Report:
        if self.passed != (not self.blocking_reasons):
            raise ValueError("passed must equal absence of blocking reasons")
        if self.manifest is not None and (
            self.manifest.identity != "paper-search-freeze-v2"
            or self.manifest.count is not None
        ):
            raise ValueError("manifest evidence is invalid")
        partition_identities = [item.identity for item in self.partitions]
        if (
            any(identity not in {"dev", "validation"} for identity in partition_identities)
            or any(item.count is None for item in self.partitions)
        ):
            raise ValueError("partition evidence is invalid")
        if self.identifier_map is not None and (
            self.identifier_map.identity != "identifier-map-v1"
            or self.identifier_map.count is None
        ):
            raise ValueError("identifier-map evidence is invalid")
        if self.passed and (
            self.manifest is None
            or partition_identities != ["dev", "validation"]
            or self.identifier_map is None
            or self.pricing_policy_sha256 is None
            or self.quality_gates_sha256 is None
            or self.readiness_report_sha256 is None
        ):
            raise ValueError("passing Gate 0 evidence is incomplete")
        return self


class _Gate0InputModel(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _ReadinessCapability(_Gate0InputModel):
    name: Literal["llm", "openalex", "semantic_scholar"]
    state: Literal["ready", "degraded", "failed"]
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def require_utc_observation(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("readiness timestamp must be UTC")
        return value.astimezone(UTC)


class _ReadinessEvidence(_Gate0InputModel):
    schema_version: Literal["gate0-readiness-v1"]
    generated_at: datetime
    capabilities: list[_ReadinessCapability]

    @field_validator("generated_at")
    @classmethod
    def require_utc_generation(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("readiness timestamp must be UTC")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_complete_safe_catalog(self) -> _ReadinessEvidence:
        names = [capability.name for capability in self.capabilities]
        expected = ["llm", "openalex", "semantic_scholar"]
        if names != expected:
            raise ValueError("readiness capability catalog is invalid")
        if any(
            capability.observed_at > self.generated_at
            for capability in self.capabilities
        ):
            raise ValueError("readiness observation follows report generation")
        return self


ExternalReason: TypeAlias = Literal[
    "pricing_policy_invalid",
    "quality_policy_invalid",
    "readiness_evidence_invalid",
]
_PRODUCTION_SOURCE = re.compile(
    r"^operator-verified-production-[A-Za-z0-9][A-Za-z0-9._:-]*$"
)
_NON_PRODUCTION_MARKER = re.compile(
    r"(?:^|[-_:/.])(unknown|mock|synthetic|test|fixture)(?:$|[-_:/.])",
    re.IGNORECASE,
)


def _path_missing(path: Path) -> bool:
    try:
        return not path.exists() and not path.is_symlink()
    except (OSError, RuntimeError):
        return False


def _enter_external_artifact(
    stack: ExitStack,
    path: Path,
) -> BoundArtifact:
    return stack.enter_context(open_confined_artifact(path.parent, path.name))


def _is_production_pricing_policy(policy: PricingPolicy) -> bool:
    if _PRODUCTION_SOURCE.fullmatch(policy.source_identity) is None:
        return False
    identities = [policy.source_identity, *(rate.model_or_adapter for rate in policy.rates)]
    return all(
        _NON_PRODUCTION_MARKER.search(identity) is None
        for identity in identities
    )


def _parse_readiness_bytes(content: bytes) -> _ReadinessEvidence:
    parse_json_object_bytes(content)
    try:
        return _ReadinessEvidence.model_validate_json(content, strict=True)
    except ValidationError:
        raise ValueError("readiness evidence is invalid") from None


def _freeze_report_evidence(
    evidence: ValidatedFreezeEvidence,
    reasons: set[Gate0ReasonCode],
) -> tuple[
    Gate0ArtifactEvidence | None,
    list[Gate0ArtifactEvidence],
    Gate0ArtifactEvidence | None,
]:
    manifest = evidence.manifest
    if not isinstance(manifest, FreezeManifestV2):
        reasons.add("manifest_invalid")
        return None, [], None

    manifest_report = Gate0ArtifactEvidence(
        identity="paper-search-freeze-v2",
        sha256=evidence.manifest_artifact.sha256,
        count=None,
    )
    partitions = [
        Gate0ArtifactEvidence(
            identity=partition.name,
            sha256=artifact.sha256,
            count=partition.query_count,
        )
        for partition, artifact in zip(
            manifest.partitions,
            evidence.partition_artifacts,
            strict=True,
        )
    ]
    identifier_report = Gate0ArtifactEvidence(
        identity="identifier-map-v1",
        sha256=manifest.identifier_map.sha256,
        count=manifest.identifier_map.entry_count,
    )
    identifier_artifact = evidence.identifier_map_artifact
    if identifier_artifact is None:
        reasons.add("identifier_map_missing")
        return manifest_report, partitions, None

    try:
        identifier_map = IdentifierMap.from_bytes(
            identifier_artifact.content,
            source="identifier map",
        )
        for _, rows in evidence.partition_rows:
            for row in rows:
                query = EvaluationQuery.model_validate(row)
                if any(
                    not identifier_map.covers(identifier)
                    for identifier in query.relevant_paper_ids
                ):
                    raise ValueError("identifier map coverage failed")
    except (ValidationError, ValueError):
        reasons.add("identifier_map_coverage_failed")

    return manifest_report, partitions, identifier_report


def verify_gate0(
    *,
    data_root: Path,
    manifest_path: Path,
    pricing_policy_path: Path,
    quality_gates_path: Path,
    readiness_report_path: Path,
    clock: Callable[[], datetime],
) -> Gate0Report:
    """Reconcile exact evidence without mutating any source or public status."""
    reasons: set[Gate0ReasonCode] = set()
    manifest_report: Gate0ArtifactEvidence | None = None
    partition_reports: list[Gate0ArtifactEvidence] = []
    identifier_report: Gate0ArtifactEvidence | None = None
    pricing_sha256: Sha256 | None = None
    quality_sha256: Sha256 | None = None
    readiness_sha256: Sha256 | None = None
    external_artifacts: list[tuple[BoundArtifact, ExternalReason]] = []
    stack = ExitStack()
    provisional: Gate0Report
    decision_constructed = False
    close_error: Exception | None = None

    try:
        if _path_missing(manifest_path):
            reasons.add("manifest_missing")
        else:
            try:
                freeze_evidence = stack.enter_context(
                    open_validated_freeze_evidence(
                        manifest_path,
                        data_root=data_root,
                    )
                )
            except FreezeEvidenceError as error:
                reasons.update(error.reasons)
            else:
                (
                    manifest_report,
                    partition_reports,
                    identifier_report,
                ) = _freeze_report_evidence(freeze_evidence, reasons)

        if _path_missing(pricing_policy_path):
            reasons.add("pricing_policy_missing")
        else:
            try:
                pricing_artifact = _enter_external_artifact(
                    stack, pricing_policy_path
                )
                external_artifacts.append(
                    (pricing_artifact, "pricing_policy_invalid")
                )
                pricing_policy = parse_pricing_policy_bytes(pricing_artifact.content)
                if not _is_production_pricing_policy(pricing_policy):
                    raise ValueError("pricing policy is not production evidence")
            except (OSError, RuntimeError, ValueError):
                reasons.add("pricing_policy_invalid")
            else:
                pricing_sha256 = pricing_artifact.sha256

        try:
            quality_artifact = _enter_external_artifact(stack, quality_gates_path)
            external_artifacts.append((quality_artifact, "quality_policy_invalid"))
            parse_quality_gate_policy_bytes(quality_artifact.content)
        except (OSError, RuntimeError, ValueError):
            reasons.add("quality_policy_invalid")
        else:
            quality_sha256 = quality_artifact.sha256

        try:
            readiness_artifact = _enter_external_artifact(
                stack, readiness_report_path
            )
            external_artifacts.append(
                (readiness_artifact, "readiness_evidence_invalid")
            )
            _parse_readiness_bytes(readiness_artifact.content)
        except (OSError, RuntimeError, ValueError):
            reasons.add("readiness_evidence_invalid")
        else:
            readiness_sha256 = readiness_artifact.sha256

        provisional = Gate0Report(
            schema_version="gate0-report-v1",
            generated_at=clock(),
            passed=not reasons,
            blocking_reasons=sorted(reasons),
            manifest=manifest_report,
            partitions=partition_reports,
            identifier_map=identifier_report,
            pricing_policy_sha256=pricing_sha256,
            quality_gates_sha256=quality_sha256,
            readiness_report_sha256=readiness_sha256,
        )
        decision_constructed = True

        for artifact, reason in external_artifacts:
            try:
                artifact.verify_path_identity()
            except (OSError, RuntimeError, ValueError):
                reasons.add(reason)
    finally:
        try:
            stack.close()
        except FreezeEvidenceError as error:
            if decision_constructed:
                reasons.update(error.reasons)
        except Exception as error:
            if decision_constructed:
                close_error = error

    if close_error is not None:
        raise ValueError("gate0 verification failed") from None

    if reasons == set(provisional.blocking_reasons):
        return provisional
    return Gate0Report(
        schema_version="gate0-report-v1",
        generated_at=provisional.generated_at,
        passed=not reasons,
        blocking_reasons=sorted(reasons),
        manifest=(
            None if "manifest_invalid" in reasons else provisional.manifest
        ),
        partitions=(
            []
            if "partition_hash_mismatch" in reasons
            else provisional.partitions
        ),
        identifier_map=(
            None
            if "identifier_map_hash_mismatch" in reasons
            else provisional.identifier_map
        ),
        pricing_policy_sha256=(
            None
            if "pricing_policy_invalid" in reasons
            else provisional.pricing_policy_sha256
        ),
        quality_gates_sha256=(
            None
            if "quality_policy_invalid" in reasons
            else provisional.quality_gates_sha256
        ),
        readiness_report_sha256=(
            None
            if "readiness_evidence_invalid" in reasons
            else provisional.readiness_report_sha256
        ),
    )


def _report_bytes(report: Gate0Report) -> bytes:
    try:
        validated = Gate0Report.model_validate(
            report.model_dump(mode="python"),
            strict=True,
        )
    except (AttributeError, ValueError):
        raise ValueError("gate0 report is invalid") from None
    return canonical_document_bytes(validated.model_dump(mode="json"))


def write_gate0_report(path: Path, report: Gate0Report) -> None:
    """Atomically create a report, accepting only a byte-identical rerun."""
    try:
        publish_confined_bytes_no_overwrite(
            path.parent,
            path.name,
            _report_bytes(report),
        )
    except FileExistsError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise ValueError("gate0 report is invalid") from None


class _InvalidArguments(ValueError):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _InvalidArguments("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Verify private Gate 0 evidence")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pricing-policy", type=Path, required=True)
    parser.add_argument("--quality-gates", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except _InvalidArguments:
        print("gate0 status=error reasons=invalid_arguments")
        return 2
    try:
        report = verify_gate0(
            data_root=args.data_root,
            manifest_path=args.manifest,
            pricing_policy_path=args.pricing_policy,
            quality_gates_path=args.quality_gates,
            readiness_report_path=args.readiness,
            clock=lambda: datetime.now(UTC),
        )
    except Exception:
        print("gate0 status=error reasons=verification_failed")
        return 2
    try:
        write_gate0_report(args.report, report)
    except Exception:
        print("gate0 status=error reasons=report_write_failed")
        return 2
    reasons = ",".join(report.blocking_reasons) if report.blocking_reasons else "none"
    status = "passed" if report.passed else "blocked"
    print(f"gate0 status={status} reasons={reasons}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
