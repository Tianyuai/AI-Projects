"""Deterministic Week-1 candidate processing and evaluation orchestration."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import BinaryIO, Literal, Protocol

import httpx
from pydantic import BaseModel, ValidationError

from paper_search.config import RuntimeConfig, load_runtime_config
from paper_search.application.composition import CompositionRoot
from paper_search.application.contracts import (
    SearchExecutionResult,
    SearchRequest,
)
from paper_search.evaluation.attempts import (
    ValidationAttemptConflictError,
    ValidationAttemptStore,
)
from paper_search.evaluation.execution_adapter import AdaptedExecution, adapt_execution
from paper_search.application.artifacts import FormalRunWorkspace, RunManifest
from paper_search.application.locks import (
    CandidateLock,
    ReplayLock,
    ValidationLock,
    load_verified_input_lock_bytes,
    lock_sha256,
)
from paper_search.control.ledger import (
    DEV_RUN_CAP_CNY,
    VALIDATION_RUN_CAP_CNY,
    LedgerReservation,
    SQLiteBudgetLedger,
)
from paper_search.control.pricing import (
    QualityGatePolicy,
    parse_quality_gate_policy_bytes,
)
from paper_search.evaluation.freeze_schema import FreezeManifestV2
from paper_search.evaluation.formal_evidence import (
    complete_policy_measures,
    configured_retrieval_endpoints,
    formal_audit_measures,
)
from paper_search.evaluation.gates import MeasureValue, evaluate_gates
from paper_search.evaluation.official_adapter import adapt_prediction_record
from paper_search.control import HardBudgetController
from paper_search.domain.models import (
    BudgetReservation,
    DomainModel,
    ErrorDetail,
    NonEmptyStr,
    NonNegativeInt,
    Paper,
    ProviderResult,
    QuerySpec,
    SearchMode,
    UsageActual,
    UsageEstimate,
)
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    IdentifierMap,
    PredictionRecord,
    read_jsonl,
    write_frozen_bytes,
    write_jsonl_atomic,
)
from paper_search.evaluation.metrics import CONTRACT_VERSION, EvaluationResult, evaluate
from paper_search.processing import (
    DEDUPLICATION_VERSION,
    DeduplicationResult,
    FUZZY_TITLE_THRESHOLD,
    FILTERING_VERSION,
    FilterResult,
    MINIMUM_UNCERTAINTY_MULTIPLIER,
    UNCERTAINTY_REASON_MULTIPLIER,
    apply_hard_filters,
    deduplicate_papers,
)
from paper_search.ranking import (
    BM25_WEIGHT,
    KEYWORD_COVERAGE_WEIGHT,
    SCORING_VERSION,
    TOKENIZER_VERSION,
    LexicalScore,
    rank_lexically,
)
from paper_search.retrieval import OpenAlexProvider
from paper_search.storage import SQLiteResponseCache, validate_snapshot_manifest
from paper_search.storage.cache import PreparedSnapshot
from paper_search.storage.dependency_snapshot import (
    DependencySnapshotManifestV2,
    DependencySnapshotReader,
)


_formal_audit_measures = formal_audit_measures


class PipelineResult(DomainModel):
    """Auditable outputs from each pure candidate-processing stage."""

    deduplication: DeduplicationResult
    filtering: FilterResult
    ranked: list[LexicalScore]


class QueryRunRecord(DomainModel):
    """One query's prediction, audit trail, usage, and provider diagnostics."""

    query_id: NonEmptyStr
    prediction: PredictionRecord
    pipeline: PipelineResult
    usage: UsageActual
    latency_ms: NonNegativeInt
    provider: NonEmptyStr
    endpoint: NonEmptyStr
    cache_keys: list[NonEmptyStr]
    page_hashes: list[NonEmptyStr]
    response_hash: NonEmptyStr
    errors: list[ErrorDetail]


class RunIdentity(DomainModel):
    """Exact frozen inputs and source revision that authorize a formal run."""

    split: NonEmptyStr
    git_sha: NonEmptyStr
    gold_sha256: NonEmptyStr
    manifest_sha256: NonEmptyStr
    dataset_revision: NonEmptyStr
    zero_answer_policy: Literal["reject", "allow"]
    id_map_sha256: NonEmptyStr | None = None


class RunResult(DomainModel):
    """Evaluation result and reproducibility data for an ordered split run."""

    evaluation: EvaluationResult
    identity: RunIdentity
    query_runs: list[QueryRunRecord]
    usage: UsageActual
    snapshot_manifest: NonEmptyStr


class FrozenSplit(DomainModel):
    """A fully validated frozen partition and its formal run identity."""

    gold_path: Path
    gold: list[EvaluationQuery]
    identity: RunIdentity


class EvaluationRunRequest(DomainModel):
    """Immutable inputs that authorize one canonical formal evaluation."""

    split: Literal["dev", "validation"]
    mode: SearchMode
    lock_path: Path
    output_root: Path
    snapshot_manifest_path: Path | None
    network_authorized: bool
    runtime_config: RuntimeConfig | None = None
    ablation_config_path: Path | None = None


class EvaluationRunResult(DomainModel):
    """Safe terminal summary for one formal evaluation."""

    run_id: NonEmptyStr
    run_path: Path
    status: Literal["complete", "failed", "interrupted"]
    gate_result: Literal["passed", "failed", "not_applicable"]


@dataclass(frozen=True)
class _FormalInputs:
    lock: CandidateLock | ValidationLock | ReplayLock
    lock_bytes: bytes
    gold: list[EvaluationQuery]
    identifier_map: IdentifierMap
    gate_policy: QualityGatePolicy
    prompt_artifact_sha256: str
    snapshot_manifest: DependencySnapshotManifestV2 | None
    snapshot_root: Path | None


def _load_formal_inputs(request: EvaluationRunRequest) -> _FormalInputs:
    artifact_root = Path.cwd().resolve()
    lock_bytes = request.lock_path.read_bytes()
    verified = load_verified_input_lock_bytes(lock_bytes, artifact_root=artifact_root)
    lock = verified.lock
    if _current_git_sha() != lock.source_git_sha:
        raise ValueError("current source SHA does not match input lock")
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    if tracked.returncode != 0 or tracked.stdout:
        raise ValueError("tracked source must be clean for formal evaluation")
    if request.split != lock.frozen_data.split:
        raise ValueError("evaluation split does not match input lock")
    if request.mode == "replay" and not isinstance(lock, ReplayLock):
        raise ValueError("replay evaluation requires a replay lock")
    if request.mode == "live" and isinstance(lock, ReplayLock):
        raise ValueError("live evaluation requires a live input lock")
    if request.mode == "live" and not request.network_authorized:
        raise ValueError("live evaluation requires explicit network authorization")
    if request.mode == "replay" and request.network_authorized:
        raise ValueError("replay evaluation cannot authorize network access")

    manifest = FreezeManifestV2.model_validate_json(
        verified.artifact_bytes[lock.frozen_data.manifest.path]
    )
    partition = next(item for item in manifest.partitions if item.name == request.split)
    if (
        partition.sha256 != lock.frozen_data.partition_sha256
        or partition.query_count != lock.frozen_data.query_count
        or manifest.identifier_map.sha256 != lock.frozen_data.identifier_map.sha256
    ):
        raise ValueError("frozen data does not match input lock")
    data_root = (artifact_root / lock.frozen_data.manifest.path).parent.resolve()
    partition_path = (data_root / partition.path).resolve(strict=True)
    if not partition_path.is_relative_to(data_root):
        raise ValueError("frozen partition path escapes data root")
    partition_bytes = partition_path.read_bytes()
    if _sha256_bytes(partition_bytes) != partition.sha256:
        raise ValueError("frozen partition hash mismatch")
    gold = [
        EvaluationQuery.model_validate_json(line)
        for line in partition_bytes.splitlines()
        if line
    ]
    if len(gold) != partition.query_count:
        raise ValueError("frozen partition count mismatch")
    if len({query.query_id for query in gold}) != len(gold):
        raise ValueError("frozen partition query IDs are duplicated")
    identifier_map = IdentifierMap.from_bytes(
        verified.artifact_bytes[lock.frozen_data.identifier_map.path]
    )
    gate_policy = parse_quality_gate_policy_bytes(
        verified.artifact_bytes[lock.quality_gates.path]
    )
    prompt_artifact_sha256 = _sha256_bytes(
        verified.artifact_bytes[lock.baseline.planner.prompt_config.path]
    )
    snapshot_manifest: DependencySnapshotManifestV2 | None = None
    snapshot_root: Path | None = None
    if isinstance(lock, ReplayLock):
        if request.snapshot_manifest_path is None:
            raise ValueError("snapshot manifest is required for replay")
        snapshot_manifest = DependencySnapshotManifestV2.model_validate_json(
            request.snapshot_manifest_path.read_bytes()
        )
        snapshot_root = request.snapshot_manifest_path.parent
    elif request.snapshot_manifest_path is not None:
        raise ValueError("live evaluation cannot accept a replay manifest")
    return _FormalInputs(
        lock=lock,
        lock_bytes=lock_bytes,
        gold=gold,
        identifier_map=identifier_map,
        gate_policy=gate_policy,
        prompt_artifact_sha256=prompt_artifact_sha256,
        snapshot_manifest=snapshot_manifest,
        snapshot_root=snapshot_root,
    )


class SearchProvider(Protocol):
    """Injected provider boundary used by the evaluation runner."""

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]: ...


class _ApplicationService(Protocol):
    async def execute(
        self,
        request: SearchRequest,
        *,
        run_id: str | None = None,
    ) -> SearchExecutionResult: ...


@dataclass
class _FormalCaptureBinding:
    run_id: str
    store: object

    def claim_snapshot_store(self) -> object:
        return self.store


class _CliInputError(ValueError):
    """A validation failure whose fixed message is safe to show to users."""


def _current_git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise _CliInputError("Git SHA is unavailable") from error
    if completed.returncode != 0:
        raise _CliInputError("Git SHA is unavailable")
    return completed.stdout.strip()


def _validate_git_sha(value: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise _CliInputError("Git SHA is invalid")
    return value.casefold()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Week-1 evaluation split")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-map", type=Path)
    return parser


def _resolve_data_file(data_root: Path, raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise _CliInputError("data split manifest is invalid")
    relative_path = Path(raw_path)
    resolved_root = data_root.resolve()
    if relative_path.is_absolute():
        raise _CliInputError(f"{label} path must stay under data")
    resolved = (resolved_root / relative_path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise _CliInputError(f"{label} path must stay under data")
    if not resolved.is_file():
        raise _CliInputError(f"{label} file does not exist")
    return resolved


def _resolve_cli_id_map(data_root: Path, raw_path: Path) -> Path:
    if raw_path.is_absolute():
        raise _CliInputError("identifier map path must stay under data")
    resolved_root = data_root.resolve()
    resolved = raw_path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise _CliInputError("identifier map path must stay under data")
    if not resolved.is_file():
        raise _CliInputError("identifier map file does not exist")
    return resolved


def _normalize_windows_final_path(raw_path: str) -> str:
    if raw_path.startswith("\\\\?\\UNC\\"):
        return "\\\\" + raw_path[8:]
    if raw_path.startswith("\\\\?\\"):
        return raw_path[4:]
    return raw_path


if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    def _windows_final_path_from_file(source: BinaryIO) -> Path:
        get_final_path = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        ).GetFinalPathNameByHandleW
        get_final_path.argtypes = (
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        get_final_path.restype = wintypes.DWORD
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(source.fileno()))
        buffer_size = 260
        while buffer_size <= 32_768:
            buffer = ctypes.create_unicode_buffer(buffer_size)
            path_length = get_final_path(handle, buffer, buffer_size, 0)
            if path_length == 0:
                error_code = ctypes.get_last_error()
                raise OSError(error_code, "final file target is unavailable")
            if path_length < buffer_size:
                return Path(_normalize_windows_final_path(buffer.value))
            buffer_size = path_length + 1
        raise OSError("final file target is too long")

else:

    def _windows_final_path_from_file(source: BinaryIO) -> Path:
        del source
        raise OSError("Windows final file target lookup is unavailable")


def _posix_final_path_from_file(source: BinaryIO) -> Path:
    descriptor = source.fileno()
    handle_stat = os.fstat(descriptor)
    for descriptor_root in (Path("/proc/self/fd"), Path("/dev/fd")):
        try:
            raw_target = os.readlink(descriptor_root / str(descriptor))
        except OSError:
            continue
        if not os.path.isabs(raw_target):
            continue
        final_path = Path(os.path.normpath(raw_target))
        try:
            target_stat = final_path.stat()
        except OSError:
            continue
        if (target_stat.st_dev, target_stat.st_ino) != (
            handle_stat.st_dev,
            handle_stat.st_ino,
        ):
            continue
        return final_path
    raise OSError("final file target is unavailable")


def _final_path_from_open_file(source: BinaryIO) -> Path:
    if os.name == "nt":
        return _windows_final_path_from_file(source)
    return _posix_final_path_from_file(source)


def _read_confined_identifier_map(data_root: Path, raw_path: Path) -> bytes:
    resolved_root = data_root.resolve()
    resolved_path = _resolve_cli_id_map(resolved_root, raw_path)
    with resolved_path.open("rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise _CliInputError("identifier map file does not exist")
        final_path = _final_path_from_open_file(source)
        if not final_path.is_relative_to(resolved_root):
            raise _CliInputError("identifier map path must stay under data")
        return source.read()


def _require_id_map_coverage(
    gold: Sequence[EvaluationQuery],
    id_map: IdentifierMap,
) -> None:
    identifiers = {
        identifier
        for record in gold
        for identifier in record.relevant_paper_ids
    }
    if any(not id_map.covers(identifier) for identifier in identifiers):
        raise _CliInputError(
            "identifier map does not cover frozen gold identifiers"
        )


def _resolve_frozen_split(data_root: Path, split: str, git_sha: str) -> FrozenSplit:
    manifest_path = data_root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest: object = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _CliInputError("data manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise _CliInputError("data manifest is invalid")
    if manifest.get("status") != "frozen":
        raise _CliInputError("data manifest is not frozen")
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        raise _CliInputError("dataset revision is invalid")

    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict) or split not in partitions:
        raise _CliInputError("unknown data split")
    partition = partitions[split]
    if not isinstance(partition, dict):
        raise _CliInputError("data split manifest is invalid")
    count = partition.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise _CliInputError("data split manifest is invalid")
    if partition.get("labels_complete") is not True:
        raise _CliInputError("data split labels must be complete")
    zero_answer_policy = partition.get("zero_answer_policy")
    if zero_answer_policy not in {"reject", "allow"}:
        raise _CliInputError("data split zero-answer policy is invalid")
    if any(
        not isinstance(partition.get(field), str) or not partition[field].strip()
        for field in ("gold_sha256", "ids_sha256")
    ):
        raise _CliInputError("data split manifest is invalid")
    gold_path = _resolve_data_file(data_root, partition.get("gold_path"), "gold")
    ids_path = _resolve_data_file(data_root, partition.get("ids_path"), "ID")

    gold_bytes = gold_path.read_bytes()
    declared_hash = partition["gold_sha256"]
    if declared_hash != _sha256_bytes(gold_bytes):
        raise _CliInputError("gold file SHA-256 mismatch")
    ids_bytes = ids_path.read_bytes()
    if partition["ids_sha256"] != _sha256_bytes(ids_bytes):
        raise _CliInputError("ID file SHA-256 mismatch")
    try:
        ids: object = json.loads(ids_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _CliInputError("ID list is invalid") from error
    if (
        not isinstance(ids, list)
        or any(not isinstance(value, str) or not value.strip() for value in ids)
        or len(ids) != len(set(ids))
    ):
        raise _CliInputError("ID list is invalid")
    if len(ids) != count:
        raise _CliInputError("ID count mismatch")
    if not gold_bytes.strip():
        raise _CliInputError("gold file must not be empty")
    gold = read_jsonl(gold_path, EvaluationQuery)
    if len(gold) != count:
        raise _CliInputError("gold record count mismatch")
    if [record.query_id for record in gold] != ids:
        raise _CliInputError("gold ordered query IDs do not match ID list")
    if zero_answer_policy == "reject" and any(
        not record.relevant_paper_ids for record in gold
    ):
        raise _CliInputError("zero-answer gold record is not allowed")
    identity = RunIdentity(
        split=split,
        git_sha=_validate_git_sha(git_sha),
        gold_sha256=declared_hash,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        dataset_revision=revision,
        zero_answer_policy=zero_answer_policy,
    )
    return FrozenSplit(gold_path=gold_path, gold=gold, identity=identity)


def process_candidates(
    query: QuerySpec,
    papers: Sequence[Paper],
    *,
    id_map: IdentifierMap | None = None,
) -> PipelineResult:
    """Deduplicate, filter, and lexically rank normalized papers."""
    deduplicated = deduplicate_papers(papers, id_map=id_map)
    filtered = apply_hard_filters(deduplicated.papers, query)
    ranked = rank_lexically(query, filtered.accepted)
    return PipelineResult(
        deduplication=deduplicated,
        filtering=filtered,
        ranked=ranked,
    )


def _fallback_query_spec(record: EvaluationQuery) -> QuerySpec:
    return QuerySpec(original_query=record.query, research_goal=record.query)


def _parse_cache_keys(provenance: dict[str, str]) -> list[str]:
    raw = provenance.get("cache_keys")
    if raw is None:
        raise ValueError("provider provenance cache_keys must be a JSON string list")
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("provider provenance cache_keys must be a JSON string list") from error
    if not isinstance(parsed, list) or any(
        not isinstance(value, str) or not value.strip() for value in parsed
    ):
        raise ValueError("provider provenance cache_keys must be a JSON string list")
    return list(parsed)


def _required_provenance(provenance: dict[str, str], key: str) -> str:
    value = provenance.get(key)
    if value is None or not value.strip():
        raise ValueError(f"provider provenance {key} is required")
    return value


def _aggregate_response_hashes(hashes: Sequence[str]) -> str:
    if len(hashes) == 1:
        return hashes[0]
    return _sha256_bytes(json.dumps(list(hashes), separators=(",", ":")).encode("utf-8"))


def _validate_query_snapshot_provenance(
    provenance: dict[str, str],
    cache: SQLiteResponseCache,
) -> tuple[str, str, list[str], list[str], str]:
    provider = _required_provenance(provenance, "provider")
    endpoint = _required_provenance(provenance, "endpoint")
    declared_hash = _required_provenance(provenance, "response_hash")
    cache_keys = _parse_cache_keys(provenance)
    page_hashes: list[str] = []
    for key in cache_keys:
        cached = cache.get_snapshot_response(key)
        if cached is None:
            raise ValueError("provider snapshot cache key is missing")
        if _sha256_bytes(cached.raw_response) != cached.response_hash:
            raise ValueError("provider snapshot cached response bytes do not match hash")
        if cached.provider != provider or cached.endpoint != endpoint:
            raise ValueError("provider snapshot provenance mismatch")
        page_hashes.append(cached.response_hash)
    if _aggregate_response_hashes(page_hashes) != declared_hash:
        raise ValueError("provider snapshot response hash mismatch")
    return provider, endpoint, cache_keys, page_hashes, declared_hash


def _validate_prepared_query_snapshot_provenance(
    records: Sequence[QueryRunRecord],
    prepared: PreparedSnapshot,
) -> None:
    responses = {response.cache_key: response for response in prepared.responses}
    for record in records:
        current_hashes: list[str] = []
        for key in record.cache_keys:
            cached = responses.get(key)
            if cached is None:
                raise ValueError("provider snapshot cache key changed after query")
            if cached.provider != record.provider or cached.endpoint != record.endpoint:
                raise ValueError("provider snapshot provenance changed after query")
            current_hashes.append(cached.response_hash)
        if (
            current_hashes != record.page_hashes
            or _aggregate_response_hashes(current_hashes) != record.response_hash
        ):
            raise ValueError("provider snapshot response hash changed after query")


def _aggregate_usage(records: Sequence[QueryRunRecord]) -> UsageActual:
    costs = [record.usage.cost_cny for record in records]
    cost = (
        sum(value for value in costs if value is not None)
        if costs and all(value is not None for value in costs)
        else None
    )
    return UsageActual(
        search_api_calls=sum(record.usage.search_api_calls for record in records),
        llm_calls=sum(record.usage.llm_calls for record in records),
        input_tokens=sum(record.usage.input_tokens for record in records),
        output_tokens=sum(record.usage.output_tokens for record in records),
        cost_cny=cost,
        elapsed_ms=sum(record.usage.elapsed_ms for record in records),
    )


def _jsonl_bytes(records: Sequence[BaseModel]) -> bytes:
    return b"".join(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


def _frozen_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_frozen_json(path: Path, payload: object) -> None:
    write_frozen_bytes(path, _frozen_json_bytes(payload))


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _preflight_frozen(path: Path, content: bytes) -> bool:
    """Return whether an identical artifact exists, rejecting changed content."""
    if not path.exists():
        return False
    if path.read_bytes() != content:
        raise FileExistsError(f"refusing to overwrite frozen file: {path}")
    return True


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _artifact_payloads(
    gold: Sequence[EvaluationQuery],
    result: RunResult,
    config: RuntimeConfig,
    snapshot_manifest_sha256: str,
) -> tuple[
    list[PredictionRecord],
    bytes,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    del gold
    predictions = [record.prediction for record in result.query_runs]
    prediction_bytes = _jsonl_bytes(predictions)
    input_hashes = {
        "gold": result.identity.gold_sha256,
        "predictions": _sha256_bytes(prediction_bytes),
    }
    if result.identity.id_map_sha256 is not None:
        input_hashes["id_map"] = result.identity.id_map_sha256
    metrics_payload: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "input_hashes": input_hashes,
        "snapshot_manifest": result.snapshot_manifest,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "per_query": {
            query_id: result.evaluation.per_query[query_id].model_dump(mode="json")
            for query_id in sorted(result.evaluation.per_query)
        },
        "summary": result.evaluation.summary.model_dump(mode="json"),
    }
    usage_payload: dict[str, object] = {
        "queries": [
            {
                "errors": [
                    {
                        "code": error.code,
                        "provider": error.provider,
                        "retryable": error.retryable,
                    }
                    for error in record.errors
                ],
                "latency_ms": record.latency_ms,
                "query_id": record.query_id,
                "usage": record.usage.model_dump(mode="json"),
            }
            for record in result.query_runs
        ],
        "total": result.usage.model_dump(mode="json"),
    }
    artifact_paths = {
        "deduplication": "deduplication.jsonl",
        "filtering": "filtering.jsonl",
        "metrics": "metrics.json",
        "predictions": "predictions.jsonl",
        "snapshot_manifest": result.snapshot_manifest,
        "usage": "usage.json",
    }
    run_payload: dict[str, object] = {
        "artifacts": artifact_paths,
        "config_hash": config.config_hash(),
        "contract_version": "week1-run-v2",
        "identity": result.identity.model_dump(mode="json", exclude_none=True),
        "input_hashes": input_hashes,
        "rules": {
            "deduplication": {
                "fuzzy_title_threshold": FUZZY_TITLE_THRESHOLD,
                "version": DEDUPLICATION_VERSION,
            },
            "filtering": {
                "minimum_uncertainty_multiplier": MINIMUM_UNCERTAINTY_MULTIPLIER,
                "uncertainty_reason_multiplier": UNCERTAINTY_REASON_MULTIPLIER,
                "version": FILTERING_VERSION,
            },
            "scoring": {
                "bm25_weight": BM25_WEIGHT,
                "keyword_coverage_weight": KEYWORD_COVERAGE_WEIGHT,
                "scoring_version": SCORING_VERSION,
                "tokenizer_version": TOKENIZER_VERSION,
            },
        },
        "query_snapshots": [
            {
                "cache_keys": record.cache_keys,
                "endpoint": record.endpoint,
                "page_hashes": record.page_hashes,
                "provider": record.provider,
                "query_id": record.query_id,
                "response_hash": record.response_hash,
            }
            for record in result.query_runs
        ],
        "scoring_version": SCORING_VERSION,
        "snapshot_manifest": result.snapshot_manifest,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
    }
    deduplication_records: list[dict[str, object]] = [
        {
            "merge_decisions": [
                decision.model_dump(mode="json")
                for decision in record.pipeline.deduplication.decisions
            ],
            "paper_ids": [paper.canonical_id for paper in record.pipeline.deduplication.papers],
            "query_id": record.query_id,
        }
        for record in result.query_runs
    ]
    filtering_records: list[dict[str, object]] = [
        {
            "accepted": [
                {
                    "paper_id": accepted.paper.canonical_id,
                    "uncertainty_reasons": accepted.uncertainty_reasons,
                }
                for accepted in record.pipeline.filtering.accepted
            ],
            "query_id": record.query_id,
            "rejected": [
                {
                    "paper_id": rejected.paper.canonical_id,
                    "reason_code": rejected.reason_code,
                }
                for rejected in record.pipeline.filtering.rejected
            ],
        }
        for record in result.query_runs
    ]
    return (
        predictions,
        prediction_bytes,
        metrics_payload,
        usage_payload,
        run_payload,
        deduplication_records,
        filtering_records,
    )


def _dictionary_jsonl_bytes(records: Sequence[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


def _write_artifacts(
    gold: Sequence[EvaluationQuery],
    result: RunResult,
    *,
    cache: SQLiteResponseCache,
    config: RuntimeConfig,
    output: Path,
) -> None:
    cache_keys = _ordered_unique([key for record in result.query_runs for key in record.cache_keys])
    try:
        prepared_snapshot = cache.prepare_snapshot(cache_keys)
    except KeyError as error:
        raise ValueError("provider snapshot cache key changed after query") from error
    _validate_prepared_query_snapshot_provenance(result.query_runs, prepared_snapshot)
    snapshot_manifest_sha256 = _sha256_bytes(prepared_snapshot.manifest_content)
    (
        predictions,
        prediction_bytes,
        metrics_payload,
        usage_payload,
        run_payload,
        deduplication_records,
        filtering_records,
    ) = _artifact_payloads(gold, result, config, snapshot_manifest_sha256)
    prepared = {
        output / "predictions.jsonl": prediction_bytes,
        output / "metrics.json": _frozen_json_bytes(metrics_payload),
        output / "usage.json": _frozen_json_bytes(usage_payload),
        output / "run.json": _frozen_json_bytes(run_payload),
        output / "deduplication.jsonl": _dictionary_jsonl_bytes(deduplication_records),
        output / "filtering.jsonl": _dictionary_jsonl_bytes(filtering_records),
    }
    existing = {path: _preflight_frozen(path, content) for path, content in prepared.items()}

    manifest_path = cache.write_snapshot(prepared_snapshot, output)
    validate_snapshot_manifest(manifest_path)
    predictions_path = output / "predictions.jsonl"
    if not existing[predictions_path]:
        write_jsonl_atomic(predictions_path, predictions)
    for filename, payload in (
        ("metrics.json", metrics_payload),
        ("usage.json", usage_payload),
        ("run.json", run_payload),
    ):
        _write_frozen_json(output / filename, payload)
    write_frozen_bytes(
        output / "deduplication.jsonl",
        prepared[output / "deduplication.jsonl"],
    )
    write_frozen_bytes(
        output / "filtering.jsonl",
        prepared[output / "filtering.jsonl"],
    )


async def _run_legacy_evaluation(
    gold: Sequence[EvaluationQuery],
    *,
    identity: RunIdentity,
    provider: SearchProvider,
    cache: SQLiteResponseCache,
    config: RuntimeConfig,
    output: Path,
    id_map: IdentifierMap | None = None,
) -> RunResult:
    """Run an evaluation split in order through an injected search provider."""
    query_runs: list[QueryRunRecord] = []
    for gold_record in gold:
        budget = HardBudgetController(config.budget)
        reservation = budget.reserve(
            f"evaluation-search:{gold_record.query_id}",
            UsageEstimate(
                search_api_calls=config.budget.max_search_api_calls,
                elapsed_ms=config.budget.max_elapsed_seconds * 1000,
            ),
        )
        provider_result = await provider.search(
            gold_record.query,
            {},
            config.budget.max_output_papers,
            reservation,
        )
        budget.settle(reservation, provider_result.usage)
        provider_name, endpoint, cache_keys, page_hashes, response_hash = (
            _validate_query_snapshot_provenance(provider_result.provenance, cache)
        )
        pipeline = process_candidates(
            _fallback_query_spec(gold_record),
            provider_result.data,
            id_map=id_map,
        )
        prediction = PredictionRecord(
            query_id=gold_record.query_id,
            predicted_paper_ids=[
                item.paper.canonical_id
                for item in pipeline.ranked[: config.budget.max_output_papers]
            ],
        )
        query_runs.append(
            QueryRunRecord(
                query_id=gold_record.query_id,
                prediction=prediction,
                pipeline=pipeline,
                usage=provider_result.usage,
                latency_ms=provider_result.latency_ms,
                provider=provider_name,
                endpoint=endpoint,
                cache_keys=cache_keys,
                page_hashes=page_hashes,
                response_hash=response_hash,
                errors=provider_result.errors,
            )
        )

    predictions = [record.prediction for record in query_runs]
    result = RunResult(
        evaluation=evaluate(gold, predictions, id_map=id_map),
        identity=identity,
        query_runs=query_runs,
        usage=_aggregate_usage(query_runs),
        snapshot_manifest="snapshot_manifest.json",
    )
    _write_artifacts(gold, result, cache=cache, config=config, output=output)
    return result


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def _execute_service_batch(
    gold: Sequence[EvaluationQuery],
    *,
    service: _ApplicationService,
    run_id: str,
    mode: SearchMode,
    on_start: Callable[[int], None] | None = None,
    on_record: Callable[[int, AdaptedExecution], None] | None = None,
) -> list[AdaptedExecution]:
    records: list[AdaptedExecution] = []
    for index, query in enumerate(gold):
        if on_start is not None:
            on_start(index)
        request = SearchRequest(
            query_id=query.query_id,
            query=query.query,
            mode=mode,
        )
        try:
            result = await service.execute(request, run_id=run_id)
        except asyncio.CancelledError:
            raise
        adapted = adapt_execution(expected_query_id=query.query_id, result=result)
        records.append(adapted)
        if on_record is not None:
            on_record(index, adapted)
        if adapted.failure is not None and adapted.failure.error_code == "integrity_failure":
            raise ValueError("formal evaluation stopped on integrity failure")
    return records


def _close_outstanding_reservations(
    *,
    ledger: SQLiteBudgetLedger,
    reservations: Sequence[LedgerReservation],
    estimates: Sequence[UsageEstimate],
    settled: set[int],
    current_index: int | None,
) -> None:
    """Fail every open reservation, conservatively charging the active query."""
    for index, reservation in enumerate(reservations):
        if index in settled:
            continue
        estimate = (
            estimates[index]
            if index == current_index
            else UsageEstimate(cost_cny=Decimal("0"))
        )
        try:
            ledger.fail(
                reservation,
                UsageActual.model_validate(estimate.model_dump()),
            )
        except Exception:  # noqa: BLE001
            continue


def _stop_on_formal_evidence_failure(
    measures: dict[str, MeasureValue],
) -> None:
    required_zero = (
        "integrity_failures",
        "provenance_failures",
        "sanitization_failures",
        "unaccounted_usage_failures",
    )
    if any(
        (measure := measures.get(name)) is None or measure.value != 0
        for name in required_zero
    ):
        raise ValueError("formal evidence integrity check failed")


def _validate_formal_workspace(path: Path) -> None:
    from paper_search.evaluation.validator import validate_run_directory

    result = validate_run_directory(path)
    if not result.valid:
        raise ValueError("formal workspace validation failed before publication")


def _reject_or_recover_existing_attempt(
    *,
    store: ValidationAttemptStore,
    validation_lock_sha256: str,
    output_root: Path,
) -> None:
    """Recover a published run's claim, then reject every consumed lock hash."""
    digest = validation_lock_sha256.removeprefix("sha256:")
    attempts_root = output_root / "validation-attempts"
    if not (
        (attempts_root / f"{digest}.claim").exists()
        or (attempts_root / f".{digest}.terminal").exists()
    ):
        return
    claim = store.read(validation_lock_sha256)
    if claim.state == "claimed":
        try:
            manifest = RunManifest.model_validate_json(
                (output_root / claim.run_id / "run.json").read_bytes()
            )
        except (OSError, ValueError):
            manifest = None
        if (
            manifest is not None
            and manifest.run_id == claim.run_id
            and manifest.status == "complete"
            and manifest.ended_at is not None
        ):
            store.transition(
                validation_lock_sha256=validation_lock_sha256,
                target="complete",
                completed_at=manifest.ended_at,
            )
        elif manifest is None:
            incomplete_paths = sorted(
                output_root.glob(f".incomplete-{claim.run_id}-*")
            )
            for incomplete_path in incomplete_paths:
                try:
                    incomplete_manifest = RunManifest.model_validate_json(
                        (incomplete_path / "run.json").read_bytes()
                    )
                except (OSError, ValueError):
                    continue
                if (
                    incomplete_manifest.run_id == claim.run_id
                    and incomplete_manifest.status == "incomplete"
                ):
                    completed_at = max(datetime.now(UTC), claim.claimed_at)
                    store.transition(
                        validation_lock_sha256=validation_lock_sha256,
                        target="interrupted",
                        completed_at=completed_at,
                        incident_ref=f"automatic-recovery:{claim.run_id}",
                    )
                    break
    raise ValidationAttemptConflictError(
        "validation lock already has an irrevocable attempt"
    )


async def _run_formal_evaluation(
    request: EvaluationRunRequest,
    *,
    composition_root: type[CompositionRoot],
    attempt_store_factory: Callable[[Path], ValidationAttemptStore],
    clock: Callable[[], datetime],
) -> EvaluationRunResult:
    inputs = _load_formal_inputs(request)
    started_at = clock()
    if started_at.tzinfo is None:
        raise ValueError("formal runner clock must be timezone-aware")
    run_id = (
        f"{request.split}-{started_at.astimezone(UTC):%Y%m%dT%H%M%SZ}-"
        f"{lock_sha256(inputs.lock).removeprefix('sha256:')[:12]}"
    )
    attempt_store: ValidationAttemptStore | None = None
    attempt_hash: str | None = None
    replacement_binding: tuple[str, str] | None = None
    if request.mode == "live" and isinstance(inputs.lock, ValidationLock):
        attempt_store = attempt_store_factory(request.output_root)
        attempt_hash = lock_sha256(inputs.lock)
        _reject_or_recover_existing_attempt(
            store=attempt_store,
            validation_lock_sha256=attempt_hash,
            output_root=request.output_root,
        )
        replacement_binding = attempt_store.replacement_binding(attempt_hash)
    bundle = composition_root.compose(
        lock_path=request.lock_path,
        mode=request.mode,
        artifact_root=Path.cwd(),
        output_root=request.output_root / ".runtime",
        snapshot_manifest_path=request.snapshot_manifest_path,
        network_authorized=request.network_authorized,
        lock_bytes=inputs.lock_bytes,
        runtime_config=request.runtime_config,
        ablation_config=request.ablation_config_path or Path("configs/ablations.yaml"),
    )
    try:
        readiness = bundle.readiness_probe()
    except BaseException:
        await bundle.aclose()
        raise
    if request.mode == "live" and readiness.status != "ready":
        await bundle.aclose()
        raise ValueError("authorized live readiness is not current")

    if isinstance(inputs.lock, ReplayLock):
        if inputs.snapshot_manifest is None:
            raise ValueError("replay snapshot manifest is unavailable")
        snapshot_manifest = inputs.snapshot_manifest
        snapshot_sha256 = inputs.lock.snapshot_manifest_sha256
        snapshot_set_id = inputs.lock.snapshot_set_id
    else:
        snapshot_set_id = _sha256_bytes(b"[]")
        snapshot_manifest = DependencySnapshotManifestV2(
            snapshot_set_id=snapshot_set_id,
            sealed_at=started_at,
            entries=[],
        )
        snapshot_sha256 = _sha256_bytes(_frozen_json_bytes(snapshot_manifest.model_dump(mode="json")))

    manifest = RunManifest(
        run_id=run_id,
        status="incomplete",
        gate_result="not_applicable",
        execution_mode=request.mode,
        split=request.split,
        frozen_manifest_sha256=inputs.lock.frozen_data.manifest.sha256,
        partition_sha256=inputs.lock.frozen_data.partition_sha256,
        identifier_map_sha256=inputs.lock.frozen_data.identifier_map.sha256,
        source_git_sha=inputs.lock.source_git_sha,
        tracked_source_dirty=False,
        config_hash=bundle.config_hash,
        input_lock_sha256=_sha256_bytes(inputs.lock_bytes),
        prompt_version=inputs.lock.baseline.prompt_version,
        snapshot_set_id=snapshot_set_id,
        snapshot_manifest_sha256=snapshot_sha256,
        experiment_name=bundle.experiment_id,
        optional_modules=bundle.optional_modules,
        started_at=started_at,
        ended_at=None,
        readiness_summary=readiness.dependencies,
        failure_count=0,
    )
    workspace: FormalRunWorkspace | None = None
    ledger: SQLiteBudgetLedger | None = None
    try:
        workspace = FormalRunWorkspace(
            runs_root=request.output_root,
            manifest=manifest,
            input_lock_bytes=inputs.lock_bytes,
            nonce_factory=lambda: hashlib.sha256(run_id.encode()).hexdigest()[:12],
            clock=clock,
            validator=_validate_formal_workspace,
            replay_snapshot_root=inputs.snapshot_root,
        )
        if request.mode == "live":
            bundle.artifact_factory._sessions[run_id.casefold()] = _FormalCaptureBinding(  # type: ignore[assignment]  # noqa: SLF001
                run_id=run_id,
                store=workspace.snapshot_store,
            )
        ledger = SQLiteBudgetLedger(
            request.output_root / ".ledger" / "formal.sqlite3",
            clock=clock,
            replay=request.mode == "replay",
        )
        if not isinstance(inputs.lock, ReplayLock):
            checkpoint_count, checkpoint_sha256 = ledger.project_checkpoint()
            if (
                checkpoint_count != inputs.lock.project_ledger.receipt_count
                or checkpoint_sha256 != inputs.lock.project_ledger.receipts_sha256
            ):
                raise ValueError("project ledger checkpoint does not match input lock")
    except BaseException:
        if ledger is not None:
            ledger.close()
        if workspace is not None and workspace.work_dir.exists():
            try:
                workspace.fail("internal_error")
            except Exception:  # noqa: BLE001
                pass
        await bundle.aclose()
        raise
    assert workspace is not None
    assert ledger is not None
    per_query_cost = (
        Decimal("0")
        if request.mode == "replay"
        else min(
            Decimal("0.30"),
            (
                DEV_RUN_CAP_CNY
                if request.split == "dev"
                else VALIDATION_RUN_CAP_CNY
            )
            / Decimal(len(inputs.gold)),
        )
    )
    estimates = [
        UsageEstimate(
            search_api_calls=(
                inputs.lock.baseline.retrieval.openalex_calls_max
                + inputs.lock.baseline.retrieval.semantic_scholar_calls_max
            ),
            llm_calls=inputs.lock.baseline.retry.max_attempts,
            input_tokens=inputs.lock.baseline.retrieval.max_raw_candidates,
            output_tokens=inputs.lock.baseline.retrieval.max_output_papers,
            cost_cny=per_query_cost,
            elapsed_ms=inputs.lock.baseline.timeout.read_seconds * 1_000,
        )
        for _ in inputs.gold
    ]
    reservations: list[LedgerReservation] = []
    settled: set[int] = set()
    current_index: int | None = None
    claim_created = False
    try:
        if attempt_store is not None and attempt_hash is not None:
            attempt_store.claim(
                validation_lock_sha256=attempt_hash,
                run_id=run_id,
                claimed_at=started_at,
                supersedes_validation_lock_sha256=(
                    replacement_binding[0] if replacement_binding is not None else None
                ),
                incident_ref=(
                    replacement_binding[1] if replacement_binding is not None else None
                ),
            )
            claim_created = True

        def on_start(index: int) -> None:
            nonlocal current_index
            current_index = index
            reservation = ledger.reserve(
                run_id=run_id,
                query_id=inputs.gold[index].query_id,
                estimate=estimates[index],
                run_cap_cny=(
                    DEV_RUN_CAP_CNY
                    if request.split == "dev"
                    else VALIDATION_RUN_CAP_CNY
                ),
            )
            reservations.append(reservation)

        def settle_and_append(index: int, record: AdaptedExecution) -> None:
            ledger.settle(reservations[index], record.execution.usage)
            settled.add(index)
            workspace.write_prediction(record.prediction)
            workspace.write_execution(record.execution)
            workspace.write_business_result(record.business_result)
            if record.failure is not None:
                workspace.write_failure(record.failure)

        adapted = await _execute_service_batch(
            inputs.gold,
            service=bundle.service,
            run_id=run_id,
            mode=request.mode,
            on_start=on_start,
            on_record=settle_and_append,
        )
        predictions = [adapt_prediction_record(record.prediction) for record in adapted]
        metrics = evaluate(inputs.gold, predictions, id_map=inputs.identifier_map)
        report = ledger.report(run_id)
        failures = [record.failure for record in adapted if record.failure is not None]
        business_results = [record.business_result for record in adapted]
        if isinstance(inputs.lock, ReplayLock):
            if inputs.snapshot_root is None:
                raise ValueError("replay snapshot root is unavailable")
            replay_lock = inputs.lock
            snapshot_reader = DependencySnapshotReader(
                inputs.snapshot_root / "snapshot-manifest.json",
                snapshot_manifest_sha256=snapshot_sha256,
                snapshot_set_id=snapshot_set_id,
            )
        else:
            sealed = workspace.seal_snapshots()
            snapshot_manifest = sealed
            snapshot_set_id = sealed.snapshot_set_id
            snapshot_sha256 = workspace.snapshot_store.manifest_sha256
            snapshot_reader = DependencySnapshotReader(
                workspace.snapshot_store.manifest_path,
                snapshot_manifest_sha256=snapshot_sha256,
                snapshot_set_id=snapshot_set_id,
            )
            replay_lock = ReplayLock(
                schema_version=inputs.lock.schema_version,
                lock_kind="replay",
                created_at=started_at,
                source_capture_run_id=run_id,
                source_git_sha=inputs.lock.source_git_sha,
                runtime_allow_live=inputs.lock.runtime_allow_live,
                frozen_data=inputs.lock.frozen_data,
                baseline=inputs.lock.baseline,
                budget_config=inputs.lock.budget_config,
                pricing_policy=inputs.lock.pricing_policy,
                quality_gates=inputs.lock.quality_gates,
                capture_policy=inputs.lock.capture_policy,
                project_ledger=inputs.lock.project_ledger,
                snapshot_set_id=snapshot_set_id,
                snapshot_manifest_sha256=snapshot_sha256,
            )
        audit_measures = complete_policy_measures(
            formal_audit_measures(
                frozen_queries=inputs.gold,
                executions=[record.execution for record in adapted],
                business_results=business_results,
                failures=failures,
                ledger_report=report,
                identifier_map=inputs.identifier_map,
                metrics=metrics,
                configured_endpoints=configured_retrieval_endpoints(
                    inputs.lock.baseline.retrieval
                ),
                snapshot_manifest=snapshot_manifest,
                snapshot_reader=snapshot_reader,
                prompt_version=inputs.lock.baseline.prompt_version,
                prompt_name=Path(
                    inputs.lock.baseline.planner.prompt_config.path
                ).stem,
                prompt_artifact_sha256=inputs.prompt_artifact_sha256,
                llm_model_allowlist=frozenset(
                    {
                        inputs.lock.baseline.primary_model,
                        inputs.lock.baseline.fallback_model,
                    }
                ),
            ),
            policy=inputs.gate_policy,
            split=request.split,
        )
        _stop_on_formal_evidence_failure(audit_measures)
        gate = evaluate_gates(
            frozen_queries=inputs.gold,
            predictions=predictions,
            failures=failures,
            metrics=metrics,
            audit_measures=audit_measures,
            ledger_report=report,
            policy=inputs.gate_policy,
        )
        workspace.write_metrics(metrics)
        workspace.bind_ledger_checkpoint(report)
        workspace.write_usage(report)
        run_path = workspace.finalize(
            gate_evaluation=gate,
            replay_lock=replay_lock,
            snapshot_manifest=snapshot_manifest,
        )
        if claim_created and attempt_store is not None and attempt_hash is not None:
            try:
                attempt_store.transition(
                    validation_lock_sha256=attempt_hash,
                    target="complete",
                    completed_at=clock(),
                )
            except Exception:  # noqa: BLE001
                pass
        return EvaluationRunResult(
            run_id=run_id,
            run_path=run_path,
            status="complete",
            gate_result=gate.gate_result,
        )
    except (asyncio.CancelledError, KeyboardInterrupt):
        _close_outstanding_reservations(
            ledger=ledger,
            reservations=reservations,
            estimates=estimates,
            settled=settled,
            current_index=current_index,
        )
        if claim_created and attempt_store is not None and attempt_hash is not None:
            try:
                attempt_store.transition(
                    validation_lock_sha256=attempt_hash,
                    target="interrupted",
                    completed_at=clock(),
                    incident_ref=f"automatic-interruption:{run_id}",
                )
            except Exception:  # noqa: BLE001
                pass
        if workspace.work_dir.exists():
            workspace.interrupt()
        raise
    except Exception:
        _close_outstanding_reservations(
            ledger=ledger,
            reservations=reservations,
            estimates=estimates,
            settled=settled,
            current_index=current_index,
        )
        if claim_created and attempt_store is not None and attempt_hash is not None:
            try:
                attempt_store.transition(
                    validation_lock_sha256=attempt_hash,
                    target="failed",
                    completed_at=clock(),
                    incident_ref=f"automatic-failure:{run_id}",
                )
            except Exception:  # noqa: BLE001
                pass
        if workspace.work_dir.exists():
            workspace.fail("internal_error")
        raise
    finally:
        ledger.close()
        await bundle.aclose()


async def run_evaluation(
    request: EvaluationRunRequest | Sequence[EvaluationQuery],
    *,
    composition_root: type[CompositionRoot] = CompositionRoot,
    attempt_store_factory: Callable[[Path], ValidationAttemptStore] = (
        ValidationAttemptStore
    ),
    clock: Callable[[], datetime] = _utc_now,
    identity: RunIdentity | None = None,
    provider: SearchProvider | None = None,
    cache: SQLiteResponseCache | None = None,
    config: RuntimeConfig | None = None,
    output: Path | None = None,
    id_map: IdentifierMap | None = None,
) -> EvaluationRunResult | RunResult:
    """Run the canonical formal path; retain the Week-1 path as compatibility."""
    if isinstance(request, EvaluationRunRequest):
        return await _run_formal_evaluation(
            request,
            composition_root=composition_root,
            attempt_store_factory=attempt_store_factory,
            clock=clock,
        )
    if identity is None or provider is None or cache is None or config is None or output is None:
        raise TypeError("legacy evaluation requires identity, provider, cache, config, and output")
    return await _run_legacy_evaluation(
        request,
        identity=identity,
        provider=provider,
        cache=cache,
        config=config,
        output=output,
        id_map=id_map,
    )


async def _run_cli_evaluation(
    gold: Sequence[EvaluationQuery],
    *,
    identity: RunIdentity,
    config: RuntimeConfig,
    api_key: str,
    output: Path,
    id_map: IdentifierMap | None,
) -> None:
    timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        cache = SQLiteResponseCache(output.parent / ".cache" / "openalex.sqlite3")
        provider = OpenAlexProvider(client=client, cache=cache, api_key=api_key)
        await run_evaluation(
            gold,
            identity=identity,
            provider=provider,
            cache=cache,
            config=config,
            output=output,
            id_map=id_map,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        frozen_split = _resolve_frozen_split(Path("data"), args.split, _current_git_sha())
        identity = frozen_split.identity
        id_map: IdentifierMap | None = None
        if args.id_map is not None:
            id_map_bytes = _read_confined_identifier_map(Path("data"), args.id_map)
            id_map = IdentifierMap.from_bytes(id_map_bytes)
            _require_id_map_coverage(frozen_split.gold, id_map)
            identity = identity.model_copy(
                update={"id_map_sha256": _sha256_bytes(id_map_bytes)}
            )
        config = load_runtime_config(args.config, env_file=None)
        if config.openalex_api_key is None:
            raise _CliInputError("OPENALEX_API_KEY is required")
        asyncio.run(
            _run_cli_evaluation(
                frozen_split.gold,
                identity=identity,
                config=config,
                api_key=config.openalex_api_key.get_secret_value(),
                output=args.output.resolve(),
                id_map=id_map,
            )
        )
    except _CliInputError as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 2
    except FileExistsError:
        print("evaluation failed: output artifacts already differ", file=sys.stderr)
        return 2
    except ValidationError:
        print("evaluation failed: invalid evaluation input", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError):
        print("evaluation failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
