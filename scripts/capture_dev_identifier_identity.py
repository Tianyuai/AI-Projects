"""Bounded, two-stage development identifier identity capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Literal, Protocol, Self, cast, runtime_checkable
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from paper_search.control.ledger import DEV_RUN_CAP_CNY, LedgerReservation, SQLiteBudgetLedger
from paper_search.domain.models import (
    DomainModel,
    NonEmptyStr,
    NonNegativeInt,
    SafeRelativePath,
    Sha256,
    UsageActual,
    UsageEstimate,
)
from paper_search.evaluation.dataset import IdentifierMap, normalize_paper_id
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencyRequestIdentity,
    DependencySnapshotReader,
)


S2_BASE_URL = "https://api.semanticscholar.org"
S2_ENDPOINT = "/graph/v1/paper/batch"
S2_API_KEY_ENV = "SEMANTIC_SCHOLAR_API_KEY"
OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_ENDPOINT_TEMPLATE = "/works"
OPENALEX_API_KEY_ENV = "OPENALEX_API_KEY"
DEFAULT_OUTPUT_ROOT = "data/annotation_work/identifier_semantics"
REQUEST_COST_CNY = Decimal("0.01")
S2_FIELDS = "paperId,externalIds"
_SAFE_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "x-request-id",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-credits-used",
        "x-ratelimit-reset",
    }
)
RequestFailureCategory = Literal[
    "timeout",
    "network_error",
    "rate_limited",
    "client_error",
    "server_error",
    "unexpected_status",
]


class IdentifierInventory(DomainModel):
    schema_version: Literal["identifier-identity-inventory-v1"]
    scope: Literal["dev"]
    source_hashes: dict[NonEmptyStr, Sha256]
    arxiv_ids: list[NonEmptyStr]
    candidate_aliases: list[NonEmptyStr]


class IdentityCaptureLock(DomainModel):
    schema_version: Literal["identifier-identity-capture-lock-v2"]
    scope: Literal["dev"]
    input_hashes: dict[NonEmptyStr, Sha256]
    semantic_scholar_arxiv_ids: list[NonEmptyStr]
    semantic_scholar_batch_max: Literal[2] = 2
    semantic_scholar_http_attempt_max: Literal[4] = 4
    semantic_scholar_base_url: Literal["https://api.semanticscholar.org"]
    semantic_scholar_endpoint: Literal["/graph/v1/paper/batch"]
    semantic_scholar_api_key_env: Literal["SEMANTIC_SCHOLAR_API_KEY"]
    openalex_exact_ids: list[NonEmptyStr]
    openalex_request_max: NonNegativeInt
    openalex_base_url: Literal["https://api.openalex.org"]
    openalex_endpoint_template: Literal["/works"]
    openalex_api_key_env: Literal["OPENALEX_API_KEY"]
    output_root: SafeRelativePath
    retry_max: Literal[1] = 1
    ledger_checkpoint_sha256: Sha256


class DerivedDoiLock(DomainModel):
    schema_version: Literal["identifier-identity-derived-doi-lock-v1"]
    parent_lock_sha256: Sha256
    arxiv_batch_snapshot_sha256: Sha256
    ids: list[NonEmptyStr]


class IdentityEvidenceRef(DomainModel):
    arxiv_id: NonEmptyStr
    alias: NonEmptyStr
    semantic_scholar_arxiv_entry_id: NonEmptyStr | None = None
    semantic_scholar_arxiv_item_index: NonNegativeInt | None = None
    semantic_scholar_doi_entry_id: NonEmptyStr | None = None
    semantic_scholar_doi_item_index: NonNegativeInt | None = None
    openalex_entry_ids: list[NonEmptyStr] = Field(default_factory=list)


class IdentityCaptureResult(DomainModel):
    schema_version: Literal["identifier-identity-evidence-v1"]
    scope: Literal["dev"]
    capture_lock_sha256: Sha256
    derived_doi_lock: DerivedDoiLock
    semantic_scholar_batch_count: NonNegativeInt
    semantic_scholar_http_attempt_count: NonNegativeInt
    openalex_request_count: NonNegativeInt
    snapshot_manifest_sha256: Sha256
    evidence_refs: list[IdentityEvidenceRef]


@dataclass(frozen=True)
class TransportResponse:
    status_code: int
    content: bytes
    headers: Mapping[str, str]


@runtime_checkable
class CaptureTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        query_params: Mapping[str, str] | None,
        json_body: object | None,
    ) -> TransportResponse: ...


class HttpxCaptureTransport:
    """Small synchronous adapter used only by an explicitly authorized CLI run."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self._timeout = timeout_seconds

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        query_params: Mapping[str, str] | None,
        json_body: object | None,
    ) -> TransportResponse:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.request(
                method,
                url,
                headers=headers,
                params=query_params,
                json=json_body,
            )
        return TransportResponse(
            status_code=response.status_code,
            content=response.content,
            headers=dict(response.headers),
        )


class IdentifierCaptureRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    project_root: Path
    inventory_path: Path
    ledger_path: Path
    expected_lock_sha256: Sha256
    output_root: str
    semantic_scholar_base_url: str
    semantic_scholar_endpoint: str
    semantic_scholar_api_key_env: str
    openalex_base_url: str
    openalex_endpoint_template: str
    openalex_api_key_env: str
    credential_values: dict[str, SecretStr]
    transport: CaptureTransport
    allow_network: bool = False
    manual_doi_ids: tuple[str, ...] = ()
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    sleeper: Callable[[float], None] = time.sleep

    def with_transport(self, transport: CaptureTransport) -> Self:
        return self.model_copy(update={"transport": transport})

    def with_manual_doi_addition(self, identifier: str) -> Self:
        return self.model_copy(update={"manual_doi_ids": (*self.manual_doi_ids, identifier)})


@dataclass
class _AttemptCounts:
    semantic_scholar: int = 0
    openalex: int = 0


class _InvalidProviderResponse(ValueError):
    pass


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _path_binding(path: Path) -> str:
    return _sha256(path.resolve().as_posix().casefold().encode("utf-8"))


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _model_bytes(model: DomainModel) -> bytes:
    return _canonical_json(model.model_dump(mode="json", exclude_none=True))


def _write_exclusive_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite sealed file: {path}") from None
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(content: bytes, *, message: str) -> dict[str, object]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(message) from None
    if not isinstance(value, dict):
        raise ValueError(message)
    return cast(dict[str, object], value)


def build_identifier_inventory(
    gold_path: Path,
    candidate_map_path: Path,
    out_path: Path,
) -> IdentifierInventory:
    """Extract only normalized identifiers from development gold and the current map."""
    gold_content = gold_path.read_bytes()
    map_content = candidate_map_path.read_bytes()
    arxiv_ids: set[str] = set()
    try:
        lines = gold_content.splitlines()
        if not lines:
            raise ValueError
        for line in lines:
            record = _read_json_object(line, message="development gold is invalid")
            identifiers = record.get("relevant_paper_ids")
            if not isinstance(identifiers, list) or not all(
                isinstance(identifier, str) for identifier in identifiers
            ):
                raise ValueError
            for identifier in identifiers:
                normalized = normalize_paper_id(identifier)
                if normalized.startswith("arxiv:"):
                    arxiv_ids.add(normalized)
        identifier_map = IdentifierMap.from_bytes(map_content)
        aliases = {
            normalize_paper_id(alias)
            for alias, _ in identifier_map.resolved_pairs()
            if not normalize_paper_id(alias).startswith("arxiv:")
        }
    except (ValidationError, ValueError):
        raise ValueError("identifier inventory inputs are invalid") from None
    if not arxiv_ids:
        raise ValueError("identifier inventory inputs are invalid")
    inventory = IdentifierInventory(
        schema_version="identifier-identity-inventory-v1",
        scope="dev",
        source_hashes={
            "dev_gold": _sha256(gold_content),
            "candidate_map": _sha256(map_content),
        },
        arxiv_ids=sorted(arxiv_ids),
        candidate_aliases=sorted(aliases),
    )
    _write_exclusive_atomic(out_path, _model_bytes(inventory))
    return inventory


def ledger_checkpoint(path: Path) -> str:
    if not path.exists():
        return _sha256(b"[]\n")
    ledger = SQLiteBudgetLedger(path)
    try:
        return ledger.project_checkpoint()[1]
    finally:
        ledger.close()


def build_capture_lock(
    *,
    inventory_path: Path,
    ledger_path: Path,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    out_lock: Path | None = None,
) -> IdentityCaptureLock:
    inventory_content = inventory_path.read_bytes()
    try:
        inventory = IdentifierInventory.model_validate_json(inventory_content)
    except ValidationError:
        raise ValueError("identifier inventory is invalid") from None
    arxiv_ids = sorted(
        {f"ARXIV:{value.removeprefix('arxiv:')}" for value in inventory.arxiv_ids}
    )
    openalex_ids = sorted(
        {
            alias.removeprefix("openalex:")
            for alias in inventory.candidate_aliases
            if alias.startswith("openalex:")
        }
    )
    lock = IdentityCaptureLock(
        schema_version="identifier-identity-capture-lock-v2",
        scope="dev",
        input_hashes={
            "identifier_inventory": _sha256(inventory_content),
            "ledger_path": _path_binding(ledger_path),
        },
        semantic_scholar_arxiv_ids=arxiv_ids,
        semantic_scholar_base_url=S2_BASE_URL,
        semantic_scholar_endpoint=S2_ENDPOINT,
        semantic_scholar_api_key_env=S2_API_KEY_ENV,
        openalex_exact_ids=openalex_ids,
        openalex_request_max=len(openalex_ids) * 2,
        openalex_base_url=OPENALEX_BASE_URL,
        openalex_endpoint_template=OPENALEX_ENDPOINT_TEMPLATE,
        openalex_api_key_env=OPENALEX_API_KEY_ENV,
        output_root=output_root,
        ledger_checkpoint_sha256=ledger_checkpoint(ledger_path),
    )
    if out_lock is not None:
        _write_exclusive_atomic(out_lock, _model_bytes(lock))
    return lock


def _lock_hash(lock: IdentityCaptureLock) -> str:
    return _sha256(_model_bytes(lock))


def _safe_output_root(lock: IdentityCaptureLock, runtime: IdentifierCaptureRuntime) -> Path:
    project_root = runtime.project_root.resolve()
    output_root = (project_root / lock.output_root).resolve()
    if not output_root.is_relative_to(project_root):
        raise ValueError("identity capture authorization mismatch")
    return output_root


def _authorize_capture(
    lock: IdentityCaptureLock,
    runtime: IdentifierCaptureRuntime,
) -> tuple[IdentifierInventory, Path]:
    try:
        lock = IdentityCaptureLock.model_validate(lock.model_dump(mode="python"))
    except ValidationError:
        raise ValueError("identity capture authorization mismatch") from None
    if _lock_hash(lock) != runtime.expected_lock_sha256:
        raise ValueError("identity capture authorization mismatch")
    locked_values = {
        "semantic_scholar_base_url": lock.semantic_scholar_base_url,
        "semantic_scholar_endpoint": lock.semantic_scholar_endpoint,
        "semantic_scholar_api_key_env": lock.semantic_scholar_api_key_env,
        "openalex_base_url": lock.openalex_base_url,
        "openalex_endpoint_template": lock.openalex_endpoint_template,
        "openalex_api_key_env": lock.openalex_api_key_env,
        "output_root": lock.output_root,
    }
    if any(getattr(runtime, name) != value for name, value in locked_values.items()):
        raise ValueError("identity capture authorization mismatch")
    if not runtime.allow_network:
        raise ValueError("explicit network authorization is required")
    inventory_content = runtime.inventory_path.read_bytes()
    if _sha256(inventory_content) != lock.input_hashes.get("identifier_inventory"):
        raise ValueError("identifier inventory hash mismatch")
    if _path_binding(runtime.ledger_path) != lock.input_hashes.get("ledger_path"):
        raise ValueError("identity capture authorization mismatch")
    try:
        inventory = IdentifierInventory.model_validate_json(inventory_content)
    except ValidationError:
        raise ValueError("identifier inventory is invalid") from None
    expected_arxiv_ids = sorted(
        {f"ARXIV:{value.removeprefix('arxiv:')}" for value in inventory.arxiv_ids}
    )
    expected_openalex_ids = sorted(
        {
            alias.removeprefix("openalex:")
            for alias in inventory.candidate_aliases
            if alias.startswith("openalex:")
        }
    )
    if (
        lock.semantic_scholar_arxiv_ids != expected_arxiv_ids
        or lock.openalex_exact_ids != expected_openalex_ids
        or lock.semantic_scholar_batch_max != 2
        or lock.semantic_scholar_http_attempt_max != 4
        or lock.retry_max != 1
        or lock.openalex_request_max != len(expected_openalex_ids) * 2
    ):
        raise ValueError("identity capture authorization mismatch")
    if len(lock.openalex_exact_ids) * (lock.retry_max + 1) > lock.openalex_request_max:
        raise ValueError("identity capture request cap exceeded")
    maximum_s2_attempts = lock.semantic_scholar_batch_max * (lock.retry_max + 1)
    if maximum_s2_attempts > lock.semantic_scholar_http_attempt_max:
        raise ValueError("identity capture request cap exceeded")
    if ledger_checkpoint(runtime.ledger_path) != lock.ledger_checkpoint_sha256:
        raise ValueError("ledger checkpoint mismatch")
    semantic_scholar_key = runtime.credential_values.get(lock.semantic_scholar_api_key_env)
    if semantic_scholar_key is None or not semantic_scholar_key.get_secret_value():
        raise ValueError("semantic scholar credential is unavailable")
    openalex_key = runtime.credential_values.get(lock.openalex_api_key_env)
    if lock.openalex_exact_ids and (
        openalex_key is None or not openalex_key.get_secret_value()
    ):
        raise ValueError("openalex credential is unavailable")
    return inventory, _safe_output_root(lock, runtime)


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name.casefold(): value
        for name, value in headers.items()
        if name.casefold() in _SAFE_RESPONSE_HEADERS
    }


def _terminalize(
    ledger: SQLiteBudgetLedger,
    reservation: LedgerReservation,
    *,
    failed: bool,
) -> None:
    actual = UsageActual(search_api_calls=1, cost_cny=REQUEST_COST_CNY)
    ledger.checkpoint_actual(reservation, actual)
    if failed:
        ledger.fail(reservation, actual)
    else:
        ledger.settle(reservation, actual)


def _exception_failure_category(error: BaseException) -> RequestFailureCategory:
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    return "network_error"


def _status_failure_category(status_code: int) -> RequestFailureCategory:
    if status_code == 429:
        return "rate_limited"
    if 400 <= status_code < 500:
        return "client_error"
    if 500 <= status_code < 600:
        return "server_error"
    return "unexpected_status"


def _s2_retry_delay(headers: Mapping[str, str] | None = None) -> float:
    retry_after = next(
        (
            value
            for name, value in (headers or {}).items()
            if name.casefold() == "retry-after"
        ),
        None,
    )
    if retry_after is not None:
        try:
            seconds = float(retry_after)
        except ValueError:
            pass
        else:
            if math.isfinite(seconds):
                return min(30.0, max(1.0, seconds))
    return 5.0


def _sleep_before_retry(
    provider: Literal["semantic_scholar", "openalex"],
    runtime: IdentifierCaptureRuntime,
    headers: Mapping[str, str] | None = None,
) -> None:
    if provider == "semantic_scholar":
        runtime.sleeper(_s2_retry_delay(headers))


def _request_json(
    *,
    provider: Literal["semantic_scholar", "openalex"],
    stage: str,
    method: Literal["GET", "POST"],
    url: str,
    headers: Mapping[str, str],
    query_params: Mapping[str, str] | None,
    json_body: object | None,
    runtime: IdentifierCaptureRuntime,
    lock: IdentityCaptureLock,
    ledger: SQLiteBudgetLedger,
    counts: _AttemptCounts,
    validator: Callable[[object], object],
) -> tuple[TransportResponse, object]:
    error_message = (
        "semantic scholar request failed"
        if provider == "semantic_scholar"
        else "openalex request failed"
    )
    invalid_message = (
        "semantic scholar response is invalid"
        if provider == "semantic_scholar"
        else "openalex response is invalid"
    )
    for retry_index in range(lock.retry_max + 1):
        if provider == "semantic_scholar":
            if counts.semantic_scholar >= lock.semantic_scholar_http_attempt_max:
                raise ValueError("identity capture request cap exceeded")
            counts.semantic_scholar += 1
            attempt_number = counts.semantic_scholar
        else:
            if counts.openalex >= lock.openalex_request_max:
                raise ValueError("identity capture request cap exceeded")
            counts.openalex += 1
            attempt_number = counts.openalex
        reservation = ledger.reserve(
            run_id=f"identifier-identity-{_lock_hash(lock).removeprefix('sha256:')[:16]}",
            query_id=f"{provider}:{stage}:{attempt_number}",
            estimate=UsageEstimate(search_api_calls=1, cost_cny=REQUEST_COST_CNY),
            run_cap_cny=DEV_RUN_CAP_CNY,
        )
        try:
            response = runtime.transport.request(
                method,
                url,
                headers=headers,
                query_params=query_params,
                json_body=json_body,
            )
        except BaseException as error:
            _terminalize(ledger, reservation, failed=True)
            if retry_index == lock.retry_max:
                category = _exception_failure_category(error)
                raise ValueError(f"{error_message}: {category}") from None
            _sleep_before_retry(provider, runtime)
            continue
        if not 200 <= response.status_code < 300:
            _terminalize(ledger, reservation, failed=True)
            if retry_index == lock.retry_max:
                category = _status_failure_category(response.status_code)
                raise ValueError(f"{error_message}: {category}")
            _sleep_before_retry(provider, runtime, response.headers)
            continue
        try:
            payload = json.loads(response.content)
            validated = validator(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, _InvalidProviderResponse):
            _terminalize(ledger, reservation, failed=True)
            if retry_index == lock.retry_max:
                raise ValueError(invalid_message) from None
            _sleep_before_retry(provider, runtime, response.headers)
            continue
        _terminalize(ledger, reservation, failed=False)
        return response, validated
    raise RuntimeError("request retry loop exhausted")


def _validate_s2_batch(payload: object, expected_ids: Sequence[str]) -> object:
    if not isinstance(payload, list) or len(payload) != len(expected_ids):
        raise _InvalidProviderResponse
    records: list[dict[str, object] | None] = []
    for expected_id, item in zip(expected_ids, payload, strict=True):
        if item is None:
            records.append(None)
            continue
        if not isinstance(item, dict):
            raise _InvalidProviderResponse
        paper_id = item.get("paperId")
        external_ids = item.get("externalIds")
        if paper_id is not None and not isinstance(paper_id, str):
            raise _InvalidProviderResponse
        if not isinstance(external_ids, dict) or not all(
            isinstance(key, str) for key in external_ids
        ):
            raise _InvalidProviderResponse
        for relevant_key in ("ArXiv", "DOI"):
            if relevant_key in external_ids and not isinstance(
                external_ids[relevant_key], str
            ):
                raise _InvalidProviderResponse
        expected_kind = "arxiv" if expected_id.startswith("ARXIV:") else "doi"
        expected_value = normalize_paper_id(expected_id, kind=expected_kind)
        provider_value = external_ids.get("ArXiv" if expected_kind == "arxiv" else "DOI")
        if (
            not isinstance(paper_id, str)
            or not paper_id.strip()
            or not isinstance(provider_value, str)
        ):
            records.append(None)
            continue
        try:
            normalized_provider_value = normalize_paper_id(
                provider_value, kind=expected_kind
            )
        except ValueError:
            records.append(None)
            continue
        if normalized_provider_value != expected_value:
            records.append(None)
            continue
        records.append(cast(dict[str, object], item))
    return records


def _validate_openalex(payload: object, expected_id: str) -> object:
    if not isinstance(payload, dict):
        raise _InvalidProviderResponse
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise _InvalidProviderResponse
    result = results[0]
    if not isinstance(result, dict):
        raise _InvalidProviderResponse
    response_id = result.get("id")
    if not isinstance(response_id, str) or response_id.rstrip("/").rsplit("/", 1)[-1] != expected_id:
        raise _InvalidProviderResponse
    locations = result.get("locations", [])
    if not isinstance(locations, list):
        raise _InvalidProviderResponse
    return cast(dict[str, object], result)


def _s2_identity(
    ids: Sequence[str], *, stage: Literal["arxiv", "doi"]
) -> DependencyRequestIdentity:
    return DependencyRequestIdentity.from_canonical_request(
        dependency="semantic_scholar",
        operation="batch",
        method="POST",
        endpoint="/paper/batch",
        model_or_adapter=f"semantic-scholar-identity-{stage}-v1",
        canonical_request={"fields": S2_FIELDS, "ids": list(ids)},
    )


def _openalex_identity(identifier: str) -> DependencyRequestIdentity:
    return DependencyRequestIdentity.from_canonical_request(
        dependency="openalex",
        operation="search",
        method="GET",
        endpoint="/works",
        model_or_adapter="openalex-identity-v1",
        canonical_request={"filter": f"openalex:{identifier}", "per_page": "1"},
    )


def _discovered_dois(
    records: Sequence[dict[str, object] | None],
    requested_arxiv_ids: Sequence[str],
) -> dict[str, tuple[str, int, str]]:
    discovered: dict[str, tuple[str, int, str]] = {}
    for index, (record, requested) in enumerate(
        zip(records, requested_arxiv_ids, strict=True)
    ):
        if record is None:
            continue
        external_ids = cast(dict[str, str], record["externalIds"])
        raw_doi = external_ids.get("DOI")
        if raw_doi is None:
            continue
        try:
            alias = normalize_paper_id(raw_doi, kind="doi")
        except ValueError:
            continue
        provider_id = f"DOI:{alias.removeprefix('doi:')}"
        arxiv_id = normalize_paper_id(requested, kind="arxiv")
        existing = discovered.get(provider_id)
        binding = (arxiv_id, index, alias)
        if existing is not None and existing != binding:
            raise ValueError("derived DOI lock mismatch")
        discovered[provider_id] = binding
    return discovered


def _openalex_arxiv_ids(payload: Mapping[str, object]) -> list[str]:
    values: set[str] = set()
    locations = payload.get("locations", [])
    if not isinstance(locations, list):
        return []
    for location in locations:
        if not isinstance(location, dict):
            continue
        for name in ("landing_page_url", "pdf_url"):
            value = location.get(name)
            if not isinstance(value, str):
                continue
            try:
                values.add(normalize_paper_id(value, kind="arxiv"))
            except ValueError:
                continue
    return sorted(values)


def capture_identity(
    lock: IdentityCaptureLock,
    runtime: IdentifierCaptureRuntime,
) -> IdentityCaptureResult:
    inventory, output_root = _authorize_capture(lock, runtime)
    del inventory
    capture_lock_sha256 = _lock_hash(lock)
    snapshot_root = output_root / "snapshots"
    derived_lock_path = output_root / "derived-doi.lock.json"
    evidence_path = output_root / "identity-evidence.json"
    store = DependencyCaptureStore(snapshot_root, clock=runtime.clock)
    ledger = SQLiteBudgetLedger(runtime.ledger_path)
    counts = _AttemptCounts()
    semantic_scholar_batch_count = 0
    evidence_refs: list[IdentityEvidenceRef] = []
    staged_identities: list[DependencyRequestIdentity] = []
    try:
        arxiv_ids = list(lock.semantic_scholar_arxiv_ids)
        arxiv_identity = _s2_identity(arxiv_ids, stage="arxiv")
        arxiv_response, raw_arxiv_records = _request_json(
            provider="semantic_scholar",
            stage="arxiv",
            method="POST",
            url=f"{lock.semantic_scholar_base_url}{lock.semantic_scholar_endpoint}",
            headers={
                "x-api-key": runtime.credential_values[
                    lock.semantic_scholar_api_key_env
                ].get_secret_value()
            },
            query_params={"fields": S2_FIELDS},
            json_body={"ids": arxiv_ids},
            runtime=runtime,
            lock=lock,
            ledger=ledger,
            counts=counts,
            validator=lambda payload: _validate_s2_batch(payload, arxiv_ids),
        )
        arxiv_ref = store.stage_success(
            arxiv_identity,
            response_bytes=arxiv_response.content,
            safe_headers=_safe_headers(arxiv_response.headers),
            captured_at=runtime.clock(),
        )
        staged_identities.append(arxiv_identity)
        semantic_scholar_batch_count += 1
        arxiv_records = cast(list[dict[str, object] | None], raw_arxiv_records)
        discovered = _discovered_dois(arxiv_records, arxiv_ids)
        derived_doi_ids = sorted(discovered)
        derived_lock = DerivedDoiLock(
            schema_version="identifier-identity-derived-doi-lock-v1",
            parent_lock_sha256=capture_lock_sha256,
            arxiv_batch_snapshot_sha256=arxiv_ref.response_sha256,
            ids=derived_doi_ids,
        )
        _write_exclusive_atomic(derived_lock_path, _model_bytes(derived_lock))
        proposed_ids = sorted({*derived_doi_ids, *runtime.manual_doi_ids})
        if proposed_ids != derived_doi_ids:
            raise ValueError("derived DOI lock mismatch")

        doi_ref = None
        doi_index_by_id: dict[str, int] = {}
        if derived_doi_ids:
            runtime.sleeper(1.0)
            doi_identity = _s2_identity(derived_doi_ids, stage="doi")
            doi_response, _ = _request_json(
                provider="semantic_scholar",
                stage="doi",
                method="POST",
                url=f"{lock.semantic_scholar_base_url}{lock.semantic_scholar_endpoint}",
                headers={
                    "x-api-key": runtime.credential_values[
                        lock.semantic_scholar_api_key_env
                    ].get_secret_value()
                },
                query_params={"fields": S2_FIELDS},
                json_body={"ids": derived_doi_ids},
                runtime=runtime,
                lock=lock,
                ledger=ledger,
                counts=counts,
                validator=lambda payload: _validate_s2_batch(payload, derived_doi_ids),
            )
            doi_ref = store.stage_success(
                doi_identity,
                response_bytes=doi_response.content,
                safe_headers=_safe_headers(doi_response.headers),
                captured_at=runtime.clock(),
            )
            staged_identities.append(doi_identity)
            semantic_scholar_batch_count += 1
            doi_index_by_id = {
                identifier: index for index, identifier in enumerate(derived_doi_ids)
            }

        for provider_doi in derived_doi_ids:
            arxiv_id, arxiv_index, alias = discovered[provider_doi]
            evidence_refs.append(
                IdentityEvidenceRef(
                    arxiv_id=arxiv_id,
                    alias=alias,
                    semantic_scholar_arxiv_entry_id=arxiv_ref.entry_id,
                    semantic_scholar_arxiv_item_index=arxiv_index,
                    semantic_scholar_doi_entry_id=(
                        doi_ref.entry_id if doi_ref is not None else None
                    ),
                    semantic_scholar_doi_item_index=doi_index_by_id.get(provider_doi),
                )
            )

        for openalex_id in lock.openalex_exact_ids:
            openalex_identity = _openalex_identity(openalex_id)
            endpoint = lock.openalex_endpoint_template
            openalex_response, raw_payload = _request_json(
                provider="openalex",
                stage=openalex_id,
                method="GET",
                url=f"{lock.openalex_base_url}{endpoint}",
                headers={},
                query_params={
                    "api_key": runtime.credential_values[
                        lock.openalex_api_key_env
                    ].get_secret_value(),
                    "filter": f"openalex:{openalex_id}",
                    "per_page": "1",
                },
                json_body=None,
                runtime=runtime,
                lock=lock,
                ledger=ledger,
                counts=counts,
                validator=partial(_validate_openalex, expected_id=openalex_id),
            )
            openalex_ref = store.stage_success(
                openalex_identity,
                response_bytes=openalex_response.content,
                safe_headers=_safe_headers(openalex_response.headers),
                captured_at=runtime.clock(),
            )
            staged_identities.append(openalex_identity)
            payload = cast(dict[str, object], raw_payload)
            for arxiv_id in _openalex_arxiv_ids(payload):
                evidence_refs.append(
                    IdentityEvidenceRef(
                        arxiv_id=arxiv_id,
                        alias=f"openalex:{openalex_id}",
                        openalex_entry_ids=[openalex_ref.entry_id],
                    )
                )

        manifest = store.seal()
        reader = DependencySnapshotReader(
            store.manifest_path,
            snapshot_manifest_sha256=store.manifest_sha256,
            snapshot_set_id=manifest.snapshot_set_id,
        )
        try:
            for identity in staged_identities:
                reader.read(identity)
        except (KeyError, OSError, ValueError):
            raise ValueError("identity snapshot is invalid") from None
        result = IdentityCaptureResult(
            schema_version="identifier-identity-evidence-v1",
            scope="dev",
            capture_lock_sha256=capture_lock_sha256,
            derived_doi_lock=derived_lock,
            semantic_scholar_batch_count=semantic_scholar_batch_count,
            semantic_scholar_http_attempt_count=counts.semantic_scholar,
            openalex_request_count=counts.openalex,
            snapshot_manifest_sha256=store.manifest_sha256,
            evidence_refs=sorted(
                evidence_refs,
                key=lambda ref: (ref.arxiv_id, ref.alias),
            ),
        )
        _write_exclusive_atomic(evidence_path, _model_bytes(result))
        return result
    finally:
        ledger.close()


def _load_lock(path: Path, *, expected_lock_sha256: str) -> IdentityCaptureLock:
    try:
        content = path.read_bytes()
        if _sha256(content) != expected_lock_sha256:
            raise ValueError
        lock = IdentityCaptureLock.model_validate_json(content)
        if content != _model_bytes(lock) or _lock_hash(lock) != expected_lock_sha256:
            raise ValueError
        return lock
    except (OSError, ValidationError):
        raise ValueError("identity capture authorization mismatch") from None
    except ValueError:
        raise ValueError("identity capture authorization mismatch") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--gold", type=Path, required=True)
    inventory.add_argument("--candidate-map", type=Path, required=True)
    inventory.add_argument("--out", type=Path, required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--inventory", type=Path, required=True)
    preflight.add_argument("--ledger", type=Path, required=True)
    preflight.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    preflight.add_argument("--out-lock", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--lock", type=Path, required=True)
    run.add_argument("--expected-lock-sha256", required=True)
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--snapshot-root", type=Path)
    run.add_argument("--out-private", type=Path)
    run.add_argument("--allow-network", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inventory":
        build_identifier_inventory(args.gold, args.candidate_map, args.out)
        return 0
    if args.command == "preflight":
        build_capture_lock(
            inventory_path=args.inventory,
            ledger_path=args.ledger,
            output_root=args.output_root,
            out_lock=args.out_lock,
        )
        return 0
    lock = _load_lock(
        args.lock,
        expected_lock_sha256=args.expected_lock_sha256,
    )
    project_root = Path.cwd()
    output_root = project_root / lock.output_root
    expected_snapshot_root = output_root / "snapshots"
    expected_private = output_root / "identity-evidence.json"
    if (
        (args.snapshot_root is not None and args.snapshot_root.resolve() != expected_snapshot_root.resolve())
        or (args.out_private is not None and args.out_private.resolve() != expected_private.resolve())
    ):
        raise ValueError("identity capture authorization mismatch")
    credential_values = {
        name: value
        for name in (lock.semantic_scholar_api_key_env, lock.openalex_api_key_env)
        if (value := os.environ.get(name)) is not None
    }
    runtime = IdentifierCaptureRuntime(
        project_root=project_root,
        inventory_path=output_root / "identifier-inventory.json",
        ledger_path=args.ledger,
        expected_lock_sha256=args.expected_lock_sha256,
        output_root=lock.output_root,
        semantic_scholar_base_url=lock.semantic_scholar_base_url,
        semantic_scholar_endpoint=lock.semantic_scholar_endpoint,
        semantic_scholar_api_key_env=lock.semantic_scholar_api_key_env,
        openalex_base_url=lock.openalex_base_url,
        openalex_endpoint_template=lock.openalex_endpoint_template,
        openalex_api_key_env=lock.openalex_api_key_env,
        credential_values=credential_values,
        transport=HttpxCaptureTransport(),
        allow_network=args.allow_network,
    )
    capture_identity(lock, runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
