"""Priced live capture and structurally offline replay for search providers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, Protocol, TypeVar
from urllib.parse import quote

import httpx

from paper_search.application.contracts import SnapshotRef
from paper_search.control.pricing import ActualCostPricer
from paper_search.domain.models import (
    BudgetReservation,
    CitationExpansion,
    ErrorDetail,
    Paper,
    ProviderName,
    ProviderPaperId,
    ProviderResult,
    UsageActual,
)
from paper_search.errors import ProtectedExecutionError
from paper_search.retrieval.openalex import (
    OPENALEX_SELECT_FIELDS,
    _filter_expression,
    _normalize_search_query,
    decode_openalex_page,
)
from paper_search.retrieval.semantic_scholar import (
    _FIELDS,
    decode_semantic_scholar_batch,
    decode_semantic_scholar_expansion,
    decode_semantic_scholar_search,
)
from paper_search.storage.cache import SAFE_RESPONSE_HEADERS
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencyRequestIdentity,
    DependencySnapshotReader,
)


Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]
Jitter = Callable[[], float]
Method = Literal["GET", "POST"]
Operation = Literal["search", "batch", "references", "citations"]
QueryValue = str | int | float | bool | None

_BASE_URLS = {
    "openalex": "https://api.openalex.org",
    "semantic_scholar": "https://api.semanticscholar.org/graph/v1",
}
_ADAPTERS = {
    "openalex": "openalex-works-v1",
    "semantic_scholar": "semantic-graph-v1",
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _aggregate_hash(hashes: list[str]) -> str:
    if not hashes:
        return _sha256(b"")
    if len(hashes) == 1:
        return hashes[0]
    return _sha256(json.dumps(hashes, separators=(",", ":")).encode("utf-8"))


def _snapshot_provenance(refs: list[SnapshotRef]) -> str:
    return json.dumps(
        [ref.model_dump(mode="json") for ref in refs],
        ensure_ascii=False,
        separators=(",", ":"),
    )


class ProviderSettlementController(Protocol):
    def mark_dispatched(self, reservation: BudgetReservation) -> None: ...

    def settle(self, reservation: BudgetReservation, actual: UsageActual) -> None: ...

    def fail_closed(
        self,
        reservation: BudgetReservation,
        actual: UsageActual,
    ) -> None: ...


class ProviderAdapterError(ProtectedExecutionError):
    """A fixed, credential-safe terminal adapter failure."""


@dataclass(frozen=True)
class _RequestOutcome:
    content: bytes | None
    calls: int
    error: ErrorDetail | None
    captured_at: datetime
    safe_headers: dict[str, str]


ResultT = TypeVar("ResultT")


def _aggregate_attempts(attempts: list[UsageActual]) -> UsageActual:
    costs = [attempt.cost_cny for attempt in attempts]
    cost = (
        None
        if any(value is None for value in costs)
        else sum((value for value in costs if value is not None), Decimal("0"))
    )
    return UsageActual(
        search_api_calls=sum(item.search_api_calls for item in attempts),
        llm_calls=sum(item.llm_calls for item in attempts),
        input_tokens=sum(item.input_tokens for item in attempts),
        output_tokens=sum(item.output_tokens for item in attempts),
        cost_cny=cost,
        elapsed_ms=sum(item.elapsed_ms for item in attempts),
    )


class _LiveOperation:
    """Track one reservation from first dispatch to exactly one terminal action."""

    def __init__(self, provider: LiveCaptureSearchProvider, reservation: BudgetReservation) -> None:
        self.provider = provider
        self.reservation = reservation
        duration = max(0.0, (reservation.expires_at - provider._clock()).total_seconds())
        self.deadline = asyncio.get_running_loop().time() + duration
        self.attempts: list[UsageActual] = []
        self._attempt_started: float | None = None
        self.terminal = False

    @property
    def dispatched(self) -> bool:
        return bool(self.attempts) or self._attempt_started is not None

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - asyncio.get_running_loop().time())

    def start_attempt(self) -> None:
        if self._attempt_started is not None:
            raise RuntimeError("provider attempt is already in flight")
        self.provider._controller.mark_dispatched(self.reservation)
        self._attempt_started = time.perf_counter()

    def finish_attempt(self) -> UsageActual:
        if self._attempt_started is None:
            raise RuntimeError("provider attempt is not in flight")
        measured = UsageActual(
            search_api_calls=1,
            elapsed_ms=max(
                0,
                round((time.perf_counter() - self._attempt_started) * 1000),
            ),
        )
        self._attempt_started = None
        self.attempts.append(measured)
        valued = self.provider._pricer.value_actual(
            dependency=self.provider._dependency,
            model_or_adapter=self.provider._adapter,
            usage=measured,
        )
        self.attempts[-1] = valued
        return valued

    def retain_in_flight_unknown(self) -> None:
        if self._attempt_started is None:
            return
        measured = UsageActual(
            search_api_calls=1,
            elapsed_ms=max(
                0,
                round((time.perf_counter() - self._attempt_started) * 1000),
            ),
        )
        self._attempt_started = None
        self.attempts.append(measured)

    def settle(self) -> UsageActual:
        actual = (
            _aggregate_attempts(self.attempts)
            if self.attempts
            else UsageActual(cost_cny=Decimal("0"))
        )
        try:
            self.provider._controller.settle(self.reservation, actual)
        except Exception:
            self.fail_closed()
            raise
        self.terminal = True
        return actual

    def fail_closed(self) -> None:
        if self.terminal or not self.dispatched:
            return
        self.retain_in_flight_unknown()
        terminal_attempts = list(self.attempts)
        batch = getattr(self.provider._controller, "fail_closed_attempts", None)
        try:
            if callable(batch):
                batch(self.reservation, terminal_attempts)
            else:
                self.provider._controller.fail_closed(
                    self.reservation,
                    _aggregate_attempts(terminal_attempts),
                )
        finally:
            self.terminal = True


def _error(
    dependency: ProviderName,
    code: str,
    message: str,
    *,
    retryable: bool,
    request_id: str | None = None,
) -> ErrorDetail:
    return ErrorDetail(
        code=code,
        message=message,
        retryable=retryable,
        provider=dependency,
        request_id=request_id,
    )


def _status_error(
    dependency: ProviderName,
    status_code: int,
    request_id: str | None,
) -> ErrorDetail:
    if status_code == 429:
        return _error(
            dependency,
            "rate_limited",
            f"{dependency} request was rate limited",
            retryable=True,
            request_id=request_id,
        )
    if 500 <= status_code <= 599:
        return _error(
            dependency,
            "server_error",
            f"{dependency} provider failed",
            retryable=True,
            request_id=request_id,
        )
    if status_code in {401, 403}:
        code, message = "authentication_error", f"{dependency} authentication failed"
    elif status_code == 400:
        code, message = "invalid_request", f"{dependency} request was rejected"
    else:
        code, message = "provider_error", f"{dependency} provider rejected the request"
    return _error(
        dependency,
        code,
        message,
        retryable=False,
        request_id=request_id,
    )


def _identity(
    *,
    dependency: ProviderName,
    operation: Operation,
    method: Method,
    endpoint: str,
    adapter: str,
    canonical_request: Mapping[str, object],
) -> DependencyRequestIdentity:
    return DependencyRequestIdentity.from_canonical_request(
        dependency=dependency,
        operation=operation,
        method=method,
        endpoint=endpoint,
        model_or_adapter=adapter,
        canonical_request=canonical_request,
    )


class LiveCaptureSearchProvider:
    """Issue bounded provider requests, price them, and capture successful bytes."""

    def __init__(
        self,
        *,
        dependency: ProviderName,
        client: httpx.AsyncClient,
        capture_store: DependencyCaptureStore,
        pricer: ActualCostPricer,
        controller: ProviderSettlementController,
        api_key: str | None = None,
        adapter_version: str | None = None,
        clock: Clock = _utc_now,
        sleep: Sleep | None = None,
        jitter: Jitter = random.random,
    ) -> None:
        self._dependency = dependency
        self._client = client
        self._capture_store = capture_store
        self._pricer = pricer
        self._controller = controller
        self._api_key = api_key
        self._adapter = adapter_version or _ADAPTERS[dependency]
        self._clock = clock
        self._sleep = sleep or asyncio.sleep
        self._jitter = jitter

    async def _run_live(
        self,
        reservation: BudgetReservation,
        call: Callable[[_LiveOperation], Awaitable[ResultT]],
    ) -> ResultT:
        operation = _LiveOperation(self, reservation)
        try:
            return await call(operation)
        except asyncio.CancelledError as cancellation:
            operation.fail_closed()
            raise cancellation from None
        except Exception:
            if not operation.dispatched:
                raise
            operation.fail_closed()
            raise ProviderAdapterError("provider live capture failed") from None

    async def _request(
        self,
        *,
        method: Method,
        endpoint: str,
        params: Mapping[str, QueryValue],
        operation: _LiveOperation,
        body: Mapping[str, object] | None = None,
        remaining_calls: int,
    ) -> _RequestOutcome:
        captured_at = self._clock()
        attempts_allowed = min(3, max(0, remaining_calls))
        if attempts_allowed == 0:
            return _RequestOutcome(
                content=None,
                calls=0,
                error=_error(
                    self._dependency,
                    "budget_exhausted",
                    f"{self._dependency} reservation exhausted",
                    retryable=False,
                ),
                captured_at=captured_at,
                safe_headers={},
            )
        headers = {"Accept": "application/json"}
        if self._api_key:
            if self._dependency == "openalex":
                request_params = dict(params)
                request_params["api_key"] = self._api_key
            else:
                request_params = dict(params)
                headers["x-api-key"] = self._api_key
        else:
            request_params = dict(params)

        last_error: ErrorDetail | None = None
        attempts_made = 0
        for retry_index in range(attempts_allowed):
            remaining_seconds = operation.remaining_seconds()
            if remaining_seconds <= 0:
                last_error = _error(
                    self._dependency,
                    "budget_exhausted",
                    f"{self._dependency} request deadline expired",
                    retryable=False,
                )
                break
            operation.start_attempt()
            attempts_made += 1
            try:
                async with asyncio.timeout(remaining_seconds):
                    response = await self._client.request(
                        method,
                        f"{_BASE_URLS[self._dependency]}{endpoint}",
                        params=request_params,
                        json=body,
                        headers=headers,
                        follow_redirects=False,
                        timeout=remaining_seconds,
                    )
            except (TimeoutError, httpx.TimeoutException):
                operation.finish_attempt()
                last_error = _error(
                    self._dependency,
                    "timeout",
                    f"{self._dependency} request timed out",
                    retryable=True,
                )
            except httpx.RequestError:
                operation.finish_attempt()
                last_error = _error(
                    self._dependency,
                    "network_error",
                    f"{self._dependency} network request failed",
                    retryable=False,
                )
            else:
                operation.finish_attempt()
                request_id = response.headers.get("x-request-id") or None
                if operation.remaining_seconds() <= 0:
                    last_error = _error(
                        self._dependency,
                        "timeout",
                        f"{self._dependency} request timed out",
                        retryable=True,
                    )
                elif response.status_code == 200:
                    safe_headers = {
                        name.casefold(): value
                        for name, value in response.headers.items()
                        if name.casefold() in SAFE_RESPONSE_HEADERS
                    }
                    return _RequestOutcome(
                        content=response.content,
                        calls=attempts_made,
                        error=None,
                        captured_at=captured_at,
                        safe_headers=safe_headers,
                    )
                else:
                    last_error = _status_error(
                        self._dependency,
                        response.status_code,
                        request_id,
                    )
            if (
                last_error is None
                or not last_error.retryable
                or retry_index + 1 >= attempts_allowed
            ):
                break
            delay_seconds = min(8, 2**retry_index) + self._jitter()
            remaining_seconds = operation.remaining_seconds()
            if delay_seconds >= remaining_seconds:
                last_error = _error(
                    self._dependency,
                    "timeout",
                    f"{self._dependency} request timed out",
                    retryable=True,
                )
                break
            async with asyncio.timeout(remaining_seconds):
                await self._sleep(delay_seconds)

        if last_error is None:
            raise RuntimeError("provider request ended without a terminal outcome")
        return _RequestOutcome(
            content=None,
            calls=attempts_made,
            error=last_error,
            captured_at=captured_at,
            safe_headers={},
        )

    @staticmethod
    def _settle(operation: _LiveOperation) -> UsageActual:
        return operation.settle()

    def _capture(
        self,
        identity: DependencyRequestIdentity,
        outcome: _RequestOutcome,
    ) -> SnapshotRef:
        if outcome.content is None:
            raise RuntimeError("cannot capture an empty provider response")
        return self._capture_store.stage_success(
            identity,
            response_bytes=outcome.content,
            safe_headers=outcome.safe_headers,
            captured_at=outcome.captured_at,
        )

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        if self._dependency == "openalex":
            return await self._run_live(
                reservation,
                lambda operation: self._openalex_search(
                    query, filters, limit, operation
                ),
            )
        return await self._run_live(
            reservation,
            lambda operation: self._semantic_search(
                query, filters, limit, operation
            ),
        )

    async def _openalex_search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        operation: _LiveOperation,
    ) -> ProviderResult[list[Paper]]:
        normalized_query = _normalize_search_query(query)
        if not normalized_query:
            raise ValueError("query must not be empty")
        if type(limit) is not int or not 1 <= limit <= 300:
            raise ValueError("limit must be an integer between 1 and 300")
        filter_value = _filter_expression(filters)
        started = time.perf_counter()
        requested_at = self._clock()
        cursor = "*"
        seen_cursors: set[str] = set()
        raw_seen = 0
        calls = 0
        papers: list[Paper] = []
        errors: list[ErrorDetail] = []
        refs: list[SnapshotRef] = []
        hashes: list[str] = []
        while raw_seen < limit:
            if cursor in seen_cursors:
                errors.append(
                    _error(
                        "openalex",
                        "pagination_cycle",
                        "OpenAlex returned a repeated page cursor",
                        retryable=False,
                    )
                )
                break
            seen_cursors.add(cursor)
            remaining = limit - raw_seen
            canonical: dict[str, object] = {
                "query": normalized_query,
                "filters": filters,
                "limit": limit,
                "cursor": cursor,
                "per_page": min(50, remaining),
                "select": OPENALEX_SELECT_FIELDS,
            }
            if filter_value is not None:
                canonical["filter"] = filter_value
            identity = _identity(
                dependency="openalex",
                operation="search",
                method="GET",
                endpoint="/works",
                adapter=self._adapter,
                canonical_request=canonical,
            )
            params: dict[str, QueryValue] = {
                "search": normalized_query,
                "cursor": cursor,
                "per_page": min(50, remaining),
                "select": OPENALEX_SELECT_FIELDS,
            }
            if filter_value is not None:
                params["filter"] = filter_value
            outcome = await self._request(
                method="GET",
                endpoint="/works",
                params=params,
                operation=operation,
                remaining_calls=operation.reservation.reserved.search_api_calls - calls,
            )
            calls += outcome.calls
            if outcome.error is not None or outcome.content is None:
                if outcome.error is not None:
                    errors.append(outcome.error)
                break
            refs.append(self._capture(identity, outcome))
            hashes.append(_sha256(outcome.content))
            if operation.remaining_seconds() <= 0:
                raise TimeoutError("provider deadline expired during capture")
            try:
                decoded = decode_openalex_page(outcome.content, limit=remaining)
            except ValueError:
                errors.append(
                    _error(
                        "openalex",
                        "invalid_response",
                        "OpenAlex returned an invalid response",
                        retryable=False,
                    )
                )
                break
            if operation.remaining_seconds() <= 0:
                raise TimeoutError("provider deadline expired during decode")
            papers.extend(decoded.papers)
            errors.extend(decoded.errors)
            raw_seen += decoded.raw_count
            if decoded.raw_count == 0 or decoded.next_cursor is None:
                break
            cursor = decoded.next_cursor
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        usage = self._settle(operation)
        return ProviderResult[list[Paper]](
            data=papers,
            usage=usage,
            provenance={
                "provider": "openalex",
                "endpoint": "/works",
                "model_id": self._adapter,
                "requested_at": requested_at.isoformat(),
                "response_hash": _aggregate_hash(hashes),
                "snapshot_refs": _snapshot_provenance(refs),
            },
            cache_hit=False,
            latency_ms=elapsed_ms,
            errors=errors,
        )

    async def _semantic_search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        operation: _LiveOperation,
    ) -> ProviderResult[list[Paper]]:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("query must not be empty")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        unknown = set(filters).difference({"year_from", "year_to"})
        if unknown:
            raise ValueError(f"unknown Semantic Scholar filters: {sorted(unknown)}")
        params: dict[str, QueryValue] = {
            "query": normalized_query,
            "limit": limit,
            "fields": _FIELDS,
        }
        canonical: dict[str, object] = {
            "query": normalized_query,
            "filters": filters,
            "limit": limit,
            "fields": _FIELDS,
        }
        if filters:
            year = f"{filters.get('year_from', '')}-{filters.get('year_to', '')}"
            params["year"] = year
            canonical["year"] = year
        return await self._semantic_papers_operation(
            request_operation="search",
            method="GET",
            endpoint="/paper/search",
            params=params,
            canonical=canonical,
            operation=operation,
            decode=lambda content: decode_semantic_scholar_search(
                content,
                limit=limit,
            ),
        )

    async def batch_details(
        self,
        paper_ids: Sequence[str],
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        if self._dependency != "semantic_scholar":
            raise ValueError("batch details require Semantic Scholar")
        ids = [value.strip() for value in paper_ids if value.strip()]
        if not ids:
            raise ValueError("paper_ids must not be empty")
        return await self._run_live(
            reservation,
            lambda operation: self._batch_details_operation(ids, operation),
        )

    async def _batch_details_operation(
        self,
        ids: list[str],
        operation: _LiveOperation,
    ) -> ProviderResult[list[Paper]]:
        canonical = {"fields": _FIELDS, "ids": ids}
        return await self._semantic_papers_operation(
            request_operation="batch",
            method="POST",
            endpoint="/paper/batch",
            params={"fields": _FIELDS},
            canonical=canonical,
            operation=operation,
            body={"ids": ids},
            decode=decode_semantic_scholar_batch,
        )

    async def _semantic_papers_operation(
        self,
        *,
        request_operation: Literal["search", "batch"],
        method: Method,
        endpoint: str,
        params: Mapping[str, QueryValue],
        canonical: Mapping[str, object],
        operation: _LiveOperation,
        decode: Callable[[bytes], Any],
        body: Mapping[str, object] | None = None,
    ) -> ProviderResult[list[Paper]]:
        started = time.perf_counter()
        requested_at = self._clock()
        identity = _identity(
            dependency="semantic_scholar",
            operation=request_operation,
            method=method,
            endpoint=endpoint,
            adapter=self._adapter,
            canonical_request=canonical,
        )
        outcome = await self._request(
            method=method,
            endpoint=endpoint,
            params=params,
            body=body,
            operation=operation,
            remaining_calls=operation.reservation.reserved.search_api_calls,
        )
        papers: list[Paper] = []
        errors: list[ErrorDetail] = []
        refs: list[SnapshotRef] = []
        hashes: list[str] = []
        if outcome.error is not None:
            errors.append(outcome.error)
        elif outcome.content is not None:
            refs.append(self._capture(identity, outcome))
            hashes.append(_sha256(outcome.content))
            if operation.remaining_seconds() <= 0:
                raise TimeoutError("provider deadline expired during capture")
            decoded = decode(outcome.content)
            if operation.remaining_seconds() <= 0:
                raise TimeoutError("provider deadline expired during decode")
            papers.extend(decoded.papers)
            errors.extend(decoded.errors)
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        usage = self._settle(operation)
        return ProviderResult[list[Paper]](
            data=papers,
            usage=usage,
            provenance={
                "provider": "semantic_scholar",
                "endpoint": endpoint,
                "model_id": self._adapter,
                "requested_at": requested_at.isoformat(),
                "response_hash": _aggregate_hash(hashes),
                "snapshot_refs": _snapshot_provenance(refs),
            },
            cache_hit=False,
            latency_ms=elapsed_ms,
            errors=errors,
        )

    async def _expansion(
        self,
        *,
        direction: Literal["references", "citations"],
        paper_id: ProviderPaperId,
        limit: int,
        operation: _LiveOperation,
    ) -> ProviderResult[CitationExpansion]:
        if self._dependency != "semantic_scholar":
            raise ValueError("citation expansion requires Semantic Scholar")
        if paper_id.provider != "semantic_scholar":
            raise ValueError("citation expansion requires a Semantic Scholar paper ID")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        started = time.perf_counter()
        requested_at = self._clock()
        endpoint = f"/paper/{quote(paper_id.value, safe='')}/{direction}"
        canonical = {
            "fields": _FIELDS,
            "limit": limit,
            "offset": 0,
            "paper_id": paper_id.value,
        }
        identity = _identity(
            dependency="semantic_scholar",
            operation=direction,
            method="GET",
            endpoint=endpoint,
            adapter=self._adapter,
            canonical_request=canonical,
        )
        outcome = await self._request(
            method="GET",
            endpoint=endpoint,
            params={"fields": _FIELDS, "limit": limit, "offset": 0},
            operation=operation,
            remaining_calls=operation.reservation.reserved.search_api_calls,
        )
        expansion = CitationExpansion(papers=[], raw_edges=[])
        errors: list[ErrorDetail] = []
        refs: list[SnapshotRef] = []
        hashes: list[str] = []
        if outcome.error is not None:
            errors.append(outcome.error)
        elif outcome.content is not None:
            refs.append(self._capture(identity, outcome))
            hashes.append(_sha256(outcome.content))
            if operation.remaining_seconds() <= 0:
                raise TimeoutError("provider deadline expired during capture")
            decoded = decode_semantic_scholar_expansion(
                outcome.content,
                direction=direction,
                paper_id=paper_id,
                limit=limit,
            )
            if operation.remaining_seconds() <= 0:
                raise TimeoutError("provider deadline expired during decode")
            expansion = decoded.expansion
            errors.extend(decoded.errors)
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        usage = self._settle(operation)
        return ProviderResult[CitationExpansion](
            data=expansion,
            usage=usage,
            provenance={
                "provider": "semantic_scholar",
                "endpoint": endpoint,
                "model_id": self._adapter,
                "requested_at": requested_at.isoformat(),
                "response_hash": _aggregate_hash(hashes),
                "snapshot_refs": _snapshot_provenance(refs),
            },
            cache_hit=False,
            latency_ms=elapsed_ms,
            errors=errors,
        )

    async def references(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        return await self._run_live(
            reservation,
            lambda operation: self._expansion(
                direction="references",
                paper_id=paper_id,
                limit=limit,
                operation=operation,
            ),
        )

    async def citations(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        return await self._run_live(
            reservation,
            lambda operation: self._expansion(
                direction="citations",
                paper_id=paper_id,
                limit=limit,
                operation=operation,
            ),
        )


class ReplaySearchProvider:
    """Decode sealed provider bytes without owning any network client."""

    def __init__(
        self,
        *,
        dependency: ProviderName,
        reader: DependencySnapshotReader,
        adapter_version: str | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        self._dependency = dependency
        self._reader = reader
        self._adapter = adapter_version or _ADAPTERS[dependency]
        self._clock = clock

    def _read(
        self,
        identity: DependencyRequestIdentity,
    ) -> tuple[bytes | None, SnapshotRef | None]:
        try:
            snapshot = self._reader.read(identity)
        except KeyError:
            return None, None
        return snapshot.response_bytes, snapshot.ref

    def _result(
        self,
        *,
        data: Any,
        endpoint: str,
        requested_at: datetime,
        hashes: list[str],
        refs: list[SnapshotRef],
        errors: list[ErrorDetail],
    ) -> ProviderResult[Any]:
        return ProviderResult[Any](
            data=data,
            usage=UsageActual(),
            provenance={
                "provider": self._dependency,
                "endpoint": endpoint,
                "model_id": self._adapter,
                "requested_at": requested_at.isoformat(),
                "response_hash": _aggregate_hash(hashes),
                "snapshot_refs": _snapshot_provenance(refs),
                "snapshot_set_id": self._reader.snapshot_set_id,
            },
            cache_hit=bool(refs),
            latency_ms=0,
            errors=errors,
        )

    def _miss(self) -> ErrorDetail:
        return _error(
            self._dependency,
            "snapshot_unavailable",
            f"{self._dependency} snapshot is unavailable",
            retryable=False,
        )

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        del reservation
        if self._dependency == "openalex":
            return await self._openalex_search(query, filters, limit)
        return await self._semantic_search(query, filters, limit)

    async def _openalex_search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
    ) -> ProviderResult[list[Paper]]:
        normalized_query = _normalize_search_query(query)
        if not normalized_query:
            raise ValueError("query must not be empty")
        if type(limit) is not int or not 1 <= limit <= 300:
            raise ValueError("limit must be an integer between 1 and 300")
        filter_value = _filter_expression(filters)
        requested_at = self._clock()
        cursor = "*"
        seen: set[str] = set()
        raw_seen = 0
        papers: list[Paper] = []
        errors: list[ErrorDetail] = []
        refs: list[SnapshotRef] = []
        hashes: list[str] = []
        while raw_seen < limit:
            if cursor in seen:
                errors.append(
                    _error(
                        "openalex",
                        "pagination_cycle",
                        "OpenAlex returned a repeated page cursor",
                        retryable=False,
                    )
                )
                break
            seen.add(cursor)
            remaining = limit - raw_seen
            canonical: dict[str, object] = {
                "query": normalized_query,
                "filters": filters,
                "limit": limit,
                "cursor": cursor,
                "per_page": min(50, remaining),
                "select": OPENALEX_SELECT_FIELDS,
            }
            if filter_value is not None:
                canonical["filter"] = filter_value
            identity = _identity(
                dependency="openalex",
                operation="search",
                method="GET",
                endpoint="/works",
                adapter=self._adapter,
                canonical_request=canonical,
            )
            content, ref = self._read(identity)
            if content is None or ref is None:
                errors.append(self._miss())
                break
            refs.append(ref)
            hashes.append(_sha256(content))
            try:
                decoded = decode_openalex_page(content, limit=remaining)
            except ValueError:
                errors.append(
                    _error(
                        "openalex",
                        "invalid_response",
                        "OpenAlex snapshot response is invalid",
                        retryable=False,
                    )
                )
                break
            papers.extend(decoded.papers)
            errors.extend(decoded.errors)
            raw_seen += decoded.raw_count
            if decoded.raw_count == 0 or decoded.next_cursor is None:
                break
            cursor = decoded.next_cursor
        return self._result(
            data=papers,
            endpoint="/works",
            requested_at=requested_at,
            hashes=hashes,
            refs=refs,
            errors=errors,
        )

    async def _semantic_search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
    ) -> ProviderResult[list[Paper]]:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("query must not be empty")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        unknown = set(filters).difference({"year_from", "year_to"})
        if unknown:
            raise ValueError(f"unknown Semantic Scholar filters: {sorted(unknown)}")
        canonical: dict[str, object] = {
            "query": normalized_query,
            "filters": filters,
            "limit": limit,
            "fields": _FIELDS,
        }
        if filters:
            canonical["year"] = (
                f"{filters.get('year_from', '')}-{filters.get('year_to', '')}"
            )
        identity = _identity(
            dependency="semantic_scholar",
            operation="search",
            method="GET",
            endpoint="/paper/search",
            adapter=self._adapter,
            canonical_request=canonical,
        )
        content, ref = self._read(identity)
        errors: list[ErrorDetail] = []
        papers: list[Paper] = []
        refs: list[SnapshotRef] = []
        hashes: list[str] = []
        if content is None or ref is None:
            errors.append(self._miss())
        else:
            decoded = decode_semantic_scholar_search(content, limit=limit)
            papers = decoded.papers
            errors.extend(decoded.errors)
            refs.append(ref)
            hashes.append(_sha256(content))
        return self._result(
            data=papers,
            endpoint="/paper/search",
            requested_at=self._clock(),
            hashes=hashes,
            refs=refs,
            errors=errors,
        )

    async def batch_details(
        self,
        paper_ids: Sequence[str],
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        del reservation
        if self._dependency != "semantic_scholar":
            raise ValueError("batch details require Semantic Scholar")
        ids = [value.strip() for value in paper_ids if value.strip()]
        if not ids:
            raise ValueError("paper_ids must not be empty")
        identity = _identity(
            dependency="semantic_scholar",
            operation="batch",
            method="POST",
            endpoint="/paper/batch",
            adapter=self._adapter,
            canonical_request={"fields": _FIELDS, "ids": ids},
        )
        content, ref = self._read(identity)
        errors: list[ErrorDetail] = []
        papers: list[Paper] = []
        refs: list[SnapshotRef] = []
        hashes: list[str] = []
        if content is None or ref is None:
            errors.append(self._miss())
        else:
            decoded = decode_semantic_scholar_batch(content)
            papers = decoded.papers
            errors.extend(decoded.errors)
            refs.append(ref)
            hashes.append(_sha256(content))
        return self._result(
            data=papers,
            endpoint="/paper/batch",
            requested_at=self._clock(),
            hashes=hashes,
            refs=refs,
            errors=errors,
        )

    async def _expansion(
        self,
        *,
        direction: Literal["references", "citations"],
        paper_id: ProviderPaperId,
        limit: int,
    ) -> ProviderResult[CitationExpansion]:
        if self._dependency != "semantic_scholar":
            raise ValueError("citation expansion requires Semantic Scholar")
        if paper_id.provider != "semantic_scholar":
            raise ValueError("citation expansion requires a Semantic Scholar paper ID")
        endpoint = f"/paper/{quote(paper_id.value, safe='')}/{direction}"
        identity = _identity(
            dependency="semantic_scholar",
            operation=direction,
            method="GET",
            endpoint=endpoint,
            adapter=self._adapter,
            canonical_request={
                "fields": _FIELDS,
                "limit": limit,
                "offset": 0,
                "paper_id": paper_id.value,
            },
        )
        content, ref = self._read(identity)
        errors: list[ErrorDetail] = []
        expansion = CitationExpansion(papers=[], raw_edges=[])
        refs: list[SnapshotRef] = []
        hashes: list[str] = []
        if content is None or ref is None:
            errors.append(self._miss())
        else:
            decoded = decode_semantic_scholar_expansion(
                content,
                direction=direction,
                paper_id=paper_id,
                limit=limit,
            )
            expansion = decoded.expansion
            errors.extend(decoded.errors)
            refs.append(ref)
            hashes.append(_sha256(content))
        return self._result(
            data=expansion,
            endpoint=endpoint,
            requested_at=self._clock(),
            hashes=hashes,
            refs=refs,
            errors=errors,
        )

    async def references(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        del reservation
        return await self._expansion(
            direction="references",
            paper_id=paper_id,
            limit=limit,
        )

    async def citations(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        del reservation
        return await self._expansion(
            direction="citations",
            paper_id=paper_id,
            limit=limit,
        )


__all__ = [
    "LiveCaptureSearchProvider",
    "ProviderAdapterError",
    "ReplaySearchProvider",
]
