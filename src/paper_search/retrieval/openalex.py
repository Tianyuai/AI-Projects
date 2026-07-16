"""Budget-bounded OpenAlex Works search with raw-response replay."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import httpx

from paper_search.domain.models import (
    BudgetReservation,
    ErrorDetail,
    Paper,
    ProviderResult,
    UsageActual,
)
from paper_search.processing.normalize import normalize_openalex_work
from paper_search.storage.cache import (
    SAFE_RESPONSE_HEADERS,
    SQLiteResponseCache,
    make_cache_key,
)


Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]
Jitter = Callable[[], float]
QueryValue = str | int | float | bool | None
OPENALEX_SELECT_FIELDS = (
    "id,doi,title,display_name,abstract_inverted_index,authorships,publication_year,"
    "primary_location,cited_by_count,is_retracted"
)
_ENDPOINT = "/works"
_CACHE_TTL = timedelta(days=7)
_COOLDOWN = timedelta(seconds=60)
_MAX_ATTEMPTS = 3


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _aggregate_hash(hashes: list[str]) -> str:
    if len(hashes) == 1:
        return hashes[0]
    encoded = json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    return _sha256(encoded)


def _validate_year(name: str, value: object) -> int:
    maximum = date.today().year + 1
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 1900 <= value <= maximum:
        raise ValueError(f"{name} must be between 1900 and {maximum}")
    return value


def _filter_expression(filters: Mapping[str, object]) -> str | None:
    unknown = set(filters).difference({"year_from", "year_to"})
    if unknown:
        raise ValueError(f"unknown OpenAlex filters: {sorted(unknown)}")
    year_from = _validate_year("year_from", filters["year_from"]) if "year_from" in filters else None
    year_to = _validate_year("year_to", filters["year_to"]) if "year_to" in filters else None
    if year_from is not None and year_to is not None and year_from > year_to:
        raise ValueError("year_from must not exceed year_to")
    parts: list[str] = []
    if year_from is not None:
        parts.append(f"from_publication_date:{year_from}-01-01")
    if year_to is not None:
        parts.append(f"to_publication_date:{year_to}-12-31")
    return ",".join(parts) or None


def _parse_page(raw_response: bytes) -> tuple[list[object], str | None]:
    try:
        payload = json.loads(raw_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("OpenAlex response is not valid JSON") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("OpenAlex response must contain a results list")
    meta = payload.get("meta")
    if meta is None:
        next_cursor = None
    elif isinstance(meta, dict):
        next_cursor = meta.get("next_cursor")
        if next_cursor is not None and not isinstance(next_cursor, str):
            raise ValueError("OpenAlex next_cursor must be a string or null")
    else:
        raise ValueError("OpenAlex response meta must be an object or null")
    return payload["results"], next_cursor


def _request_id(response: httpx.Response) -> str | None:
    value = response.headers.get("x-request-id")
    return value if value and value.strip() else None


def _provider_error(
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
        provider="openalex",
        request_id=request_id,
    )


@dataclass(frozen=True)
class _PageFetch:
    raw_response: bytes | None
    response_hash: str | None
    from_cache: bool
    calls: int
    errors: list[ErrorDetail]


class OpenAlexProvider:
    """Search OpenAlex with deterministic caching and normalized output."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        cache: SQLiteResponseCache,
        api_key: str,
        clock: Clock = _utc_now,
        sleep: Sleep = asyncio.sleep,
        jitter: Jitter = random.random,
        cache_version: str = "v1",
    ) -> None:
        if not api_key:
            raise ValueError("OpenAlex API key must not be empty")
        self._client = client
        self._cache = cache
        self._api_key = api_key
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter
        self._cache_version = cache_version

    async def _fetch_page(
        self,
        *,
        params: dict[str, QueryValue],
        key: str,
        remaining_calls: int,
    ) -> _PageFetch:
        cached = self._cache.get_response(key)
        if cached is not None:
            try:
                _parse_page(cached.raw_response)
            except ValueError:
                return _PageFetch(
                    None,
                    None,
                    True,
                    0,
                    [_provider_error("invalid_response", "Cached OpenAlex response is invalid", retryable=False)],
                )
            return _PageFetch(cached.raw_response, cached.response_hash, True, 0, [])

        if self._cache.get_cooldown(key) is not None:
            return _PageFetch(
                None,
                None,
                True,
                0,
                [_provider_error("rate_limited", "OpenAlex request is in cooldown", retryable=True)],
            )
        if remaining_calls <= 0:
            return _PageFetch(
                None,
                None,
                False,
                0,
                [_provider_error("budget_exhausted", "OpenAlex search call reservation exhausted", retryable=False)],
            )

        attempts_allowed = min(_MAX_ATTEMPTS, remaining_calls)
        last_error: ErrorDetail | None = None
        for attempt in range(attempts_allowed):
            try:
                response = await self._client.get(_ENDPOINT, params=params)
            except httpx.TimeoutException:
                last_error = _provider_error("timeout", "OpenAlex request timed out", retryable=True)
            except httpx.RequestError:
                last_error = _provider_error(
                    "network_error",
                    "OpenAlex network request failed",
                    retryable=True,
                )
            else:
                request_id = _request_id(response)
                if response.status_code == 200:
                    raw_response = response.content
                    try:
                        _parse_page(raw_response)
                    except ValueError:
                        return _PageFetch(
                            None,
                            None,
                            False,
                            attempt + 1,
                            [_provider_error("invalid_response", "OpenAlex returned an invalid response", retryable=False, request_id=request_id)],
                        )
                    safe_headers = {
                        name.casefold(): value
                        for name, value in response.headers.items()
                        if name.casefold() in SAFE_RESPONSE_HEADERS
                    }
                    self._cache.put_response(
                        key=key,
                        provider="openalex",
                        endpoint=_ENDPOINT,
                        cache_version=self._cache_version,
                        params=params,
                        raw_response=raw_response,
                        requested_at=self._clock(),
                        ttl=_CACHE_TTL,
                        safe_headers=safe_headers,
                    )
                    return _PageFetch(raw_response, _sha256(raw_response), False, attempt + 1, [])
                if response.status_code == 429:
                    self._cache.set_cooldown(key, self._clock() + _COOLDOWN)
                    last_error = _provider_error(
                        "rate_limited",
                        "OpenAlex rate limit exceeded",
                        retryable=True,
                        request_id=request_id,
                    )
                elif 500 <= response.status_code <= 599:
                    last_error = _provider_error(
                        "server_error",
                        "OpenAlex server error",
                        retryable=True,
                        request_id=request_id,
                    )
                else:
                    if response.status_code == 400:
                        code = "invalid_request"
                    elif response.status_code in {401, 403}:
                        code = "authentication_error"
                    else:
                        code = "client_error"
                    return _PageFetch(
                        None,
                        None,
                        False,
                        attempt + 1,
                        [_provider_error(code, "OpenAlex rejected the request", retryable=False, request_id=request_id)],
                    )

            if attempt + 1 < attempts_allowed:
                delay = min(8.0, float(2**attempt)) + self._jitter()
                await self._sleep(delay)

        if last_error is None:
            raise RuntimeError("OpenAlex attempt loop ended without a result")
        return _PageFetch(None, None, False, attempts_allowed, [last_error])

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        """Return normalized OpenAlex Works without exceeding the reservation."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 300:
            raise ValueError("limit must be an integer between 1 and 300")
        filter_value = _filter_expression(filters)

        started = time.perf_counter()
        requested_at = self._clock()
        remaining_calls = reservation.reserved.search_api_calls
        actual_calls = 0
        cursor = "*"
        papers: list[Paper] = []
        errors: list[ErrorDetail] = []
        cache_keys: list[str] = []
        response_hashes: list[str] = []
        successful_pages = 0
        cached_pages = 0
        seen_cursors: set[str] = set()

        while len(papers) < limit:
            if cursor in seen_cursors:
                errors.append(
                    _provider_error(
                        "pagination_cycle",
                        "OpenAlex returned a repeated page cursor",
                        retryable=False,
                    )
                )
                break
            seen_cursors.add(cursor)
            remaining = limit - len(papers)
            params: dict[str, QueryValue] = {
                "api_key": self._api_key,
                "cursor": cursor,
                "per_page": min(50, remaining),
                "search": normalized_query,
                "select": OPENALEX_SELECT_FIELDS,
            }
            if filter_value is not None:
                params["filter"] = filter_value
            key = make_cache_key("openalex", _ENDPOINT, params, self._cache_version)
            fetched = await self._fetch_page(
                params=params,
                key=key,
                remaining_calls=remaining_calls - actual_calls,
            )
            actual_calls += fetched.calls
            errors.extend(fetched.errors)
            if fetched.raw_response is None or fetched.response_hash is None:
                break
            raw_response = fetched.raw_response
            response_hash = fetched.response_hash
            if fetched.from_cache:
                cached_pages += 1

            raw_works, next_cursor = _parse_page(raw_response)
            cache_keys.append(key)
            response_hashes.append(response_hash)
            successful_pages += 1
            for raw_work in raw_works:
                if not isinstance(raw_work, Mapping):
                    errors.append(
                        ErrorDetail(
                            code="invalid_work",
                            message="OpenAlex work must be an object",
                            retryable=False,
                            provider="openalex",
                        )
                    )
                    continue
                try:
                    papers.append(normalize_openalex_work(raw_work))
                except ValueError as error:
                    errors.append(
                        ErrorDetail(
                            code="invalid_work",
                            message=str(error),
                            retryable=False,
                            provider="openalex",
                        )
                    )
                if len(papers) >= limit:
                    break
            if not raw_works or next_cursor is None or len(papers) >= limit:
                break
            cursor = next_cursor

        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        return ProviderResult[list[Paper]](
            data=papers,
            usage=UsageActual(search_api_calls=actual_calls, elapsed_ms=elapsed_ms),
            provenance={
                "provider": "openalex",
                "endpoint": _ENDPOINT,
                "model_id": "openalex-api",
                "requested_at": requested_at.isoformat(),
                "response_hash": _aggregate_hash(response_hashes),
                "cache_keys": json.dumps(cache_keys, separators=(",", ":")),
            },
            cache_hit=successful_pages > 0 and cached_pages == successful_pages,
            latency_ms=elapsed_ms,
            errors=errors,
        )
