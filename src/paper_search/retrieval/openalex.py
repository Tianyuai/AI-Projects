"""Budget-bounded OpenAlex Works search with raw-response replay."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
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
QueryValue = str | int | float | bool | None
OPENALEX_SELECT_FIELDS = (
    "id,doi,title,display_name,abstract_inverted_index,authorships,publication_year,"
    "primary_location,cited_by_count,is_retracted"
)
_ENDPOINT = "/works"
_CACHE_TTL = timedelta(days=7)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _aggregate_hash(hashes: list[str]) -> str:
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


class OpenAlexProvider:
    """Search OpenAlex with deterministic caching and normalized output."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        cache: SQLiteResponseCache,
        api_key: str,
        clock: Clock = _utc_now,
        cache_version: str = "v1",
    ) -> None:
        if not api_key:
            raise ValueError("OpenAlex API key must not be empty")
        self._client = client
        self._cache = cache
        self._api_key = api_key
        self._clock = clock
        self._cache_version = cache_version

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

        while len(papers) < limit:
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
            cached = self._cache.get_response(key)

            if cached is not None:
                raw_response = cached.raw_response
                response_hash = cached.response_hash
                cached_pages += 1
            else:
                if actual_calls >= remaining_calls:
                    errors.append(
                        ErrorDetail(
                            code="budget_exhausted",
                            message="OpenAlex search call reservation exhausted",
                            retryable=False,
                            provider="openalex",
                        )
                    )
                    break
                response = await self._client.get(_ENDPOINT, params=params)
                actual_calls += 1
                response.raise_for_status()
                raw_response = response.content
                _parse_page(raw_response)
                response_hash = _sha256(raw_response)
                safe_headers = {
                    name.casefold(): value
                    for name, value in response.headers.items()
                    if name.casefold() in SAFE_RESPONSE_HEADERS
                }
                self._cache.put_response(
                    key=key,
                    provider="openalex",
                    endpoint=_ENDPOINT,
                    params=params,
                    raw_response=raw_response,
                    requested_at=self._clock(),
                    ttl=_CACHE_TTL,
                    safe_headers=safe_headers,
                )

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
