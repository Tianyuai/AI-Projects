"""Fixture-testable Semantic Scholar Graph API adapter."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from paper_search.domain.models import (
    BudgetReservation,
    CitationEdge,
    CitationExpansion,
    ErrorDetail,
    Paper,
    ProviderPaperId,
    ProviderResult,
    UsageActual,
)
from paper_search.evaluation.dataset import normalize_paper_id
from paper_search.storage.cache import SAFE_RESPONSE_HEADERS, SQLiteResponseCache, make_cache_key


Clock = Callable[[], datetime]
QueryValue = str | int | float | bool | None
_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_SEARCH_ENDPOINT = "/paper/search"
_BATCH_ENDPOINT = "/paper/batch"
_FIELDS = (
    "paperId,title,abstract,authors,year,venue,externalIds,url,citationCount,"
    "references.paperId,citations.paperId"
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _error(code: str, message: str, *, retryable: bool) -> ErrorDetail:
    return ErrorDetail(
        code=code,
        message=message,
        retryable=retryable,
        provider="semantic_scholar",
    )


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _relation_ids(value: object) -> list[ProviderPaperId]:
    if not isinstance(value, list):
        return []
    identifiers: list[ProviderPaperId] = []
    for item in value:
        if isinstance(item, Mapping):
            identifier = _optional_string(item.get("paperId"))
            if identifier is not None:
                identifiers.append(
                    ProviderPaperId(provider="semantic_scholar", value=identifier)
                )
    return identifiers


def _normalize_record(record: Mapping[str, object]) -> Paper:
    paper_id = _optional_string(record.get("paperId"))
    title = _optional_string(record.get("title"))
    if paper_id is None or title is None:
        raise ValueError("Semantic Scholar record requires paperId and title")
    external_ids = record.get("externalIds")
    doi: str | None = None
    arxiv_id: str | None = None
    if isinstance(external_ids, Mapping):
        raw_doi = _optional_string(external_ids.get("DOI"))
        if raw_doi is not None:
            doi = normalize_paper_id(raw_doi, kind="doi").removeprefix("doi:")
        raw_arxiv = _optional_string(external_ids.get("ArXiv"))
        if raw_arxiv is not None:
            try:
                arxiv_id = normalize_paper_id(raw_arxiv, kind="arxiv").removeprefix(
                    "arxiv:"
                )
            except ValueError:
                arxiv_id = None
    authors: list[str] = []
    raw_authors = record.get("authors")
    if isinstance(raw_authors, list):
        for author in raw_authors:
            if isinstance(author, Mapping):
                name = _optional_string(author.get("name"))
                if name is not None:
                    authors.append(name)
    year = record.get("year")
    citation_count = record.get("citationCount")
    is_retracted = record.get("isRetracted")
    return Paper(
        canonical_id=f"doi:{doi}" if doi is not None else f"s2:{paper_id}",
        title=title,
        abstract=_optional_string(record.get("abstract")),
        authors=authors,
        publication_year=year if isinstance(year, int) and not isinstance(year, bool) else None,
        venue=_optional_string(record.get("venue")),
        doi=doi,
        arxiv_id=arxiv_id,
        semantic_scholar_id=paper_id,
        url=_optional_string(record.get("url"))
        or f"https://www.semanticscholar.org/paper/{paper_id}",
        citation_count=(
            citation_count
            if isinstance(citation_count, int)
            and not isinstance(citation_count, bool)
            and citation_count >= 0
            else None
        ),
        reference_ids=_relation_ids(record.get("references")),
        cited_by_ids=_relation_ids(record.get("citations")),
        is_retracted=is_retracted if isinstance(is_retracted, bool) else None,
        sources=["semantic_scholar"],
    )


@dataclass(frozen=True)
class _RawResult:
    content: bytes | None
    calls: int
    cache_hit: bool
    error: ErrorDetail | None
    response_hash: str
    requested_at: datetime


@dataclass(frozen=True)
class SemanticScholarPaperDecode:
    """Pure normalized paper response used by live and replay modes."""

    papers: list[Paper]
    errors: list[ErrorDetail]


@dataclass(frozen=True)
class SemanticScholarExpansionDecode:
    """Pure normalized citation expansion used by live and replay modes."""

    expansion: CitationExpansion
    errors: list[ErrorDetail]


def _decode_json(content: bytes) -> object:
    try:
        return json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def decode_semantic_scholar_search(
    content: bytes,
    *,
    limit: int,
) -> SemanticScholarPaperDecode:
    """Decode a Graph API search response without performing I/O."""

    payload = _decode_json(content)
    records = payload.get("data") if isinstance(payload, Mapping) else None
    errors: list[ErrorDetail] = []
    papers: list[Paper] = []
    if (
        isinstance(payload, Mapping)
        and payload.get("total") == 0
        and records is None
    ):
        records = []
    if not isinstance(records, list):
        errors.append(
            _error(
                "invalid_response",
                "Search response requires a data list",
                retryable=False,
            )
        )
    else:
        for record in records[:limit]:
            if not isinstance(record, Mapping):
                errors.append(
                    _error(
                        "invalid_record",
                        "Search record must be an object",
                        retryable=False,
                    )
                )
                continue
            try:
                papers.append(_normalize_record(record))
            except ValueError as error:
                errors.append(_error("invalid_record", str(error), retryable=False))
    return SemanticScholarPaperDecode(papers=papers, errors=errors)


def decode_semantic_scholar_batch(content: bytes) -> SemanticScholarPaperDecode:
    """Decode a Graph API batch response without performing I/O."""

    payload = _decode_json(content)
    errors: list[ErrorDetail] = []
    papers: list[Paper] = []
    if not isinstance(payload, list):
        errors.append(
            _error(
                "invalid_response",
                "Batch response requires a list",
                retryable=False,
            )
        )
    else:
        for record in payload:
            if record is None:
                errors.append(
                    _error(
                        "missing_record",
                        "Batch record was not found",
                        retryable=False,
                    )
                )
            elif isinstance(record, Mapping):
                try:
                    papers.append(_normalize_record(record))
                except ValueError as error:
                    errors.append(
                        _error("invalid_record", str(error), retryable=False)
                    )
            else:
                errors.append(
                    _error(
                        "invalid_record",
                        "Batch record must be an object",
                        retryable=False,
                    )
                )
    return SemanticScholarPaperDecode(papers=papers, errors=errors)


def decode_semantic_scholar_expansion(
    content: bytes,
    *,
    direction: str,
    paper_id: ProviderPaperId,
    limit: int,
) -> SemanticScholarExpansionDecode:
    """Decode references or citations without performing I/O."""

    if direction not in {"references", "citations"}:
        raise ValueError("direction must be references or citations")
    payload = _decode_json(content)
    records = payload.get("data") if isinstance(payload, Mapping) else None
    errors: list[ErrorDetail] = []
    papers: list[Paper] = []
    edges: list[CitationEdge] = []
    item_key = "citedPaper" if direction == "references" else "citingPaper"
    if not isinstance(records, list):
        errors.append(
            _error(
                "invalid_response",
                "Expansion response requires a data list",
                retryable=False,
            )
        )
    else:
        for item in records[:limit]:
            record = item.get(item_key) if isinstance(item, Mapping) else None
            if not isinstance(record, Mapping):
                errors.append(
                    _error(
                        "invalid_record",
                        "Expansion record is invalid",
                        retryable=False,
                    )
                )
                continue
            try:
                paper = _normalize_record(record)
            except ValueError as error:
                errors.append(_error("invalid_record", str(error), retryable=False))
                continue
            neighbor = ProviderPaperId(
                provider="semantic_scholar",
                value=paper.semantic_scholar_id or "",
            )
            papers.append(paper)
            if direction == "references":
                citing, cited = paper_id, neighbor
            else:
                citing, cited = neighbor, paper_id
            edges.append(
                CitationEdge(
                    provider="semantic_scholar",
                    citing_provider_id=citing,
                    cited_provider_id=cited,
                )
            )
    return SemanticScholarExpansionDecode(
        expansion=CitationExpansion(papers=papers, raw_edges=edges),
        errors=errors,
    )


class SemanticScholarProvider:
    """Map bounded Graph API responses into existing domain models."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        cache: SQLiteResponseCache,
        api_key: str | None = None,
        clock: Clock = _utc_now,
        cache_version: str = "v1",
    ) -> None:
        self._client = client
        self._cache = cache
        self._api_key = api_key
        self._clock = clock
        self._cache_version = cache_version

    async def _request(
        self,
        *,
        method: str,
        endpoint: str,
        params: Mapping[str, QueryValue],
        reservation: BudgetReservation,
        body: dict[str, object] | None = None,
        ttl: timedelta = timedelta(days=7),
    ) -> _RawResult:
        requested_at = self._clock()
        identity = dict(params)
        if body is not None:
            identity["body"] = json.dumps(body, ensure_ascii=False, sort_keys=True)
        key = make_cache_key(
            "semantic_scholar",
            endpoint,
            identity,
            self._cache_version,
        )
        cached = self._cache.get_response(key)
        if cached is not None:
            return _RawResult(
                cached.raw_response,
                0,
                True,
                None,
                cached.response_hash,
                requested_at,
            )
        if reservation.reserved.search_api_calls < 1:
            return _RawResult(
                None,
                0,
                False,
                _error("budget_exhausted", "Semantic Scholar reservation exhausted", retryable=False),
                _sha256(b""),
                requested_at,
            )

        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        try:
            response = await self._client.request(
                method,
                f"{_BASE_URL}{endpoint}",
                params=params,
                json=body,
                headers=headers,
                follow_redirects=False,
            )
        except httpx.TimeoutException:
            return _RawResult(
                None,
                1,
                False,
                _error("timeout", "Semantic Scholar request timed out", retryable=True),
                _sha256(b""),
                requested_at,
            )
        except httpx.RequestError:
            return _RawResult(
                None,
                1,
                False,
                _error("network_error", "Semantic Scholar network request failed", retryable=True),
                _sha256(b""),
                requested_at,
            )
        if response.status_code != 200:
            if response.status_code == 429:
                code, retryable = "rate_limited", True
            elif response.status_code >= 500:
                code, retryable = "server_error", True
            elif response.status_code == 400:
                code, retryable = "invalid_request", False
            elif response.status_code in {401, 403}:
                code, retryable = "authentication_error", False
            else:
                code, retryable = "provider_error", False
            return _RawResult(
                None,
                1,
                False,
                _error(code, f"Semantic Scholar returned HTTP {response.status_code}", retryable=retryable),
                _sha256(response.content),
                requested_at,
            )

        safe_headers = {
            name.casefold(): value
            for name, value in response.headers.items()
            if name.casefold() in SAFE_RESPONSE_HEADERS
        }
        self._cache.put_response(
            key=key,
            provider="semantic_scholar",
            endpoint=endpoint,
            cache_version=self._cache_version,
            params=identity,
            raw_response=response.content,
            requested_at=requested_at,
            ttl=ttl,
            safe_headers=safe_headers,
        )
        return _RawResult(
            response.content,
            1,
            False,
            None,
            _sha256(response.content),
            requested_at,
        )

    @staticmethod
    def _result(
        *,
        data: Any,
        raw: _RawResult,
        endpoint: str,
        started: float,
        errors: list[ErrorDetail],
    ) -> ProviderResult[Any]:
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        if raw.error is not None:
            errors.insert(0, raw.error)
        return ProviderResult[Any](
            data=data,
            usage=UsageActual(search_api_calls=raw.calls, elapsed_ms=elapsed_ms),
            provenance={
                "provider": "semantic_scholar",
                "endpoint": endpoint,
                "model_id": "semantic-scholar-graph-api",
                "requested_at": raw.requested_at.isoformat(),
                "response_hash": raw.response_hash,
            },
            cache_hit=raw.cache_hit,
            latency_ms=elapsed_ms,
            errors=errors,
        )

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        started = time.perf_counter()
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("query must not be empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        unknown = set(filters).difference({"year_from", "year_to"})
        if unknown:
            raise ValueError(f"unknown Semantic Scholar filters: {sorted(unknown)}")
        params: dict[str, QueryValue] = {
            "query": normalized_query,
            "limit": limit,
            "fields": _FIELDS,
        }
        if filters:
            start = filters.get("year_from", "")
            end = filters.get("year_to", "")
            params["year"] = f"{start}-{end}"
        raw = await self._request(
            method="GET",
            endpoint=_SEARCH_ENDPOINT,
            params=params,
            reservation=reservation,
        )
        decoded = (
            decode_semantic_scholar_search(raw.content, limit=limit)
            if raw.content is not None
            else SemanticScholarPaperDecode(papers=[], errors=[])
        )
        return self._result(
            data=decoded.papers,
            raw=raw,
            endpoint=_SEARCH_ENDPOINT,
            started=started,
            errors=decoded.errors,
        )

    async def batch_details(
        self,
        paper_ids: Sequence[str],
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        started = time.perf_counter()
        ids = [value.strip() for value in paper_ids if value.strip()]
        if not ids:
            raise ValueError("paper_ids must not be empty")
        raw = await self._request(
            method="POST",
            endpoint=_BATCH_ENDPOINT,
            params={"fields": _FIELDS},
            body={"ids": ids},
            reservation=reservation,
            ttl=timedelta(days=30),
        )
        decoded = (
            decode_semantic_scholar_batch(raw.content)
            if raw.content is not None
            else SemanticScholarPaperDecode(papers=[], errors=[])
        )
        return self._result(
            data=decoded.papers,
            raw=raw,
            endpoint=_BATCH_ENDPOINT,
            started=started,
            errors=decoded.errors,
        )

    async def _expansion(
        self,
        *,
        direction: str,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        started = time.perf_counter()
        if paper_id.provider != "semantic_scholar":
            raise ValueError("Semantic Scholar expansion requires a Semantic Scholar paper ID")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        endpoint = f"/paper/{paper_id.value}/{direction}"
        raw = await self._request(
            method="GET",
            endpoint=endpoint,
            params={"fields": _FIELDS, "limit": limit},
            reservation=reservation,
            ttl=timedelta(days=30),
        )
        decoded = (
            decode_semantic_scholar_expansion(
                raw.content,
                direction=direction,
                paper_id=paper_id,
                limit=limit,
            )
            if raw.content is not None
            else SemanticScholarExpansionDecode(
                expansion=CitationExpansion(papers=[], raw_edges=[]),
                errors=[],
            )
        )
        return self._result(
            data=decoded.expansion,
            raw=raw,
            endpoint=endpoint,
            started=started,
            errors=decoded.errors,
        )

    async def references(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        return await self._expansion(
            direction="references",
            paper_id=paper_id,
            limit=limit,
            reservation=reservation,
        )

    async def citations(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        return await self._expansion(
            direction="citations",
            paper_id=paper_id,
            limit=limit,
            reservation=reservation,
        )
