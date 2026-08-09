"""Optional LLM title-candidate recall stage (default off).

The stage asks the LLM for concrete real paper titles that answer the
research query, verifies each candidate against the search provider, and
returns the merged papers as an independent ranking source for fusion.
Every LLM and provider call goes through the snapshot-bound adapters, so
live capture and deterministic replay behave like the query-analysis path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import Field

from paper_search.application.contracts import DependencyDiagnostic, SnapshotRef
from paper_search.control.budget import (
    BudgetExceededError,
    HardBudgetController,
    ReservationError,
)
from paper_search.domain.models import (
    BudgetReservation,
    DomainModel,
    ErrorDetail,
    NonEmptyStr,
    NonNegativeInt,
    Paper,
    ProviderResult,
    QuerySpec,
    UsageActual,
    UsageEstimate,
)
from paper_search.retrieval.base import SearchProvider

__all__ = [
    "AsyncTitleCandidateStage",
    "LLMTitleCandidateStage",
    "TitleCandidateRecallResult",
    "extract_title_candidates",
]

_TITLE_KEYS = (
    "titles",
    "title_candidates",
    "candidate_titles",
    "paper_titles",
    "papers",
    "candidates",
)
_MAX_TITLES = 20
_DEFAULT_INSTRUCTIONS = (
    'Respond with a JSON object whose only key is "titles", a list of 20 '
    "strings. Each string must be the full, exact title of a specific real "
    "published academic paper that directly answers the research query. "
    "Titles must read like genuine paper titles and use precise academic "
    "terminology, including the specific methods, datasets, phenomena, or "
    "problem formulations named in the research goal and constraints. "
    "Cover distinct aspects and facets of the research goal across the "
    "candidates: method variants, dataset-specific works, and problem "
    "reformulations, so the set as a whole spans the query's semantic space. "
    "Prefer well-known, verifiable papers; vary the phrasing across "
    "candidates, e.g., synonyms, alternate problem framings, and "
    "method-first versus task-first titles. Only include papers you are "
    "confident exist; never invent, paraphrase, or describe papers. "
    "Do not add explanations."
)


class _Analyzer(Protocol):
    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]: ...


def extract_title_candidates(data: object, *, limit: int = 5) -> list[str]:
    """Extract concrete paper titles from flexible LLM output, deduped."""

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= _MAX_TITLES
    ):
        raise ValueError("limit must be an integer between 1 and 20")
    collected: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key in _TITLE_KEYS:
                nested = value.get(key)
                if isinstance(nested, list):
                    for item in nested:
                        if isinstance(item, str):
                            collected.append(item)
                        elif isinstance(item, dict):
                            title = item.get("title") or item.get("name")
                            if isinstance(title, str):
                                collected.append(title)
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    walk(nested)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    result: list[str] = []
    seen: set[str] = set()
    for title in collected:
        text = " ".join(title.split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
        if len(result) == limit:
            break
    return result


class TitleCandidateRecallResult(DomainModel):
    provider_result: ProviderResult[list[Paper]]
    status: Literal["applied", "degraded"]
    diagnostics: list[DependencyDiagnostic] = Field(default_factory=list)
    warnings: list[NonEmptyStr] = Field(default_factory=list)
    titles_generated: NonNegativeInt = 0
    titles_searched: NonNegativeInt = 0
    truncated: bool = False


class AsyncTitleCandidateStage(Protocol):
    async def recall(
        self,
        spec: QuerySpec,
        *,
        controller: HardBudgetController,
    ) -> TitleCandidateRecallResult: ...


def _add_usage(left: UsageActual, right: UsageActual) -> UsageActual:
    return UsageActual(
        search_api_calls=left.search_api_calls + right.search_api_calls,
        llm_calls=left.llm_calls + right.llm_calls,
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cost_cny=(
            left.cost_cny + right.cost_cny
            if left.cost_cny is not None and right.cost_cny is not None
            else None
        ),
        elapsed_ms=left.elapsed_ms + right.elapsed_ms,
    )


def _llm_snapshot_ref(provenance: dict[str, Any]) -> SnapshotRef | None:
    names = (
        "snapshot_entry_id",
        "snapshot_cache_key",
        "snapshot_response_sha256",
        "snapshot_path",
    )
    if not any(name in provenance for name in names):
        return None
    if not all(name in provenance for name in names):
        raise ValueError("incomplete LLM snapshot provenance")
    return SnapshotRef(
        entry_id=provenance["snapshot_entry_id"],
        dependency="llm",
        cache_key=provenance["snapshot_cache_key"],
        response_sha256=provenance["snapshot_response_sha256"],
        captured_at=datetime.fromisoformat(provenance["requested_at"]),
        snapshot_path=provenance["snapshot_path"],
    )


def _provider_diagnostic(
    result: ProviderResult[list[Paper]],
) -> DependencyDiagnostic:
    raw_refs = result.provenance.get("snapshot_refs", "[]")
    decoded = json.loads(raw_refs) if isinstance(raw_refs, str) else raw_refs
    if not isinstance(decoded, list) or any(
        not isinstance(raw, dict) for raw in decoded
    ):
        raise ValueError("invalid provider snapshot provenance")
    refs = [SnapshotRef.model_validate(raw) for raw in decoded]
    return DependencyDiagnostic(
        dependency="openalex",
        endpoint=result.provenance["endpoint"],
        model_id=result.provenance.get("model_id"),
        usage=result.usage,
        latency_ms=result.latency_ms,
        cache_hit=result.cache_hit,
        snapshot_refs=refs,
        errors=result.errors,
    )


def _llm_diagnostic(
    result: ProviderResult[dict[str, Any]],
    ref: SnapshotRef | None,
) -> DependencyDiagnostic:
    return DependencyDiagnostic(
        dependency="llm",
        endpoint="title_candidates",
        model_id=result.provenance.get("model_id"),
        usage=result.usage,
        latency_ms=result.latency_ms,
        cache_hit=result.cache_hit,
        snapshot_refs=[] if ref is None else [ref],
        errors=[
            ErrorDetail(
                code=error.code,
                message="Title candidate dependency reported an error",
                retryable=error.retryable,
                provider="llm",
            )
            for error in result.errors
        ],
    )


def _provider_failure_diagnostic() -> DependencyDiagnostic:
    return DependencyDiagnostic(
        dependency="openalex",
        endpoint="title_candidates",
        model_id=None,
        usage=UsageActual(search_api_calls=1),
        latency_ms=0,
        cache_hit=False,
        snapshot_refs=[],
        errors=[
            ErrorDetail(
                code="provider_error",
                message="Title candidate search failed",
                retryable=False,
                provider="openalex",
            )
        ],
    )


def _llm_failure_diagnostic() -> DependencyDiagnostic:
    return DependencyDiagnostic(
        dependency="llm",
        endpoint="title_candidates",
        model_id=None,
        usage=UsageActual(llm_calls=1),
        latency_ms=0,
        cache_hit=False,
        snapshot_refs=[],
        errors=[
            ErrorDetail(
                code="provider_error",
                message="Title candidate generation failed",
                retryable=False,
                provider="llm",
            )
        ],
    )


def _empty_provider_result(
    usage: UsageActual,
) -> ProviderResult[list[Paper]]:
    return ProviderResult[list[Paper]](
        data=[],
        usage=usage,
        provenance={
            "provider": "title_candidates",
            "endpoint": "title_candidates",
            "model_id": "title_candidates",
            "requested_at": datetime.now(UTC).isoformat(),
            "response_hash": "sha256:empty",
        },
        cache_hit=False,
        latency_ms=usage.elapsed_ms,
        errors=[],
    )


def _degraded(
    diagnostics: list[DependencyDiagnostic],
    warnings: list[str],
    usage: UsageActual,
) -> TitleCandidateRecallResult:
    return TitleCandidateRecallResult(
        provider_result=_empty_provider_result(usage),
        status="degraded",
        diagnostics=diagnostics,
        warnings=warnings,
        titles_generated=0,
        titles_searched=0,
        truncated=False,
    )


def _settle_or_verify(
    controller: HardBudgetController,
    reservation: BudgetReservation,
    result: ProviderResult[Any],
) -> None:
    terminal = controller.terminal_outcome(reservation)
    if terminal is None:
        controller.settle(reservation, result.usage)
        return
    mode, recorded = terminal
    if mode != "settled" or recorded != result.usage:
        raise ReservationError("title candidate settlement receipt does not match result")


class LLMTitleCandidateStage:
    """Generate LLM title candidates and verify them through a provider."""

    def __init__(
        self,
        *,
        analyzer: _Analyzer,
        provider: SearchProvider,
        llm_estimate: UsageEstimate,
        search_estimate: UsageEstimate,
        max_titles: int = 20,
        max_results_per_title: int = 10,
        instructions: str | None = None,
    ) -> None:
        if (
            isinstance(max_titles, bool)
            or not isinstance(max_titles, int)
            or not 1 <= max_titles <= _MAX_TITLES
        ):
            raise ValueError("max_titles must be an integer between 1 and 10")
        if (
            isinstance(max_results_per_title, bool)
            or not isinstance(max_results_per_title, int)
            or max_results_per_title < 1
        ):
            raise ValueError("max_results_per_title must be a positive integer")
        if llm_estimate.llm_calls < 1 or llm_estimate.cost_cny is None:
            raise ValueError(
                "title candidate generation requires a valued LLM estimate"
            )
        if search_estimate.search_api_calls < 1:
            raise ValueError("title candidate search requires a search estimate")
        self._analyzer = analyzer
        self._provider = provider
        self._llm_estimate = llm_estimate
        self._search_estimate = search_estimate
        self._max_titles = max_titles
        self._max_results_per_title = max_results_per_title
        self._instructions = instructions or _DEFAULT_INSTRUCTIONS

    async def recall(
        self,
        spec: QuerySpec,
        *,
        controller: HardBudgetController,
    ) -> TitleCandidateRecallResult:
        if not spec.original_query:
            raise ValueError("title candidates require a nonempty query")
        try:
            llm_reservation = controller.reserve(
                "llm.title_candidates",
                self._llm_estimate,
            )
        except BudgetExceededError:
            return _degraded([], ["unavailable"], UsageActual())
        diagnostics: list[DependencyDiagnostic] = []
        try:
            llm_result = await self._analyzer.generate_json(
                prompt_name="title_candidates",
                payload={
                    "query": spec.original_query,
                    "research_goal": spec.research_goal,
                    "topics": list(spec.topics),
                    "must_have": list(spec.must_have),
                    "instructions": self._instructions,
                },
                reservation=llm_reservation,
            )
            _settle_or_verify(controller, llm_reservation, llm_result)
        except asyncio.CancelledError:
            if controller.terminal_outcome(llm_reservation) is None:
                controller.fail_closed(llm_reservation, UsageActual())
            raise
        except ReservationError:
            if controller.terminal_outcome(llm_reservation) is None:
                controller.fail_closed(llm_reservation)
            raise
        except Exception:
            if controller.terminal_outcome(llm_reservation) is None:
                try:
                    controller.fail_closed(
                        llm_reservation,
                        UsageActual(llm_calls=1),
                    )
                except Exception:
                    pass
            diagnostics.append(_llm_failure_diagnostic())
            return _degraded(
                diagnostics,
                ["unavailable"],
                UsageActual(llm_calls=1),
            )
        llm_ref = _llm_snapshot_ref(llm_result.provenance)
        diagnostics.append(_llm_diagnostic(llm_result, llm_ref))
        if llm_result.errors:
            return _degraded(diagnostics, ["unavailable"], llm_result.usage)
        titles = extract_title_candidates(
            llm_result.data,
            limit=self._max_titles,
        )
        if not titles:
            return _degraded(diagnostics, ["malformed"], llm_result.usage)

        papers: list[Paper] = []
        seen: set[str] = set()
        searches = 0
        search_errors = 0
        truncated = False
        usage = UsageActual(
            llm_calls=llm_result.usage.llm_calls,
            input_tokens=llm_result.usage.input_tokens,
            output_tokens=llm_result.usage.output_tokens,
            cost_cny=llm_result.usage.cost_cny,
            elapsed_ms=llm_result.usage.elapsed_ms,
        )
        for index, title in enumerate(titles):
            try:
                reservation = controller.reserve(
                    f"openalex.title:{index + 1}",
                    self._search_estimate,
                )
            except BudgetExceededError:
                truncated = True
                break
            try:
                search = await self._provider.search(
                    title,
                    {},
                    self._max_results_per_title,
                    reservation,
                )
                _settle_or_verify(controller, reservation, search)
            except asyncio.CancelledError:
                if controller.terminal_outcome(reservation) is None:
                    controller.fail_closed(reservation, UsageActual())
                raise
            except ReservationError:
                searches += 1
                if controller.terminal_outcome(reservation) is None:
                    try:
                        controller.release(reservation)
                    except Exception:
                        pass
                diagnostics.append(_provider_failure_diagnostic())
                search_errors += 1
                continue
            except Exception:
                searches += 1
                if controller.terminal_outcome(reservation) is None:
                    try:
                        controller.release(
                            reservation,
                        )
                    except Exception:
                        pass
                diagnostics.append(_provider_failure_diagnostic())
                search_errors += 1
                continue
            searches += 1
            usage = _add_usage(usage, search.usage)
            diagnostics.append(_provider_diagnostic(search))
            if search.errors:
                search_errors += 1
            for paper in search.data:
                if paper.canonical_id in seen:
                    continue
                seen.add(paper.canonical_id)
                papers.append(paper)

        if not papers:
            status: Literal["applied", "degraded"] = "degraded"
            warnings = ["unavailable" if search_errors else "malformed"]
        else:
            status = "applied"
            warnings = []
        return TitleCandidateRecallResult(
            provider_result=ProviderResult[list[Paper]](
                data=papers,
                usage=usage,
                provenance={
                    "provider": "title_candidates",
                    "endpoint": "title_candidates",
                    "model_id": "title_candidates",
                    "requested_at": datetime.now(UTC).isoformat(),
                    "response_hash": (
                        "sha256:"
                        + hashlib.sha256(
                            "\n".join(
                                paper.canonical_id for paper in papers
                            ).encode("utf-8")
                        ).hexdigest()
                    ),
                },
                cache_hit=False,
                latency_ms=usage.elapsed_ms,
                errors=[],
            ),
            status=status,
            diagnostics=diagnostics,
            warnings=warnings,
            titles_generated=len(titles),
            titles_searched=searches,
            truncated=truncated,
        )
