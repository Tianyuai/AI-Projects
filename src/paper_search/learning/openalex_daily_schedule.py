"""Deterministic daily OpenAlex work allocation without secret material."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import unicodedata
from collections.abc import Mapping, Sequence, Set
from collections.abc import Callable
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Literal

from pydantic import Field, field_validator, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, QuerySpec, Sha256
from paper_search.learning.candidate_ceiling import Core4SemanticBooleanQueryGenerator
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    RecallGenerationContext,
)
from paper_search.recall_experiments.generation.base import GenerationResult
from paper_search.recall_experiments.generation.base import QueryGenerator
from paper_search.retrieval.snapshot_adapters import SearchAttemptQuotaExceededError


_BEIJING = timezone(timedelta(hours=8), name="Asia/Shanghai")


class OpenAlexDailyQuotaExceededError(SearchAttemptQuotaExceededError):
    """The selected key slot has consumed its configured daily hard cap."""


def current_openalex_quota_window(now: datetime) -> date:
    """Map an instant to the Beijing-local window that resets at 08:00."""

    if now.tzinfo is None:
        raise ValueError("OpenAlex quota clock must be timezone-aware")
    local = now.astimezone(_BEIJING)
    return local.date() if local.hour >= 8 else local.date() - timedelta(days=1)


class SQLiteOpenAlexDailyQuotaLedger:
    """Atomically count every attempted request for one daily key slot."""

    def __init__(
        self,
        path: str | Path,
        *,
        window: date,
        key_slot: int,
        max_search_calls: int,
        clock: Callable[[], datetime],
    ) -> None:
        if type(key_slot) is not int or key_slot <= 0:
            raise ValueError("OpenAlex key slot must be positive")
        if type(max_search_calls) is not int or max_search_calls <= 0:
            raise ValueError("OpenAlex daily call cap must be positive")
        self._path = Path(path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._window = window
        self._key_slot = key_slot
        self._max_search_calls = max_search_calls
        self._clock = clock
        with self._transaction() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS quota_usage (
                    window TEXT NOT NULL,
                    key_slot INTEGER NOT NULL CHECK (key_slot > 0),
                    max_search_calls INTEGER NOT NULL CHECK (max_search_calls > 0),
                    used_search_calls INTEGER NOT NULL DEFAULT 0
                        CHECK (used_search_calls >= 0),
                    PRIMARY KEY (window, key_slot)
                )
                """
            )
            row = connection.execute(
                """
                SELECT max_search_calls FROM quota_usage
                WHERE window = ? AND key_slot = ?
                """,
                (window.isoformat(), key_slot),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO quota_usage(
                        window, key_slot, max_search_calls, used_search_calls
                    ) VALUES (?, ?, ?, 0)
                    """,
                    (window.isoformat(), key_slot, max_search_calls),
                )
            elif int(row[0]) != max_search_calls:
                raise ValueError("OpenAlex daily ledger hard cap does not match")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        try:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()
        finally:
            connection.close()

    def _validate_current_window(self) -> None:
        observed = current_openalex_quota_window(self._clock())
        if observed != self._window:
            raise ValueError(
                "configured window is not the current OpenAlex quota window"
            )

    def claim_attempt(self) -> int:
        """Irreversibly reserve one attempt before network dispatch."""

        self._validate_current_window()
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT used_search_calls FROM quota_usage
                WHERE window = ? AND key_slot = ?
                """,
                (self._window.isoformat(), self._key_slot),
            ).fetchone()
            if row is None:
                raise RuntimeError("OpenAlex daily ledger row is unavailable")
            used = int(row[0])
            if used >= self._max_search_calls:
                raise OpenAlexDailyQuotaExceededError(
                    "OpenAlex daily search hard cap reached"
                )
            used += 1
            connection.execute(
                """
                UPDATE quota_usage SET used_search_calls = ?
                WHERE window = ? AND key_slot = ?
                """,
                (used, self._window.isoformat(), self._key_slot),
            )
            return used

    @property
    def used_search_calls(self) -> int:
        with sqlite3.connect(self._path) as connection:
            row = connection.execute(
                """
                SELECT used_search_calls FROM quota_usage
                WHERE window = ? AND key_slot = ?
                """,
                (self._window.isoformat(), self._key_slot),
            ).fetchone()
        if row is None:
            raise RuntimeError("OpenAlex daily ledger row is unavailable")
        return int(row[0])


def canonical_action_text(value: str) -> str:
    """Return the normalized text used by retrieval action identity."""

    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def search_action_identity(
    action: Mapping[str, object],
) -> SearchActionIdentity | None:
    """Read a canonical identity from one saved search action."""

    action_type = action.get("action_type")
    if action_type not in {"text_search", "title_search"}:
        return None
    payload = action.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("saved search action payload is invalid")
    if action_type == "text_search":
        text = payload.get("query_text")
        mode = payload.get("search_mode", "lexical")
    else:
        text = payload.get("title_text")
        mode = "lexical"
    if not isinstance(text, str) or mode not in {"lexical", "semantic"}:
        raise ValueError("saved search action identity is invalid")
    return SearchActionIdentity(
        action_type=action_type,
        search_mode=mode,
        normalized_text=text,
    )


class SearchActionIdentity(DomainModel):
    action_type: Literal["text_search", "title_search"]
    search_mode: Literal["lexical", "semantic"]
    normalized_text: NonEmptyStr

    @field_validator("normalized_text", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        return canonical_action_text(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_title_mode(self) -> SearchActionIdentity:
        if self.action_type == "title_search" and self.search_mode != "lexical":
            raise ValueError("title search identity must be lexical")
        return self


def estimate_max_openalex_search_api_calls(
    actions: Sequence[SearchActionIdentity],
    *,
    max_results_per_action: int,
) -> int:
    """Estimate the fail-closed raw request cap, including lexical pagination."""

    if (
        type(max_results_per_action) is not int
        or not 1 <= max_results_per_action <= 300
    ):
        raise ValueError("max results per action must be between 1 and 300")
    lexical_pages = math.ceil(max_results_per_action / 50)
    return sum(
        1 if action.search_mode == "semantic" else lexical_pages
        for action in actions
    )


class ScheduledQueryActions(DomainModel):
    query_id: NonEmptyStr
    missing_actions: tuple[SearchActionIdentity, ...] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_unique_actions(self) -> ScheduledQueryActions:
        if len(set(self.missing_actions)) != len(self.missing_actions):
            raise ValueError("duplicate missing action identity")
        return self


class ScheduledMissingActionGenerator:
    """Emit only the required identities from one frozen source generator."""

    generator_type = "fixed_actions"
    candidate_policy = "core4-semantic-boolean-missing-actions-v1"
    source_sha256 = "sha256:" + hashlib.sha256(
        b"core4-semantic-boolean-missing-actions-v1"
    ).hexdigest()

    def __init__(
        self,
        work: Sequence[ScheduledQueryActions],
        *,
        source: QueryGenerator | None = None,
    ) -> None:
        self._missing_by_query: dict[str, frozenset[SearchActionIdentity]] = {}
        for item in work:
            if item.query_id in self._missing_by_query:
                raise ValueError(f"duplicate scheduled query: {item.query_id}")
            self._missing_by_query[item.query_id] = frozenset(item.missing_actions)
        if not self._missing_by_query:
            raise ValueError("scheduled missing action work must not be empty")
        self._source = source or Core4SemanticBooleanQueryGenerator()

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        expected = self._missing_by_query.get(context.query_id)
        if expected is None:
            raise ValueError(f"query is not in the scheduled shard: {context.query_id}")
        source = await self._source.generate(context)
        selected = []
        observed: set[SearchActionIdentity] = set()
        for action in source.action_batch.actions:
            identity = search_action_identity(action.model_dump(mode="json"))
            if identity in expected:
                selected.append(action)
                observed.add(identity)
        if observed != expected or len(selected) != len(expected):
            raise ValueError(
                f"scheduled action identity is unavailable for query: {context.query_id}"
            )
        batch = RecallActionBatch(actions=selected)
        artifact_bytes = json.dumps(
            batch.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return source.model_copy(
            update={
                "action_batch": batch,
                "artifact_bytes": artifact_bytes,
                "provenance": {
                    **source.provenance,
                    "collection_mode": "scheduled_missing_actions",
                    "scheduled_missing_action_count": str(len(selected)),
                    "source_candidate_policy": str(
                        getattr(self._source, "candidate_policy", "unknown")
                    ),
                },
            }
        )


class OpenAlexDailyShard(DomainModel):
    window: date
    key_slot: int = Field(strict=True, gt=0)
    max_search_calls: int = Field(strict=True, gt=0)
    planned_search_calls: int = Field(strict=True, gt=0)
    queries: tuple[ScheduledQueryActions, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_call_count(self) -> OpenAlexDailyShard:
        observed = sum(len(query.missing_actions) for query in self.queries)
        if self.planned_search_calls != observed:
            raise ValueError("planned OpenAlex calls do not match missing actions")
        if observed > self.max_search_calls:
            raise ValueError("planned OpenAlex calls exceed the per-key hard cap")
        return self


class OpenAlexDailyTrainingSchedule(DomainModel):
    schema_version: Literal["openalex-daily-training-schedule-v1"]
    timezone: Literal["Asia/Shanghai"] = "Asia/Shanghai"
    reset_hour_local: Literal[8] = 8
    first_window: date
    last_training_window: date
    final_test_window: date
    key_count: int = Field(strict=True, gt=0)
    max_search_calls_per_key: int = Field(strict=True, gt=0)
    planned_search_calls: int = Field(strict=True, ge=0)
    available_training_search_calls: int = Field(strict=True, gt=0)
    shards: tuple[OpenAlexDailyShard, ...]

    @model_validator(mode="after")
    def validate_boundaries(self) -> OpenAlexDailyTrainingSchedule:
        if self.first_window > self.last_training_window:
            raise ValueError("first training window must not follow the last window")
        if self.final_test_window <= self.last_training_window:
            raise ValueError("final test window must be reserved after training")
        if self.planned_search_calls != sum(
            shard.planned_search_calls for shard in self.shards
        ):
            raise ValueError("schedule call count does not match its shards")
        return self


class OpenAlexDailyTrainingPlan(DomainModel):
    schema_version: Literal["openalex-daily-training-plan-v1"]
    partition_sha256: Sha256
    receipt_inventory_sha256: Sha256
    required_query_count: int = Field(strict=True, gt=0)
    required_search_actions: int = Field(strict=True, gt=0)
    reused_search_actions: int = Field(strict=True, ge=0)
    missing_search_actions: int = Field(strict=True, ge=0)
    schedule: OpenAlexDailyTrainingSchedule

    @model_validator(mode="after")
    def validate_action_counts(self) -> OpenAlexDailyTrainingPlan:
        if self.required_search_actions != (
            self.reused_search_actions + self.missing_search_actions
        ):
            raise ValueError("OpenAlex training plan action counts do not balance")
        if self.schedule.planned_search_calls != self.missing_search_actions:
            raise ValueError("OpenAlex training plan does not match its schedule")
        return self


def _paired_retrieval_path(root: Path, generation_path: Path) -> Path:
    relative = generation_path.relative_to(root)
    parts = list(relative.parts)
    try:
        generation_index = parts.index("generation")
    except ValueError as error:
        raise ValueError("generation receipt path is invalid") from error
    parts[generation_index] = "retrieval"
    return root.joinpath(*parts)


def load_completed_search_action_identities(
    roots: Sequence[Path],
    *,
    _terminal_error_codes: frozenset[str] = frozenset(),
) -> dict[str, frozenset[SearchActionIdentity]]:
    """Inventory successful saved actions without trusting action IDs as identity."""

    completed: dict[str, set[SearchActionIdentity]] = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"OpenAlex receipt root is unavailable: {root}")
        for generation_path in sorted(root.rglob("generation/attempt-01/*.json")):
            retrieval_path = _paired_retrieval_path(root, generation_path)
            if not retrieval_path.is_file():
                continue
            generation = json.loads(generation_path.read_text(encoding="utf-8"))
            retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
            query_id = generation.get("query_id")
            retrieval_status = retrieval.get("attempt_status")
            if (
                not isinstance(query_id, str)
                or retrieval.get("query_id") != query_id
                or generation.get("attempt_status") != "succeeded"
                or retrieval_status not in {"succeeded", "failed"}
            ):
                continue
            raw_actions = generation.get("actions")
            raw_results = retrieval.get("results")
            if not isinstance(raw_actions, list) or not isinstance(raw_results, list):
                raise ValueError(f"invalid action receipts for query: {query_id}")
            by_action_id: dict[str, SearchActionIdentity] = {}
            saved_action_ids: set[str] = set()
            for raw_action in raw_actions:
                if not isinstance(raw_action, Mapping):
                    raise ValueError(f"invalid saved action for query: {query_id}")
                action_id = raw_action.get("action_id")
                identity = search_action_identity(raw_action)
                if not isinstance(action_id, str):
                    raise ValueError(f"saved action ID is invalid for query: {query_id}")
                saved_action_ids.add(action_id)
                if identity is not None:
                    previous = by_action_id.get(action_id)
                    if previous is not None and previous != identity:
                        raise ValueError(f"conflicting saved action ID for query: {query_id}")
                    by_action_id[action_id] = identity
            selected = completed.setdefault(query_id, set())
            for result in raw_results:
                if not isinstance(result, Mapping):
                    raise ValueError(f"invalid retrieval result for query: {query_id}")
                action_id = result.get("action_id")
                errors = result.get("errors")
                if (
                    not isinstance(action_id, str)
                    or action_id not in saved_action_ids
                    or not isinstance(errors, list)
                ):
                    raise ValueError(f"retrieval action result is invalid for query: {query_id}")
                if action_id not in by_action_id:
                    continue
                terminal_error = bool(errors) and all(
                    isinstance(error, Mapping)
                    and error.get("code") in _terminal_error_codes
                    and error.get("retryable") is False
                    for error in errors
                )
                if (
                    result.get("infrastructure_failure") is not True
                    and (not errors or terminal_error)
                ):
                    selected.add(by_action_id[action_id])
    return {
        query_id: frozenset(identities)
        for query_id, identities in sorted(completed.items())
        if identities
    }


def load_settled_search_action_identities(
    roots: Sequence[Path],
) -> dict[str, frozenset[SearchActionIdentity]]:
    """Inventory success plus terminal decode failures that must not be retried."""

    return load_completed_search_action_identities(
        roots,
        _terminal_error_codes=frozenset({"invalid_work"}),
    )


def build_missing_action_work(
    required_by_query: Mapping[str, Sequence[SearchActionIdentity]],
    completed_by_query: Mapping[str, Set[SearchActionIdentity]],
) -> list[ScheduledQueryActions]:
    """Subtract successful saved identities from the frozen required actions."""

    work: list[ScheduledQueryActions] = []
    for query_id in sorted(required_by_query):
        completed = completed_by_query.get(query_id, frozenset())
        missing = tuple(
            identity
            for identity in required_by_query[query_id]
            if identity not in completed
        )
        if missing:
            work.append(
                ScheduledQueryActions(query_id=query_id, missing_actions=missing)
            )
    return work


async def required_a_prime_action_identities(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[SearchActionIdentity, ...]]:
    """Generate the frozen A-prime action contract for isolated training rows."""

    generator = Core4SemanticBooleanQueryGenerator()
    required: dict[str, tuple[SearchActionIdentity, ...]] = {}
    for row in rows:
        if row.get("split") != "auto_train" or row.get("role") != "training":
            raise ValueError("A-prime daily scheduling requires isolated auto_train rows")
        query_id = row.get("query_id")
        query = row.get("query")
        if not isinstance(query_id, str) or not query_id or not isinstance(query, str):
            raise ValueError("A-prime training row identity is invalid")
        if query_id in required:
            raise ValueError(f"duplicate A-prime training query: {query_id}")
        generation = await generator.generate(
            RecallGenerationContext(
                query_id=query_id,
                original_query=query,
                query_spec=QuerySpec(original_query=query, research_goal=query),
            )
        )
        identities = tuple(
            identity
            for action in generation.action_batch.actions
            if (
                identity := search_action_identity(
                    action.model_dump(mode="json")
                )
            )
            is not None
        )
        if not identities or len(identities) != len(set(identities)):
            raise ValueError(f"A-prime actions are invalid for query: {query_id}")
        required[query_id] = identities
    if not required:
        raise ValueError("A-prime daily scheduling requires training rows")
    return required


def _window_dates(first: date, last: date) -> tuple[date, ...]:
    return tuple(
        first + timedelta(days=offset)
        for offset in range((last - first).days + 1)
    )


def build_daily_openalex_schedule(
    work: list[ScheduledQueryActions],
    *,
    first_window: date,
    last_training_window: date,
    final_test_window: date,
    key_count: int,
    max_search_calls_per_key: int,
) -> OpenAlexDailyTrainingSchedule:
    """Pack complete query action sets into deterministic daily key shards."""

    if type(key_count) is not int or key_count <= 0:
        raise ValueError("OpenAlex key count must be positive")
    if type(max_search_calls_per_key) is not int or max_search_calls_per_key <= 0:
        raise ValueError("per-key OpenAlex call cap must be positive")
    windows = _window_dates(first_window, last_training_window)
    capacity = len(windows) * key_count * max_search_calls_per_key
    ordered = sorted(work, key=lambda item: item.query_id)
    if len({item.query_id for item in ordered}) != len(ordered):
        raise ValueError("scheduled OpenAlex query IDs must be unique")
    required = sum(len(item.missing_actions) for item in ordered)
    if required > capacity:
        raise ValueError(
            "insufficient OpenAlex training capacity: "
            f"required={required}, available={capacity}"
        )
    if any(len(item.missing_actions) > max_search_calls_per_key for item in ordered):
        raise ValueError("one query exceeds the per-key OpenAlex call cap")

    buckets: list[tuple[date, int, list[ScheduledQueryActions], int]] = [
        (window, key_slot, [], 0)
        for window in windows
        for key_slot in range(1, key_count + 1)
    ]
    bucket_index = 0
    for item in ordered:
        action_count = len(item.missing_actions)
        while buckets[bucket_index][3] + action_count > max_search_calls_per_key:
            bucket_index += 1
        window, key_slot, queries, used = buckets[bucket_index]
        queries.append(item)
        buckets[bucket_index] = (window, key_slot, queries, used + action_count)

    shards = tuple(
        OpenAlexDailyShard(
            window=window,
            key_slot=key_slot,
            max_search_calls=max_search_calls_per_key,
            planned_search_calls=used,
            queries=tuple(queries),
        )
        for window, key_slot, queries, used in buckets
        if queries
    )
    return OpenAlexDailyTrainingSchedule(
        schema_version="openalex-daily-training-schedule-v1",
        first_window=first_window,
        last_training_window=last_training_window,
        final_test_window=final_test_window,
        key_count=key_count,
        max_search_calls_per_key=max_search_calls_per_key,
        planned_search_calls=required,
        available_training_search_calls=capacity,
        shards=shards,
    )


async def build_openalex_daily_training_plan(
    rows: Sequence[Mapping[str, object]],
    *,
    partition_sha256: Sha256,
    completed_by_query: Mapping[str, Set[SearchActionIdentity]],
    first_window: date,
    last_training_window: date,
    final_test_window: date,
    key_count: int,
    max_search_calls_per_key: int,
) -> OpenAlexDailyTrainingPlan:
    """Freeze required, reused, and missing A-prime work into one plan."""

    required = await required_a_prime_action_identities(rows)
    relevant_completed = {
        query_id: tuple(
            sorted(
                set(required_actions) & set(completed_by_query.get(query_id, set())),
                key=lambda item: (
                    item.action_type,
                    item.search_mode,
                    item.normalized_text,
                ),
            )
        )
        for query_id, required_actions in sorted(required.items())
    }
    inventory_bytes = json.dumps(
        {
            query_id: [item.model_dump(mode="json") for item in identities]
            for query_id, identities in relevant_completed.items()
            if identities
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    work = build_missing_action_work(required, completed_by_query)
    schedule = build_daily_openalex_schedule(
        work,
        first_window=first_window,
        last_training_window=last_training_window,
        final_test_window=final_test_window,
        key_count=key_count,
        max_search_calls_per_key=max_search_calls_per_key,
    )
    required_count = sum(len(actions) for actions in required.values())
    missing_count = sum(len(item.missing_actions) for item in work)
    return OpenAlexDailyTrainingPlan(
        schema_version="openalex-daily-training-plan-v1",
        partition_sha256=partition_sha256,
        receipt_inventory_sha256=(
            "sha256:" + hashlib.sha256(inventory_bytes).hexdigest()
        ),
        required_query_count=len(required),
        required_search_actions=required_count,
        reused_search_actions=required_count - missing_count,
        missing_search_actions=missing_count,
        schedule=schedule,
    )


__all__ = [
    "OpenAlexDailyQuotaExceededError",
    "OpenAlexDailyShard",
    "OpenAlexDailyTrainingSchedule",
    "OpenAlexDailyTrainingPlan",
    "SQLiteOpenAlexDailyQuotaLedger",
    "ScheduledMissingActionGenerator",
    "ScheduledQueryActions",
    "SearchActionIdentity",
    "build_missing_action_work",
    "build_daily_openalex_schedule",
    "build_openalex_daily_training_plan",
    "canonical_action_text",
    "current_openalex_quota_window",
    "estimate_max_openalex_search_api_calls",
    "load_completed_search_action_identities",
    "load_settled_search_action_identities",
    "required_a_prime_action_identities",
    "search_action_identity",
]
