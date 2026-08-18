"""Deterministic, training-only sampling for Gold retrievability audits."""

from __future__ import annotations

import hashlib
import asyncio
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import Field, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, Sha256


IntentFamily = Literal["which_what", "request", "how_why", "other"]
LengthBucket = Literal["q1", "q2", "q3", "q4"]
GoldCountBucket = Literal["1", "2", "3_plus"]
GoldAvailabilityStatus = Literal[
    "available", "exact_not_found", "unknown_transient", "integrity_failure"
]


class GoldAvailabilityRecord(DomainModel):
    gold_id: NonEmptyStr
    status: GoldAvailabilityStatus
    attempts: int = Field(gt=0)
    openalex_id: str | None = None
    resolved_doi: str | None = None
    http_status: int | None = None


class AuditSampleItem(DomainModel):
    query_id: NonEmptyStr
    stratum: NonEmptyStr
    intent_family: IntentFamily
    length_bucket: LengthBucket
    gold_count_bucket: GoldCountBucket
    query_token_count: int = Field(gt=0)
    gold_paper_count: int = Field(gt=0)
    fold: int = Field(ge=1, le=3)
    selection_sha256: Sha256


class FrozenAuditManifest(DomainModel):
    schema_version: Literal[
        "gold-retrievability-audit-freeze-v1",
        "gold-retrievability-audit-freeze-v2",
    ]
    dataset: Literal["pasa"]
    split: Literal["auto_train", "auto_dev"]
    role: Literal["training", "development"]
    revision: NonEmptyStr
    source_path: NonEmptyStr
    source_sha256: Sha256
    seed: NonEmptyStr
    sampling_method: Literal["proportional_hamilton_sha256"]
    stratification_fields: tuple[
        Literal["intent_family"],
        Literal["query_length_quartile"],
        Literal["gold_count_bucket"],
    ]
    population_query_count: int = Field(gt=0)
    source_query_count: int | None = Field(default=None, gt=0)
    excluded_query_count: int = Field(default=0, ge=0)
    excluded_query_ids_sha256: Sha256 | None = None
    sample_query_count: int = Field(gt=0)
    length_cut_points: tuple[int, int, int]
    stratum_population_counts: dict[str, int]
    stratum_sample_counts: dict[str, int]
    fold_counts: dict[int, int]
    sample: list[AuditSampleItem]

    @model_validator(mode="after")
    def validate_totals(self) -> FrozenAuditManifest:
        expected_role = "training" if self.split == "auto_train" else "development"
        if self.role != expected_role:
            raise ValueError("partition split and role do not match")
        if self.source_query_count is not None and self.source_query_count != (
            self.population_query_count + self.excluded_query_count
        ):
            raise ValueError("source count must equal eligible plus excluded queries")
        if (self.excluded_query_count > 0) != (
            self.excluded_query_ids_sha256 is not None
        ):
            raise ValueError("excluded query count and hash must be present together")
        if len(self.sample) != self.sample_query_count:
            raise ValueError("sample count differs from sample records")
        if sum(self.stratum_population_counts.values()) != self.population_query_count:
            raise ValueError("population strata do not sum to population")
        if sum(self.stratum_sample_counts.values()) != self.sample_query_count:
            raise ValueError("sample strata do not sum to sample")
        if sum(self.fold_counts.values()) != self.sample_query_count:
            raise ValueError("folds do not sum to sample")
        return self


class _PartitionRow(DomainModel):
    dataset: NonEmptyStr
    split: NonEmptyStr
    role: NonEmptyStr
    revision: NonEmptyStr
    query_id: NonEmptyStr
    query: NonEmptyStr
    gold_paper_ids: list[NonEmptyStr]
    source_components: list[NonEmptyStr] = Field(default_factory=list)


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _arxiv_value(gold_id: str) -> str:
    prefix, separator, value = gold_id.strip().partition(":")
    if not separator or prefix.casefold() != "arxiv" or not value.strip():
        raise ValueError("Gold availability audit currently requires arxiv identifiers")
    return re.sub(r"v\d+$", "", value.strip(), flags=re.IGNORECASE).casefold()


def openalex_exact_url(gold_id: str) -> str:
    arxiv_id = _arxiv_value(gold_id)
    doi = f"https://doi.org/10.48550/arxiv.{arxiv_id}"
    return f"https://api.openalex.org/works/{quote(doi, safe=':.')}"


async def probe_gold_identifier(
    gold_id: str,
    *,
    client: httpx.AsyncClient,
    api_key: str | None = None,
    max_attempts: int = 3,
) -> GoldAvailabilityRecord:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    _arxiv_value(gold_id)
    params = {"select": "id,doi"}
    if api_key:
        params["api_key"] = api_key
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.get(openalex_exact_url(gold_id), params=params)
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == max_attempts:
                return GoldAvailabilityRecord(
                    gold_id=gold_id,
                    status="unknown_transient",
                    attempts=attempt,
                )
            await asyncio.sleep(float(attempt))
            continue
        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if not isinstance(payload, dict):
                return GoldAvailabilityRecord(
                    gold_id=gold_id,
                    status="integrity_failure",
                    attempts=attempt,
                    http_status=200,
                )
            openalex_id = payload.get("id")
            response_doi = payload.get("doi")
            if (
                not isinstance(openalex_id, str)
                or not openalex_id.startswith("https://openalex.org/W")
            ):
                return GoldAvailabilityRecord(
                    gold_id=gold_id,
                    status="integrity_failure",
                    attempts=attempt,
                    http_status=200,
                )
            return GoldAvailabilityRecord(
                gold_id=gold_id,
                status="available",
                attempts=attempt,
                openalex_id=openalex_id,
                resolved_doi=response_doi if isinstance(response_doi, str) else None,
                http_status=200,
            )
        if response.status_code == 404:
            return GoldAvailabilityRecord(
                gold_id=gold_id,
                status="exact_not_found",
                attempts=attempt,
                http_status=404,
            )
        if response.status_code == 429 or 500 <= response.status_code <= 599:
            if attempt < max_attempts:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = min(max(float(retry_after or attempt), 0.0), 10.0)
                except ValueError:
                    delay = float(attempt)
                await asyncio.sleep(delay)
                continue
            return GoldAvailabilityRecord(
                gold_id=gold_id,
                status="unknown_transient",
                attempts=attempt,
                http_status=response.status_code,
            )
        response.raise_for_status()
    raise AssertionError("unreachable")


def _selection_hash(seed: str, query_id: str) -> str:
    return _sha256_bytes(f"{seed}\0{query_id}".encode("utf-8"))


def _read_partition(path: Path) -> tuple[list[_PartitionRow], bytes]:
    source = path.read_bytes()
    rows: list[_PartitionRow] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"blank partition line: {line_number}")
        try:
            row = _PartitionRow.model_validate_json(raw_line)
        except ValueError as error:
            raise ValueError(f"invalid partition record: {line_number}") from error
        if row.query_id in seen:
            raise ValueError(f"duplicate query_id: {row.query_id}")
        if not row.gold_paper_ids:
            raise ValueError(f"query has no Gold papers: {row.query_id}")
        seen.add(row.query_id)
        rows.append(row)
    if not rows:
        raise ValueError("partition is empty")
    identities = {(row.dataset, row.split, row.role) for row in rows}
    if identities not in (
        {("pasa", "auto_train", "training")},
        {("pasa", "auto_dev", "development")},
    ):
        raise ValueError("audit requires a PaSa auto_train or auto_dev partition")
    if len({row.revision for row in rows}) != 1:
        raise ValueError("partition contains multiple revisions")
    return rows, source


def _token_count(query: str) -> int:
    count = len(re.findall(r"\w+", query, flags=re.UNICODE))
    if count == 0:
        raise ValueError("query contains no tokens")
    return count


def _quartile_cut_points(values: list[int]) -> tuple[int, int, int]:
    ordered = sorted(values)
    return tuple(
        ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]
        for fraction in (0.25, 0.50, 0.75)
    )  # type: ignore[return-value]


def _intent_family(query: str) -> IntentFamily:
    normalized = " ".join(query.casefold().split())
    if re.match(r"^(which|what)\b", normalized):
        return "which_what"
    if re.match(r"^(could|can|please|find|provide|list|give|show|recommend)\b", normalized):
        return "request"
    if re.match(r"^(how|why)\b", normalized):
        return "how_why"
    return "other"


def _length_bucket(value: int, cuts: tuple[int, int, int]) -> LengthBucket:
    if value <= cuts[0]:
        return "q1"
    if value <= cuts[1]:
        return "q2"
    if value <= cuts[2]:
        return "q3"
    return "q4"


def _gold_bucket(count: int) -> GoldCountBucket:
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3_plus"


def _hamilton_quotas(counts: dict[str, int], sample_size: int) -> dict[str, int]:
    population = sum(counts.values())
    exact = {key: sample_size * value / population for key, value in counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = sample_size - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return {key: value for key, value in quotas.items() if value > 0}


def build_frozen_audit_manifest(
    partition_path: Path,
    *,
    sample_size: int = 385,
    seed: str = "pasa-gold-retrievability-v1",
    excluded_query_ids: frozenset[str] = frozenset(),
) -> FrozenAuditManifest:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not seed.strip():
        raise ValueError("seed must not be empty")
    rows, source = _read_partition(partition_path)
    source_query_count = len(rows)
    known_query_ids = {row.query_id for row in rows}
    unknown_exclusions = excluded_query_ids.difference(known_query_ids)
    if unknown_exclusions:
        raise ValueError("excluded query IDs are missing from the partition")
    rows = [row for row in rows if row.query_id not in excluded_query_ids]
    if sample_size > len(rows):
        raise ValueError("sample_size exceeds the eligible population")

    lengths = {row.query_id: _token_count(row.query) for row in rows}
    cuts = _quartile_cut_points(list(lengths.values()))
    strata: dict[
        str, list[tuple[_PartitionRow, IntentFamily, LengthBucket, GoldCountBucket]]
    ] = defaultdict(list)
    for row in rows:
        intent = _intent_family(row.query)
        length_bucket = _length_bucket(lengths[row.query_id], cuts)
        gold_bucket = _gold_bucket(len(row.gold_paper_ids))
        stratum = f"intent={intent}|length={length_bucket}|gold={gold_bucket}"
        strata[stratum].append((row, intent, length_bucket, gold_bucket))

    population_counts = {key: len(value) for key, value in sorted(strata.items())}
    quotas = _hamilton_quotas(population_counts, sample_size)
    selected: list[
        tuple[_PartitionRow, IntentFamily, LengthBucket, GoldCountBucket, str]
    ] = []
    for stratum, quota in sorted(quotas.items()):
        ranked = sorted(
            strata[stratum],
            key=lambda item: (_selection_hash(seed, item[0].query_id), item[0].query_id),
        )
        selected.extend((*item, stratum) for item in ranked[:quota])

    selected.sort(key=lambda item: (_selection_hash(seed + "\0fold", item[0].query_id), item[0].query_id))
    sample: list[AuditSampleItem] = []
    for index, (row, intent, length_bucket, gold_bucket, stratum) in enumerate(selected):
        sample.append(
            AuditSampleItem(
                query_id=row.query_id,
                stratum=stratum,
                intent_family=intent,
                length_bucket=length_bucket,
                gold_count_bucket=gold_bucket,
                query_token_count=lengths[row.query_id],
                gold_paper_count=len(row.gold_paper_ids),
                fold=index % 3 + 1,
                selection_sha256=_selection_hash(seed, row.query_id),
            )
        )

    return FrozenAuditManifest(
        schema_version=(
            "gold-retrievability-audit-freeze-v2"
            if excluded_query_ids or rows[0].role == "development"
            else "gold-retrievability-audit-freeze-v1"
        ),
        dataset="pasa",
        split=rows[0].split,
        role=rows[0].role,
        revision=rows[0].revision,
        source_path=partition_path.as_posix(),
        source_sha256=_sha256_bytes(source),
        seed=seed,
        sampling_method="proportional_hamilton_sha256",
        stratification_fields=(
            "intent_family",
            "query_length_quartile",
            "gold_count_bucket",
        ),
        population_query_count=len(rows),
        source_query_count=source_query_count,
        excluded_query_count=len(excluded_query_ids),
        excluded_query_ids_sha256=(
            _sha256_bytes(
                ("\n".join(sorted(excluded_query_ids)) + "\n").encode("utf-8")
            )
            if excluded_query_ids
            else None
        ),
        sample_query_count=len(sample),
        length_cut_points=cuts,
        stratum_population_counts=population_counts,
        stratum_sample_counts=dict(sorted(Counter(item.stratum for item in sample).items())),
        fold_counts=dict(sorted(Counter(item.fold for item in sample).items())),
        sample=sample,
    )


def selected_gold_by_query(
    partition_path: Path,
    manifest: FrozenAuditManifest,
) -> dict[str, frozenset[str]]:
    rows, source = _read_partition(partition_path)
    if _sha256_bytes(source) != manifest.source_sha256:
        raise ValueError("partition bytes do not match the frozen manifest")
    selected_ids = {item.query_id for item in manifest.sample}
    by_query = {
        row.query_id: frozenset(row.gold_paper_ids)
        for row in rows
        if row.query_id in selected_ids
    }
    if set(by_query) != selected_ids:
        raise ValueError("frozen sample query IDs are missing from the partition")
    return by_query


def _wilson_interval(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total))
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "estimate": estimate,
        "ci95_low": max(0.0, center - margin),
        "ci95_high": min(1.0, center + margin),
    }


def summarize_gold_availability(
    manifest: FrozenAuditManifest,
    gold_by_query: dict[str, frozenset[str]],
    records: dict[str, GoldAvailabilityRecord],
) -> dict[str, object]:
    expected_gold = set().union(*gold_by_query.values()) if gold_by_query else set()
    if set(records) != expected_gold:
        raise ValueError("availability records do not match the frozen Gold set")
    if any(record.gold_id != key for key, record in records.items()):
        raise ValueError("availability record key mismatch")

    status_counts = dict(sorted(Counter(record.status for record in records.values()).items()))
    association_statuses = [
        records[gold_id].status
        for query_id in sorted(gold_by_query)
        for gold_id in sorted(gold_by_query[query_id])
    ]
    available_unique = status_counts.get("available", 0)
    available_associations = sum(status == "available" for status in association_statuses)
    fold_by_query = {item.query_id: item.fold for item in manifest.sample}
    fold_payload: dict[str, dict[str, object]] = {}
    for fold in (1, 2, 3):
        query_ids = sorted(
            query_id for query_id, assigned_fold in fold_by_query.items() if assigned_fold == fold
        )
        fold_associations = [
            records[gold_id].status
            for query_id in query_ids
            for gold_id in sorted(gold_by_query[query_id])
        ]
        fold_payload[str(fold)] = {
            "query_count": len(query_ids),
            "gold_association_count": len(fold_associations),
            "available_association_count": sum(
                status == "available" for status in fold_associations
            ),
            "all_gold_available_query_count": sum(
                all(records[gold_id].status == "available" for gold_id in gold_by_query[query_id])
                for query_id in query_ids
            ),
        }
    all_available_queries = sum(
        all(records[gold_id].status == "available" for gold_id in gold_ids)
        for gold_ids in gold_by_query.values()
    )
    any_available_queries = sum(
        any(records[gold_id].status == "available" for gold_id in gold_ids)
        for gold_ids in gold_by_query.values()
    )
    return {
        "schema_version": "gold-retrievability-audit-result-v1",
        "source_sha256": manifest.source_sha256,
        "query_count": len(gold_by_query),
        "unique_gold_count": len(expected_gold),
        "gold_association_count": len(association_statuses),
        "status_counts": status_counts,
        "unique_gold_availability": _wilson_interval(available_unique, len(expected_gold)),
        "association_availability": _wilson_interval(
            available_associations, len(association_statuses)
        ),
        "all_gold_available_queries": _wilson_interval(
            all_available_queries, len(gold_by_query)
        ),
        "any_gold_available_queries": _wilson_interval(
            any_available_queries, len(gold_by_query)
        ),
        "folds": fold_payload,
        "diagnostic_complete": not any(
            status in {"unknown_transient", "integrity_failure"}
            for status in status_counts
        ),
    }


def shard_frozen_audit_manifest(
    manifest: FrozenAuditManifest,
    *,
    skip_query_ids: set[str] | frozenset[str] = frozenset(),
    shard_count: int,
) -> list[FrozenAuditManifest]:
    validated = FrozenAuditManifest.model_validate(manifest)
    if shard_count <= 0:
        raise ValueError("shard count must be positive")
    sample_ids = {item.query_id for item in validated.sample}
    unknown = set(skip_query_ids).difference(sample_ids)
    if unknown:
        raise ValueError("skip query IDs are absent from the frozen sample")
    remaining = [
        item for item in validated.sample if item.query_id not in skip_query_ids
    ]
    if shard_count > len(remaining):
        raise ValueError("shard count exceeds remaining query count")
    shards: list[FrozenAuditManifest] = []
    base = validated.model_dump(mode="json")
    for shard_index in range(shard_count):
        sample = remaining[shard_index::shard_count]
        strata = Counter(item.stratum for item in sample)
        folds = Counter(item.fold for item in sample)
        shards.append(
            FrozenAuditManifest.model_validate(
                {
                    **base,
                    "sample_query_count": len(sample),
                    "stratum_sample_counts": dict(sorted(strata.items())),
                    "fold_counts": dict(sorted(folds.items())),
                    "sample": [item.model_dump(mode="json") for item in sample],
                }
            )
        )
    return shards


def _canonical_bytes(manifest: FrozenAuditManifest) -> bytes:
    payload = manifest.model_dump(mode="json")
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def freeze_audit_manifest(path: Path, manifest: FrozenAuditManifest) -> str:
    content = _canonical_bytes(manifest)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"different frozen audit manifest already exists: {path}")
        return _sha256_bytes(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return _sha256_bytes(content)


__all__ = [
    "AuditSampleItem",
    "FrozenAuditManifest",
    "GoldAvailabilityRecord",
    "build_frozen_audit_manifest",
    "freeze_audit_manifest",
    "openalex_exact_url",
    "probe_gold_identifier",
    "selected_gold_by_query",
    "shard_frozen_audit_manifest",
    "summarize_gold_availability",
]
