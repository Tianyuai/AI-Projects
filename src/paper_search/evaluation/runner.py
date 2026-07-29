"""Deterministic Week-1 candidate processing and evaluation orchestration."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ValidationError

from paper_search.config import RuntimeConfig, load_runtime_config
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


class SearchProvider(Protocol):
    """Injected provider boundary used by the evaluation runner."""

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]: ...


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


async def run_evaluation(
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
            id_map_path = _resolve_cli_id_map(Path("data"), args.id_map)
            id_map_bytes = id_map_path.read_bytes()
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
