"""Deterministic Week-1 candidate processing and evaluation orchestration."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

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
    DeduplicationResult,
    FilterResult,
    apply_hard_filters,
    deduplicate_papers,
)
from paper_search.ranking import SCORING_VERSION, LexicalScore, rank_lexically
from paper_search.retrieval import OpenAlexProvider
from paper_search.storage import SQLiteResponseCache, validate_snapshot_manifest


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
    cache_keys: list[NonEmptyStr]
    errors: list[ErrorDetail]


class RunResult(DomainModel):
    """Evaluation result and reproducibility data for an ordered split run."""

    evaluation: EvaluationResult
    query_runs: list[QueryRunRecord]
    usage: UsageActual
    snapshot_manifest: NonEmptyStr


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Week-1 evaluation split")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve_frozen_gold(data_root: Path, split: str) -> Path:
    manifest_path = data_root / "manifest.json"
    try:
        manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise _CliInputError("data manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise _CliInputError("data manifest is invalid")
    if manifest.get("status") != "frozen":
        raise _CliInputError("data manifest is not frozen")

    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict) or split not in partitions:
        raise _CliInputError("unknown data split")
    partition = partitions[split]
    if not isinstance(partition, dict):
        raise _CliInputError("data split manifest is invalid")
    raw_gold_path = partition.get("gold_path")
    if not isinstance(raw_gold_path, str) or not raw_gold_path.strip():
        raise _CliInputError("data split manifest is invalid")

    relative_path = Path(raw_gold_path)
    resolved_root = data_root.resolve()
    if relative_path.is_absolute():
        raise _CliInputError("gold path must stay under data")
    gold_path = (resolved_root / relative_path).resolve()
    if not gold_path.is_relative_to(resolved_root):
        raise _CliInputError("gold path must stay under data")
    if not gold_path.is_file():
        raise _CliInputError("gold file does not exist")

    declared_hash = partition.get("gold_sha256", partition.get("sha256"))
    if declared_hash is not None:
        if not isinstance(declared_hash, str) or declared_hash != _sha256_bytes(
            gold_path.read_bytes()
        ):
            raise _CliInputError("gold file SHA-256 mismatch")
    return gold_path


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
) -> tuple[
    list[PredictionRecord],
    bytes,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    predictions = [record.prediction for record in result.query_runs]
    gold_bytes = _jsonl_bytes(list(gold))
    prediction_bytes = _jsonl_bytes(predictions)
    input_hashes = {
        "gold": _sha256_bytes(gold_bytes),
        "predictions": _sha256_bytes(prediction_bytes),
    }
    metrics_payload: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "input_hashes": input_hashes,
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
        "contract_version": "week1-run-v1",
        "input_hashes": input_hashes,
        "rules": {
            "deduplication": {"fuzzy_title_threshold": 0.98},
            "filtering": {"minimum_uncertainty_multiplier": 0.7},
        },
        "scoring_version": SCORING_VERSION,
        "snapshot_manifest": result.snapshot_manifest,
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
    (
        predictions,
        prediction_bytes,
        metrics_payload,
        usage_payload,
        run_payload,
        deduplication_records,
        filtering_records,
    ) = _artifact_payloads(gold, result, config)
    prepared = {
        output / "predictions.jsonl": prediction_bytes,
        output / "metrics.json": _frozen_json_bytes(metrics_payload),
        output / "usage.json": _frozen_json_bytes(usage_payload),
        output / "run.json": _frozen_json_bytes(run_payload),
        output / "deduplication.jsonl": _dictionary_jsonl_bytes(deduplication_records),
        output / "filtering.jsonl": _dictionary_jsonl_bytes(filtering_records),
    }
    existing = {path: _preflight_frozen(path, content) for path, content in prepared.items()}

    cache_keys = _ordered_unique([key for record in result.query_runs for key in record.cache_keys])
    manifest_path = cache.export_snapshot(cache_keys, output)
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
                cache_keys=_parse_cache_keys(provider_result.provenance),
                errors=provider_result.errors,
            )
        )

    predictions = [record.prediction for record in query_runs]
    result = RunResult(
        evaluation=evaluate(gold, predictions, id_map=id_map),
        query_runs=query_runs,
        usage=_aggregate_usage(query_runs),
        snapshot_manifest="snapshot_manifest.json",
    )
    _write_artifacts(gold, result, cache=cache, config=config, output=output)
    return result


async def _run_cli_evaluation(
    gold: Sequence[EvaluationQuery],
    *,
    config: RuntimeConfig,
    api_key: str,
    output: Path,
) -> None:
    timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        cache = SQLiteResponseCache(output.parent / ".cache" / "openalex.sqlite3")
        provider = OpenAlexProvider(client=client, cache=cache, api_key=api_key)
        await run_evaluation(
            gold,
            provider=provider,
            cache=cache,
            config=config,
            output=output,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        gold_path = _resolve_frozen_gold(Path("data"), args.split)
        gold = read_jsonl(gold_path, EvaluationQuery)
        config = load_runtime_config(args.config, env_file=None)
        if config.openalex_api_key is None:
            raise _CliInputError("OPENALEX_API_KEY is required")
        asyncio.run(
            _run_cli_evaluation(
                gold,
                config=config,
                api_key=config.openalex_api_key.get_secret_value(),
                output=args.output.resolve(),
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
