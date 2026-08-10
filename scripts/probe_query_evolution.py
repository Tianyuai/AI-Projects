"""Offline-first bounded Query Evolution probe CLI and runtime boundaries."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from decimal import Decimal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import httpx
from dotenv import dotenv_values
from pydantic import Field, StringConstraints

from paper_search.control.budget import HardBudgetController
from paper_search.control.ledger import (
    DEV_RUN_CAP_CNY,
    LedgerReservation,
    LedgerReservationError,
    SQLiteBudgetLedger,
)
from paper_search.control.pricing import ActualCostPricer, load_pricing_policy
from paper_search.domain.models import (
    DomainModel,
    Paper,
    ProviderResult,
    QuerySpec,
    SafeRelativePath,
    SearchBudget,
    SearchPlan,
    Sha256,
    UsageActual,
    UsageEstimate,
)
from paper_search.evolution.query_evolution import (
    QueryEvolutionGenerator,
    build_query_evolution_context,
)
from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap, read_jsonl
from paper_search.evaluation.query_evolution_probe import (
    FrozenProbeInputs,
    FrozenQueryRecord,
    ProbeIntegrity,
    evaluate_probe,
    merge_probe_results,
    public_probe_report,
    reconstruct_frozen_baseline,
)
from paper_search.llm.client import OpenAICompatibleLLMClient
from paper_search.llm.prompt_artifacts import (
    load_prompt_artifact,
    render_prompt_system_message,
)
from paper_search.llm.snapshot_adapters import (
    HardBudgetSettlementAdapter,
    LiveCaptureLLMAnalyzer,
    ReplayLLMAnalyzer,
)
from paper_search.retrieval.snapshot_adapters import (
    LiveCaptureSearchProvider,
    ReplaySearchProvider,
)
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencySnapshotReader,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs" / "dev-20260809T061903Z-9bd861e90299"
DEFAULT_GOLD = ROOT / "data" / "dev" / "gold.jsonl"
DEFAULT_ID_MAP = ROOT / "data" / "identifier-map.json"
DEFAULT_AVAILABILITY = ROOT / "docs" / "evidence" / "gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json"
DEFAULT_PROMPT_CONFIG = ROOT / "configs" / "prompts" / "query_evolve.yaml"
DEFAULT_BUDGET_CONFIG = ROOT / "configs" / "budget_balanced.yaml"
DEFAULT_PRICING_POLICY = ROOT / "data" / "annotation_work" / "pricing_v1.yaml"
DEFAULT_LOCK = ROOT / "runs" / "_diag_query_evolution_preflight" / "probe.lock.json"
EXPECTED_AVAILABILITY_SHA256 = "sha256:3f445486d5cf590f3f11a51930153a45916023880e856def379e0f01d053ad04"
PROBE_GLOBAL_TIMEOUT_SECONDS = 3600
PROBE_LEDGER_TTL_SECONDS = 3900
OPERATIONS = ("evolve", "search-1", "search-2")
SafeRunId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$"),
]


class LiveNotAuthorized(RuntimeError):
    pass


class ProbePromptBinding(DomainModel):
    path: SafeRelativePath
    sha256: Sha256
    name: Literal["query_evolve"]
    version: Literal["query-evolve-v1"]


class ProbeLock(DomainModel):
    schema_version: Literal["query-evolution-probe-lock-v2"] = (
        "query-evolution-probe-lock-v2"
    )
    preflight_complete: bool
    probe_run_id: SafeRunId
    source_run_id: str = Field(min_length=1)
    source_hashes: dict[str, str]
    source_git_sha: str = Field(min_length=1)
    gold_sha256: str
    identifier_map_sha256: str
    availability_sha256: str
    query_ids: tuple[str, ...]
    query_count: int = Field(strict=True, ge=0)
    total_selected: int = Field(strict=True, ge=0)
    baseline_candidate_gold_count: int = Field(strict=True, ge=0)
    baseline_top50_gold_count: int = Field(strict=True, ge=0)
    prompt: ProbePromptBinding
    model_id: Literal["deepseek-v4-flash"]
    endpoint: Literal["https://api.deepseek.com/v1"]
    probe_code_sha256: str
    limits: dict[str, int]
    estimates: dict[str, dict[str, object]]
    ledger_checkpoint_sha256: str
    expected_run_directory: str
    lock_sha256: str


class ProbeRuntime(DomainModel):
    allow_live: bool = False
    env_file: Path = Path(r"D:\AI Projects\Projects\.env")
    ledger_path: Path = ROOT / "data" / "budget_ledger.sqlite3"


class ReplayTrace(DomainModel):
    query_ids: tuple[str, ...]
    terminals: dict[str, str]


class CapturedProbe(DomainModel):
    trace: ReplayTrace
    snapshot_set_id: str | None = None


class ReplayedProbe(DomainModel):
    replay_business_sha256: str | None
    capture_replay_match: Literal["matched", "mismatched", "not_evaluated"]


@dataclass(frozen=True)
class ProbeReservations:
    values: dict[tuple[str, str], LedgerReservation]


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _source_git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _probe_code_sha256() -> str:
    result = subprocess.run(["git", "ls-files", "src"], cwd=ROOT, check=True, capture_output=True, text=True)
    paths = sorted({line.replace("\\", "/") for line in result.stdout.splitlines() if line.endswith(".py")} | {"scripts/probe_query_evolution.py"})
    content = bytearray()
    for relative in paths:
        raw = (ROOT / relative).read_bytes()
        content.extend(relative.encode("utf-8"))
        content.extend(b"\0")
        content.extend(str(len(raw)).encode("ascii"))
        content.extend(b"\0")
        content.extend(raw)
    return _sha256_bytes(bytes(content))


def _jsonl_objects(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object record")
            records.append(value)
    return records


def _resolve_all(identifier_map: IdentifierMap, values: Sequence[object]) -> set[str]:
    return {identifier_map.resolve(value) for value in values if isinstance(value, str)}


def _select_queue(
    business: Sequence[Mapping[str, object]],
    gold: Sequence[EvaluationQuery],
    identifier_map: IdentifierMap,
    *,
    retrieved_by_query: Mapping[str, Sequence[object]] | None = None,
) -> tuple[str, ...]:
    gold_by_query = {record.query_id: record for record in gold}
    selected: list[str] = []
    for record in business:
        query_id = record.get("query_id")
        if not isinstance(query_id, str):
            raise ValueError("business result query_id is invalid")
        gold_record = gold_by_query.get(query_id)
        if gold_record is None:
            raise ValueError(f"gold is missing query {query_id}")
        selected_ids = (
            retrieved_by_query.get(query_id, [])
            if retrieved_by_query is not None
            else record.get("selected_paper_ids", [])
        )
        if not isinstance(selected_ids, (list, tuple)):
            raise ValueError("retrieved paper IDs must be a list")
        retrieved = _resolve_all(identifier_map, selected_ids)
        gold_ids = _resolve_all(identifier_map, gold_record.relevant_paper_ids)
        if gold_ids.difference(retrieved):
            selected.append(query_id)
    return tuple(selected)


def _build_prompt_binding(prompt_config: Path) -> ProbePromptBinding:
    root = ROOT.resolve(strict=True)
    try:
        prompt_path = (ROOT / prompt_config).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("prompt path is invalid") from error
    if not prompt_path.is_file() or not prompt_path.is_relative_to(root):
        raise ValueError("prompt path is invalid")
    prompt_bytes = prompt_path.read_bytes()
    artifact = load_prompt_artifact(prompt_bytes)
    return ProbePromptBinding(
        path=prompt_path.relative_to(root).as_posix(),
        sha256=_sha256_bytes(prompt_bytes),
        name=artifact.name,
        version=artifact.version,
    )


def _build_lock_payload(
    *,
    frozen_run: Path,
    gold_path: Path,
    id_map_path: Path,
    availability_path: Path,
    prompt_config: Path,
    probe_run_id: SafeRunId,
    ledger_path: Path,
) -> dict[str, object]:
    business_path = frozen_run / "business-results.jsonl"
    executions_path = frozen_run / "executions.jsonl"
    run_path = frozen_run / "run.json"
    snapshot_path = frozen_run / "snapshot-manifest.json"
    if not snapshot_path.exists():
        snapshot_path = frozen_run / "snapshots" / "snapshot-manifest.json"
    required = [business_path, executions_path, run_path, snapshot_path, gold_path, id_map_path, availability_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"frozen input missing: {missing}")
    business = _jsonl_objects(business_path)
    if len(business) != 60:
        raise ValueError("frozen business result query count must be 60")
    total_selected = 0
    for value in business:
        selected_ids = value.get("selected_paper_ids", [])
        if not isinstance(selected_ids, list):
            raise ValueError("selected_paper_ids must be a list")
        total_selected += len(selected_ids)
    if total_selected != 2910:
        raise ValueError("frozen selected total must be 2910")
    gold = read_jsonl(gold_path, EvaluationQuery)
    identifier_map = IdentifierMap.from_path(id_map_path)
    executions = _jsonl_objects(executions_path)
    retrieved_by_query: dict[str, Sequence[object]] = {}
    for value in executions:
        query_id = value.get("query_id")
        retrieved_ids = value.get("retrieved_paper_ids", [])
        if isinstance(query_id, str) and isinstance(retrieved_ids, list):
            retrieved_by_query[query_id] = retrieved_ids
    availability_sha256 = _sha256_file(availability_path)
    if availability_sha256 != EXPECTED_AVAILABILITY_SHA256:
        raise ValueError("availability evidence hash mismatch")
    availability = json.loads(availability_path.read_text(encoding="utf-8"))
    if not isinstance(availability, dict) or availability.get("availability", {}).get("available") != 134:
        raise ValueError("availability evidence does not prove 134 available works")
    query_ids = _select_queue(business, gold, identifier_map, retrieved_by_query=retrieved_by_query)
    if len(query_ids) != 55:
        raise ValueError("frozen queue must contain 55 available-but-not-retrieved queries")
    prompt_binding = _build_prompt_binding(prompt_config)
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    source_git_sha = run_record.get("source_git_sha")
    if not isinstance(source_git_sha, str) or not source_git_sha:
        raise ValueError("frozen source git SHA is missing")
    source_hashes = {
        "business_results_sha256": _sha256_file(business_path),
        "executions_sha256": _sha256_file(executions_path),
        "run_sha256": _sha256_file(run_path),
        "snapshot_manifest_sha256": _sha256_file(snapshot_path),
    }
    if ledger_path.exists():
        ledger = SQLiteBudgetLedger(ledger_path, reservation_ttl_seconds=PROBE_LEDGER_TTL_SECONDS)
        _, empty_checkpoint = ledger.project_checkpoint()
    else:
        empty_checkpoint = _sha256_bytes(b"")
    return {
        "schema_version": "query-evolution-probe-lock-v2",
        "preflight_complete": True,
        "probe_run_id": probe_run_id,
        "source_run_id": str(run_record.get("run_id", frozen_run.name)),
        "source_hashes": source_hashes,
        "source_git_sha": source_git_sha,
        "gold_sha256": _sha256_file(gold_path),
        "identifier_map_sha256": _sha256_file(id_map_path),
        "availability_sha256": availability_sha256,
        "query_ids": list(query_ids),
        "query_count": 60,
        "total_selected": 2910,
        "baseline_candidate_gold_count": 14,
        "baseline_top50_gold_count": 8,
        "prompt": prompt_binding.model_dump(mode="json"),
        "model_id": "deepseek-v4-flash",
        "endpoint": "https://api.deepseek.com/v1",
        "probe_code_sha256": _probe_code_sha256(),
        "limits": {
            "query_count": 55,
            "llm_logical_operations": 55,
            "openalex_logical_operations": 110,
            "llm_attempts": 165,
            "openalex_attempts": 330,
            "global_timeout_seconds": PROBE_GLOBAL_TIMEOUT_SECONDS,
            "ledger_ttl_seconds": PROBE_LEDGER_TTL_SECONDS,
        },
        "estimates": {
            "evolve": {"llm_calls": 3, "input_tokens": 20000, "output_tokens": 4000, "elapsed_ms": 60000},
            "search-1": {"search_api_calls": 3, "elapsed_ms": 60000},
            "search-2": {"search_api_calls": 3, "elapsed_ms": 60000},
        },
        "ledger_checkpoint_sha256": empty_checkpoint,
        "expected_run_directory": f"runs/_diag_query_evolution_{probe_run_id}",
        "lock_sha256": "sha256:" + "0" * 64,
    }


def _self_hash(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned["lock_sha256"] = "sha256:" + "0" * 64
    return _sha256_bytes(_canonical_json(unsigned))


def preflight_probe(
    *,
    frozen_run: Path,
    gold_path: Path,
    id_map_path: Path,
    availability_path: Path,
    prompt_config: Path,
    probe_run_id: SafeRunId,
    ledger_path: Path,
    output_path: Path,
) -> ProbeLock:
    """Build the immutable lock with zero reservation, environment, or network access."""
    if output_path.exists():
        raise FileExistsError(f"preflight output already exists: {output_path}")
    payload = _build_lock_payload(
        frozen_run=frozen_run,
        gold_path=gold_path,
        id_map_path=id_map_path,
        availability_path=availability_path,
        prompt_config=prompt_config,
        probe_run_id=probe_run_id,
        ledger_path=ledger_path,
    )
    payload["lock_sha256"] = _self_hash(payload)
    lock = ProbeLock.model_validate(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json(lock.model_dump(mode="json")))
    return lock


def load_probe_lock(path: Path) -> ProbeLock:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("probe lock must be a JSON object")
    lock = ProbeLock.model_validate(payload)
    if lock.lock_sha256 != _self_hash(payload):
        raise ValueError("probe lock self-hash mismatch")
    return lock


def _load_locked_prompt(lock: ProbeLock) -> tuple[bytes, str]:
    path = (ROOT / lock.prompt.path).resolve(strict=True)
    root = ROOT.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("locked prompt path is invalid")
    prompt_bytes = path.read_bytes()
    if _sha256_bytes(prompt_bytes) != lock.prompt.sha256:
        raise ValueError("locked prompt hash mismatch")
    artifact = load_prompt_artifact(prompt_bytes)
    if artifact.name != lock.prompt.name or artifact.version != lock.prompt.version:
        raise ValueError("locked prompt identity mismatch")
    return prompt_bytes, render_prompt_system_message(prompt_bytes)


def _derive_run_directory(lock: ProbeLock) -> Path:
    root = ROOT.resolve(strict=True)
    expected_run_directory = f"runs/_diag_query_evolution_{lock.probe_run_id}"
    if lock.expected_run_directory != expected_run_directory:
        raise ValueError("expected run directory mismatch")
    run_dir = (root / expected_run_directory).resolve()
    if not run_dir.is_relative_to(root):
        raise ValueError("expected run directory escapes root")
    return run_dir


def reserve_probe_operations(lock: ProbeLock, ledger: SQLiteBudgetLedger) -> ProbeReservations:
    """Reserve every logical slot before any future live request."""
    expected: dict[tuple[str, str], str] = {
        (query_id, operation): f"{hashlib.sha256(query_id.encode('utf-8')).hexdigest()[:16]}:{operation}"
        for query_id in lock.query_ids
        for operation in OPERATIONS
    }
    try:
        receipts = ledger.report(lock.probe_run_id).receipts
    except LedgerReservationError:
        receipts = []
    if receipts:
        if (
            len(receipts) == len(expected)
            and all(receipt.state == "reserved" and receipt.actual is None for receipt in receipts)
            and {receipt.query_id for receipt in receipts} == set(expected.values())
        ):
            return ProbeReservations(
                {
                    next(key for key, query_id in expected.items() if query_id == receipt.query_id): LedgerReservation(
                        reservation_id=receipt.reservation_id,
                        run_id=receipt.run_id,
                        query_id=receipt.query_id,
                        estimate=receipt.estimate,
                        state="reserved",
                    )
                    for receipt in receipts
                }
            )
        raise LedgerReservationError("probe ledger contains an incomplete or terminal reservation set")
    created: dict[tuple[str, str], LedgerReservation] = {}
    try:
        for query_id in lock.query_ids:
            private_id = hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:16]
            for operation in OPERATIONS:
                raw = lock.estimates[operation]
                estimate = UsageEstimate.model_validate({**raw, "cost_cny": "0.01"})
                created[(query_id, operation)] = ledger.reserve(
                    run_id=lock.probe_run_id,
                    query_id=f"{private_id}:{operation}",
                    estimate=estimate,
                    run_cap_cny=DEV_RUN_CAP_CNY,
                )
    except BaseException:
        for reservation in created.values():
            ledger.fail(reservation, UsageActual(cost_cny="0"))
        raise
    return ProbeReservations(created)


def _estimate(lock: ProbeLock, operation: str) -> UsageEstimate:
    return UsageEstimate.model_validate({**lock.estimates[operation], "cost_cny": "0.01"})


def _controller(operation: str, estimate: UsageEstimate) -> HardBudgetController:
    is_llm = operation == "evolve"
    budget = SearchBudget(
        max_search_api_calls=estimate.search_api_calls if not is_llm else 1,
        target_search_api_calls=estimate.search_api_calls if not is_llm else 0,
        max_llm_calls=estimate.llm_calls if is_llm else 1,
        target_llm_calls=estimate.llm_calls if is_llm else 0,
        max_total_tokens=max(1, estimate.input_tokens + estimate.output_tokens),
        max_cost_cny=0.30,
        max_elapsed_seconds=800,
        soft_deadline_seconds=790,
    )
    return HardBudgetController(budget, reservation_ttl_seconds=800, formal_live=True)


def _placeholder_results(ids: Sequence[object]) -> ProviderResult[list[Paper]]:
    papers = [
        Paper(canonical_id=str(identifier), title=str(identifier))
        for identifier in ids
        if isinstance(identifier, str)
    ]
    return ProviderResult(
        data=papers,
        usage=UsageActual(),
        provenance={
            "provider": "openalex",
            "endpoint": "frozen-baseline",
            "model_id": "openalex-works-v1",
            "requested_at": "2026-08-10T00:00:00+00:00",
            "response_hash": _sha256_bytes(
                _canonical_json([paper.model_dump(mode="json") for paper in papers])
            ),
        },
        cache_hit=True,
        latency_ms=0,
        errors=[],
    )


def _frozen_inputs(lock: ProbeLock) -> tuple[FrozenProbeInputs, dict[str, dict[str, object]]]:
    business = {record["query_id"]: record for record in _jsonl_objects(DEFAULT_RUN / "business-results.jsonl") if isinstance(record.get("query_id"), str)}
    executions = {record["query_id"]: record for record in _jsonl_objects(DEFAULT_RUN / "executions.jsonl") if isinstance(record.get("query_id"), str)}
    records: list[FrozenQueryRecord] = []
    for index, query_id in enumerate(
        [record["query_id"] for record in _jsonl_objects(DEFAULT_RUN / "business-results.jsonl")]
    ):
        raw = business[query_id]
        analysis = raw.get("query_analysis")
        if not isinstance(analysis, Mapping):
            raise ValueError(f"frozen query analysis is missing for {query_id}")
        spec = QuerySpec.model_validate(analysis.get("query_spec"))
        plan = SearchPlan.model_validate(analysis.get("search_plan"))
        selected = raw.get("selected_paper_ids", [])
        if not isinstance(selected, list):
            raise ValueError(f"frozen selected paper IDs are invalid for {query_id}")
        records.append(
            FrozenQueryRecord(
                query_id=query_id,
                query_spec=spec,
                search_plan=plan,
                baseline_results=[_placeholder_results(selected)],
                source_index=index,
            )
        )
    return (
        FrozenProbeInputs(
            queries=records,
            source_run_id=lock.source_run_id,
            source_hashes=lock.source_hashes,
            expected_query_count=lock.query_count,
            expected_total_selected=lock.total_selected,
        ),
        {query_id: {"business": business[query_id], "execution": executions.get(query_id, {})} for query_id in lock.query_ids},
    )


def _load_secrets(env_file: Path) -> tuple[str, tuple[str, ...], str | None]:
    values = dotenv_values(env_file)
    llm_key = values.get("LLM_API_KEY") or os.environ.get("LLM_API_KEY")
    openalex_keys = [
        values.get(name) or os.environ.get(name)
        for name in ("OPENALEX_API_KEY", *[f"OPENALEX_API_KEY_{index}" for index in range(2, 8)])
    ]
    keys = tuple(value for value in openalex_keys if value)
    if not llm_key:
        raise ValueError("LLM_API_KEY is missing from the authorized env file")
    if not keys:
        raise ValueError("OPENALEX_API_KEY is missing from the authorized env file")
    return llm_key, keys, values.get("OPENALEX_MAILTO") or os.environ.get("OPENALEX_MAILTO")


def _settle_ledger(
    ledger: SQLiteBudgetLedger,
    reservation: LedgerReservation,
    actual: UsageActual,
) -> None:
    ledger.checkpoint_actual(reservation, actual)
    ledger.settle(reservation, actual)


def _fail_ledger(ledger: SQLiteBudgetLedger, reservation: LedgerReservation) -> None:
    ledger.fail(reservation, UsageActual(cost_cny=Decimal("0")))


def _comparable(value: object) -> object:
    if isinstance(value, ProviderResult):
        return {
            "data": [item.model_dump(mode="json") for item in value.data],
            "errors": [
                {"provider": item.provider, "code": item.code, "retryable": item.retryable}
                for item in value.errors
            ],
        }
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _capture_replay_hash(lock: ProbeLock, outcomes: Mapping[str, Mapping[str, object]]) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "lock_sha256": lock.lock_sha256,
                "queries": [
                    {
                        "query_id": query_id,
                        **{
                            key: value
                            for key, value in outcomes[query_id].items()
                            if key != "additions"
                        },
                    }
                    for query_id in lock.query_ids
                ],
            }
        )
    )


def _can_resume_partial_run(run_dir: Path) -> bool:
    lock_copy = run_dir / "probe.lock.json"
    snapshot_root = run_dir / "snapshots"
    if not run_dir.is_dir() or not lock_copy.is_file() or not snapshot_root.is_dir():
        return False
    return not any(snapshot_root.iterdir()) and not any(
        item.name not in {"probe.lock.json", "snapshots"}
        for item in run_dir.iterdir()
    )


async def _run_live_probe(
    lock: ProbeLock,
    runtime: ProbeRuntime,
    run_dir: Path,
    *,
    prompt_sha: str,
    prompt_instructions: str,
) -> None:
    if runtime.ledger_path.resolve() == run_dir.resolve():
        raise ValueError("ledger path must not be the probe output directory")
    llm_key, openalex_keys, openalex_mailto = _load_secrets(runtime.env_file)
    frozen_inputs, raw_records = _frozen_inputs(lock)
    business_by_id: dict[str, Mapping[str, object]] = {
        query_id: cast(Mapping[str, object], raw_records[query_id]["business"])
        for query_id in lock.query_ids
    }
    policy = load_pricing_policy(DEFAULT_PRICING_POLICY)
    pricer = ActualCostPricer(policy)
    ledger = SQLiteBudgetLedger(runtime.ledger_path, reservation_ttl_seconds=PROBE_LEDGER_TTL_SECONDS)
    persistent = reserve_probe_operations(lock, ledger).values
    capture_root = run_dir / "snapshots"
    capture_root.mkdir(parents=True, exist_ok=True)
    capture_store = DependencyCaptureStore(capture_root)
    llm_http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0))
    openalex_http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0))
    captured: dict[str, dict[str, object]] = {}
    try:
        client = OpenAICompatibleLLMClient(
            client=llm_http,
            base_url="https://api.deepseek.com/v1",
            model=lock.model_id,
            api_key=llm_key,
            prompt_version=lock.prompt.version,
        )
        for query_id in lock.query_ids:
            raw = business_by_id[query_id]
            analysis = raw.get("query_analysis")
            if not isinstance(analysis, Mapping):
                raise ValueError(f"frozen query analysis is missing for {query_id}")
            spec = QuerySpec.model_validate(analysis["query_spec"])
            plan = SearchPlan.model_validate(analysis["search_plan"])
            execution = raw_records[query_id].get("execution")
            if not isinstance(execution, Mapping):
                execution = {}
            retrieved = execution.get("retrieved_paper_ids", [])
            candidate_count = len(retrieved) if isinstance(retrieved, list) else 0
            context = build_query_evolution_context(spec, plan, candidate_count, [])
            evolve_controller = _controller("evolve", _estimate(lock, "evolve"))
            evolve_reservation = evolve_controller.reserve("query-evolution", _estimate(lock, "evolve"))
            evolve_persistent = persistent[(query_id, "evolve")]
            generator = QueryEvolutionGenerator(
                analyzer=LiveCaptureLLMAnalyzer(
                    client=client,
                    capture_store=capture_store,
                    pricer=pricer,
                    controller=HardBudgetSettlementAdapter(evolve_controller),
                    prompt_artifact_sha256=prompt_sha,
                    prompt_instructions=prompt_instructions,
                )
            )
            try:
                generated = await generator.generate(context, evolve_reservation)
                _settle_ledger(ledger, evolve_persistent, generated.usage)
            except BaseException:
                _fail_ledger(ledger, evolve_persistent)
                raise
            searches: list[ProviderResult[list[Paper]]] = []
            status = generated.status
            if generated.status == "generated" and generated.proposal is not None:
                for index, subquery in enumerate(generated.proposal.subqueries, start=1):
                    operation = f"search-{index}"
                    search_controller = _controller(operation, _estimate(lock, operation))
                    search_reservation = search_controller.reserve(operation, _estimate(lock, operation))
                    search_persistent = persistent[(query_id, operation)]
                    provider = LiveCaptureSearchProvider(
                        dependency="openalex",
                        client=openalex_http,
                        capture_store=capture_store,
                        pricer=pricer,
                        controller=HardBudgetSettlementAdapter(search_controller),
                        api_key=openalex_keys[0],
                        additional_api_keys=openalex_keys[1:],
                        mailto=openalex_mailto,
                    )
                    try:
                        result = await provider.search(
                            subquery.text,
                            dict(plan.inherited_hard_filters),
                            50,
                            search_reservation,
                        )
                        _settle_ledger(ledger, search_persistent, result.usage)
                    except BaseException:
                        _fail_ledger(ledger, search_persistent)
                        raise
                    searches.append(result)
                for index in range(len(searches) + 1, 3):
                    _fail_ledger(ledger, persistent[(query_id, f"search-{index}")])
            else:
                _fail_ledger(ledger, persistent[(query_id, "search-1")])
                _fail_ledger(ledger, persistent[(query_id, "search-2")])
            captured[query_id] = {
                "terminal": status,
                "proposal": _comparable(generated.proposal),
                "searches": [_comparable(result) for result in searches],
                "additions": searches,
            }
    finally:
        await llm_http.aclose()
        await openalex_http.aclose()
    manifest = capture_store.seal()
    reader = DependencySnapshotReader(
        capture_store.manifest_path,
        snapshot_manifest_sha256=capture_store.manifest_sha256,
        snapshot_set_id=manifest.snapshot_set_id,
    )
    replayed: dict[str, dict[str, object]] = {}
    for query_id in lock.query_ids:
        raw = business_by_id[query_id]
        analysis = raw.get("query_analysis")
        if not isinstance(analysis, Mapping):
            raise ValueError(f"frozen query analysis is missing for {query_id}")
        spec = QuerySpec.model_validate(analysis["query_spec"])
        plan = SearchPlan.model_validate(analysis["search_plan"])
        execution = raw_records[query_id].get("execution")
        if not isinstance(execution, Mapping):
            execution = {}
        retrieved = execution.get("retrieved_paper_ids", [])
        candidate_count = len(retrieved) if isinstance(retrieved, list) else 0
        context = build_query_evolution_context(spec, plan, candidate_count, [])
        controller = _controller("evolve", _estimate(lock, "evolve"))
        reservation = controller.reserve("replay", _estimate(lock, "evolve"))
        generator = QueryEvolutionGenerator(
            analyzer=ReplayLLMAnalyzer(
                reader=reader,
                model_id=lock.model_id,
                prompt_artifact_sha256=prompt_sha,
                prompt_version=lock.prompt.version,
            )
        )
        generated = await generator.generate(context, reservation)
        replay_searches: list[ProviderResult[list[Paper]]] = []
        if generated.status == "generated" and generated.proposal is not None:
            replay_provider = ReplaySearchProvider(dependency="openalex", reader=reader)
            for subquery in generated.proposal.subqueries:
                replay_searches.append(await replay_provider.search(subquery.text, dict(plan.inherited_hard_filters), 50, reservation))
        replayed[query_id] = {
            "terminal": generated.status,
            "proposal": _comparable(generated.proposal),
            "searches": [_comparable(result) for result in replay_searches],
            "additions": replay_searches,
        }
    capture_hash = _capture_replay_hash(lock, captured)
    replay_hash = _capture_replay_hash(lock, replayed)
    match = capture_hash == replay_hash
    additions: dict[str, list[ProviderResult[list[Paper]]]] = {}
    for query_id in lock.query_ids:
        raw_additions = replayed[query_id].get("additions", [])
        additions[query_id] = cast(list[ProviderResult[list[Paper]]], raw_additions)
    baseline = reconstruct_frozen_baseline(frozen_inputs, None)
    projection = merge_probe_results(baseline, additions)
    gold = read_jsonl(DEFAULT_GOLD, EvaluationQuery)
    id_map = IdentifierMap.from_path(DEFAULT_ID_MAP)
    integrity = ProbeIntegrity(
        capture_replay_match="matched" if match else "mismatched",
        locked_query_count=baseline.expected_query_count,
        terminal_count=len(replayed),
        request_failures=sum(
            1 for query_id in lock.query_ids if replayed[query_id]["terminal"] in {"dependency_failure", "integrity_failure"}
        ),
        balanced_production_estimate=Decimal("0.01"),
        run_reason=None if match else "replay_mismatch",
    )
    evaluation = evaluate_probe(baseline=baseline, projection=projection, gold=gold, id_map=id_map, integrity=integrity)
    (run_dir / "outcomes.jsonl").write_text(
        "".join(_canonical_json({"query_id": query_id, **{key: value for key, value in captured[query_id].items() if key != "additions"}}).decode("utf-8") for query_id in lock.query_ids),
        encoding="utf-8",
    )
    (run_dir / "result.json").write_bytes(
        _canonical_json(
            {
                "schema_version": "query-evolution-probe-result-v1",
                "capture_business_sha256": capture_hash,
                "replay_business_sha256": replay_hash if match else None,
                "capture_replay_match": "matched" if match else "mismatched",
                "public_report": public_probe_report(evaluation).model_dump(mode="json"),
                "snapshot_manifest_sha256": capture_store.manifest_sha256,
                "snapshot_set_id": manifest.snapshot_set_id,
                "ledger_checkpoint_sha256": ledger.project_checkpoint()[1],
            }
        )
    )


def capture_probe(lock: ProbeLock, runtime: ProbeRuntime, reservations: ProbeReservations | None = None) -> CapturedProbe:
    del reservations
    if not runtime.allow_live:
        raise LiveNotAuthorized("bounded live capture requires --allow-live")
    raise NotImplementedError("capture_probe is only available through run_probe")


def replay_probe(lock: ProbeLock, replay_trace: ReplayTrace, snapshot_reader: object) -> ReplayedProbe:
    del snapshot_reader
    if replay_trace.query_ids != lock.query_ids:
        return ReplayedProbe(replay_business_sha256=None, capture_replay_match="mismatched")
    return ReplayedProbe(replay_business_sha256=None, capture_replay_match="not_evaluated")


def run_probe(lock_path: Path, runtime: ProbeRuntime) -> None:
    lock = load_probe_lock(lock_path)
    if not runtime.allow_live:
        raise LiveNotAuthorized("run --lock requires --allow-live")
    if lock.probe_code_sha256 != _probe_code_sha256():
        raise ValueError("probe code hash mismatch")
    _, prompt_instructions = _load_locked_prompt(lock)
    run_dir = _derive_run_directory(lock)
    if runtime.ledger_path.exists():
        ledger = SQLiteBudgetLedger(runtime.ledger_path, reservation_ttl_seconds=PROBE_LEDGER_TTL_SECONDS)
        if ledger.project_checkpoint()[1] != lock.ledger_checkpoint_sha256:
            raise ValueError("ledger checkpoint mismatch")
    lock_bytes = lock_path.read_bytes()
    if run_dir.exists():
        if not _can_resume_partial_run(run_dir):
            raise FileExistsError(f"probe output already exists: {run_dir}")
        (run_dir / "probe.lock.json").write_bytes(lock_bytes)
    else:
        run_dir.mkdir(parents=True)
        (run_dir / "probe.lock.json").write_bytes(lock_bytes)
    try:
        asyncio.run(
            _run_live_probe(
                lock,
                runtime,
                run_dir,
                prompt_sha=lock.prompt.sha256,
                prompt_instructions=prompt_instructions,
            )
        )
    except BaseException:
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probe_query_evolution")
    subparsers = parser.add_subparsers(dest="command")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--run", type=Path, default=DEFAULT_RUN)
    preflight.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    preflight.add_argument("--id-map", type=Path, default=DEFAULT_ID_MAP)
    preflight.add_argument("--availability", type=Path, default=DEFAULT_AVAILABILITY)
    preflight.add_argument("--prompt-config", type=Path, default=DEFAULT_PROMPT_CONFIG)
    preflight.add_argument("--budget-config", type=Path, default=DEFAULT_BUDGET_CONFIG)
    preflight.add_argument("--pricing-policy", type=Path, default=DEFAULT_PRICING_POLICY)
    preflight.add_argument("--ledger", type=Path, default=ROOT / "data" / "budget_ledger.sqlite3")
    preflight.add_argument("--out", type=Path, default=DEFAULT_LOCK)
    run = subparsers.add_parser("run")
    run.add_argument("--lock", type=Path, required=True)
    run.add_argument("--allow-live", action="store_true")
    run.add_argument("--env-file", type=Path, default=ProbeRuntime.model_fields["env_file"].default)
    run.add_argument("--ledger", type=Path, default=ProbeRuntime.model_fields["ledger_path"].default)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "preflight"
    try:
        if command == "preflight":
            for config_path in (args.budget_config, args.pricing_policy):
                if not config_path.is_file():
                    raise FileNotFoundError(f"preflight config missing: {config_path}")
            preflight_probe(
                frozen_run=args.run,
                gold_path=args.gold,
                id_map_path=args.id_map,
                availability_path=args.availability,
                prompt_config=args.prompt_config,
                probe_run_id="query-evolution-preflight",
                ledger_path=args.ledger,
                output_path=args.out,
            )
            return 0
        run_probe(
            args.lock,
            ProbeRuntime(
                allow_live=args.allow_live,
                env_file=args.env_file,
                ledger_path=args.ledger,
            ),
        )
        return 0
    except LiveNotAuthorized:
        return 2
    except (FileExistsError, NotImplementedError, OSError, ValueError) as error:
        print(f"probe failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
