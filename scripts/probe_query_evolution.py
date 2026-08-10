"""Offline-first bounded Query Evolution probe CLI and runtime boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from paper_search.control.ledger import DEV_RUN_CAP_CNY, LedgerReservation, SQLiteBudgetLedger
from paper_search.domain.models import DomainModel, UsageActual, UsageEstimate
from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs" / "dev-20260809T061903Z-9bd861e90299"
DEFAULT_GOLD = ROOT / "data" / "dev" / "gold.jsonl"
DEFAULT_ID_MAP = ROOT / "data" / "identifier-map.json"
DEFAULT_AVAILABILITY = ROOT / "docs" / "evidence" / "gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json"
DEFAULT_LOCK = ROOT / "runs" / "_diag_query_evolution_preflight" / "probe.lock.json"
EXPECTED_AVAILABILITY_SHA256 = "sha256:3f445486d5cf590f3f11a51930153a45916023880e856def379e0f01d053ad04"
PROBE_GLOBAL_TIMEOUT_SECONDS = 3600
PROBE_LEDGER_TTL_SECONDS = 3900
OPERATIONS = ("evolve", "search-1", "search-2")


class LiveNotAuthorized(RuntimeError):
    pass


class ProbeLock(DomainModel):
    schema_version: Literal["query-evolution-probe-lock-v1"] = "query-evolution-probe-lock-v1"
    preflight_complete: bool
    probe_run_id: str = Field(min_length=1)
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
    prompt_version: Literal["query-evolve-v1"]
    prompt_sha256: str
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


def _build_lock_payload(
    *,
    frozen_run: Path,
    gold_path: Path,
    id_map_path: Path,
    availability_path: Path,
    ledger_path: Path,
    output_path: Path,
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
    prompt_path = ROOT / "configs" / "prompts" / "query_evolve.yaml"
    if not prompt_path.exists():
        raise ValueError("query evolution prompt artifact is missing")
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
    empty_checkpoint = _sha256_file(ledger_path) if ledger_path.exists() else _sha256_bytes(b"")
    return {
        "schema_version": "query-evolution-probe-lock-v1",
        "preflight_complete": True,
        "probe_run_id": "query-evolution-preflight",
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
        "prompt_version": "query-evolve-v1",
        "prompt_sha256": _sha256_file(prompt_path),
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
        "expected_run_directory": "runs/_diag_query_evolution_query-evolution-preflight",
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
        ledger_path=ledger_path,
        output_path=output_path,
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


def reserve_probe_operations(lock: ProbeLock, ledger: SQLiteBudgetLedger) -> ProbeReservations:
    """Reserve every logical slot before any future live request."""
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


def capture_probe(lock: ProbeLock, runtime: ProbeRuntime, reservations: ProbeReservations | None = None) -> CapturedProbe:
    del reservations
    if not runtime.allow_live:
        raise LiveNotAuthorized("bounded live capture requires --allow-live")
    raise NotImplementedError("live capture is intentionally not executed in the offline implementation phase")


def replay_probe(lock: ProbeLock, replay_trace: ReplayTrace, snapshot_reader: object) -> ReplayedProbe:
    del snapshot_reader
    if replay_trace.query_ids != lock.query_ids:
        return ReplayedProbe(replay_business_sha256=None, capture_replay_match="mismatched")
    return ReplayedProbe(replay_business_sha256=None, capture_replay_match="not_evaluated")


def run_probe(lock_path: Path, runtime: ProbeRuntime) -> None:
    lock = load_probe_lock(lock_path)
    if not runtime.allow_live:
        raise LiveNotAuthorized("run --lock requires --allow-live")
    del lock
    raise NotImplementedError("live capture is outside the authorized offline phase")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probe_query_evolution")
    subparsers = parser.add_subparsers(dest="command")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--run", type=Path, default=DEFAULT_RUN)
    preflight.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    preflight.add_argument("--id-map", type=Path, default=DEFAULT_ID_MAP)
    preflight.add_argument("--availability", type=Path, default=DEFAULT_AVAILABILITY)
    preflight.add_argument("--ledger", type=Path, default=ROOT / "data" / "budget_ledger.sqlite3")
    preflight.add_argument("--out", type=Path, default=DEFAULT_LOCK)
    run = subparsers.add_parser("run")
    run.add_argument("--lock", type=Path, required=True)
    run.add_argument("--allow-live", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "preflight"
    try:
        if command == "preflight":
            preflight_probe(
                frozen_run=args.run,
                gold_path=args.gold,
                id_map_path=args.id_map,
                availability_path=args.availability,
                ledger_path=args.ledger,
                output_path=args.out,
            )
            return 0
        run_probe(args.lock, ProbeRuntime(allow_live=args.allow_live))
        return 0
    except LiveNotAuthorized:
        return 2
    except (FileExistsError, NotImplementedError, OSError, ValueError) as error:
        print(f"probe failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
