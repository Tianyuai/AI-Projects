from __future__ import annotations

import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

import scripts.probe_query_evolution as probe
import scripts.rescore_identifier_semantics as rescore
from scripts.probe_query_evolution import (
    DEFAULT_AVAILABILITY,
    DEFAULT_GOLD,
    DEFAULT_ID_MAP,
    DEFAULT_PROMPT_CONFIG,
    DEFAULT_RUN,
    LiveNotAuthorized,
    ProbeRuntime,
    load_probe_lock,
    preflight_probe,
    run_probe,
)


def _sha(value: str) -> str:
    return probe._sha256_bytes(value.encode("utf-8"))


def _run_id(label: str, root: Path) -> str:
    digest = hashlib.sha256(f"{label}:{root}".encode("utf-8")).hexdigest()[:12]
    return f"task4-{label}-{digest}"


def _synthetic_probe_lock(query_ids: tuple[str, ...]) -> probe.ProbeLock:
    prompt_bytes = DEFAULT_PROMPT_CONFIG.read_bytes()
    return probe.ProbeLock(
        preflight_complete=True,
        probe_run_id="task-3-probe-lock",
        source_run_id=DEFAULT_RUN.name,
        source_hashes={
            "business_results_sha256": _sha("business"),
            "executions_sha256": _sha("executions"),
            "run_sha256": _sha("run"),
            "snapshot_manifest_sha256": _sha("snapshot"),
        },
        source_git_sha="deadbeef",
        gold_sha256=_sha("gold"),
        identifier_map_sha256=_sha("identifier-map"),
        availability_sha256=_sha("availability"),
        query_ids=query_ids,
        query_count=60,
        total_selected=2910,
        baseline_candidate_gold_count=14,
        baseline_top50_gold_count=8,
        prompt=probe.ProbePromptBinding(
            path=DEFAULT_PROMPT_CONFIG.relative_to(probe.ROOT).as_posix(),
            sha256=probe._sha256_bytes(prompt_bytes),
            name="query_evolve",
            version="query-evolve-v2",
        ),
        model_id="deepseek-v4-flash",
        endpoint="https://api.deepseek.com/v1",
        probe_code_sha256=probe._probe_code_sha256(),
        limits={
            "query_count": 55,
            "llm_logical_operations": 55,
            "openalex_logical_operations": 110,
            "llm_attempts": 165,
            "openalex_attempts": 330,
            "global_timeout_seconds": probe.PROBE_GLOBAL_TIMEOUT_SECONDS,
            "ledger_ttl_seconds": probe.PROBE_LEDGER_TTL_SECONDS,
        },
        estimates={
            "evolve": {
                "llm_calls": 3,
                "input_tokens": 20000,
                "output_tokens": 4000,
                "cost_cny": "0.01",
                "elapsed_ms": 60000,
            },
            "search-1": {"search_api_calls": 3, "cost_cny": "0.01", "elapsed_ms": 60000},
            "search-2": {"search_api_calls": 3, "cost_cny": "0.01", "elapsed_ms": 60000},
        },
        ledger_checkpoint_sha256=_sha("ledger"),
        expected_run_directory="runs/_diag_query_evolution_task-3-probe-lock",
        lock_sha256=_sha("lock"),
    )


def _synthetic_raw_record(query_label: str, extra_topics: int, candidate_count: int = 2) -> dict[str, object]:
    topics = [f"{query_label}-topic-{index}" for index in range(extra_topics)]
    return {
        "business": {
            "query_id": query_label,
            "query_analysis": {
                "query_spec": {
                    "original_query": f"{query_label} original query",
                    "research_goal": f"{query_label} research goal",
                    "topics": topics,
                    "methods": [f"{query_label} method"],
                    "tasks": [f"{query_label} task"],
                    "datasets": [f"{query_label} dataset"],
                    "domains": [f"{query_label} domain"],
                    "venues": [f"{query_label} venue"],
                    "must_have": [f"{query_label} must-have"],
                    "should_have": [f"{query_label} should-have"],
                },
                "search_plan": {
                    "subqueries": [
                        {
                            "query_id": f"{query_label}-subquery-1",
                            "text": f"{query_label} exact",
                            "query_type": "exact",
                            "target_constraints": [f"{query_label} constraint 1"],
                            "priority": 1,
                            "provider_hint": "openalex",
                        },
                        {
                            "query_id": f"{query_label}-subquery-2",
                            "text": f"{query_label} expanded",
                            "query_type": "expanded",
                            "target_constraints": [f"{query_label} constraint 2"],
                            "priority": 2,
                            "provider_hint": "openalex",
                        },
                        {
                            "query_id": f"{query_label}-subquery-3",
                            "text": f"{query_label} decomposed",
                            "query_type": "decomposed",
                            "target_constraints": [f"{query_label} constraint 3"],
                            "priority": 3,
                            "provider_hint": "openalex",
                        },
                    ],
                    "inherited_hard_filters": {"from_year": 2020},
                    "rationale": f"{query_label} rationale",
                },
            },
            "selected_paper_ids": [f"{query_label}-paper-{index}" for index in range(candidate_count)],
        },
        "execution": {
            "query_id": query_label,
            "retrieved_paper_ids": [f"{query_label}-paper-{index}" for index in range(candidate_count)],
        },
    }


def _copy_frozen_run(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "snapshots").mkdir()
    for name in ("business-results.jsonl", "executions.jsonl", "run.json"):
        shutil.copy2(DEFAULT_RUN / name, destination / name)
    shutil.copy2(
        DEFAULT_RUN / "snapshots" / "snapshot-manifest.json",
        destination / "snapshots" / "snapshot-manifest.json",
    )


def _seed_settled_receipt(
    ledger: probe.SQLiteBudgetLedger,
    *,
    run_id: str = "prior-run",
    query_id: str = "prior:evolve",
) -> probe.LedgerReservation:
    estimate = probe.UsageEstimate(llm_calls=1, cost_cny=Decimal("0.01"))
    actual = probe.UsageActual(llm_calls=1, cost_cny=Decimal("0.001"))
    reservation = ledger.reserve(
        run_id=run_id,
        query_id=query_id,
        estimate=estimate,
        run_cap_cny=probe.DEV_RUN_CAP_CNY,
    )
    ledger.checkpoint_actual(reservation, actual)
    ledger.settle(reservation, actual)
    return reservation


def _write_canary_lock(
    tmp_path: Path,
    label: str,
    *,
    ledger_path: Path | None = None,
) -> tuple[Path, probe.CanaryLock, Path]:
    probe_lock_path = tmp_path / f"{label}-probe.lock.json"
    probe.preflight_probe(
        frozen_run=DEFAULT_RUN,
        gold_path=DEFAULT_GOLD,
        id_map_path=DEFAULT_ID_MAP,
        availability_path=DEFAULT_AVAILABILITY,
        prompt_config=DEFAULT_PROMPT_CONFIG,
        probe_run_id=_run_id(f"{label}-probe", tmp_path),
        ledger_path=tmp_path / f"{label}-probe-ledger.sqlite3",
        output_path=probe_lock_path,
    )
    canary_lock_path = tmp_path / f"{label}-canary.lock.json"
    canary_ledger_path = ledger_path or tmp_path / f"{label}-canary-ledger.sqlite3"
    lock = probe.preflight_canary(
        probe_lock_path=probe_lock_path,
        ledger_path=canary_ledger_path,
        canary_run_id=_run_id(label, tmp_path),
        output_path=canary_lock_path,
    )
    run_dir = (probe.ROOT / lock.expected_run_directory).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return canary_lock_path, lock, run_dir


def _install_mock_llm_client(
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.MockTransport,
) -> None:
    real_async_client = httpx.AsyncClient

    def build_async_client(*_: object, **kwargs: object) -> httpx.AsyncClient:
        timeout = kwargs.get("timeout")
        return real_async_client(transport=transport, timeout=timeout)

    monkeypatch.setattr(probe.httpx, "AsyncClient", build_async_client)


def _install_openalex_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    class GuardedEnv(dict[str, str]):
        def get(self, key: str, default: str | None = None) -> str | None:
            if key.startswith("OPENALEX"):
                raise AssertionError("canary must not read OpenAlex secrets")
            return super().get(key, default)

    def fail_live_capture_search_provider(*_: object, **__: object) -> None:
        raise AssertionError("canary must not construct LiveCaptureSearchProvider")

    monkeypatch.setattr(probe.os, "environ", GuardedEnv(), raising=False)
    monkeypatch.setattr(probe, "LiveCaptureSearchProvider", fail_live_capture_search_provider)


def _response_envelope(
    content: dict[str, object],
    *,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
) -> dict[str, object]:
    return {
        "id": "resp-1",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": json.dumps(content)}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def _mock_llm_transport(
    behaviors: list[str],
    call_log: list[str],
) -> httpx.MockTransport:
    state = {"index": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        message = json.loads(body["messages"][1]["content"])
        payload = message["payload"]
        facets = payload["facets"]
        behavior = behaviors[min(state["index"], len(behaviors) - 1)]
        state["index"] += 1
        call_log.append(behavior)
        if behavior == "generated":
            content = {
                "subqueries": [
                    {
                        "text": f"{facets[-1]} validation cohort {state['index']}",
                        "source_facets": [facets[-1]],
                        "strategy": "synonym",
                    }
                ],
                "no_op_reason": None,
            }
            return httpx.Response(200, json=_response_envelope(content))
        if behavior == "no_op":
            return httpx.Response(
                200,
                json=_response_envelope(
                    {"subqueries": [], "no_op_reason": "no_novel_query"}
                ),
            )
        if behavior == "integrity":
            content = {
                "subqueries": [
                    {
                        "text": f"{facets[-1]} invalid {state['index']}",
                        "source_facets": ["not-from-context"],
                        "strategy": "synonym",
                    }
                ],
                "no_op_reason": None,
            }
            return httpx.Response(200, json=_response_envelope(content))
        if behavior == "server_error":
            return httpx.Response(500, json={"error": {"message": "retry me"}})
        raise AssertionError(f"unexpected mock behavior: {behavior}")

    return httpx.MockTransport(handler)


def _tamper_canary_lock(
    path: Path,
    mutate: callable[[dict[str, object]], None],  # type: ignore[valid-type]
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    payload["lock_sha256"] = probe._self_hash(payload)
    path.write_text(probe._canonical_json(payload).decode("utf-8"), encoding="utf-8")


def _read_result(path: Path) -> dict[str, object]:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def _read_outcomes(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with (path / "outcomes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def test_offline_preflight_reconstructs_fixed_queue_and_self_hash(tmp_path: Path) -> None:
    output = tmp_path / "probe.lock.json"

    lock = preflight_probe(
        frozen_run=DEFAULT_RUN,
        gold_path=DEFAULT_GOLD,
        id_map_path=DEFAULT_ID_MAP,
        availability_path=DEFAULT_AVAILABILITY,
        prompt_config=DEFAULT_PROMPT_CONFIG,
        probe_run_id="query-evolution-preflight",
        ledger_path=tmp_path / "ledger.sqlite3",
        output_path=output,
    )

    assert lock.preflight_complete is True
    assert lock.query_count == 60
    assert lock.total_selected == 2910
    assert lock.baseline_candidate_gold_count == 14
    assert lock.baseline_top50_gold_count == 8
    assert len(lock.query_ids) == 55
    assert lock.query_ids == tuple(lock.query_ids)
    assert lock.probe_code_sha256.startswith("sha256:")
    assert lock.lock_sha256.startswith("sha256:")
    assert output.exists()
    assert load_probe_lock(output).lock_sha256 == lock.lock_sha256


def test_preflight_rejects_derived_baseline_gold_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = iter((13, 8))
    monkeypatch.setattr(probe, "count_gold_associations", lambda *_: next(counts))

    with pytest.raises(ValueError, match="baseline gold counts"):
        preflight_probe(
            frozen_run=DEFAULT_RUN,
            gold_path=DEFAULT_GOLD,
            id_map_path=DEFAULT_ID_MAP,
            availability_path=DEFAULT_AVAILABILITY,
            prompt_config=DEFAULT_PROMPT_CONFIG,
            probe_run_id="query-evolution-preflight-drift",
            ledger_path=tmp_path / "ledger.sqlite3",
            output_path=tmp_path / "probe.lock.json",
        )


def test_preflight_rejects_incomplete_execution_record_set(tmp_path: Path) -> None:
    frozen_run = tmp_path / "frozen-run"
    _copy_frozen_run(frozen_run)
    execution_path = frozen_run / "executions.jsonl"
    records = execution_path.read_text(encoding="utf-8").splitlines()
    execution_path.write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="execution record set"):
        preflight_probe(
            frozen_run=frozen_run,
            gold_path=DEFAULT_GOLD,
            id_map_path=DEFAULT_ID_MAP,
            availability_path=DEFAULT_AVAILABILITY,
            prompt_config=DEFAULT_PROMPT_CONFIG,
            probe_run_id="query-evolution-incomplete-executions",
            ledger_path=tmp_path / "ledger.sqlite3",
            output_path=tmp_path / "probe.lock.json",
        )


def test_preflight_rejects_missing_retrieved_stream(tmp_path: Path) -> None:
    frozen_run = tmp_path / "frozen-run"
    _copy_frozen_run(frozen_run)
    execution_path = frozen_run / "executions.jsonl"
    executions = [json.loads(line) for line in execution_path.read_text(encoding="utf-8").splitlines()]
    executions[0].pop("retrieved_paper_ids")
    execution_path.write_text(
        "".join(json.dumps(record) + "\n" for record in executions),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid retrieved IDs"):
        preflight_probe(
            frozen_run=frozen_run,
            gold_path=DEFAULT_GOLD,
            id_map_path=DEFAULT_ID_MAP,
            availability_path=DEFAULT_AVAILABILITY,
            prompt_config=DEFAULT_PROMPT_CONFIG,
            probe_run_id="query-evolution-missing-retrieved-stream",
            ledger_path=tmp_path / "ledger.sqlite3",
            output_path=tmp_path / "probe.lock.json",
        )


def test_frozen_inputs_read_the_lock_bound_source_run(tmp_path: Path) -> None:
    with TemporaryDirectory(dir=probe.ROOT / "runs") as source_directory:
        frozen_run = Path(source_directory)
        _copy_frozen_run(frozen_run)
        run_path = frozen_run / "run.json"
        run_record = json.loads(run_path.read_text(encoding="utf-8"))
        run_record["run_id"] = frozen_run.name
        run_path.write_text(json.dumps(run_record), encoding="utf-8")
        execution_path = frozen_run / "executions.jsonl"
        executions = [json.loads(line) for line in execution_path.read_text(encoding="utf-8").splitlines()]
        sentinel = "openalex:W999999999999"
        executions[0]["retrieved_paper_ids"].append(sentinel)
        execution_path.write_text(
            "".join(json.dumps(record) + "\n" for record in executions),
            encoding="utf-8",
        )
        lock_path = tmp_path / "bound-source.lock.json"
        lock = preflight_probe(
            frozen_run=frozen_run,
            gold_path=DEFAULT_GOLD,
            id_map_path=DEFAULT_ID_MAP,
            availability_path=DEFAULT_AVAILABILITY,
            prompt_config=DEFAULT_PROMPT_CONFIG,
            probe_run_id="query-evolution-bound-source",
            ledger_path=tmp_path / "ledger.sqlite3",
            output_path=lock_path,
        )

        frozen_inputs, _ = probe._frozen_inputs(lock)

        assert sentinel in frozen_inputs.queries[0].retrieved_paper_ids


def test_run_rejects_source_hash_drift_before_creating_output_or_live_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(dir=probe.ROOT / "runs") as source_directory:
        frozen_run = Path(source_directory)
        _copy_frozen_run(frozen_run)
        run_path = frozen_run / "run.json"
        run_record = json.loads(run_path.read_text(encoding="utf-8"))
        run_record["run_id"] = frozen_run.name
        run_path.write_text(json.dumps(run_record), encoding="utf-8")
        probe_run_id = _run_id("source-drift", tmp_path)
        lock_path = tmp_path / "source-drift.lock.json"
        lock = preflight_probe(
            frozen_run=frozen_run,
            gold_path=DEFAULT_GOLD,
            id_map_path=DEFAULT_ID_MAP,
            availability_path=DEFAULT_AVAILABILITY,
            prompt_config=DEFAULT_PROMPT_CONFIG,
            probe_run_id=probe_run_id,
            ledger_path=tmp_path / "ledger.sqlite3",
            output_path=lock_path,
        )
        execution_path = frozen_run / "executions.jsonl"
        execution_path.write_text(
            execution_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        run_dir = (probe.ROOT / lock.expected_run_directory).resolve()
        if run_dir.exists():
            shutil.rmtree(run_dir)
        live_started = False

        async def fail_live(*_: object, **__: object) -> None:
            nonlocal live_started
            live_started = True

        monkeypatch.setattr(probe, "_run_live_probe", fail_live)
        try:
            with pytest.raises(ValueError, match="frozen executions hash mismatch"):
                run_probe(
                    lock_path,
                    ProbeRuntime(
                        allow_live=True,
                        env_file=tmp_path / "unused.env",
                        ledger_path=tmp_path / "runtime-ledger.sqlite3",
                    ),
                )
            assert live_started is False
            assert not run_dir.exists()
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir)


@pytest.mark.parametrize(
    ("field", "label", "message"),
    [
        ("gold_sha256", "gold", "frozen gold hash mismatch"),
        ("identifier_map_sha256", "id-map", "frozen identifier map hash mismatch"),
        ("availability_sha256", "availability", "frozen availability hash mismatch"),
    ],
)
def test_run_rejects_external_evidence_hash_drift_before_output_or_live_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    label: str,
    message: str,
) -> None:
    probe_run_id = _run_id(label, tmp_path)
    lock_path = tmp_path / f"{field}.lock.json"
    lock = preflight_probe(
        frozen_run=DEFAULT_RUN,
        gold_path=DEFAULT_GOLD,
        id_map_path=DEFAULT_ID_MAP,
        availability_path=DEFAULT_AVAILABILITY,
        prompt_config=DEFAULT_PROMPT_CONFIG,
        probe_run_id=probe_run_id,
        ledger_path=tmp_path / "ledger.sqlite3",
        output_path=lock_path,
    )
    payload = lock.model_dump(mode="json")
    payload[field] = _sha("drifted")
    payload["lock_sha256"] = probe._self_hash(payload)
    lock_path.write_bytes(probe._canonical_json(payload))
    run_dir = (probe.ROOT / lock.expected_run_directory).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    live_started = False

    async def fail_live(*_: object, **__: object) -> None:
        nonlocal live_started
        live_started = True

    monkeypatch.setattr(probe, "_run_live_probe", fail_live)
    try:
        with pytest.raises(ValueError, match=message):
            run_probe(
                lock_path,
                ProbeRuntime(
                    allow_live=True,
                    env_file=tmp_path / "unused.env",
                    ledger_path=tmp_path / "runtime-ledger.sqlite3",
                ),
            )
        assert live_started is False
        assert not run_dir.exists()
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir)


def test_probe_integrity_uses_locked_queue_count() -> None:
    lock = _synthetic_probe_lock(("q1", "q2"))

    integrity = probe._probe_integrity(
        lock,
        capture_replay_match="matched",
        terminal_count=2,
        request_failures=0,
    )

    assert integrity.locked_query_count == 2
    assert integrity.terminal_count == 2


def test_lock_bytes_are_canonical_and_do_not_contain_gold_content(tmp_path: Path) -> None:
    output = tmp_path / "probe.lock.json"
    lock = preflight_probe(
        frozen_run=DEFAULT_RUN,
        gold_path=DEFAULT_GOLD,
        id_map_path=DEFAULT_ID_MAP,
        availability_path=DEFAULT_AVAILABILITY,
        prompt_config=DEFAULT_PROMPT_CONFIG,
        probe_run_id="query-evolution-preflight",
        ledger_path=tmp_path / "ledger.sqlite3",
        output_path=output,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["query_ids"] == list(lock.query_ids)
    serialized = output.read_text(encoding="utf-8").casefold()
    assert "relevant_paper_ids" not in serialized
    assert "gold_content" not in serialized
    assert "label" not in serialized


def test_run_requires_explicit_live_authorization_without_network(tmp_path: Path) -> None:
    output = tmp_path / "probe.lock.json"
    preflight_probe(
        frozen_run=DEFAULT_RUN,
        gold_path=DEFAULT_GOLD,
        id_map_path=DEFAULT_ID_MAP,
        availability_path=DEFAULT_AVAILABILITY,
        prompt_config=DEFAULT_PROMPT_CONFIG,
        probe_run_id="query-evolution-preflight",
        ledger_path=tmp_path / "ledger.sqlite3",
        output_path=output,
    )

    with pytest.raises(LiveNotAuthorized):
        run_probe(output, ProbeRuntime(allow_live=False))


def test_preflight_rejects_availability_hash_drift(tmp_path: Path) -> None:
    drifted = tmp_path / "availability.json"
    drifted.write_text(DEFAULT_AVAILABILITY.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="availability"):
        preflight_probe(
            frozen_run=DEFAULT_RUN,
            gold_path=DEFAULT_GOLD,
            id_map_path=DEFAULT_ID_MAP,
            availability_path=drifted,
            prompt_config=DEFAULT_PROMPT_CONFIG,
            probe_run_id="query-evolution-preflight",
            ledger_path=tmp_path / "ledger.sqlite3",
            output_path=tmp_path / "probe.lock.json",
        )


def test_preflight_stores_caller_supplied_prompt_binding(tmp_path: Path) -> None:
    with TemporaryDirectory(dir=probe.ROOT) as prompt_dir:
        prompt_copy = Path(prompt_dir) / "query_evolve_copy.yaml"
        prompt_copy.write_bytes(DEFAULT_PROMPT_CONFIG.read_bytes())
        relative_prompt = prompt_copy.relative_to(probe.ROOT)
        output = tmp_path / "probe.lock.json"

        lock = preflight_probe(
            frozen_run=DEFAULT_RUN,
            gold_path=DEFAULT_GOLD,
            id_map_path=DEFAULT_ID_MAP,
            availability_path=DEFAULT_AVAILABILITY,
            prompt_config=relative_prompt,
            probe_run_id="task-2-binding",
            ledger_path=tmp_path / "ledger.sqlite3",
            output_path=output,
        )

        assert lock.schema_version == "query-evolution-probe-lock-v2"
        assert lock.prompt.path == relative_prompt.as_posix()
        assert lock.prompt.sha256 == probe._sha256_bytes(prompt_copy.read_bytes())
        assert lock.prompt.name == "query_evolve"
        assert lock.prompt.version == "query-evolve-v2"
        assert lock.expected_run_directory == "runs/_diag_query_evolution_task-2-binding"
        assert load_probe_lock(output).prompt == lock.prompt


def test_preflight_rejects_prompt_path_traversal(tmp_path: Path) -> None:
    with TemporaryDirectory() as prompt_dir:
        escaped_prompt = Path(prompt_dir) / "query_evolve_escape.yaml"
        escaped_prompt.write_bytes(DEFAULT_PROMPT_CONFIG.read_bytes())
        prompt_config = Path("..") / escaped_prompt.name

        with pytest.raises(ValueError, match="prompt"):
            preflight_probe(
                frozen_run=DEFAULT_RUN,
                gold_path=DEFAULT_GOLD,
                id_map_path=DEFAULT_ID_MAP,
                availability_path=DEFAULT_AVAILABILITY,
                prompt_config=prompt_config,
                probe_run_id="task-2-traversal",
                ledger_path=tmp_path / "ledger.sqlite3",
                output_path=tmp_path / "probe.lock.json",
            )


def test_preflight_rejects_prompt_name_drift(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="query_evolve"):
        preflight_probe(
            frozen_run=DEFAULT_RUN,
            gold_path=DEFAULT_GOLD,
            id_map_path=DEFAULT_ID_MAP,
            availability_path=DEFAULT_AVAILABILITY,
            prompt_config=Path("configs/prompts/query_analyze.yaml"),
            probe_run_id="task-2-name-drift",
            ledger_path=tmp_path / "ledger.sqlite3",
            output_path=tmp_path / "probe.lock.json",
        )


def test_run_rejects_tampered_expected_run_directory_before_live_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "probe.lock.json"
    lock = preflight_probe(
        frozen_run=DEFAULT_RUN,
        gold_path=DEFAULT_GOLD,
        id_map_path=DEFAULT_ID_MAP,
        availability_path=DEFAULT_AVAILABILITY,
        prompt_config=DEFAULT_PROMPT_CONFIG,
        probe_run_id="task-2-run-dir",
        ledger_path=tmp_path / "ledger.sqlite3",
        output_path=output,
    )
    payload = lock.model_dump(mode="json")
    payload["expected_run_directory"] = "runs/_diag_query_evolution_other"
    payload["lock_sha256"] = probe._self_hash(payload)
    output.write_text(
        probe._canonical_json(payload).decode("utf-8"),
        encoding="utf-8",
    )
    live_probe_started = False

    async def fail_live_probe(*_: object, **__: object) -> None:
        nonlocal live_probe_started
        live_probe_started = True
        raise AssertionError("live probe should not start")

    monkeypatch.setattr(probe, "_run_live_probe", fail_live_probe)

    with pytest.raises(ValueError, match="expected run directory mismatch"):
        run_probe(
            output,
            ProbeRuntime(
                allow_live=True,
                env_file=tmp_path / "unused.env",
                ledger_path=tmp_path / "ledger.sqlite3",
            ),
        )

    assert live_probe_started is False


def test_run_rejects_expected_run_directory_path_traversal_before_live_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "probe.lock.json"
    lock = preflight_probe(
        frozen_run=DEFAULT_RUN,
        gold_path=DEFAULT_GOLD,
        id_map_path=DEFAULT_ID_MAP,
        availability_path=DEFAULT_AVAILABILITY,
        prompt_config=DEFAULT_PROMPT_CONFIG,
        probe_run_id="task-2-run-dir",
        ledger_path=tmp_path / "ledger.sqlite3",
        output_path=output,
    )
    payload = lock.model_dump(mode="json")
    payload["expected_run_directory"] = "..\\..\\outside"
    payload["lock_sha256"] = probe._self_hash(payload)
    output.write_text(
        probe._canonical_json(payload).decode("utf-8"),
        encoding="utf-8",
    )
    live_probe_started = False

    async def fail_live_probe(*_: object, **__: object) -> None:
        nonlocal live_probe_started
        live_probe_started = True
        raise AssertionError("live probe should not start")

    monkeypatch.setattr(probe, "_run_live_probe", fail_live_probe)

    with pytest.raises(ValueError, match="expected run directory mismatch"):
        run_probe(
            output,
            ProbeRuntime(
                allow_live=True,
                env_file=tmp_path / "unused.env",
                ledger_path=tmp_path / "ledger.sqlite3",
            ),
        )

    assert live_probe_started is False


def test_preflight_rejects_prompt_version_drift(tmp_path: Path) -> None:
    with TemporaryDirectory(dir=probe.ROOT) as prompt_dir:
        prompt_copy = Path(prompt_dir) / "query_evolve_version_drift.yaml"
        prompt_copy.write_text(
            DEFAULT_PROMPT_CONFIG.read_text(encoding="utf-8").replace(
                "version: query-evolve-v2",
                "version: query-evolve-v1",
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="query_evolve prompt version"):
            preflight_probe(
                frozen_run=DEFAULT_RUN,
                gold_path=DEFAULT_GOLD,
                id_map_path=DEFAULT_ID_MAP,
                availability_path=DEFAULT_AVAILABILITY,
                prompt_config=prompt_copy.relative_to(probe.ROOT),
                probe_run_id="task-2-version-drift",
                ledger_path=tmp_path / "ledger.sqlite3",
                output_path=tmp_path / "probe.lock.json",
            )


def test_run_rejects_locked_prompt_hash_drift_before_live_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(dir=probe.ROOT) as prompt_dir:
        prompt_copy = Path(prompt_dir) / "query_evolve_hash_drift.yaml"
        prompt_copy.write_bytes(DEFAULT_PROMPT_CONFIG.read_bytes())
        output = tmp_path / "probe.lock.json"
        preflight_probe(
            frozen_run=DEFAULT_RUN,
            gold_path=DEFAULT_GOLD,
            id_map_path=DEFAULT_ID_MAP,
            availability_path=DEFAULT_AVAILABILITY,
            prompt_config=prompt_copy.relative_to(probe.ROOT),
            probe_run_id="task-2-hash-drift",
            ledger_path=tmp_path / "ledger.sqlite3",
            output_path=output,
        )
        prompt_copy.write_text(
            prompt_copy.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        live_probe_started = False
        ledger_path = tmp_path / "runtime-ledger.sqlite3"

        async def fail_live_probe(*_: object, **__: object) -> None:
            nonlocal live_probe_started
            live_probe_started = True
            raise AssertionError("live probe should not start")

        monkeypatch.setattr(probe, "_run_live_probe", fail_live_probe)

        with pytest.raises(ValueError, match="locked prompt hash mismatch"):
            run_probe(
                output,
                ProbeRuntime(
                    allow_live=True,
                    env_file=tmp_path / "unused.env",
                    ledger_path=ledger_path,
                ),
            )

        assert live_probe_started is False
        assert not ledger_path.exists()


def test_canary_selects_minimum_median_and_maximum_canonical_payload() -> None:
    assert hasattr(probe, "select_canary_query_ids")
    assert hasattr(probe, "build_probe_context")

    lock = _synthetic_probe_lock(("qa", "qb", "qc", "qd", "qe"))
    raw_records = {
        "qa": _synthetic_raw_record("qa", extra_topics=1),
        "qb": _synthetic_raw_record("qa", extra_topics=1),
        "qc": _synthetic_raw_record("qc", extra_topics=3),
        "qd": _synthetic_raw_record("qd", extra_topics=5),
        "qe": _synthetic_raw_record("qe", extra_topics=7),
    }

    selected = probe.select_canary_query_ids(lock, raw_records)
    ranked = sorted(
        lock.query_ids,
        key=lambda query_id: (
            len(
                probe._canonical_json(
                    probe.build_probe_context(query_id, raw_records[query_id]).model_dump(mode="json")
                )
            ),
            query_id,
        ),
    )

    assert ranked[:2] == ["qa", "qb"]
    assert selected == (ranked[0], ranked[len(ranked) // 2], ranked[-1])
    assert len(set(selected)) == 3


def test_canary_preflight_is_gold_blind_and_writes_self_hashed_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert hasattr(probe, "preflight_canary")

    probe_lock_path = tmp_path / "probe.lock.json"
    probe_lock = preflight_probe(
        frozen_run=DEFAULT_RUN,
        gold_path=DEFAULT_GOLD,
        id_map_path=DEFAULT_ID_MAP,
        availability_path=DEFAULT_AVAILABILITY,
        prompt_config=DEFAULT_PROMPT_CONFIG,
        probe_run_id="task-3-source-lock",
        ledger_path=tmp_path / "probe-ledger.sqlite3",
        output_path=probe_lock_path,
    )

    def fail_read_jsonl(*_: object, **__: object) -> None:
        raise AssertionError("canary preflight must not read gold data")

    def fail_identifier_map(*_: object, **__: object) -> None:
        raise AssertionError("canary preflight must not read identifier data")

    monkeypatch.setattr(probe, "read_jsonl", fail_read_jsonl)
    monkeypatch.setattr(probe.IdentifierMap, "from_path", fail_identifier_map)

    output = tmp_path / "canary.lock.json"
    lock = probe.preflight_canary(
        probe_lock_path=probe_lock_path,
        ledger_path=tmp_path / "canary-ledger.sqlite3",
        canary_run_id="task-3-canary-lock",
        output_path=output,
    )

    assert lock.schema_version == "query-evolution-contract-canary-lock-v1"
    assert lock.source_probe_lock_sha256 == probe._sha256_bytes(probe_lock_path.read_bytes())
    assert lock.source_run_id == probe_lock.source_run_id
    assert lock.prompt == probe_lock.prompt
    assert lock.query_ids == tuple(lock.query_ids)
    assert len(lock.query_ids) == 3
    assert lock.limits.query_count == 3
    assert lock.limits.llm_logical_operations == 3
    assert lock.limits.llm_attempts == 9
    assert lock.limits.global_timeout_seconds == 600
    assert output.exists()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["lock_sha256"] == lock.lock_sha256
    assert probe._self_hash(payload) == lock.lock_sha256


def test_canary_run_promotes_three_offline_outcomes_without_openalex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock, run_dir = _write_canary_lock(tmp_path, "promote")
    env_file = tmp_path / "canary.env"
    env_file.write_text("LLM_API_KEY=test-llm-key\n", encoding="utf-8")
    calls: list[str] = []
    _install_mock_llm_client(
        monkeypatch,
        _mock_llm_transport(["generated", "no_op", "generated"], calls),
    )
    _install_openalex_guards(monkeypatch)

    probe.run_canary(
        lock_path,
        ProbeRuntime(
            allow_live=True,
            env_file=env_file,
            ledger_path=tmp_path / "ledger.sqlite3",
        ),
    )

    assert calls == ["generated", "no_op", "generated"]
    assert run_dir.exists()
    assert {item.name for item in run_dir.iterdir()} == {
        "canary.lock.json",
        "outcomes.jsonl",
        "result.json",
        "snapshots",
    }
    assert (run_dir / "canary.lock.json").read_bytes() == lock_path.read_bytes()
    assert not list(run_dir.rglob("*.tmp"))

    outcomes = _read_outcomes(run_dir)
    manifest = json.loads(
        (run_dir / "snapshots" / "snapshot-manifest.json").read_text(encoding="utf-8")
    )
    manifest_entry_ids = {entry["entry_id"] for entry in manifest["entries"]}
    assert [record["terminal"] for record in outcomes] == ["generated", "no_op", "generated"]
    for record in outcomes:
        for snapshot_ref in record["snapshot_refs"]:
            assert snapshot_ref["entry_id"] in manifest_entry_ids

    result = _read_result(run_dir)
    assert result["reason"] == "passed"
    assert result["promoted"] is True
    assert result["terminal_counts"] == {"generated": 2, "no_op": 1, "failed": 0}
    assert result["aggregate_usage"]["llm_calls"] == 3
    assert result["snapshot_manifest_sha256"].startswith("sha256:")
    assert result["snapshot_set_id"].startswith("sha256:")
    assert result["ledger_checkpoint_sha256"].startswith("sha256:")
    assert lock.query_ids == tuple(record["query_id"] for record in outcomes)

    ledger = probe.SQLiteBudgetLedger(
        tmp_path / "ledger.sqlite3",
        reservation_ttl_seconds=probe.PROBE_LEDGER_TTL_SECONDS,
    )
    receipts = ledger.report(lock.canary_run_id).receipts
    assert len(receipts) == 3
    assert all(receipt.state in {"settled", "failed"} for receipt in receipts)
    assert all(receipt.query_id.endswith(":evolve") for receipt in receipts)


def test_canary_run_records_contract_failure_for_strict_schema_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, _, run_dir = _write_canary_lock(tmp_path, "integrity")
    env_file = tmp_path / "canary.env"
    env_file.write_text("LLM_API_KEY=test-llm-key\n", encoding="utf-8")
    calls: list[str] = []
    _install_mock_llm_client(
        monkeypatch,
        _mock_llm_transport(["generated", "integrity", "no_op"], calls),
    )
    _install_openalex_guards(monkeypatch)

    probe.run_canary(
        lock_path,
        ProbeRuntime(
            allow_live=True,
            env_file=env_file,
            ledger_path=tmp_path / "ledger.sqlite3",
        ),
    )

    result = _read_result(run_dir)
    assert calls == ["generated", "integrity", "no_op"]
    assert result["reason"] == "contract_canary_failed"
    assert result["promoted"] is False
    assert result["terminal_counts"]["failed"] == 1


@pytest.mark.parametrize(
    ("behaviors", "expected_reason", "expected_promoted"),
    [
        pytest.param(
            ["generated", "no_op", "generated"],
            "passed",
            True,
            id="promoted",
        ),
        pytest.param(
            ["generated", "integrity", "generated"],
            "contract_canary_failed",
            False,
            id="contract-failed",
        ),
    ],
)
def test_canary_uses_only_current_run_receipts_when_project_has_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behaviors: list[str],
    expected_reason: str,
    expected_promoted: bool,
) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = probe.SQLiteBudgetLedger(
        ledger_path,
        reservation_ttl_seconds=probe.PROBE_LEDGER_TTL_SECONDS,
    )
    prior = _seed_settled_receipt(ledger)
    lock_path, lock, run_dir = _write_canary_lock(
        tmp_path,
        "project-history",
        ledger_path=ledger_path,
    )
    env_file = tmp_path / "canary.env"
    env_file.write_text("LLM_API_KEY=test-llm-key\n", encoding="utf-8")
    _install_mock_llm_client(monkeypatch, _mock_llm_transport(behaviors, []))
    _install_openalex_guards(monkeypatch)

    probe.run_canary(
        lock_path,
        ProbeRuntime(
            allow_live=True,
            env_file=env_file,
            ledger_path=ledger_path,
        ),
    )

    result = _read_result(run_dir)
    receipts = ledger.report(lock.canary_run_id).receipts
    current = [receipt for receipt in receipts if receipt.run_id == lock.canary_run_id]
    assert result["reason"] == expected_reason
    assert result["promoted"] is expected_promoted
    assert len(receipts) == 4
    assert len(current) == 3
    assert all(receipt.state == "settled" for receipt in current)
    assert (
        next(
            receipt
            for receipt in receipts
            if receipt.reservation_id == prior.reservation_id
        ).state
        == "settled"
    )


def test_canary_reservation_recovery_ignores_other_runs(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = probe.SQLiteBudgetLedger(ledger_path)
    _seed_settled_receipt(ledger)
    _, lock, _ = _write_canary_lock(
        tmp_path,
        "canary-resume",
        ledger_path=ledger_path,
    )
    created = probe.reserve_canary_operations(lock, ledger)

    restored = probe.reserve_canary_operations(lock, ledger)

    assert set(restored) == set(created)
    assert {
        key: value.reservation_id for key, value in restored.items()
    } == {
        key: value.reservation_id for key, value in created.items()
    }


def test_full_probe_reservation_recovery_ignores_other_runs(tmp_path: Path) -> None:
    ledger = probe.SQLiteBudgetLedger(tmp_path / "ledger.sqlite3")
    _seed_settled_receipt(ledger)
    lock = _synthetic_probe_lock(("q1", "q2"))
    created = probe.reserve_probe_operations(lock, ledger)

    restored = probe.reserve_probe_operations(lock, ledger)

    assert {
        key: value.reservation_id for key, value in restored.values.items()
    } == {
        key: value.reservation_id for key, value in created.values.items()
    }


def test_canary_run_stops_after_nine_retryable_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, _, run_dir = _write_canary_lock(tmp_path, "dependency")
    env_file = tmp_path / "canary.env"
    env_file.write_text("LLM_API_KEY=test-llm-key\n", encoding="utf-8")
    calls: list[str] = []
    _install_mock_llm_client(
        monkeypatch,
        _mock_llm_transport(["server_error"] * 9, calls),
    )
    _install_openalex_guards(monkeypatch)

    probe.run_canary(
        lock_path,
        ProbeRuntime(
            allow_live=True,
            env_file=env_file,
            ledger_path=tmp_path / "ledger.sqlite3",
        ),
    )

    result = _read_result(run_dir)
    assert len(calls) == 9
    assert result["reason"] == "canary_dependency_failed"
    assert result["promoted"] is False


def test_canary_run_marks_accounting_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = probe.SQLiteBudgetLedger(
        ledger_path,
        reservation_ttl_seconds=probe.PROBE_LEDGER_TTL_SECONDS,
    )
    prior = _seed_settled_receipt(ledger)
    lock_path, lock, run_dir = _write_canary_lock(
        tmp_path,
        "accounting",
        ledger_path=ledger_path,
    )
    env_file = tmp_path / "canary.env"
    env_file.write_text("LLM_API_KEY=test-llm-key\n", encoding="utf-8")
    _install_mock_llm_client(
        monkeypatch,
        _mock_llm_transport(["generated", "generated", "generated"], []),
    )
    _install_openalex_guards(monkeypatch)

    def fail_settlement(*_: object, **__: object) -> None:
        raise RuntimeError("ledger broke")

    monkeypatch.setattr(probe, "_settle_ledger", fail_settlement)

    probe.run_canary(
        lock_path,
        ProbeRuntime(
            allow_live=True,
            env_file=env_file,
            ledger_path=ledger_path,
        ),
    )

    result = _read_result(run_dir)
    assert result["reason"] == "canary_accounting_failed"
    assert result["promoted"] is False
    receipts = ledger.report(lock.canary_run_id).receipts
    current = [receipt for receipt in receipts if receipt.run_id == lock.canary_run_id]
    assert len(receipts) == 4
    assert len(current) == 3
    assert all(receipt.state == "failed" for receipt in current)
    assert all(receipt.actual is not None for receipt in current)
    assert next(receipt for receipt in receipts if receipt.reservation_id == prior.reservation_id).state == "settled"


def test_canary_run_marks_snapshot_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, _, run_dir = _write_canary_lock(tmp_path, "snapshot")
    env_file = tmp_path / "canary.env"
    env_file.write_text("LLM_API_KEY=test-llm-key\n", encoding="utf-8")
    _install_mock_llm_client(
        monkeypatch,
        _mock_llm_transport(["generated", "generated", "generated"], []),
    )
    _install_openalex_guards(monkeypatch)

    def fail_seal(self: object) -> None:
        raise OSError("seal failed")

    monkeypatch.setattr(probe.DependencyCaptureStore, "seal", fail_seal)

    probe.run_canary(
        lock_path,
        ProbeRuntime(
            allow_live=True,
            env_file=env_file,
            ledger_path=tmp_path / "ledger.sqlite3",
        ),
    )

    result = _read_result(run_dir)
    assert result["reason"] == "canary_snapshot_failed"
    assert result["promoted"] is False


def test_canary_run_marks_timeout_as_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = probe.SQLiteBudgetLedger(
        ledger_path,
        reservation_ttl_seconds=probe.PROBE_LEDGER_TTL_SECONDS,
    )
    prior = _seed_settled_receipt(ledger)
    lock_path, lock, run_dir = _write_canary_lock(
        tmp_path,
        "cancelled",
        ledger_path=ledger_path,
    )
    env_file = tmp_path / "canary.env"
    env_file.write_text("LLM_API_KEY=test-llm-key\n", encoding="utf-8")
    calls: list[str] = []
    _install_mock_llm_client(
        monkeypatch,
        _mock_llm_transport(["generated", "generated", "generated"], calls),
    )
    _install_openalex_guards(monkeypatch)

    class ExplodingTimeout:
        async def __aenter__(self) -> None:
            raise TimeoutError("expired")

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    monkeypatch.setattr(probe.asyncio, "timeout", lambda _: ExplodingTimeout())

    probe.run_canary(
        lock_path,
        ProbeRuntime(
            allow_live=True,
            env_file=env_file,
            ledger_path=ledger_path,
        ),
    )

    result = _read_result(run_dir)
    assert calls == []
    assert result["reason"] == "canary_cancelled"
    assert result["promoted"] is False
    receipts = ledger.report(lock.canary_run_id).receipts
    current = [receipt for receipt in receipts if receipt.run_id == lock.canary_run_id]
    assert len(receipts) == 4
    assert len(current) == 3
    assert all(receipt.state == "failed" for receipt in current)
    assert all(receipt.actual is not None for receipt in current)
    assert next(receipt for receipt in receipts if receipt.reservation_id == prior.reservation_id).state == "settled"


@pytest.mark.parametrize(
    ("label", "mutate", "expected_reason"),
    [
        (
            "source-hash",
            lambda payload: payload["source_hashes"].__setitem__(  # type: ignore[index]
                "business_results_sha256", _sha("drifted")
            ),
            "canary_preflight_failed",
        ),
        (
            "probe-code",
            lambda payload: payload.__setitem__("probe_code_sha256", _sha("drifted")),
            "canary_preflight_failed",
        ),
        (
            "prompt",
            lambda payload: payload["prompt"].__setitem__("sha256", _sha("drifted")),  # type: ignore[index]
            "prompt_binding_failed",
        ),
        (
            "prompt-version",
            lambda payload: payload["prompt"].__setitem__("version", "query-evolve-v1"),  # type: ignore[index]
            "prompt_binding_failed",
        ),
        (
            "limits",
            lambda payload: payload["limits"].__setitem__("llm_attempts", 8),  # type: ignore[index]
            "canary_preflight_failed",
        ),
        (
            "checkpoint",
            lambda payload: payload.__setitem__("ledger_checkpoint_sha256", _sha("drifted")),
            "canary_preflight_failed",
        ),
    ],
)
def test_canary_run_rejects_lock_drift_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    mutate: callable[[dict[str, object]], None],  # type: ignore[valid-type]
    expected_reason: str,
) -> None:
    lock_path, lock, run_dir = _write_canary_lock(tmp_path, label)
    env_file = tmp_path / "canary.env"
    env_file.write_text("LLM_API_KEY=test-llm-key\n", encoding="utf-8")
    _tamper_canary_lock(lock_path, mutate)

    def fail_async_client(*_: object, **__: object) -> None:
        raise AssertionError("network must not start for pre-dispatch drift")

    _install_openalex_guards(monkeypatch)
    monkeypatch.setattr(probe.httpx, "AsyncClient", fail_async_client)

    ledger_path = tmp_path / "runtime-ledger.sqlite3"

    probe.run_canary(
        lock_path,
        ProbeRuntime(
            allow_live=True,
            env_file=env_file,
            ledger_path=ledger_path,
        ),
    )

    assert not ledger_path.exists()
    result = _read_result(run_dir)
    assert result["reason"] == expected_reason
    assert result["promoted"] is False
    assert result["aggregate_usage"]["llm_calls"] == 0
    if expected_reason == "canary_preflight_failed":
        assert result["snapshot_manifest_sha256"] is None
        assert result["snapshot_set_id"] is None
    assert lock.canary_run_id in run_dir.as_posix()


def test_canary_cli_requires_explicit_live_flag(tmp_path: Path) -> None:
    lock_path, _, _ = _write_canary_lock(tmp_path, "cli-auth")
    env_file = tmp_path / "canary.env"
    env_file.write_text("LLM_API_KEY=test-llm-key\n", encoding="utf-8")

    assert (
        probe.main(
            [
                "canary-run",
                "--lock",
                str(lock_path),
                "--env-file",
                str(env_file),
                "--ledger",
                str(tmp_path / "ledger.sqlite3"),
            ]
        )
        == 2
    )


def test_canary_cli_returns_zero_only_for_promoted_canary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, _, _ = _write_canary_lock(tmp_path, "cli-success")
    env_file = tmp_path / "canary.env"
    env_file.write_text("LLM_API_KEY=test-llm-key\n", encoding="utf-8")
    _install_mock_llm_client(
        monkeypatch,
        _mock_llm_transport(["generated", "no_op", "generated"], []),
    )
    _install_openalex_guards(monkeypatch)

    assert (
        probe.main(
            [
                "canary-run",
                "--lock",
                str(lock_path),
                "--env-file",
                str(env_file),
                "--ledger",
                str(tmp_path / "ledger.sqlite3"),
                "--allow-live",
            ]
        )
        == 0
    )


SEALED_PROBE = (
    Path(__file__).resolve().parents[2]
    / "runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810"
)


def _expected_rescore_query_ids() -> tuple[str, ...]:
    return tuple(
        json.loads(line)["query_id"]
        for line in DEFAULT_GOLD.read_text(encoding="utf-8").splitlines()
    )


def _copy_probe_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    destination = (
        tmp_path
        / "runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810"
    )
    shutil.copytree(SEALED_PROBE, destination)
    monkeypatch.setattr(rescore, "ROOT", tmp_path)
    return destination


def _rewrite_probe_json(path: Path, update) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    update(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rewrite_probe_outcomes(path: Path, update) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    update(rows)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_formal_source_validation_runs_before_artifact_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ValidationSentinel(RuntimeError):
        pass

    monkeypatch.setattr(
        rescore,
        "validate_run_directory",
        lambda path: (_ for _ in ()).throw(ValidationSentinel(path)),
    )

    with pytest.raises(ValidationSentinel):
        rescore.load_formal_source(
            "formal_baseline_2026_08_10",
            tmp_path / "missing-run",
            ("q-1",),
        )


@pytest.mark.parametrize("artifact", ["business-results.jsonl", "executions.jsonl"])
def test_formal_source_requires_exact_business_and_execution_query_order(
    artifact: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).resolve().parents[2] / "runs/dev-20260810T104256Z-d9e89476d484"
    run_dir = tmp_path / "formal"
    run_dir.mkdir()
    for name in (
        "run.json",
        "gates.json",
        "predictions.jsonl",
        "executions.jsonl",
        "business-results.jsonl",
    ):
        shutil.copy2(source / name, run_dir / name)
    lines = (run_dir / artifact).read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    (run_dir / artifact).write_text("".join(lines), encoding="utf-8")
    monkeypatch.setattr(
        rescore,
        "validate_run_directory",
        lambda path: SimpleNamespace(valid=True, issues=()),
    )

    with pytest.raises(ValueError, match="query order"):
        rescore.load_formal_source(
            "formal_baseline_2026_08_10",
            run_dir,
            _expected_rescore_query_ids(),
        )


def test_formal_source_rejects_business_execution_query_set_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).resolve().parents[2] / "runs/dev-20260810T104256Z-d9e89476d484"
    run_dir = tmp_path / "formal"
    run_dir.mkdir()
    for name in (
        "run.json",
        "gates.json",
        "predictions.jsonl",
        "executions.jsonl",
        "business-results.jsonl",
    ):
        shutil.copy2(source / name, run_dir / name)
    path = run_dir / "executions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["query_id"] = rows[1]["query_id"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(
        rescore,
        "validate_run_directory",
        lambda path: SimpleNamespace(valid=True, issues=()),
    )

    with pytest.raises(ValueError, match="query IDs"):
        rescore.load_formal_source(
            "formal_baseline_2026_08_10",
            run_dir,
            _expected_rescore_query_ids(),
        )


def test_legacy_source_rejects_hash_drift_before_parsing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "runs/dev-20260805T035209Z-7af4b103f6cc"
    run_dir = tmp_path / source.name
    run_dir.mkdir()
    for name in ("business-results.jsonl", "executions.jsonl"):
        shutil.copy2(source / name, run_dir / name)
    evidence = tmp_path / "evidence.json"
    shutil.copy2(
        root / "docs/evidence/title-retention-offline-2026-08-09.json",
        evidence,
    )
    with (run_dir / "business-results.jsonl").open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="business results hash mismatch"):
        rescore.load_legacy_source(
            run_dir,
            evidence,
            _expected_rescore_query_ids(),
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda value: value.__setitem__("schema_version", "unknown"), "result"),
        (
            lambda value: value.__setitem__("capture_replay_match", "mismatched"),
            "capture_replay_match",
        ),
        (
            lambda value: value.__setitem__(
                "replay_business_sha256", "sha256:" + "0" * 64
            ),
            "business hashes",
        ),
        (lambda value: value.__setitem__("unexpected", True), "result"),
        (
            lambda value: value.__setitem__(
                "snapshot_set_id", "sha256:" + "0" * 64
            ),
            "snapshot set identity",
        ),
    ],
)
def test_probe_source_requires_closed_matched_result_contract(
    update,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _copy_probe_source(tmp_path, monkeypatch)
    _rewrite_probe_json(run_dir / "result.json", update)

    with pytest.raises(ValueError, match=message):
        rescore.load_probe_source(run_dir, _expected_rescore_query_ids())


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda value: value.__setitem__("source_git_sha", "changed"), "self-hash"),
        (
            lambda value: value.__setitem__(
                "expected_run_directory", "runs/unexpected-probe"
            ),
            "expected directory",
        ),
        (
            lambda value: value["source_hashes"].__setitem__(
                "business_results_sha256", "sha256:" + "0" * 64
            ),
            "business results hash mismatch",
        ),
    ],
)
def test_probe_source_requires_lock_identity_directory_and_source_hashes(
    update,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _copy_probe_source(tmp_path, monkeypatch)
    lock_path = run_dir / "probe.lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    update(payload)
    if message != "self-hash":
        payload["lock_sha256"] = probe._self_hash(payload)
    lock_path.write_bytes(probe._canonical_json(payload))

    with pytest.raises(ValueError, match=message):
        rescore.load_probe_source(run_dir, _expected_rescore_query_ids())


def test_probe_source_verifies_every_snapshot_response_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _copy_probe_source(tmp_path, monkeypatch)
    manifest = json.loads(
        (run_dir / "snapshots/snapshot-manifest.json").read_text(encoding="utf-8")
    )
    response = run_dir / "snapshots" / manifest["entries"][0]["response_path"]
    with response.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="response hash mismatch"):
        rescore.load_probe_source(run_dir, _expected_rescore_query_ids())


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda rows: rows.reverse(), "outcome order"),
        (
            lambda rows: rows[1].__setitem__("query_id", rows[0]["query_id"]),
            "duplicate",
        ),
        (lambda rows: rows[0].__setitem__("query_id", "unknown-query"), "unknown"),
        (
            lambda rows: rows[0].__setitem__("query_id", 7),
            "outcome query ID must be a string",
        ),
        (
            lambda rows: next(row for row in rows if row["searches"])["searches"][0][
                "errors"
            ].append(
                {"provider": "openalex", "code": "provider_error", "retryable": False}
            ),
            "search errors",
        ),
        (
            lambda rows: rows[0]["proposal"].__setitem__("no_op_reason", "changed"),
            "outcome hash mismatch",
        ),
    ],
)
def test_probe_source_requires_exact_error_free_ordered_outcomes(
    update,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _copy_probe_source(tmp_path, monkeypatch)
    _rewrite_probe_outcomes(run_dir / "outcomes.jsonl", update)

    with pytest.raises(ValueError, match=message):
        rescore.load_probe_source(run_dir, _expected_rescore_query_ids())


def test_probe_source_rejects_stage_subset_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _copy_probe_source(tmp_path, monkeypatch)
    real_merge = rescore.merge_probe_results

    def invalid_merge(*args, **kwargs):
        projection = real_merge(*args, **kwargs)
        query_id = next(iter(projection.by_query))
        row = projection.by_query[query_id]
        invalid_row = row.model_copy(
            update={"post_filter_ids": [*row.post_filter_ids, "openalex:UNKNOWN"]}
        )
        return projection.model_copy(
            update={"by_query": {**projection.by_query, query_id: invalid_row}}
        )

    monkeypatch.setattr(rescore, "merge_probe_results", invalid_merge)

    with pytest.raises(ValueError, match="subset"):
        rescore.load_probe_source(run_dir, _expected_rescore_query_ids())


def test_fixed_source_orchestrator_loads_exact_four_real_sealed_labels() -> None:
    sources = rescore.load_fixed_sources(_expected_rescore_query_ids())

    assert tuple(source.label for source in sources) == (
        "formal_baseline_2026_08_10",
        "formal_baseline_2026_08_09",
        "legacy_title_2026_08_05",
        "query_evolution_prompt_v2",
    )
    assert all(source.query_ids == _expected_rescore_query_ids() for source in sources)
