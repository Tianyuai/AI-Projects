from __future__ import annotations

import hashlib
import json
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

import scripts.probe_query_evolution as probe
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
            version="query-evolve-v1",
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


def _write_canary_lock(tmp_path: Path, label: str) -> tuple[Path, probe.CanaryLock, Path]:
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
    lock = probe.preflight_canary(
        probe_lock_path=probe_lock_path,
        ledger_path=tmp_path / f"{label}-canary-ledger.sqlite3",
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
    assert len(lock.query_ids) == 55
    assert lock.query_ids == tuple(lock.query_ids)
    assert lock.probe_code_sha256.startswith("sha256:")
    assert lock.lock_sha256.startswith("sha256:")
    assert output.exists()
    assert load_probe_lock(output).lock_sha256 == lock.lock_sha256


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
        assert lock.prompt.version == "query-evolve-v1"
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
                "version: query-evolve-v1",
                "version: query-evolve-v2",
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
                    ledger_path=tmp_path / "ledger.sqlite3",
                ),
            )

        assert live_probe_started is False


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
    lock_path, lock, run_dir = _write_canary_lock(tmp_path, "accounting")
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
            ledger_path=tmp_path / "ledger.sqlite3",
        ),
    )

    result = _read_result(run_dir)
    assert result["reason"] == "canary_accounting_failed"
    assert result["promoted"] is False
    ledger = probe.SQLiteBudgetLedger(
        tmp_path / "ledger.sqlite3",
        reservation_ttl_seconds=probe.PROBE_LEDGER_TTL_SECONDS,
    )
    receipts = ledger.report(lock.canary_run_id).receipts
    assert len(receipts) == 3
    assert all(receipt.state == "failed" for receipt in receipts)
    assert all(receipt.actual is not None for receipt in receipts)


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
    lock_path, lock, run_dir = _write_canary_lock(tmp_path, "cancelled")
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
            ledger_path=tmp_path / "ledger.sqlite3",
        ),
    )

    result = _read_result(run_dir)
    assert calls == []
    assert result["reason"] == "canary_cancelled"
    assert result["promoted"] is False
    ledger = probe.SQLiteBudgetLedger(
        tmp_path / "ledger.sqlite3",
        reservation_ttl_seconds=probe.PROBE_LEDGER_TTL_SECONDS,
    )
    receipts = ledger.report(lock.canary_run_id).receipts
    assert len(receipts) == 3
    assert all(receipt.state == "failed" for receipt in receipts)
    assert all(receipt.actual is not None for receipt in receipts)


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

    probe.run_canary(
        lock_path,
        ProbeRuntime(
            allow_live=True,
            env_file=env_file,
            ledger_path=tmp_path / "ledger.sqlite3",
        ),
    )

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
