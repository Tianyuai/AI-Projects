from __future__ import annotations

import json
from tempfile import TemporaryDirectory
from pathlib import Path

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
