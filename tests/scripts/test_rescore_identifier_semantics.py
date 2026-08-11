from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import scripts.probe_query_evolution as probe
import scripts.rescore_identifier_semantics as rescore


ROOT = Path(__file__).resolve().parents[2]


def test_probe_script_public_helpers_are_thin_wrappers(monkeypatch) -> None:
    marker = object()
    lock = object()
    outcomes = object()

    monkeypatch.setattr(probe, "_verify_probe_source_bindings", lambda value: (marker, value))
    monkeypatch.setattr(probe, "_frozen_inputs", lambda value: (marker, value))
    monkeypatch.setattr(
        probe,
        "_capture_replay_hash",
        lambda lock_value, outcome_value: (marker, lock_value, outcome_value),
    )

    assert probe.verify_probe_source_bindings(lock) == (marker, lock)
    assert probe.frozen_probe_inputs(lock) == (marker, lock)
    assert probe.probe_outcome_hash(lock, outcomes) == (marker, lock, outcomes)


def test_rescore_adapter_module_invocation_works() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.rescore_identifier_semantics"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_fixed_source_orchestrator_uses_only_the_four_sealed_sources(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        rescore,
        "load_formal_source",
        lambda *args: calls.append(("formal", *args)) or args[0],
    )
    monkeypatch.setattr(
        rescore,
        "load_legacy_source",
        lambda *args: calls.append(("legacy", *args)) or "legacy_title_2026_08_05",
    )
    monkeypatch.setattr(
        rescore,
        "load_probe_source",
        lambda *args: calls.append(("probe", *args)) or "query_evolution_prompt_v2",
    )
    expected_query_ids = ("q-1", "q-2")

    sources = rescore.load_fixed_sources(expected_query_ids, root=ROOT)

    assert sources == (
        "formal_baseline_2026_08_10",
        "formal_baseline_2026_08_09",
        "legacy_title_2026_08_05",
        "query_evolution_prompt_v2",
    )
    assert calls == [
        (
            "formal",
            "formal_baseline_2026_08_10",
            ROOT / "runs/dev-20260810T104256Z-d9e89476d484",
            expected_query_ids,
        ),
        (
            "formal",
            "formal_baseline_2026_08_09",
            ROOT / "runs/dev-20260809T061903Z-9bd861e90299",
            expected_query_ids,
        ),
        (
            "legacy",
            ROOT / "runs/dev-20260805T035209Z-7af4b103f6cc",
            ROOT / "docs/evidence/title-retention-offline-2026-08-09.json",
            expected_query_ids,
        ),
        (
            "probe",
            ROOT / "runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810",
            expected_query_ids,
        ),
    ]
