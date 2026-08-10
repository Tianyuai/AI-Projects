from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.probe_query_evolution import (
    DEFAULT_AVAILABILITY,
    DEFAULT_GOLD,
    DEFAULT_ID_MAP,
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
            ledger_path=tmp_path / "ledger.sqlite3",
            output_path=tmp_path / "probe.lock.json",
        )
