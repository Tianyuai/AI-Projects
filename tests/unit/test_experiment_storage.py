from pathlib import Path

import pytest

from paper_search.storage.experiment import ExperimentRecordStore


def test_record_store_writes_canonical_recoverable_json(tmp_path: Path) -> None:
    store = ExperimentRecordStore(tmp_path)
    record = {
        "config_hash": "sha256:" + "a" * 64,
        "prompt_version": "query-analyze-v1",
        "provider_results": {"openalex": {"errors": []}},
        "stop_reason": "completed",
    }

    path = store.write("run-001", record)

    assert path.name == "run-001.json"
    assert store.read("run-001") == record
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError):
        store.write("run-001", {**record, "stop_reason": "changed"})


@pytest.mark.parametrize("run_id", ["", "../escape", "nested/run"])
def test_record_store_rejects_unsafe_run_ids(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError):
        ExperimentRecordStore(tmp_path).write(run_id, {"value": 1})
