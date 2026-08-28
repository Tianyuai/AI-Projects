from __future__ import annotations

from scripts.prepare_query_native_phrase_validation import _balanced_sample


def _row(query_id: str, signal: str) -> dict[str, object]:
    return {"query_id": query_id, "signal": signal}


def test_balanced_sample_prefers_equal_unconstrained_and_structured_rows() -> None:
    rows = [*[_row(f"task-{i}", "task") for i in range(20)]]
    rows.extend(_row(f"unc-{i}", "unconstrained") for i in range(12))

    selected = _balanced_sample(rows, limit=24, unconstrained_target=12)

    assert len(selected) == 24
    assert len({str(row["query_id"]) for row in selected}) == 24
    assert sum(row["signal"] == "unconstrained" for row in selected) == 12
    assert sum(row["signal"] != "unconstrained" for row in selected) == 12


def test_balanced_sample_fills_short_unconstrained_stratum_without_duplicates() -> None:
    rows = [*[_row(f"task-{i}", "task") for i in range(24)]]
    rows.extend(_row(f"unc-{i}", "unconstrained") for i in range(5))

    selected = _balanced_sample(rows, limit=24, unconstrained_target=12)

    assert len(selected) == 24
    assert len({str(row["query_id"]) for row in selected}) == 24
    assert sum(row["signal"] == "unconstrained" for row in selected) == 5
