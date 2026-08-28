from __future__ import annotations

import json
from pathlib import Path

from scripts.freeze_provider_continuation import (
    remaining_query_ids,
    successful_query_ids,
)


def _retrieval(
    path: Path,
    *,
    query_id: str,
    status: str,
    error_code: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = (
        []
        if error_code is None
        else [{"code": error_code, "message": error_code, "retryable": True}]
    )
    path.write_text(
        json.dumps(
            {
                "query_id": query_id,
                "attempt_status": status,
                "results": [
                    {
                        "errors": errors,
                        "infrastructure_failure": bool(errors),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_successful_query_ids_excludes_only_proven_successes(tmp_path: Path) -> None:
    _retrieval(
        tmp_path / "batch-1" / "retrieval" / "attempt-01" / "q1.json",
        query_id="q1",
        status="failed",
        error_code="rate_limited",
    )
    _retrieval(
        tmp_path / "batch-1-retry" / "retrieval" / "attempt-01" / "q1.json",
        query_id="q1",
        status="succeeded",
    )
    _retrieval(
        tmp_path / "batch-2" / "retrieval" / "attempt-01" / "q2.json",
        query_id="q2",
        status="failed",
        error_code="timeout",
    )

    assert successful_query_ids(tmp_path) == {"q1"}


def test_remaining_query_ids_excludes_explicitly_exhausted_failures() -> None:
    assert remaining_query_ids(
        ["q1", "q2", "q3"],
        succeeded={"q1"},
        exhausted={"q2"},
    ) == ["q3"]
