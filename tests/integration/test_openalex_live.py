from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from paper_search.domain.models import BudgetReservation, UsageEstimate
from paper_search.retrieval.openalex import OpenAlexProvider
from paper_search.storage.cache import SQLiteResponseCache, validate_snapshot_manifest


LIVE_QUERIES = (
    "retrieval augmented generation evaluation",
    "dense scholarly search benchmark",
    "academic paper recommendation evidence",
)
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "openalex"
SMOKE_CONTRACT_VERSION = "openalex-smoke-v1"


def fixture_transport(filename: str) -> httpx.MockTransport:
    payload = (FIXTURE_ROOT / filename).read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "application/json"},
            request=request,
        )

    return httpx.MockTransport(handler)


def reservation(index: int) -> BudgetReservation:
    return BudgetReservation(
        reservation_id=f"live-{index}",
        action="openalex-live-smoke",
        reserved=UsageEstimate(search_api_calls=3),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def publish_summary(path: Path, value: dict[str, object]) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


async def run_live_queries(
    api_key: str,
    smoke_root: Path,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Path:
    run_id = new_run_id()
    run_dir = smoke_root / "runs" / run_id
    cache = SQLiteResponseCache(run_dir / "openalex-cache.sqlite3")
    summaries: list[dict[str, object]] = []
    ordered_keys: list[str] = []
    async with httpx.AsyncClient(
        base_url="https://api.openalex.org",
        timeout=httpx.Timeout(30.0),
        transport=transport,
    ) as client:
        provider = OpenAlexProvider(client=client, cache=cache, api_key=api_key)
        for index, query in enumerate(LIVE_QUERIES, start=1):
            result = await provider.search(query, {}, 3, reservation(index))
            assert result.data
            ordered_keys.extend(json.loads(result.provenance["cache_keys"]))
            summaries.append(
                {
                    "error_codes": [error.code for error in result.errors],
                    "latency_ms": result.latency_ms,
                    "paper_count": len(result.data),
                    "response_hash": result.provenance["response_hash"],
                }
            )
    manifest = cache.export_snapshot(list(dict.fromkeys(ordered_keys)), run_dir)
    validate_snapshot_manifest(manifest)
    summary: dict[str, object] = {
        "contract_version": SMOKE_CONTRACT_VERSION,
        "run_id": run_id,
        "manifest": manifest.relative_to(smoke_root).as_posix(),
        "queries": summaries,
    }
    provider_path = smoke_root / "provider.json"
    publish_summary(provider_path, summary)
    return provider_path


def test_smoke_artifacts_are_persistent_versioned_and_secret_free(
    tmp_path: Path,
) -> None:
    smoke_root = tmp_path / "experiments" / "smoke"
    key = "sentinel-openalex-key"

    first_provider = asyncio.run(
        run_live_queries(
            key,
            smoke_root,
            transport=fixture_transport("works_page_1.json"),
        )
    )
    first_summary = json.loads(first_provider.read_text(encoding="utf-8"))
    first_manifest = smoke_root / first_summary["manifest"]
    validate_snapshot_manifest(first_manifest)

    second_provider = asyncio.run(
        run_live_queries(
            key,
            smoke_root,
            transport=fixture_transport("works_page_1.json"),
        )
    )
    second_summary = json.loads(second_provider.read_text(encoding="utf-8"))
    second_manifest = smoke_root / second_summary["manifest"]

    assert first_summary["contract_version"] == "openalex-smoke-v1"
    assert first_summary["run_id"] != second_summary["run_id"]
    assert first_manifest.exists()
    assert second_manifest.exists()
    assert len(second_summary["queries"]) == 3
    assert all(item["paper_count"] > 0 for item in second_summary["queries"])
    assert key.encode() not in b"".join(
        path.read_bytes() for path in smoke_root.rglob("*") if path.is_file()
    )


def test_failed_smoke_does_not_replace_last_accepted_summary(tmp_path: Path) -> None:
    smoke_root = tmp_path / "experiments" / "smoke"
    provider = asyncio.run(
        run_live_queries(
            "sentinel-openalex-key",
            smoke_root,
            transport=fixture_transport("works_page_1.json"),
        )
    )
    accepted = provider.read_bytes()

    with pytest.raises(AssertionError):
        asyncio.run(
            run_live_queries(
                "sentinel-openalex-key",
                smoke_root,
                transport=fixture_transport("works_empty.json"),
            )
        )

    assert provider.read_bytes() == accepted


@pytest.mark.online
def test_three_live_queries_produce_safe_snapshot(tmp_path: Path) -> None:
    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        pytest.skip("OPENALEX_API_KEY is not set in the process environment")

    provider_path = asyncio.run(run_live_queries(api_key, tmp_path))

    serialized = provider_path.read_text(encoding="utf-8")
    summary = json.loads(serialized)
    assert api_key not in serialized
    assert len(summary["queries"]) == 3
    validate_snapshot_manifest(tmp_path / summary["manifest"])
