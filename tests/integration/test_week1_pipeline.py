from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from paper_search.config import RuntimeConfig, load_budget
from paper_search.domain.models import (
    BudgetReservation,
    Paper,
    ProviderResult,
    UsageActual,
)
from paper_search.evaluation.annotation import AgreementReport, FieldAgreement
from paper_search.evaluation.dataset import EvaluationQuery, read_jsonl
from paper_search.evaluation.freeze import (
    FreezeAuditReport,
    FreezeAuditResult,
    PartitionFreezeAudit,
    approve_freeze,
    build_approval_plan,
)
from paper_search.evaluation.runner import (
    RunIdentity,
    _resolve_frozen_split,
    run_evaluation,
)
from paper_search.storage import SQLiteResponseCache
from paper_search.storage.cache import validate_snapshot_manifest


FIXTURES = Path(__file__).parents[1] / "fixtures" / "week1"
CONFIGS = Path(__file__).parents[2] / "configs"


class FixedProvider:
    def __init__(
        self,
        results: dict[str, list[Paper]],
        raw_responses: dict[str, bytes],
    ) -> None:
        self._results = results
        self._raw_responses = raw_responses
        self.responses: list[ProviderResult[list[Paper]]] = []

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        del filters, reservation
        query_index = list(self._results).index(query) + 1
        response = ProviderResult(
            data=self._results[query][:limit],
            usage=UsageActual(search_api_calls=0, elapsed_ms=query_index),
            provenance={
                "provider": "openalex",
                "endpoint": "/works",
                "model_id": "openalex-api",
                "requested_at": "2026-07-17T00:00:00+00:00",
                "response_hash": (
                    f"sha256:{hashlib.sha256(self._raw_responses[query]).hexdigest()}"
                ),
                "cache_keys": json.dumps([f"fixture-page-{query_index}"]),
            },
            cache_hit=True,
            latency_ms=query_index,
            errors=[],
        )
        self.responses.append(response)
        return response


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        budget_profile="low",
        budget=load_budget(CONFIGS / "budget_low.yaml"),
        llm_base_url="https://llm.invalid/v1",
        llm_model_primary="test-primary",
        llm_model_fallback="test-fallback",
    )


def test_fixed_week1_fixture_runs_full_pipeline_and_snapshot(tmp_path: Path) -> None:
    gold = read_jsonl(FIXTURES / "gold.jsonl", EvaluationQuery)
    fixture_bytes = (FIXTURES / "openalex_results.json").read_bytes()
    payload = json.loads(fixture_bytes)
    results = {
        query: [Paper.model_validate(record) for record in records]
        for query, records in payload.items()
    }
    raw_responses = {
        query: json.dumps(
            {"query": query, "results": payload[query]},
            sort_keys=True,
        ).encode("utf-8")
        for query in results
    }
    cache = SQLiteResponseCache(tmp_path / ".cache" / "openalex.sqlite3")
    now = datetime(2026, 7, 17, tzinfo=UTC)
    for index, query in enumerate(results, start=1):
        cache.put_response(
            key=f"fixture-page-{index}",
            provider="openalex",
            endpoint="/works",
            cache_version="v1",
            params={"search": query},
            raw_response=raw_responses[query],
            requested_at=now,
            ttl=timedelta(days=7),
            safe_headers={},
        )
    output = tmp_path / "run"
    provider = FixedProvider(results, raw_responses)

    result = asyncio.run(
        run_evaluation(
            gold,
            identity=RunIdentity(
                split="dev",
                git_sha="a" * 40,
                gold_sha256=(
                    f"sha256:{hashlib.sha256((FIXTURES / 'gold.jsonl').read_bytes()).hexdigest()}"
                ),
                manifest_sha256=f"sha256:{'b' * 64}",
                dataset_revision="fixture-r1",
                zero_answer_policy="reject",
            ),
            provider=provider,
            cache=cache,
            config=_runtime_config(),
            output=output,
        )
    )

    assert result.evaluation.summary.query_count == 2
    assert result.evaluation.summary.macro_f1 > 0
    assert [
        (response.usage.search_api_calls, response.provenance["response_hash"])
        for response in provider.responses
    ] == [
        (0, f"sha256:{hashlib.sha256(raw_response).hexdigest()}")
        for raw_response in raw_responses.values()
    ]
    assert result.query_runs[0].pipeline.deduplication.decisions
    assert result.query_runs[0].pipeline.filtering.rejected[0].reason_code == "retracted"
    assert any(
        accepted.paper.publication_year is None
        for accepted in result.query_runs[1].pipeline.filtering.accepted
    )
    validate_snapshot_manifest(output / "snapshot_manifest.json")


def test_approved_synthetic_manifest_is_accepted_by_week1_runner(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    gold_path = data_root / "dev" / "gold.jsonl"
    ids_path = data_root / "splits" / "dev.ids.json"
    gold_path.parent.mkdir(parents=True)
    ids_path.parent.mkdir(parents=True)
    query = EvaluationQuery(
        query_id="q1",
        query="Synthetic integration query",
        relevant_paper_ids=["arxiv:1706.03762"],
    )
    gold_bytes = (
        json.dumps(query.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + chr(10)
    ).encode()
    ids_bytes = (json.dumps(["q1"], indent=2) + chr(10)).encode()
    gold_path.write_bytes(gold_bytes)
    ids_path.write_bytes(ids_bytes)

    prepared_bytes = (
        json.dumps({"status": "waiting_for_human_label_freeze"}, separators=(",", ":"))
        + chr(10)
    ).encode()
    (data_root / "manifest.json").write_bytes(prepared_bytes)
    partition = PartitionFreezeAudit(
        count=1,
        gold_path="dev/gold.jsonl",
        gold_sha256=f"sha256:{hashlib.sha256(gold_bytes).hexdigest()}",
        ids_path="splits/dev.ids.json",
        ids_sha256=f"sha256:{hashlib.sha256(ids_bytes).hexdigest()}",
        zero_answer_policy="reject",
        labels_complete=True,
    )
    agreement = AgreementReport(
        compared_query_count=1,
        fields={
            field: FieldAgreement(kappa=1.0, threshold=0.8, accepted=True)
            for field in ("query_type", "domain")
        },
    )
    prepared_hash = f"sha256:{hashlib.sha256(prepared_bytes).hexdigest()}"
    report = FreezeAuditReport(
        prepared_manifest_sha256=prepared_hash,
        dataset_revision="fixture-r1",
        source_file_count=1,
        type_domain_count=1,
        type_domain_sha256=f"sha256:{'1' * 64}",
        constraint_count=1,
        constraint_sha256=f"sha256:{'2' * 64}",
        overlap_count=1,
        overlap_sha256=f"sha256:{'3' * 64}",
        agreement=agreement,
        partitions={"dev": partition},
        approval_requested=False,
    )
    audit = FreezeAuditResult(
        prepared_manifest_bytes=prepared_bytes,
        frozen_manifest_payload={
            "status": "frozen",
            "revision": "fixture-r1",
            "prepared_manifest_sha256": prepared_hash,
            "partitions": {"dev": partition.model_dump(mode="json")},
        },
        report=report,
    )
    plan = build_approval_plan(
        audit,
        report_relative_path="freeze_reports/integration.json",
    )

    approve_freeze(data_root=data_root, plan=plan)
    frozen = _resolve_frozen_split(data_root, "dev", "a" * 40)

    assert frozen.identity.split == "dev"
    assert frozen.identity.git_sha == "a" * 40
    assert frozen.identity.gold_sha256 == partition.gold_sha256
    assert frozen.identity.manifest_sha256 == (
        f"sha256:{hashlib.sha256(plan.frozen_manifest_bytes).hexdigest()}"
    )
    assert frozen.identity.dataset_revision == "fixture-r1"
    assert frozen.identity.zero_answer_policy == "reject"