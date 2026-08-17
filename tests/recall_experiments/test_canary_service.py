from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from paper_search.domain.models import Paper
from paper_search.recall_experiments.canary_inputs import load_canary_input
from paper_search.recall_experiments.canary_runtime import (
    RecallRuntimeProfile,
    RecallRuntimeSecrets,
    build_live_runtime_bundle,
)
from paper_search.recall_experiments.canary_service import RecallCanaryService
from paper_search.recall_experiments.composition import RecallRuntime
from paper_search.recall_experiments.recipes import load_recall_recipe
from paper_search.recall_experiments.retrieval.backends import (
    BackendCitationResult,
    BackendSearchResult,
)


class _Search:
    async def search(
        self, action_id: str, query: str, filters: dict[str, object], limit: int
    ) -> BackendSearchResult:
        del action_id, query, filters, limit
        return BackendSearchResult(
            hits=[
                Paper(
                    canonical_id="doi:10.1234/example",
                    title="Candidate",
                    sources=["openalex"],
                )
            ]
        )


class _Citation:
    async def expand(self, *args: object, **kwargs: object) -> BackendCitationResult:
        del args, kwargs
        return BackendCitationResult(direction="references")


class _UnusedLLM:
    async def generate(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("fixed generator must not call LLM")


def _transport(request: httpx.Request) -> httpx.Response:
    if request.url.host != "api.openalex.org":
        raise AssertionError(f"unexpected dependency request: {request.url}")
    return httpx.Response(
        200,
        json={
            "meta": {"count": 1, "per_page": 5, "next_cursor": None},
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.1234/example",
                    "title": "Candidate",
                    "display_name": "Candidate",
                    "authorships": [],
                    "publication_year": 2026,
                    "primary_location": None,
                    "cited_by_count": 0,
                    "is_retracted": False,
                }
            ],
        },
        request=request,
    )


async def _run_canary(
    tmp_path: Path,
    *,
    loaded_input: object,
    output_name: str,
    baseline: Path | None = None,
) -> object:
    recipe_path = _recipe(tmp_path)
    loaded_recipe = load_recall_recipe(recipe_path)
    profile = RecallRuntimeProfile(
        schema_version="recall-runtime-profile-v1",
        env_file=tmp_path / "unused.env",
        pricing_policy=Path("configs/pricing_v1.yaml").resolve(),
        budget=Path("configs/budget_low.yaml").resolve(),
        capture_responses=True,
        llm_model="deepseek-v4-flash",
        llm_reservation_input_tokens=2500,
        llm_reservation_output_tokens=1000,
    )
    secrets = RecallRuntimeSecrets(
        llm_api_key=SecretStr("test"),
        openalex_api_key=SecretStr("test"),
        semantic_scholar_api_key=SecretStr("test"),
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(_transport))
    bundle = await build_live_runtime_bundle(
        profile=profile,
        secrets=secrets,
        loaded_recipe=loaded_recipe,
        capture_root=tmp_path / f"{output_name}-capture",
        client=client,
    )
    try:
        return await RecallCanaryService(workspace_root=tmp_path).run(
            loaded_recipe=loaded_recipe,
            loaded_input=loaded_input,  # type: ignore[arg-type]
            runtime_bundle=bundle,
            output_path=tmp_path / output_name,
            baseline_report_path=baseline,
        )
    finally:
        await bundle.aclose()


def _recipe(tmp_path: Path) -> Path:
    actions = tmp_path / "actions.json"
    actions.write_text(
        json.dumps(
            {
                "q-1": {
                    "actions": [
                        {
                            "action_id": "a-1",
                            "action_type": "text_search",
                            "strategy": "fixed",
                            "payload": {"query_text": "graph retrieval"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "method_id: fixed-canary\n"
        "generator:\n"
        "  type: fixed_actions\n"
        "  actions: actions.json\n"
        "  gold_visibility: blind\n"
        "retrieval:\n"
        "  allowed_actions: [text_search]\n"
        "  backend: live_provider\n"
        "  max_results_per_action: 5\n"
        "  max_total_actions: 1\n"
        "candidate_pool:\n"
        "  policy_version: production-dedup-v1\n"
        "evaluation:\n"
        "  repeat_count: 1\n"
        "  max_repeat_attempts: 1\n",
        encoding="utf-8",
    )
    return recipe


def test_service_runs_unscored_query_and_writes_fixed_report(tmp_path: Path) -> None:
    loaded_input = load_canary_input(
        input_kind="single",
        query="graph retrieval",
        query_id="q-1",
        source_path=None,
        identifier_map_path=None,
        workspace_root=tmp_path,
    )
    report = asyncio.run(_run_canary(tmp_path, loaded_input=loaded_input, output_name="run"))

    assert report.result.evaluation_status == "not_available"
    assert report.result.gold_hit_count is None
    assert report.result.per_query[0].candidate_count == 1
    assert report.actions_by_query["q-1"][0].payload.model_dump(mode="json") == {
        "query_text": "graph retrieval"
    }
    assert report.usage.search_api_calls == 1
    saved = json.loads((tmp_path / "run" / "canary-report.json").read_text(encoding="utf-8"))
    assert set(saved["result"]) == set(report.result.model_dump(mode="json"))
    assert saved["input"]["evaluation_status"] == "not_available"


def test_service_scores_gold_and_embeds_a_baseline_comparison(tmp_path: Path) -> None:
    identifier_map = tmp_path / "id-map.json"
    identifier_map.write_text("{}", encoding="utf-8")
    input_jsonl = tmp_path / "queries.jsonl"
    input_jsonl.write_text(
        '{"query_id":"q-1","query":"graph retrieval",'
        '"gold_paper_ids":["doi:10.1234/example"]}\n',
        encoding="utf-8",
    )
    loaded_input = load_canary_input(
        input_kind="jsonl",
        query=None,
        query_id=None,
        source_path=input_jsonl,
        identifier_map_path=identifier_map,
        workspace_root=tmp_path,
    )
    baseline = asyncio.run(
        _run_canary(tmp_path, loaded_input=loaded_input, output_name="baseline")
    )
    current = asyncio.run(
        _run_canary(
            tmp_path,
            loaded_input=loaded_input,
            output_name="current",
            baseline=tmp_path / "baseline" / "canary-report.json",
        )
    )

    assert baseline.result.gold_hit_count == 1
    assert current.execution_identity.identifier_map_sha256 == loaded_input.identifier_map_sha256
    assert current.result.macro_candidate_recall == 1
    assert current.comparison is not None
    assert current.comparison.evidence_level == "strict"
    assert current.comparison.per_query[0].jaccard == 1


def test_service_rejects_unverified_runtime_before_dispatch(tmp_path: Path) -> None:
    loaded_input = load_canary_input(
        input_kind="single",
        query="graph retrieval",
        query_id="q-1",
        source_path=None,
        identifier_map_path=None,
        workspace_root=tmp_path,
    )
    fake = RecallRuntime(_Search(), _Citation(), _UnusedLLM(), identity={"mode": "test"})
    from paper_search.recall_experiments.canary_runtime import RecallLiveRuntimeBundle

    with pytest.raises(ValueError, match="fixed factory"):
        RecallLiveRuntimeBundle(
            runtime=fake,
            client=httpx.AsyncClient(transport=httpx.MockTransport(_transport)),
            capture_store=object(),  # type: ignore[arg-type]
            _capability=object(),
        )
    loaded_recipe = load_recall_recipe(_recipe(tmp_path))
    profile = RecallRuntimeProfile(
        schema_version="recall-runtime-profile-v1", env_file=tmp_path / "none",
        pricing_policy=Path("configs/pricing_v1.yaml").resolve(),
        budget=Path("configs/budget_low.yaml").resolve(), capture_responses=True,
        llm_model="deepseek-v4-flash", llm_reservation_input_tokens=2500,
        llm_reservation_output_tokens=1000,
    )
    trusted = asyncio.run(build_live_runtime_bundle(
        profile=profile,
        secrets=RecallRuntimeSecrets(
            llm_api_key=SecretStr("x"), openalex_api_key=SecretStr("x"),
            semantic_scholar_api_key=SecretStr("x")
        ), loaded_recipe=loaded_recipe, capture_root=tmp_path / "capture",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_transport)),
    ))
    object.__setattr__(trusted, "runtime", fake)
    with pytest.raises(Exception, match="resources do not match"):
        asyncio.run(
            RecallCanaryService(workspace_root=tmp_path).run(
                loaded_recipe=loaded_recipe,
                loaded_input=loaded_input,
                runtime_bundle=trusted,
                output_path=tmp_path / "rejected",
            )
        )
    asyncio.run(trusted.client.aclose())
