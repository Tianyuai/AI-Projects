from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from paper_search.recall_experiments.canary_runtime import (
    RecallRuntimeProfile,
    build_live_runtime_bundle,
    load_runtime_profile,
    resolve_runtime_secrets,
)
from paper_search.recall_experiments.recipes import load_recall_recipe


def _write_profile(tmp_path: Path) -> Path:
    profile = tmp_path / "runtime.yaml"
    profile.write_text(
        "\n".join(
            [
                "schema_version: recall-runtime-profile-v1",
                "env_file: secrets.env",
                "pricing_policy: pricing.yaml",
                "budget: budget.yaml",
                "capture_responses: true",
                "llm_model: deepseek-v4-flash",
                "llm_reservation_input_tokens: 2500",
                "llm_reservation_output_tokens: 1000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return profile


def test_runtime_profile_resolves_relative_paths_and_allows_explicit_overrides(
    tmp_path: Path,
) -> None:
    profile = _write_profile(tmp_path)

    loaded = load_runtime_profile(
        profile,
        env_file=tmp_path / "override.env",
        budget=tmp_path / "override-budget.yaml",
    )

    assert loaded == RecallRuntimeProfile(
        schema_version="recall-runtime-profile-v1",
        env_file=(tmp_path / "override.env").resolve(),
        pricing_policy=(tmp_path / "pricing.yaml").resolve(),
        budget=(tmp_path / "override-budget.yaml").resolve(),
        capture_responses=True,
        llm_model="deepseek-v4-flash",
        llm_reservation_input_tokens=2500,
        llm_reservation_output_tokens=1000,
    )


def test_runtime_profile_forbids_inline_secrets(tmp_path: Path) -> None:
    profile = tmp_path / "runtime.yaml"
    profile.write_text(
        "schema_version: recall-runtime-profile-v1\n"
        "env_file: secrets.env\n"
        "pricing_policy: pricing.yaml\n"
        "budget: budget.yaml\n"
        "capture_responses: true\n"
        "llm_reservation_input_tokens: 2500\n"
        "llm_reservation_output_tokens: 1000\n"
        "llm_api_key: forbidden\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="secret fields are not allowed"):
        load_runtime_profile(profile)


def test_runtime_secrets_are_loaded_but_never_serialized(tmp_path: Path) -> None:
    env_file = tmp_path / "secrets.env"
    env_file.write_text(
        "LLM_API_KEY=llm-secret\n"
        "OPENALEX_API_KEY=openalex-secret\n"
        "SEMANTIC_SCHOLAR_API_KEY=s2-secret\n",
        encoding="utf-8",
    )
    profile = RecallRuntimeProfile(
        schema_version="recall-runtime-profile-v1",
        env_file=env_file,
        pricing_policy=tmp_path / "pricing.yaml",
        budget=tmp_path / "budget.yaml",
        capture_responses=True,
        llm_model="deepseek-v4-flash",
        llm_reservation_input_tokens=2500,
        llm_reservation_output_tokens=1000,
    )

    secrets = resolve_runtime_secrets(profile, environ={})

    assert secrets.llm_api_key.get_secret_value() == "llm-secret"
    assert "llm-secret" not in profile.model_dump_json()
    assert "llm-secret" not in secrets.model_dump_json()


def test_runtime_secrets_fail_closed_when_required_key_is_missing(tmp_path: Path) -> None:
    env_file = tmp_path / "secrets.env"
    env_file.write_text("LLM_API_KEY=only-one\n", encoding="utf-8")
    profile = RecallRuntimeProfile(
        schema_version="recall-runtime-profile-v1",
        env_file=env_file,
        pricing_policy=tmp_path / "pricing.yaml",
        budget=tmp_path / "budget.yaml",
        capture_responses=True,
        llm_model="deepseek-v4-flash",
        llm_reservation_input_tokens=2500,
        llm_reservation_output_tokens=1000,
    )

    with pytest.raises(ValueError, match="missing required runtime secrets"):
        resolve_runtime_secrets(profile, environ={})


def test_fixed_factory_builds_one_owned_live_runtime_without_dispatch(tmp_path: Path) -> None:
    (tmp_path / "pricing.yaml").write_bytes(
        Path("configs/pricing_v1.yaml").read_bytes()
    )
    (tmp_path / "budget.yaml").write_bytes(Path("configs/budget_low.yaml").read_bytes())
    (tmp_path / "secrets.env").write_text(
        "LLM_API_KEY=llm-test\nOPENALEX_API_KEY=oa-test\n"
        "SEMANTIC_SCHOLAR_API_KEY=s2-test\n",
        encoding="utf-8",
    )
    profile = load_runtime_profile(_write_profile(tmp_path))
    secrets = resolve_runtime_secrets(profile, environ={})
    loaded_recipe = load_recall_recipe(
        Path("configs/recall_experiments/methods/scheme-b-blind-live.yaml")
    )

    bundle = asyncio.run(
        build_live_runtime_bundle(
            profile=profile,
            secrets=secrets,
            loaded_recipe=loaded_recipe,
            capture_root=tmp_path / "capture",
        )
    )
    try:
        assert bundle.runtime.controller is not None
        assert bundle.runtime.controller.formal_live is True
        assert set(bundle.runtime.identity["dependencies"]) == {"search", "citation", "llm"}
        assert (
            bundle.runtime.identity["dependencies"]["citation"]["dependency"]
            == "openalex"
        )
    finally:
        asyncio.run(bundle.aclose())
    assert (tmp_path / "capture" / "snapshot-manifest.json").is_file()


def test_factory_can_select_semantic_scholar_as_search_provider(tmp_path: Path) -> None:
    (tmp_path / "pricing.yaml").write_bytes(
        Path("data/annotation_work/pricing_v1.yaml").read_bytes()
    )
    (tmp_path / "budget.yaml").write_bytes(Path("configs/budget_low.yaml").read_bytes())
    (tmp_path / "secrets.env").write_text(
        "LLM_API_KEY=llm-test\nOPENALEX_API_KEY=oa-test\n"
        "SEMANTIC_SCHOLAR_API_KEY=s2-test\n",
        encoding="utf-8",
    )
    profile = load_runtime_profile(_write_profile(tmp_path))
    loaded_recipe = load_recall_recipe(
        Path("configs/recall_experiments/methods/scheme-b-blind-live.yaml")
    )

    bundle = asyncio.run(
        build_live_runtime_bundle(
            profile=profile,
            secrets=resolve_runtime_secrets(profile, environ={}),
            loaded_recipe=loaded_recipe,
            capture_root=tmp_path / "capture-s2",
            search_dependency="semantic_scholar",
        )
    )
    try:
        assert (
            bundle.runtime.identity["dependencies"]["search"]["dependency"]
            == "semantic_scholar"
        )
    finally:
        asyncio.run(bundle.aclose())


def test_runtime_model_is_independent_of_a_fixed_action_method(tmp_path: Path) -> None:
    actions = tmp_path / "actions.json"
    actions.write_text('{"q-1":{"actions":[]}}', encoding="utf-8")
    recipe_path = tmp_path / "fixed.yaml"
    recipe_path.write_text(
        "method_id: fixed\n"
        "generator:\n  type: fixed_actions\n  actions: actions.json\n  gold_visibility: blind\n"
        "retrieval:\n  allowed_actions: [text_search]\n  backend: live_provider\n"
        "  max_results_per_action: 5\n  max_total_actions: 1\n"
        "candidate_pool:\n  policy_version: production-dedup-v1\n"
        "evaluation:\n  repeat_count: 1\n  max_repeat_attempts: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "pricing.yaml").write_bytes(
        Path("configs/pricing_v1.yaml").read_bytes()
    )
    (tmp_path / "budget.yaml").write_bytes(Path("configs/budget_low.yaml").read_bytes())
    (tmp_path / "secrets.env").write_text(
        "LLM_API_KEY=llm-test\nOPENALEX_API_KEY=oa-test\n"
        "SEMANTIC_SCHOLAR_API_KEY=s2-test\n",
        encoding="utf-8",
    )
    profile = load_runtime_profile(_write_profile(tmp_path))
    bundle = asyncio.run(
        build_live_runtime_bundle(
            profile=profile,
            secrets=resolve_runtime_secrets(profile, environ={}),
            loaded_recipe=load_recall_recipe(recipe_path),
            capture_root=tmp_path / "fixed-capture",
        )
    )
    asyncio.run(bundle.aclose())


def test_repository_default_profile_resolves_the_fixed_external_interfaces() -> None:
    profile = load_runtime_profile(
        Path("configs/recall_experiments/runtime/default-live.yaml")
    )

    root = Path.cwd().resolve()
    assert profile.env_file == root / ".env"
    assert profile.pricing_policy == root / "configs/pricing_v1.yaml"
    assert profile.budget == root / "configs/budget_low.yaml"
