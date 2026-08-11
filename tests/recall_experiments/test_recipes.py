from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_search.recall_experiments.recipes import (
    RecallMethodRecipe,
    SampleBinding,
    authorize_live_backend,
    load_recall_recipe,
    load_sample_binding,
    validate_recipe_sample_preflight,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _sha256(content: bytes) -> str:
    return f"sha256:{sha256(content).hexdigest()}"


def _manual_recipe(*, actions: str = "runs/actions.json") -> dict[str, object]:
    return {
        "method_id": "manual-example",
        "generator": {
            "type": "manual_actions",
            "actions": actions,
            "gold_visibility": "oracle",
        },
        "retrieval": {
            "allowed_actions": ["text_search", "title_search"],
            "backend": "live_provider",
            "max_results_per_action": 50,
            "max_total_actions": 3,
        },
        "candidate_pool": {},
        "evaluation": {"repeat_count": 1, "max_repeat_attempts": 1},
    }


def _comparison_recipe() -> dict[str, object]:
    return {
        "method_id": "reproduce-history",
        "generator": {
            "type": "deepseek_prompt",
            "prompt": "prompts/reproduce.yaml",
            "model": "deepseek-v4-flash",
            "temperature": 0,
            "gold_visibility": "historical",
            "max_generated_actions": 2,
            "repair_attempts": 1,
        },
        "retrieval": {
            "allowed_actions": ["text_search"],
            "backend": "snapshot_replay",
            "max_results_per_action": 50,
            "max_total_actions": 3,
        },
        "candidate_pool": {"policy_version": "canonical-id-first-v1"},
        "evaluation": {
            "repeat_count": 3,
            "max_repeat_attempts": 5,
            "compare_with": "historical-query-evolution",
            "gold_count_tolerance": 1,
            "macro_recall_tolerance": 0.02,
            "retained_gold_min": 0.90,
            "required_passing_repeats": 2,
        },
    }


def _sample_binding(*, query_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "sample_id": "sample-a",
        "query_ids": query_ids or ["query-a"],
        "gold_document_catalog": {
            "path": "catalogs/gold.jsonl",
            "sha256": _sha256(b"catalog bytes"),
        },
        "gold_document_catalog_manifest": {
            "path": "catalogs/gold.manifest.json",
            "sha256": _sha256(b"catalog manifest bytes"),
        },
        "gold_ids": ["gold-a"],
        "seed_canonical_ids": ["seed-a"],
        "legacy_candidate_pool_policy": "canonical-id-first-v1",
    }


def test_recipe_models_reject_unknown_fields_and_unsafe_relative_paths() -> None:
    payload = _manual_recipe(actions="../actions.json")
    with pytest.raises(ValidationError, match="safe relative path"):
        RecallMethodRecipe.model_validate(payload)

    payload = _manual_recipe()
    payload["generator"] = {**payload["generator"], "implementation": "module:callable"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RecallMethodRecipe.model_validate(payload)


def test_recipe_defaults_to_production_dedup_and_has_hashable_canonical_serialization() -> None:
    recipe = RecallMethodRecipe.model_validate(_manual_recipe())

    assert recipe.candidate_pool.policy_version == "production-dedup-v1"


def test_equivalent_recipe_files_have_equal_canonical_identity(tmp_path: Path) -> None:
    first_path = _write(
        tmp_path / "first.yaml",
        """method_id: manual-example
generator:
  type: manual_actions
  actions: runs/actions.json
  gold_visibility: oracle
retrieval:
  allowed_actions: [text_search, title_search]
  backend: live_provider
  max_results_per_action: 50
  max_total_actions: 3
candidate_pool: {}
evaluation:
  repeat_count: 1
  max_repeat_attempts: 1
""",
    )
    second_path = _write(tmp_path / "second.yaml", first_path.read_text(encoding="utf-8"))
    different_path = _write(
        tmp_path / "different.yaml",
        first_path.read_text(encoding="utf-8").replace("manual-example", "manual-other"),
    )

    first = load_recall_recipe(first_path)
    second = load_recall_recipe(second_path)
    different = load_recall_recipe(different_path)

    assert first.recipe.canonical_bytes() == second.recipe.canonical_bytes()
    assert hash(first.recipe) == hash(second.recipe)
    assert first.recipe_sha256 == second.recipe_sha256
    assert first.recipe.canonical_bytes() != different.recipe.canonical_bytes()
    assert first.recipe_sha256 != different.recipe_sha256


def test_manual_and_fixed_generators_require_actions_artifacts() -> None:
    payload = _manual_recipe()
    payload["generator"] = {"type": "manual_actions", "gold_visibility": "blind"}

    with pytest.raises(ValidationError, match="actions"):
        RecallMethodRecipe.model_validate(payload)


def test_deepseek_recipe_requires_frozen_generation_settings() -> None:
    payload = _comparison_recipe()
    payload["generator"] = {**payload["generator"], "temperature": 0.1}

    with pytest.raises(ValidationError, match="temperature"):
        RecallMethodRecipe.model_validate(payload)

    payload = _comparison_recipe()
    payload["generator"] = {**payload["generator"], "repair_attempts": 2}
    with pytest.raises(ValidationError, match="repair_attempts"):
        RecallMethodRecipe.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", False),
        ("temperature", "0"),
        ("temperature", 0.0),
        ("repair_attempts", True),
        ("repair_attempts", "1"),
        ("repair_attempts", 1.0),
    ],
)
def test_deepseek_recipe_rejects_coercible_non_integer_lock_values(
    field: str, value: object
) -> None:
    payload = _comparison_recipe()
    payload["generator"] = {**payload["generator"], field: value}

    with pytest.raises(ValidationError, match=field):
        RecallMethodRecipe.model_validate(payload)


def test_retrieval_actions_are_non_empty_and_limited_to_phase_one_types() -> None:
    payload = _manual_recipe()
    payload["retrieval"] = {**payload["retrieval"], "allowed_actions": []}
    with pytest.raises(ValidationError, match="at least 1"):
        RecallMethodRecipe.model_validate(payload)

    payload = _manual_recipe()
    payload["retrieval"] = {**payload["retrieval"], "allowed_actions": ["web_search"]}
    with pytest.raises(ValidationError):
        RecallMethodRecipe.model_validate(payload)


def test_historical_comparison_uses_only_the_approved_thresholds() -> None:
    recipe = RecallMethodRecipe.model_validate(_comparison_recipe())
    assert recipe.evaluation.repeat_count == 3

    payload = _comparison_recipe()
    payload["evaluation"] = {**payload["evaluation"], "macro_recall_tolerance": float("inf")}
    with pytest.raises(ValidationError, match="macro_recall_tolerance"):
        RecallMethodRecipe.model_validate(payload)

    payload = _comparison_recipe()
    payload["evaluation"] = {**payload["evaluation"], "required_passing_repeats": 1}
    with pytest.raises(ValidationError, match="required_passing_repeats"):
        RecallMethodRecipe.model_validate(payload)


def test_load_recall_recipe_binds_exact_recipe_and_prompt_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prompt = _write(tmp_path / "prompt.yaml", "version: one\n")
    recipe_path = _write(
        tmp_path / "recipe.yaml",
        """method_id: prompted-example
generator:
  type: deepseek_prompt
  prompt: prompt.yaml
  model: deepseek-v4-flash
  temperature: 0
  gold_visibility: blind
  max_generated_actions: 2
  repair_attempts: 1
retrieval:
  allowed_actions: [text_search]
  backend: snapshot_replay
  max_results_per_action: 50
  max_total_actions: 3
candidate_pool: {}
evaluation:
  repeat_count: 1
  max_repeat_attempts: 1
""",
    )

    loaded = load_recall_recipe(recipe_path)

    assert loaded.recipe_bytes == recipe_path.read_bytes()
    assert loaded.recipe_sha256 == _sha256(recipe_path.read_bytes())
    assert loaded.prompt_bytes == prompt.read_bytes()
    assert loaded.prompt_sha256 == _sha256(prompt.read_bytes())


def test_live_backend_needs_separate_runtime_authorization() -> None:
    recipe = RecallMethodRecipe.model_validate(_manual_recipe())

    with pytest.raises(PermissionError, match="runtime authorization"):
        authorize_live_backend(recipe, allow_live=False)
    authorize_live_backend(recipe, allow_live=True)


def test_loader_does_not_require_the_manual_actions_file(tmp_path: Path) -> None:
    recipe_path = _write(
        tmp_path / "recipe.yaml",
        """method_id: manual-example
generator:
  type: manual_actions
  actions: runs/not-yet-pasted.json
  gold_visibility: blind
retrieval:
  allowed_actions: [text_search]
  backend: live_provider
  max_results_per_action: 50
  max_total_actions: 3
candidate_pool: {}
evaluation:
  repeat_count: 1
  max_repeat_attempts: 1
""",
    )

    assert load_recall_recipe(recipe_path).recipe.method_id == "manual-example"


def test_combined_preflight_requires_oracle_catalog_and_legacy_binding() -> None:
    oracle_recipe = RecallMethodRecipe.model_validate(_manual_recipe())
    sample = SampleBinding.model_validate(_sample_binding())
    validate_recipe_sample_preflight(oracle_recipe, sample)

    missing_catalog = sample.model_copy(update={"gold_document_catalog": None})
    with pytest.raises(ValueError, match="Gold-document catalog"):
        validate_recipe_sample_preflight(oracle_recipe, missing_catalog)

    legacy_recipe = RecallMethodRecipe.model_validate(_comparison_recipe())
    without_legacy_proof = sample.model_copy(update={"legacy_candidate_pool_policy": None})
    with pytest.raises(ValueError, match="legacy policy"):
        validate_recipe_sample_preflight(legacy_recipe, without_legacy_proof)


def test_combined_preflight_rejects_oracle_blind_overlap_and_gold_seed_overlap() -> None:
    oracle = SampleBinding.model_validate(_sample_binding(query_ids=["query-a", "query-b"]))
    blind_payload = _sample_binding(query_ids=["query-b"])
    blind_payload["sample_id"] = "sample-b"
    blind = SampleBinding.model_validate(blind_payload)

    with pytest.raises(ValueError, match="overlap"):
        validate_recipe_sample_preflight(
            RecallMethodRecipe.model_validate(_manual_recipe()), oracle, blind_sample=blind
        )

    invalid_payload = _sample_binding()
    invalid_payload["seed_canonical_ids"] = ["gold-a"]
    invalid_sample = SampleBinding.model_validate(invalid_payload)
    with pytest.raises(ValueError, match="Gold"):
        validate_recipe_sample_preflight(
            RecallMethodRecipe.model_validate(_manual_recipe()), invalid_sample
        )


def test_new_method_uses_generator_type_as_the_only_implementation_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "prompt.yaml", "prompt: custom ordinary method\n")
    loaded = load_recall_recipe(
        _write(
            tmp_path / "ordinary-method.yaml",
            """method_id: ordinary-search-terms
generator:
  type: deepseek_prompt
  prompt: prompt.yaml
  model: deepseek-v4-flash
  temperature: 0
  gold_visibility: blind
  max_generated_actions: 1
  repair_attempts: 1
retrieval:
  allowed_actions: [text_search]
  backend: snapshot_replay
  max_results_per_action: 50
  max_total_actions: 1
candidate_pool: {}
evaluation:
  repeat_count: 1
  max_repeat_attempts: 1
""",
        )
    )
    generator_factories = {"deepseek_prompt": lambda recipe: recipe.method_id}

    assert generator_factories[loaded.recipe.generator.type](loaded.recipe) == "ordinary-search-terms"


def test_load_sample_binding_binds_exact_bytes(tmp_path: Path) -> None:
    binding_path = _write(
        tmp_path / "sample.yaml",
        f"""sample_id: sample-a
query_ids: [query-a]
gold_document_catalog:
  path: catalogs/gold.jsonl
  sha256: {_sha256(b"catalog bytes")}
gold_ids: [gold-a]
seed_canonical_ids: [seed-a]
""",
    )

    loaded = load_sample_binding(binding_path)

    assert loaded.binding_bytes == binding_path.read_bytes()
    assert loaded.binding_sha256 == _sha256(binding_path.read_bytes())
