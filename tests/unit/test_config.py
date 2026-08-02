from pathlib import Path

import pytest
from pydantic import ValidationError

import paper_search.config as config_module
from paper_search.config import RuntimeConfig, validate_mode_authorization


def load_runtime_config(
    path: str | Path, *, env_file: str | Path | None = ".env"
) -> RuntimeConfig:
    return config_module.load_runtime_config(path, env_file=env_file)


def write_runtime_files(directory: Path, *, extra_base: str = "", budget: str = "") -> Path:
    base = directory / "base.yaml"
    base.write_text(
        "\n".join(
            [
                "budget_profile: balanced",
                "llm_base_url: https://example.test/v1",
                "llm_model_primary: base-primary",
                "llm_model_fallback: base-fallback",
                "runtime:",
                "  allow_live: false",
                "  artifact_root: artifacts",
                extra_base,
            ]
        ),
        encoding="utf-8",
    )
    (directory / "budget_balanced.yaml").write_text(
        budget
        or "\n".join(
            [
                "max_total_tokens: 24000",
                "max_cost_cny: 0.30",
            ]
        ),
        encoding="utf-8",
    )
    return base


def test_environment_variables_load_secrets_but_cannot_override_frozen_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = write_runtime_files(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-secret")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://env.example.test/v1")
    monkeypatch.setenv("LLM_MODEL_PRIMARY", "env-primary")
    monkeypatch.setenv("LLM_MODEL_FALLBACK", "env-fallback")

    with pytest.raises(ValueError, match="frozen"):
        load_runtime_config(config_path, env_file=None)

    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.delenv("LLM_MODEL_PRIMARY")
    monkeypatch.delenv("LLM_MODEL_FALLBACK")
    config = load_runtime_config(config_path, env_file=None)

    assert config.llm_base_url == "https://example.test/v1"
    assert config.llm_model_primary == "base-primary"
    assert config.llm_model_fallback == "base-fallback"
    assert config.openalex_api_key is not None
    assert config.openalex_api_key.get_secret_value() == "openalex-secret"
    assert config.llm_api_key is not None
    assert config.llm_api_key.get_secret_value() == "llm-secret"
    assert config.budget.max_total_tokens == 24_000


def test_config_hash_is_stable_and_excludes_secrets(tmp_path: Path) -> None:
    config_path = write_runtime_files(tmp_path)
    first = load_runtime_config(config_path, env_file=None)
    first_hash = first.config_hash()
    changed_secret = first.model_copy(update={"llm_api_key": "different-secret"})

    assert first_hash.startswith("sha256:")
    assert changed_secret.config_hash() == first_hash


def test_config_hash_excludes_operational_artifact_root_but_not_runtime_behavior(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(write_runtime_files(tmp_path), env_file=None)
    relocated = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"artifact_root": Path("elsewhere")})}
    )
    changed_runtime_behavior = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(update={"allow_live": True})
        }
    )

    assert relocated.config_hash() == config.config_hash()
    assert changed_runtime_behavior.config_hash() != config.config_hash()


def test_embedding_config_defaults_off_and_is_hashed(tmp_path: Path) -> None:
    config_path = write_runtime_files(tmp_path)

    config = load_runtime_config(config_path, env_file=None)
    enabled = config.model_copy(
        update={
            "embedding": config.embedding.model_copy(
                update={"enabled": True}
            )
        }
    )

    assert config.embedding.enabled is False
    assert config.embedding.device == "cpu"
    assert config.embedding.batch_size == 16
    assert config.embedding.fallback_to_cpu is True
    assert enabled.config_hash() != config.config_hash()


def test_runtime_settings_and_policy_bindings_are_reproducible(tmp_path: Path) -> None:
    config_path = write_runtime_files(tmp_path)

    config = load_runtime_config(config_path, env_file=None)
    changed_runtime = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"allow_live": True})}
    )
    changed_policy = config.model_copy(
        update={
            "policy_bindings": config.policy_bindings.model_copy(
                update={"pricing_policy": "configs/pricing_v2.yaml"}
            )
        }
    )

    assert config.runtime.allow_live is False
    assert config.runtime.artifact_root == Path("artifacts")
    assert config.runtime.connect_timeout_seconds == 5
    assert config.runtime.read_timeout_seconds == 20
    assert config.runtime.write_timeout_seconds == 20
    assert config.runtime.pool_timeout_seconds == 5
    assert config.runtime.max_attempts == 3
    assert config.capture_policy.snapshot_schema == "dependency-snapshot-v2"
    assert config.routing.openalex_calls_min == 3
    assert config.routing.openalex_calls_max == 6
    assert config.retry.retryable_statuses == (429, "5xx")
    assert changed_runtime.config_hash() != config.config_hash()
    assert changed_policy.config_hash() != config.config_hash()


def test_base_config_preserves_the_balanced_integrated_baseline() -> None:
    config = load_runtime_config(Path("configs/base.yaml"), env_file=None)

    assert config.budget.max_search_api_calls == 12
    assert config.budget.max_llm_calls == 5
    assert config.budget.max_iterations == 2
    assert config.routing.openalex_calls_max == 6
    assert config.budget.max_elapsed_seconds == 90
    assert config.budget.soft_deadline_seconds == 80
    assert config.budget.max_total_tokens == 24_000
    assert config.budget.max_output_papers == 50
    assert config.budget.max_cost_cny == pytest.approx(0.30)


def test_embedding_config_accepts_explicit_cuda_with_cpu_fallback(
    tmp_path: Path,
) -> None:
    config_path = write_runtime_files(
        tmp_path,
        extra_base="\n".join(
            [
                "embedding:",
                "  enabled: true",
                "  model_id: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "  device: cuda",
                "  batch_size: 8",
                "  fallback_to_cpu: true",
            ]
        ),
    )

    config = load_runtime_config(config_path, env_file=None)

    assert config.embedding.enabled is True
    assert config.embedding.device == "cuda"
    assert config.embedding.batch_size == 8


@pytest.mark.parametrize("batch_size", [0, 129, True])
def test_embedding_config_rejects_unsafe_batch_size(
    tmp_path: Path,
    batch_size: object,
) -> None:
    config_path = write_runtime_files(
        tmp_path,
        extra_base="\n".join(
            [
                "embedding:",
                "  enabled: false",
                "  model_id: fixture/model",
                "  device: cpu",
                f"  batch_size: {str(batch_size).lower()}",
                "  fallback_to_cpu: true",
            ]
        ),
    )

    with pytest.raises(ValidationError):
        load_runtime_config(config_path, env_file=None)


def test_unknown_yaml_field_is_rejected(tmp_path: Path) -> None:
    config_path = write_runtime_files(tmp_path, extra_base="unexpected: true")

    with pytest.raises(ValidationError):
        load_runtime_config(config_path, env_file=None)


def test_missing_token_limit_is_rejected_at_load_time(tmp_path: Path) -> None:
    config_path = write_runtime_files(tmp_path, budget="max_cost_cny: 0.30")

    with pytest.raises(ValidationError):
        load_runtime_config(config_path, env_file=None)


def test_runtime_config_rejects_unknown_fields() -> None:
    assert hasattr(config_module, "RuntimeConfig"), "RuntimeConfig must be implemented"
    with pytest.raises(ValidationError):
        config_module.RuntimeConfig.model_validate(
            {
                "budget_profile": "balanced",
                "budget": {"max_total_tokens": 100, "max_cost_cny": 0.1},
                "llm_base_url": "https://example.test/v1",
                "llm_model_primary": "primary",
                "llm_model_fallback": "fallback",
                "runtime": {"artifact_root": "artifacts"},
                "unexpected": True,
            }
        )


def test_direct_runtime_config_constructors_receive_safe_runtime_defaults() -> None:
    config = config_module.RuntimeConfig.model_validate(
        {
            "budget_profile": "balanced",
            "budget": {"max_total_tokens": 100, "max_cost_cny": 0.1},
            "llm_base_url": "https://example.test/v1",
            "llm_model_primary": "primary",
            "llm_model_fallback": "fallback",
        }
    )

    assert config.runtime.allow_live is False
    assert config.runtime.artifact_root == Path("artifacts")


@pytest.mark.parametrize(
    ("mode", "runtime_allow_live", "network_authorized", "message"),
    [
        ("replay", True, True, "network"),
        ("live", False, True, "allow live"),
        ("live", True, False, "network"),
    ],
)
def test_mode_authorization_fails_closed(
    mode: str,
    runtime_allow_live: bool,
    network_authorized: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_mode_authorization(
            mode=mode,  # type: ignore[arg-type]
            runtime_allow_live=runtime_allow_live,
            network_authorized=network_authorized,
        )


@pytest.mark.parametrize("mode", ["replay", "live"])
def test_mode_authorization_accepts_only_the_exact_allowed_matrix(mode: str) -> None:
    validate_mode_authorization(
        mode=mode,  # type: ignore[arg-type]
        runtime_allow_live=mode == "live",
        network_authorized=mode == "live",
    )


@pytest.mark.parametrize(
    "secret_field",
    ["openalex_api_key", "semantic_scholar_api_key", "llm_api_key"],
)
def test_yaml_cannot_contain_api_secrets(tmp_path: Path, secret_field: str) -> None:
    config_path = write_runtime_files(tmp_path, extra_base=f"{secret_field}: leaked")

    with pytest.raises(ValueError, match="secret"):
        load_runtime_config(config_path, env_file=None)


def test_process_environment_overrides_dotenv_for_secrets_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = write_runtime_files(tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_MODEL_PRIMARY=dotenv-primary\nLLM_API_KEY=dotenv-secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LLM_MODEL_PRIMARY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="frozen"):
        load_runtime_config(config_path, env_file=env_path)

    env_path.write_text("LLM_API_KEY=dotenv-secret\n", encoding="utf-8")
    dotenv_config = load_runtime_config(config_path, env_file=env_path)
    monkeypatch.setenv("LLM_API_KEY", "process-secret")
    process_config = load_runtime_config(config_path, env_file=env_path)

    assert dotenv_config.llm_api_key is not None
    assert dotenv_config.llm_api_key.get_secret_value() == "dotenv-secret"
    assert process_config.llm_api_key is not None
    assert process_config.llm_api_key.get_secret_value() == "process-secret"
