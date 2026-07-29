from pathlib import Path

import pytest
from pydantic import ValidationError

import paper_search.config as config_module


def load_runtime_config(*args: object, **kwargs: object) -> object:
    assert hasattr(config_module, "load_runtime_config"), "load_runtime_config must be implemented"
    return config_module.load_runtime_config(*args, **kwargs)


def write_runtime_files(directory: Path, *, extra_base: str = "", budget: str = "") -> Path:
    base = directory / "base.yaml"
    base.write_text(
        "\n".join(
            [
                "budget_profile: balanced",
                "llm_base_url: https://example.test/v1",
                "llm_model_primary: base-primary",
                "llm_model_fallback: base-fallback",
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


def test_environment_variables_override_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = write_runtime_files(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-secret")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://env.example.test/v1")
    monkeypatch.setenv("LLM_MODEL_PRIMARY", "env-primary")
    monkeypatch.setenv("LLM_MODEL_FALLBACK", "env-fallback")

    config = load_runtime_config(config_path, env_file=None)

    assert config.llm_base_url == "https://env.example.test/v1"
    assert config.llm_model_primary == "env-primary"
    assert config.llm_model_fallback == "env-fallback"
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
                "unexpected": True,
            }
        )


@pytest.mark.parametrize(
    "secret_field",
    ["openalex_api_key", "semantic_scholar_api_key", "llm_api_key"],
)
def test_yaml_cannot_contain_api_secrets(tmp_path: Path, secret_field: str) -> None:
    config_path = write_runtime_files(tmp_path, extra_base=f"{secret_field}: leaked")

    with pytest.raises(ValueError, match="secret"):
        load_runtime_config(config_path, env_file=None)


def test_process_environment_overrides_dotenv_and_dotenv_overrides_yaml(
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

    dotenv_config = load_runtime_config(config_path, env_file=env_path)
    monkeypatch.setenv("LLM_MODEL_PRIMARY", "process-primary")
    process_config = load_runtime_config(config_path, env_file=env_path)

    assert dotenv_config.llm_model_primary == "dotenv-primary"
    assert dotenv_config.llm_api_key is not None
    assert dotenv_config.llm_api_key.get_secret_value() == "dotenv-secret"
    assert process_config.llm_model_primary == "process-primary"
