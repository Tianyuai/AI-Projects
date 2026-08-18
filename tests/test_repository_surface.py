from pathlib import Path

import pytest

from paper_search.application.experiments import expected_experiment_flags


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "relative_path",
    (
        "configs/ablations.yaml",
        "configs/prompts/query_evolve.yaml",
        "configs/title_candidates.yaml",
        "src/paper_search/evolution",
        "src/paper_search/ranking/embedding.py",
        "src/paper_search/ranking/llm_stage.py",
        "src/paper_search/ranking/rerank.py",
        "src/paper_search/ranking/sentence_transformer.py",
        "src/paper_search/retrieval/title_candidates.py",
        "scripts/probe_query_evolution.py",
        "docs/superpowers",
    ),
)
def test_deprecated_repository_surfaces_are_absent(relative_path: str) -> None:
    candidate = ROOT / relative_path
    if candidate.is_dir():
        assert not any(candidate.rglob("*.py"))
    else:
        assert not candidate.exists()


@pytest.mark.parametrize(
    "name",
    (
        "embedding",
        "citation-expansion",
        "llm-rerank",
        "title-candidates",
        "fixed-two-round",
        "adaptive-evolution",
    ),
)
def test_deprecated_production_experiments_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="unknown experiment name"):
        expected_experiment_flags(name)
