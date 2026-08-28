from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from paper_search.application.composition import CompositionRoot
from paper_search.application.locks import load_verified_input_lock
from paper_search.control.budget import HardBudgetController
from paper_search.control.pricing import parse_pricing_policy_bytes
from paper_search.domain.models import SearchBudget


ROOT = Path(__file__).resolve().parents[2]


def test_demo_replay_lock_binds_the_full_production_ranker_chain() -> None:
    verified = load_verified_input_lock(
        ROOT / "deliverables/demo/replay_demo.lock.yaml", artifact_root=ROOT
    )

    binding = verified.lock.baseline.document_ranker
    assert binding is not None
    assert binding.manifest.path == (
        "artifacts/models/gated-feature-fusion-18314-unified-context-v3-v1/manifest.json"
    )
    assert binding.fallback_manifest is not None
    assert binding.fallback_manifest.path == (
        "artifacts/models/reliability-fusion-18314-unified-context-v3-v1/manifest.json"
    )
    assert binding.emergency_manifest is not None
    assert binding.emergency_manifest.path == (
        "artifacts/models/cpu-pairwise-document-ranker-expanded2385-v1/manifest.json"
    )


def test_evaluator_live_lock_is_verified_and_binds_the_same_ranker_chain() -> None:
    verified = load_verified_input_lock(
        ROOT / "deliverables/evaluator/live-evaluator.lock.yaml", artifact_root=ROOT
    )

    assert verified.lock.lock_kind == "candidate"
    assert verified.lock.runtime_allow_live is True
    assert verified.lock.baseline.primary_model == "deepseek-v4-flash"
    assert verified.lock.baseline.retrieval.semantic_scholar_calls_max == 2
    bridge = verified.lock.baseline.supervised_lexical_bridge
    aliases = verified.lock.baseline.pasa_identity_aliases
    low_confidence = verified.lock.baseline.low_confidence_llm_supplement
    assert bridge is not None
    assert aliases is not None
    assert low_confidence is not None
    assert low_confidence.prompt_version == "query-analyze-protected-actions-v3"
    assert low_confidence.max_total_deduplicated_candidates == 250
    assert aliases.alias_count == 2571
    assert bridge.manifest.path in verified.artifact_bytes
    assert bridge.model.path in verified.artifact_bytes
    assert aliases.alias_map.path in verified.artifact_bytes
    assert low_confidence.prompt_config.path in verified.artifact_bytes
    pricing = parse_pricing_policy_bytes(
        verified.artifact_bytes[verified.lock.pricing_policy.path]
    )
    assert pricing.billing_schedule == "beijing-weekday-peak-offpeak-v1"
    assert pricing.concurrency_limits["deepseek-v4-flash"] == 2500
    binding = verified.lock.baseline.document_ranker
    assert binding is not None
    assert binding.manifest.path.endswith(
        "gated-feature-fusion-18314-unified-context-v3-v1/manifest.json"
    )
    assert binding.fallback_manifest is not None
    assert binding.fallback_manifest.path.endswith(
        "reliability-fusion-18314-unified-context-v3-v1/manifest.json"
    )
    assert binding.emergency_manifest is not None
    assert binding.emergency_manifest.path.endswith(
        "cpu-pairwise-document-ranker-expanded2385-v1/manifest.json"
    )


def test_evaluator_live_lock_composes_all_offline_production_extensions(
    tmp_path: Path,
) -> None:
    bundle = CompositionRoot.compose(
        lock_path=ROOT / "deliverables/evaluator/live-evaluator.lock.yaml",
        mode="live",
        artifact_root=ROOT,
        output_root=tmp_path / "output",
        network_authorized=True,
        environ={"LLM_API_KEY": "offline-fixture-secret"},
    )
    factory = bundle.service._orchestrator_factory  # noqa: SLF001

    assert factory._query_plan_enricher is not None  # noqa: SLF001
    assert factory._identifier_map is not None  # noqa: SLF001
    assert factory._identifier_alias_count >= 2571  # noqa: SLF001
    assert factory._document_ranker.deployment_role == "F5-gated-fusion"  # noqa: SLF001
    assert factory._lock.baseline.low_confidence_llm_supplement is not None  # noqa: SLF001
    assert factory._low_confidence_prompt_instructions is not None  # noqa: SLF001
    budget = SearchBudget.model_validate(
        yaml.safe_load((ROOT / "configs/budget_balanced.yaml").read_bytes())
    )
    runtime = factory(
        HardBudgetController(budget, formal_live=True),
        "offline-production-wiring",
    )
    orchestrator = runtime._orchestrator  # noqa: SLF001
    assert orchestrator._low_confidence_analyzer is not None  # noqa: SLF001
    assert orchestrator._max_low_confidence_raw_candidates == 50  # noqa: SLF001
    assert orchestrator._max_low_confidence_deduplicated_candidates == 50  # noqa: SLF001
    asyncio.run(bundle.aclose())
