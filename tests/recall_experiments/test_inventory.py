"""Offline evidence inventory tests for the candidate-recall harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from paper_search.recall_experiments.inventory import (
    InventoryError,
    build_inventory,
    classify_request,
    frozen_request_identities,
    main,
)


WORKSPACE_ROOT = Path(__file__).parents[2]
CONFIG_ROOT = WORKSPACE_ROOT / "configs" / "recall_experiments" / "historical"


def test_workspace_bindings_inventory_all_methods_and_evidence_classes() -> None:
    report = build_inventory(CONFIG_ROOT, workspace_root=WORKSPACE_ROOT)

    assert {method["method_id"] for method in report["methods"]} == {
        "query-rewrite",
        "llm-query-variants",
        "query-evolution",
        "title-candidates",
        "citation-expansion",
    }
    assert set(report["source_statuses"]) == {
        "exact",
        "aggregate_only",
        "insufficient",
        "not_comparable",
    }
    for method in report["methods"]:
        assert method["source_paths"]
        assert method["source_sha256"]
        assert method["method_id"]
        assert method["query_ids"]
        assert method["action_family"]
        assert method["candidate_pool_policy_version"]
        assert set(method["evidence"]) == {
            "actions",
            "provider_responses",
            "candidate_pool",
            "per_query_gold_hits",
            "aggregate_metrics",
            "prompt_model_visibility",
            "gold_denominator",
        }


def test_report_inventories_gold_associations_and_catalog_coverage() -> None:
    report = build_inventory(CONFIG_ROOT, workspace_root=WORKSPACE_ROOT)

    gold = report["gold_catalog"]
    assert gold["association_source"]["path"] == "data/dev/gold.jsonl"
    assert gold["association_source"]["kind"] == "association_only"
    assert "document_sources" in gold
    assert gold["per_query_coverage"]
    for coverage in gold["per_query_coverage"]:
        assert set(coverage) == {
            "query_id",
            "gold_count",
            "all_gold_have_titles",
            "title_count",
            "abstract_count",
            "author_count",
            "year_count",
        }
    residual_networks = next(
        item for item in gold["per_query_coverage"] if item["query_id"] == "AutoScholarQuery_dev_287"
    )
    assert residual_networks["title_count"] == 1
    assert residual_networks["abstract_count"] == 1
    assert residual_networks["author_count"] == 1
    assert residual_networks["year_count"] == 1


def test_frozen_and_novel_requests_have_distinct_backend_classifications() -> None:
    frozen = {"sha256:known"}

    assert classify_request("sha256:known", frozen) == "exact_snapshot_replay"
    assert (
        classify_request("sha256:novel", frozen)
        == "requires_live_provider_or_local_index"
    )


def test_overall_compatibility_accepts_citation_as_the_non_text_family() -> None:
    from paper_search.recall_experiments.inventory import _has_overall_compatibility

    assert _has_overall_compatibility(
        [{"action_family": "text_search"}, {"action_family": "citation_expand"}]
    )


def test_bound_snapshot_entries_are_verified_and_reported() -> None:
    manifest = (
        WORKSPACE_ROOT
        / "runs"
        / "_diag_query_evolution_query-evolution-prompt-v2-full-20260810"
        / "snapshots"
        / "snapshot-manifest.json"
    )
    frozen_requests = frozen_request_identities(manifest, workspace_root=WORKSPACE_ROOT)
    assert frozen_requests

    report = build_inventory(CONFIG_ROOT, workspace_root=WORKSPACE_ROOT)
    reported = report["backend_classifications"]["frozen_requests"]
    assert reported[0]["classification"] == "exact_snapshot_replay"
    assert {item["request_identity"] for item in reported} == set(frozen_requests)


@pytest.mark.parametrize("failure", ["missing", "changed_hash"])
def test_snapshot_verification_fails_closed_for_invalid_response(
    tmp_path: Path, failure: str
) -> None:
    manifest = tmp_path / "snapshot-manifest.json"
    response = tmp_path / "responses" / "response.bin"
    response.parent.mkdir()
    response.write_bytes(b"actual response")
    response_path = "responses/response.bin"
    expected_hash = hashlib.sha256(response.read_bytes()).hexdigest()
    if failure == "missing":
        response_path = "responses/missing.bin"
    else:
        expected_hash = "0" * 64
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "request": {"canonical_request_sha256": "sha256:known"},
                        "response_path": response_path,
                        "response_sha256": "sha256:" + expected_hash,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InventoryError):
        frozen_request_identities(manifest, workspace_root=tmp_path)


@pytest.mark.parametrize("failure", ["duplicate", "missing", "changed_hash", "unbound"])
def test_invalid_source_binding_fails_closed(tmp_path: Path, failure: str) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"query_ids": ["q1"]}), encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source_path = "source.json"
    if failure == "missing":
        source_path = "missing.json"
    if failure == "changed_hash":
        digest = "0" * 64

    binding = {
        "method_id": "method-a",
        "source_paths": [source_path],
        "source_sha256": [digest],
        "query_ids": ["q1"],
        "action_family": "text_search",
        "evidence": {
            "actions": {"path": source_path, "available": True},
            "provider_responses": {"available": False},
            "candidate_pool": {"available": False},
            "per_query_gold_hits": {"available": False},
            "aggregate_metrics": {"available": False},
            "prompt_model_visibility": {"available": False},
            "gold_denominator": {"available": False},
        },
        "generation_recipe_reconstructability": "not_reconstructable",
        "candidate_pool_policy_version": "production-dedup-v1",
        "status": "insufficient",
        "missing_or_misaligned": ["test"],
        "gold_catalog": {
            "association_source": {"path": source_path, "sha256": digest},
            "document_sources": [],
        },
    }
    if failure == "duplicate":
        duplicate = dict(binding)
        duplicate["method_id"] = "method-b"
        (tmp_path / "second.yaml").write_text(yaml.safe_dump(duplicate), encoding="utf-8")
    if failure == "unbound":
        binding["evidence"]["actions"]["path"] = "unbound.json"

    (tmp_path / "first.yaml").write_text(yaml.safe_dump(binding), encoding="utf-8")

    with pytest.raises(InventoryError):
        build_inventory(tmp_path, workspace_root=tmp_path)


def test_inventory_module_uses_only_offline_standard_library_adapters() -> None:
    source = (WORKSPACE_ROOT / "src" / "paper_search" / "recall_experiments" / "inventory.py").read_text(
        encoding="utf-8"
    )

    for forbidden in ("paper_search.llm", "OpenAlexProvider", "dotenv", "sqlite3"):
        assert forbidden not in source


def test_command_rejects_output_outside_the_diagnostic_directory(tmp_path: Path) -> None:
    with pytest.raises(InventoryError):
        main(
            [
                "--config-root",
                str(CONFIG_ROOT),
                "--out",
                str(tmp_path / "not-the-diagnostic-directory"),
            ]
        )
