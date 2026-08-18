from __future__ import annotations

import json
import asyncio
from pathlib import Path

import httpx
import pytest

from paper_search.learning.gold_retrievability_audit import (
    FrozenAuditManifest,
    GoldAvailabilityRecord,
    build_frozen_audit_manifest,
    freeze_audit_manifest,
    openalex_exact_url,
    probe_gold_identifier,
    selected_gold_by_query,
    summarize_gold_availability,
)


def _write_partition(path: Path, *, count: int = 48, split: str = "auto_train") -> None:
    rows = []
    leads = ("Which works", "Could you provide", "How do papers", "Studies that")
    for index in range(count):
        rows.append(
            {
                "dataset": "pasa",
                "split": split,
                "role": "training" if split == "auto_train" else "development",
                "revision": "a" * 40,
                "query_id": f"q-{index:03d}",
                "query": f"{leads[index % len(leads)]} discuss token {index} "
                + "long " * (index % 9),
                "gold_paper_ids": [f"arxiv:2001.{index:05d}"]
                * (1 + index % 4),
                "source_components": [],
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_manifest_is_deterministic_stratified_and_fold_balanced(tmp_path: Path) -> None:
    partition = tmp_path / "train.jsonl"
    _write_partition(partition)

    first = build_frozen_audit_manifest(partition, sample_size=24, seed="audit-v1")
    second = build_frozen_audit_manifest(partition, sample_size=24, seed="audit-v1")

    assert first == second
    assert first.dataset == "pasa"
    assert first.split == "auto_train"
    assert first.role == "training"
    assert first.population_query_count == 48
    assert first.sample_query_count == 24
    assert sum(first.fold_counts.values()) == 24
    assert max(first.fold_counts.values()) - min(first.fold_counts.values()) <= 1
    assert len({item.query_id for item in first.sample}) == 24
    assert all(item.stratum for item in first.sample)
    assert all(item.fold in {1, 2, 3} for item in first.sample)
    assert sum(first.stratum_sample_counts.values()) == 24
    assert set(first.stratum_sample_counts) <= set(first.stratum_population_counts)


def test_manifest_supports_development_and_excludes_prior_method_queries(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "dev.jsonl"
    _write_partition(partition, split="auto_dev")

    manifest = build_frozen_audit_manifest(
        partition,
        sample_size=12,
        seed="router-dev-v1",
        excluded_query_ids=frozenset({"q-000", "q-001", "q-002"}),
    )

    assert manifest.split == "auto_dev"
    assert manifest.role == "development"
    assert manifest.source_query_count == 48
    assert manifest.population_query_count == 45
    assert manifest.excluded_query_count == 3
    assert manifest.excluded_query_ids_sha256 is not None
    assert not {"q-000", "q-001", "q-002"}.intersection(
        item.query_id for item in manifest.sample
    )


def test_freeze_is_idempotent_but_refuses_changed_bytes(tmp_path: Path) -> None:
    partition = tmp_path / "train.jsonl"
    output = tmp_path / "manifest.json"
    _write_partition(partition)
    manifest = build_frozen_audit_manifest(partition, sample_size=24, seed="audit-v1")

    first_hash = freeze_audit_manifest(output, manifest)
    second_hash = freeze_audit_manifest(output, manifest)

    assert first_hash == second_hash
    persisted = FrozenAuditManifest.model_validate_json(output.read_text(encoding="utf-8"))
    assert persisted == manifest

    changed = manifest.model_copy(update={"seed": "changed"})
    with pytest.raises(FileExistsError, match="different frozen audit manifest"):
        freeze_audit_manifest(output, changed)


def test_arxiv_gold_uses_canonical_datacite_doi_endpoint() -> None:
    assert openalex_exact_url("arxiv:1810.09726") == (
        "https://api.openalex.org/works/"
        "https:%2F%2Fdoi.org%2F10.48550%2Farxiv.1810.09726"
    )


def test_probe_accepts_exact_doi_resolution_and_classifies_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "1810.09726" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "id": "https://openalex.org/W1",
                    # OpenAlex may resolve the requested DOI alias while returning
                    # a different canonical DOI for the same Work.
                    "doi": "https://doi.org/10.52202/068431-1689",
                },
            )
        return httpx.Response(404)

    async def run() -> tuple[str, str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            available = await probe_gold_identifier("arxiv:1810.09726", client=client)
            missing = await probe_gold_identifier("arxiv:2001.00001", client=client)
        return available.status, missing.status

    assert asyncio.run(run()) == ("available", "exact_not_found")


def test_availability_summary_uses_query_association_and_fold_denominators(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "train.jsonl"
    _write_partition(partition, count=12)
    manifest = build_frozen_audit_manifest(partition, sample_size=12, seed="audit-v1")
    gold_by_query = selected_gold_by_query(partition, manifest)
    unique_gold = sorted(set().union(*gold_by_query.values()))
    records = {
        gold_id: GoldAvailabilityRecord(
            gold_id=gold_id,
            status="available" if index % 2 == 0 else "exact_not_found",
            attempts=1,
            http_status=200 if index % 2 == 0 else 404,
            openalex_id=f"https://openalex.org/W{index}" if index % 2 == 0 else None,
            resolved_doi=(
                f"https://doi.org/10.48550/{gold_id.replace(':', '.')}"
                if index % 2 == 0
                else None
            ),
        )
        for index, gold_id in enumerate(unique_gold)
    }

    summary = summarize_gold_availability(manifest, gold_by_query, records)

    assert summary["unique_gold_count"] == 12
    assert summary["gold_association_count"] == 12
    assert summary["status_counts"] == {"available": 6, "exact_not_found": 6}
    assert summary["query_count"] == 12
    assert sum(item["query_count"] for item in summary["folds"].values()) == 12
    assert summary["unique_gold_availability"]["estimate"] == 0.5
    assert summary["diagnostic_complete"] is True
