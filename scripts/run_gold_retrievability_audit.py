"""Run the resumable OpenAlex exact-availability stage of a frozen Gold audit."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path

import httpx

from paper_search.learning.gold_retrievability_audit import (
    FrozenAuditManifest,
    GoldAvailabilityRecord,
    probe_gold_identifier,
    selected_gold_by_query,
    summarize_gold_availability,
)


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_atomic(path: Path, payload: object) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_cache(path: Path, manifest_sha256: str) -> dict[str, GoldAvailabilityRecord]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("cache belongs to a different frozen manifest")
    return {
        item["gold_id"]: GoldAvailabilityRecord.model_validate(item)
        for item in payload.get("records", [])
    }


def _save_cache(
    path: Path,
    manifest_sha256: str,
    records: dict[str, GoldAvailabilityRecord],
) -> None:
    _write_atomic(
        path,
        {
            "schema_version": "gold-retrievability-openalex-cache-v1",
            "manifest_sha256": manifest_sha256,
            "records": [
                records[key].model_dump(mode="json") for key in sorted(records)
            ],
        },
    )


async def _run(args: argparse.Namespace) -> dict[str, object]:
    manifest = FrozenAuditManifest.model_validate_json(
        args.manifest.read_text(encoding="utf-8")
    )
    manifest_sha256 = _sha256(args.manifest)
    gold_by_query = selected_gold_by_query(args.partition, manifest)
    gold_ids = sorted(set().union(*gold_by_query.values()))
    records = _load_cache(args.cache, manifest_sha256)
    unexpected = set(records).difference(gold_ids)
    if unexpected:
        raise ValueError("cache contains Gold IDs outside the frozen sample")

    retryable = {
        gold_id
        for gold_id, record in records.items()
        if record.status in {"unknown_transient", "integrity_failure"}
    }
    pending = [gold_id for gold_id in gold_ids if gold_id not in records or gold_id in retryable]
    api_key = os.environ.get("OPENALEX_API_KEY")
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            semaphore = asyncio.Semaphore(args.concurrency)

            async def probe(gold_id: str) -> GoldAvailabilityRecord:
                async with semaphore:
                    return await probe_gold_identifier(
                        gold_id,
                        client=client,
                        api_key=api_key,
                    )

            results = await asyncio.gather(*(probe(gold_id) for gold_id in batch))
            records.update({record.gold_id: record for record in results})
            _save_cache(args.cache, manifest_sha256, records)
            print(f"completed={len(records)}/{len(gold_ids)}", flush=True)

    if set(records) != set(gold_ids):
        raise ValueError("availability cache is incomplete")
    summary = summarize_gold_availability(manifest, gold_by_query, records)
    summary["manifest_sha256"] = manifest_sha256
    summary["cache_sha256"] = _sha256(args.cache)
    summary["provider"] = "openalex"
    summary["lookup_contract"] = "arxiv_to_datacite_doi_exact_work_endpoint"
    _write_atomic(args.output, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()
    if args.concurrency <= 0 or args.batch_size <= 0:
        parser.error("concurrency and batch-size must be positive")
    summary = asyncio.run(_run(args))
    print(f"diagnostic_complete={summary['diagnostic_complete']}")
    print(f"status_counts={summary['status_counts']}")
    print(f"result_sha256={_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
