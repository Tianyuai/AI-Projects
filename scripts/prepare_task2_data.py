from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import ValidationError

from paper_search.evaluation.dataset import (
    EvaluationQuery,
    answer_count_bucket,
    stratified_sample,
    write_frozen_bytes,
)
from paper_search.evaluation.official_adapter import PaSaRecord, adapt_pasa_record


PASA_REPO_ID = "CarlanLark/pasa-dataset"
PASA_REVISION = "232428b0c867268c3b8ded90db4d98c1b30501d6"
PASA_FILES = (
    "AutoScholarQuery/dev.jsonl",
    "AutoScholarQuery/test.jsonl",
    "RealScholarQuery/test.jsonl",
)
PASA_EXPECTED_COUNTS = {
    "AutoScholarQuery/dev.jsonl": 1000,
    "AutoScholarQuery/test.jsonl": 1000,
    "RealScholarQuery/test.jsonl": 50,
}
RANDOM_SEED = 20260714
SAMPLING_ALGORITHM = "answer-count-largest-remainder-v1"
HUMAN_LABEL_STATUS = "waiting_for_human_label_freeze"


Downloader = Callable[[str, str, str, str], bytes]


def download_file(repo_id: str, revision: str, path: str, token: str) -> bytes:
    """Download one fixed-revision Hugging Face dataset file."""
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        content = response.read()
        if not isinstance(content, bytes):
            raise TypeError("dataset response body must be bytes")
        return content


def _parse_source(content: bytes, source_path: str) -> list[PaSaRecord]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{source_path}: invalid UTF-8") from None

    records: list[PaSaRecord] = []
    seen_query_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{source_path}:{line_number}: blank line is not allowed")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{source_path}:{line_number}: invalid JSON: {error.msg}"
            ) from None
        if not isinstance(payload, dict):
            raise ValueError(f"{source_path}:{line_number}: expected a JSON object")
        try:
            record = PaSaRecord.model_validate(payload)
        except ValidationError:
            raise ValueError(
                f"{source_path}:{line_number}: record validation failed"
            ) from None
        if record.qid in seen_query_ids:
            raise ValueError(
                f"{source_path}:{line_number}: duplicate query_id: {record.qid}"
            )
        seen_query_ids.add(record.qid)
        records.append(record)
    return records


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    ).encode()


def _jsonl_bytes(records: Sequence[EvaluationQuery]) -> bytes:
    lines = [
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode()


def _adapt_records(
    records: Sequence[PaSaRecord],
    *,
    source: str,
    split: str,
) -> list[EvaluationQuery]:
    return [
        adapt_pasa_record(
            record,
            source=source,
            split=split,
            revision=PASA_REVISION,
        )
        for record in records
    ]


def _sample_queries(records: Sequence[EvaluationQuery], size: int) -> list[EvaluationQuery]:
    return stratified_sample(
        records,
        size,
        seed=RANDOM_SEED,
        key=lambda record: record.query_id,
        stratum=lambda record: answer_count_bucket(len(record.relevant_paper_ids)),
    )


def prepare(
    *,
    output_root: Path,
    token: str | None,
    downloader: Downloader = download_file,
    expected_counts: dict[str, int] | None = None,
    dev_size: int = 60,
    validation_size: int = 30,
    simulated_test_size: int = 50,
) -> dict[str, object]:
    """Download, validate, sample, and freeze the approved PaSa inputs."""
    if token is None or not token.strip():
        raise RuntimeError("HF_TOKEN is required for PaSa data preparation")

    expected = PASA_EXPECTED_COUNTS if expected_counts is None else expected_counts
    downloaded = {
        path: downloader(PASA_REPO_ID, PASA_REVISION, path, token)
        for path in PASA_FILES
    }
    parsed = {path: _parse_source(downloaded[path], path) for path in PASA_FILES}
    for path in PASA_FILES:
        expected_count = expected.get(path)
        if expected_count is None:
            raise ValueError(f"missing expected count for {path}")
        actual_count = len(parsed[path])
        if actual_count != expected_count:
            raise ValueError(
                f"{path}: expected {expected_count} records, found {actual_count}"
            )

    dev_candidates = _adapt_records(
        parsed["AutoScholarQuery/dev.jsonl"],
        source="AutoScholarQuery",
        split="dev",
    )
    validation_candidates = _adapt_records(
        parsed["AutoScholarQuery/test.jsonl"],
        source="AutoScholarQuery",
        split="validation",
    )
    simulated_test = _adapt_records(
        parsed["RealScholarQuery/test.jsonl"],
        source="RealScholarQuery",
        split="simulated_test",
    )
    if simulated_test_size != len(simulated_test):
        raise ValueError("simulated test size must equal the complete frozen source")

    partitions = {
        "dev": _sample_queries(dev_candidates, dev_size),
        "validation": _sample_queries(validation_candidates, validation_size),
        "simulated_test": simulated_test,
    }

    source_files: list[dict[str, object]] = []
    for path in PASA_FILES:
        content = downloaded[path]
        source_files.append(
            {
                "path": path,
                "raw_path": f"raw/{path}",
                "row_count": len(parsed[path]),
                "byte_count": len(content),
                "sha256": _sha256_bytes(content),
            }
        )

    partition_manifest: dict[str, dict[str, object]] = {}
    output_payloads: list[tuple[Path, bytes]] = []
    for name, records in partitions.items():
        gold_path = output_root / name / "gold.jsonl"
        ids_path = output_root / "splits" / f"{name}.ids.json"
        ids_content = _json_bytes([record.query_id for record in records])
        output_payloads.extend(
            [
                (gold_path, _jsonl_bytes(records)),
                (ids_path, ids_content),
            ]
        )
        partition_manifest[name] = {
            "count": len(records),
            "gold_path": f"{name}/gold.jsonl",
            "ids_path": f"splits/{name}.ids.json",
            "ids_sha256": _sha256_bytes(ids_content),
        }

    manifest: dict[str, object] = {
        "repo_id": PASA_REPO_ID,
        "revision": PASA_REVISION,
        "license": "CC-BY-NC-SA-4.0",
        "access": "gated-hugging-face-dataset",
        "random_seed": RANDOM_SEED,
        "sampling_algorithm": SAMPLING_ALGORITHM,
        "status": HUMAN_LABEL_STATUS,
        "source_files": source_files,
        "partitions": partition_manifest,
    }

    for source_path in PASA_FILES:
        write_frozen_bytes(
            output_root / "raw" / source_path,
            downloaded[source_path],
        )
    for output_path, content in output_payloads:
        write_frozen_bytes(output_path, content)
    write_frozen_bytes(output_root / "manifest.json", _json_bytes(manifest))
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the frozen Task 2 PaSa data")
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        prepare(output_root=args.output_root, token=os.environ.get("HF_TOKEN"))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"data preparation failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
