"""Freeze objective proxy statistics for a blinded task-validity packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.task_validity_audit import (
    summarize_objective_task_validity_proxies,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    review = json.loads(args.review.read_text(encoding="utf-8"))
    key = json.loads(args.key.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "pasa-task-validity-objective-proxies-v1",
        "proxy_warning": "Objective proxies do not replace blinded relevance adjudication.",
        "groups": summarize_objective_task_validity_proxies(
            review_cases=review["review_cases"],
            private_key=key["cases"],
        ),
        "input_sha256": {"review": _sha256(args.review), "key": _sha256(args.key)},
        "test_partition_touched": False,
    }
    content = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    write_frozen_bytes(args.output, content)
    print(json.dumps(payload["groups"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
