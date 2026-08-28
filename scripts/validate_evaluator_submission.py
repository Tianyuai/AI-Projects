"""Validate the minimal evaluator query/prediction JSONL exchange."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = str(_REPOSITORY_ROOT / "src")
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

from paper_search.evaluation.dataset import read_jsonl  # noqa: E402
from paper_search.evaluation.official_adapter import (  # noqa: E402
    AstaPaperFindingQuery,
    InternalPredictionRecord,
)
from paper_search.evaluation.submission_contract import (  # noqa: E402
    validate_submission_records,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = validate_submission_records(
        read_jsonl(args.queries, AstaPaperFindingQuery),
        read_jsonl(args.predictions, InternalPredictionRecord),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
