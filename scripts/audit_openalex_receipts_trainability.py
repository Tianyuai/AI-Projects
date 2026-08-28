"""Run the local incremental OpenAlex receipt trainability audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_search.learning.openalex_receipt_audit import audit_openalex_receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-root", action="append", required=True, type=Path)
    parser.add_argument("--partition", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--query-output", type=Path)
    args = parser.parse_args()
    summary = audit_openalex_receipts(
        receipt_roots=args.receipt_root,
        partition_path=args.partition,
        state_path=args.state,
        include_query_rows=args.query_output is not None,
    )
    query_rows = summary.pop("query_rows", None)
    if args.query_output is not None:
        if not isinstance(query_rows, list):
            raise ValueError("query-level audit rows were not produced")
        args.query_output.parent.mkdir(parents=True, exist_ok=True)
        args.query_output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in query_rows
            ),
            encoding="utf-8",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
