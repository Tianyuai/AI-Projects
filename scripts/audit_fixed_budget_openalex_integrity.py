from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_search.learning.fixed_budget_integrity import (
    audit_fixed_budget_receipts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_fixed_budget_receipts(
        structured_root=args.structured_root,
        semantic_root=args.semantic_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
