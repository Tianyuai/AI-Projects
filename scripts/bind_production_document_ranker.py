"""Bind the frozen F5/F4/B0 chain into an evaluator input lock."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.application.production_ranker_binding import (  # noqa: E402
    bind_production_ranker_selection,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("artifacts/models/production-document-ranker-selection.json"),
    )
    parser.add_argument("--selection-root", default="artifacts/models")
    args = parser.parse_args(argv)
    raw_lock = yaml.safe_load(args.input_lock.read_bytes())
    selection = json.loads(args.selection.read_bytes())
    bound = bind_production_ranker_selection(
        raw_lock, selection, selection_root=args.selection_root
    )
    args.output_lock.parent.mkdir(parents=True, exist_ok=True)
    args.output_lock.write_text(
        yaml.safe_dump(bound, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
