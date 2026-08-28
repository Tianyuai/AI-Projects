"""Build the versioned, self-verifying VivaAI evaluator runtime package."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.evaluation.release_package import (  # noqa: E402
    RELEASE_NAME,
    build_release,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=_ROOT / "deliverables" / "submission" / RELEASE_NAME,
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=_ROOT / "deliverables" / "submission" / f"{RELEASE_NAME}.zip",
    )
    args = parser.parse_args(argv)
    summary = build_release(_ROOT, args.release_root, archive_path=args.archive)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
