"""Local environment health checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any


RETRIEVAL_DEPENDENCIES = {
    "rank_bm25": "rank-bm25",
}


def _dependency_report() -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for module_name, distribution_name in RETRIEVAL_DEPENDENCIES.items():
        try:
            import_module(module_name)
            package_version = version(distribution_name)
        except (ImportError, PackageNotFoundError) as exc:
            report[module_name] = {
                "available": False,
                "version": None,
                "error": type(exc).__name__,
            }
        else:
            report[module_name] = {
                "available": True,
                "version": package_version,
                "error": None,
            }
    return report


def collect_local_health() -> dict[str, Any]:
    """Collect secret-free health evidence for the supported runtime."""
    dependencies = _dependency_report()
    errors = [
        f"dependency_missing:{name}"
        for name, item in dependencies.items()
        if not item["available"]
    ]

    return {
        "status": "ready" if not errors else "degraded",
        "python": {
            "version": ".".join(map(str, sys.version_info[:3])),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "executable": sys.executable,
        },
        "core": {
            "dependencies": dependencies,
        },
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the local health report as JSON without reading application secrets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report = collect_local_health()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
