"""Local environment health checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any


RETRIEVAL_DEPENDENCIES = {
    "faiss": "faiss-cpu",
    "transformers": "transformers",
    "sentence_transformers": "sentence-transformers",
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


def _cpu_smoke(torch_module: Any, matrix_size: int) -> dict[str, object]:
    with torch_module.inference_mode():
        left = torch_module.ones((matrix_size, matrix_size), device="cpu")
        right = torch_module.ones((matrix_size, matrix_size), device="cpu")
        product = left @ right
        finite = bool(torch_module.isfinite(product).all().item())
        checksum = float(product.sum().item())
    return {
        "shape": [matrix_size, matrix_size],
        "finite": finite,
        "checksum": checksum,
    }


def _cuda_smoke(torch_module: Any, matrix_size: int) -> dict[str, object]:
    device_index = torch_module.cuda.current_device()
    with torch_module.inference_mode():
        left = torch_module.ones((matrix_size, matrix_size), device="cuda")
        right = torch_module.ones((matrix_size, matrix_size), device="cuda")
        product = left @ right
        finite = bool(torch_module.isfinite(product).all().item())
        checksum = float(product.sum().item())
    torch_module.cuda.synchronize(device_index)
    smoke = {
        "shape": [matrix_size, matrix_size],
        "finite": finite,
        "checksum": checksum,
    }
    del left, right, product
    torch_module.cuda.empty_cache()
    return {
        "device": str(torch_module.cuda.get_device_name(device_index)),
        "matrix_smoke": smoke,
    }


def _smoke_result_is_valid(smoke: object, matrix_size: int) -> bool:
    if not isinstance(smoke, Mapping):
        return False
    return (
        smoke.get("shape") == [matrix_size, matrix_size]
        and smoke.get("finite") is True
        and smoke.get("checksum") == float(matrix_size**3)
    )


def collect_local_health(
    matrix_size: int = 64,
    require_accelerator: str | None = None,
) -> dict[str, Any]:
    """Collect secret-free core and optional accelerator health evidence."""
    if matrix_size < 1:
        raise ValueError("matrix_size must be positive")
    if require_accelerator not in (None, "cuda"):
        raise ValueError("unsupported accelerator requirement")

    import torch

    dependencies = _dependency_report()
    errors = [
        f"dependency_missing:{name}"
        for name, item in dependencies.items()
        if not item["available"]
    ]

    cpu_smoke: dict[str, object] | None = None
    try:
        cpu_smoke = _cpu_smoke(torch, matrix_size)
        if not _smoke_result_is_valid(cpu_smoke, matrix_size):
            errors.append("cpu_smoke:invalid_result")
    except RuntimeError as exc:
        errors.append(f"cpu_smoke:{type(exc).__name__}")

    accelerator: dict[str, object] = {
        "backend": "cuda",
        "status": "unavailable",
        "build": torch.version.cuda,
        "device": None,
        "matrix_smoke": None,
    }
    if torch.cuda.is_available():
        try:
            cuda_result = _cuda_smoke(torch, matrix_size)
            accelerator.update(cuda_result)
            accelerator["status"] = (
                "available"
                if _smoke_result_is_valid(cuda_result.get("matrix_smoke"), matrix_size)
                else "error"
            )
        except RuntimeError:
            accelerator["status"] = "error"

    if require_accelerator == "cuda" and accelerator["status"] != "available":
        errors.append("accelerator_required:cuda")

    return {
        "status": "ready" if not errors else "degraded",
        "python": {
            "version": ".".join(map(str, sys.version_info[:3])),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "executable": sys.executable,
        },
        "core": {
            "torch_version": str(torch.__version__),
            "matrix_smoke": cpu_smoke,
            "dependencies": dependencies,
        },
        "accelerator": accelerator,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the local health report as JSON without reading application secrets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-accelerator", choices=("cuda",))
    args = parser.parse_args(argv)
    report = collect_local_health(require_accelerator=args.require_accelerator)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())