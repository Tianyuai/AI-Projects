"""Local environment health checks."""

from __future__ import annotations

import json
import sys
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


def collect_local_health(matrix_size: int = 64) -> dict[str, Any]:
    """Collect a secret-free local Python, CUDA, and retrieval-stack report."""
    if matrix_size < 1:
        raise ValueError("matrix_size must be positive")

    import torch

    dependencies = _dependency_report()
    cuda_available = torch.cuda.is_available()
    torch_report: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": None,
        "compute_capability": None,
        "total_memory_mb": None,
        "matrix_smoke": None,
    }

    errors: list[str] = []
    if cuda_available:
        try:
            device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            torch.cuda.reset_peak_memory_stats(device_index)
            with torch.inference_mode():
                left = torch.ones((matrix_size, matrix_size), device="cuda")
                right = torch.ones((matrix_size, matrix_size), device="cuda")
                product = left @ right
                finite = bool(torch.isfinite(product).all().item())
                checksum = float(product.sum().item())
            torch.cuda.synchronize(device_index)
            peak_memory_mb = torch.cuda.max_memory_allocated(device_index) / (1024**2)
            torch_report.update(
                {
                    "device": properties.name,
                    "compute_capability": list(torch.cuda.get_device_capability(device_index)),
                    "total_memory_mb": round(properties.total_memory / (1024**2), 1),
                    "matrix_smoke": {
                        "shape": [matrix_size, matrix_size],
                        "finite": finite,
                        "checksum": checksum,
                        "peak_memory_mb": round(peak_memory_mb, 2),
                    },
                }
            )
            del left, right, product
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            errors.append(f"gpu_smoke:{type(exc).__name__}")
    else:
        errors.append("cuda_unavailable")

    missing_dependencies = [
        name for name, item in dependencies.items() if not item["available"]
    ]
    errors.extend(f"dependency_missing:{name}" for name in missing_dependencies)

    return {
        "status": "ready" if not errors else "degraded",
        "python": {
            "version": ".".join(map(str, sys.version_info[:3])),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "executable": sys.executable,
        },
        "torch": torch_report,
        "dependencies": dependencies,
        "errors": errors,
    }


def main() -> int:
    """Print the local health report as JSON without reading application secrets."""
    report = collect_local_health()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
