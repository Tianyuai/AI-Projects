"""Shared protected failure boundaries that composition must never downgrade."""


class ProtectedExecutionError(RuntimeError):
    """A typed execution failure that must propagate without reinterpretation."""

    search_error_code = "integrity_failure"


__all__ = ["ProtectedExecutionError"]
