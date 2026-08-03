"""Shared protected failure boundaries that composition must never downgrade."""


class ProtectedExecutionError(RuntimeError):
    """A typed execution failure that must propagate without reinterpretation."""


__all__ = ["ProtectedExecutionError"]
