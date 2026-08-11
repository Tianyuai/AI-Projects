"""An explicit per-composition action registry with no import-time mutation."""

from __future__ import annotations

from collections.abc import Iterator

from paper_search.recall_experiments.contracts import ActionType, RetrievalActionHandler


class RetrievalActionRegistry:
    """Map action types to handlers in registration order.

    Each composition root creates and populates its own instance.  A module
    global would leak handlers across formal runs and make replay composition
    order-dependent, so this module intentionally declares none.
    """

    def __init__(self) -> None:
        self._handlers: dict[ActionType, RetrievalActionHandler] = {}

    def register(self, action_type: ActionType, handler: RetrievalActionHandler) -> None:
        if action_type in self._handlers:
            raise ValueError(f"retrieval action already registered: {action_type}")
        self._handlers[action_type] = handler

    def unregister(self, action_type: ActionType) -> RetrievalActionHandler:
        try:
            return self._handlers.pop(action_type)
        except KeyError as error:
            raise KeyError(f"unknown retrieval action: {action_type}") from error

    def resolve(self, action_type: ActionType) -> RetrievalActionHandler:
        try:
            return self._handlers[action_type]
        except KeyError as error:
            raise KeyError(f"unknown retrieval action: {action_type}") from error

    def __iter__(self) -> Iterator[ActionType]:
        return iter(self._handlers)

    def __len__(self) -> int:
        return len(self._handlers)


__all__ = ["RetrievalActionRegistry"]
