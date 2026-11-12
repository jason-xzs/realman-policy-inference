"""Deterministic local FIFO only; no timing, retry, or watchdog policy."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any


class ChunkScheduler:
    def __init__(self) -> None:
        self._queue: deque[list[float]] = deque()

    @property
    def pending(self) -> int:
        return len(self._queue)

    def enqueue(self, actions: Sequence[Sequence[float]]) -> int:
        added = 0
        for action in actions:
            self._queue.append(list(action))
            added += 1
        return added

    def pop_next(self) -> list[float] | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def clear(self) -> None:
        self._queue.clear()
