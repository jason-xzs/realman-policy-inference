"""Canonical action splitter with injection-only backend dispatch."""

from __future__ import annotations

from typing import Any

from .schemas import ACTION_SCHEMA_ID, IntegrationNotEnabledError
from .action_validator import ActionValidator


class RealManDualArmAdapter:
    def __init__(self, backend: Any | None = None, *, backend_order: str = "left_first"):
        if backend_order not in {"left_first", "right_first"}:
            raise ValueError("backend_order must be left_first or right_first")
        self.backend = backend
        self.backend_order = backend_order
        self._validator = ActionValidator()

    def split(self, action: Any) -> tuple[list[float], list[float]]:
        canonical = self._validator.validate(action, schema_id=ACTION_SCHEMA_ID)
        return canonical[0:8], canonical[8:16]

    def dispatch(self, action: Any) -> Any:
        if self.backend is None:
            raise IntegrationNotEnabledError("no injected RealMan backend is configured")
        left, right = self.split(action)
        if self.backend_order == "left_first":
            return self.backend.execute_targets(left, right, order="left_first")
        return self.backend.execute_targets(right, left, order="right_first")
