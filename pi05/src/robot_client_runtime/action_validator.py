"""Canonical action validation with opt-in limits only."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any

from .schemas import ACTION_DIMENSION, ACTION_SCHEMA_ID, ValidationError, coerce_finite_row


class ActionValidator:
    def __init__(self, *, limits: Mapping[int, tuple[float, float]] | None = None):
        self.limits = self._validate_limits(limits or {})
        self.rejection_records: list[dict[str, Any]] = []
        self.execution_acknowledgements: list[dict[str, Any]] = []

    @staticmethod
    def _validate_limits(limits: Mapping[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
        validated: dict[int, tuple[float, float]] = {}
        for index, bounds in limits.items():
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < ACTION_DIMENSION:
                raise ValidationError("invalid_limit_index", f"limit index {index!r} is invalid")
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValidationError("invalid_limit", f"limit {index} must be a (minimum, maximum) tuple")
            minimum, maximum = bounds
            if (
                isinstance(minimum, bool)
                or isinstance(maximum, bool)
                or not isinstance(minimum, Real)
                or not isinstance(maximum, Real)
            ):
                raise ValidationError("invalid_limit", f"limit {index} bounds must be numeric")
            minimum_float = float(minimum)
            maximum_float = float(maximum)
            if not math.isfinite(minimum_float) or not math.isfinite(maximum_float):
                raise ValidationError("invalid_limit", f"limit {index} bounds must be finite")
            if minimum_float > maximum_float:
                raise ValidationError("invalid_limit", f"limit {index} minimum exceeds maximum")
            validated[index] = (minimum_float, maximum_float)
        return validated

    def validate(self, action: Any, *, schema_id: str = ACTION_SCHEMA_ID) -> list[float]:
        try:
            if schema_id != ACTION_SCHEMA_ID:
                raise ValidationError(
                    "wrong_schema",
                    f"expected {ACTION_SCHEMA_ID}, received {schema_id}",
                )
            row = coerce_finite_row(action, ACTION_DIMENSION, field="canonical_action")
            for index, (minimum, maximum) in self.limits.items():
                if not minimum <= row[index] <= maximum:
                    raise ValidationError(
                        "explicit_limit_rejection",
                        f"canonical_action[{index}]={row[index]} outside [{minimum}, {maximum}]",
                    )
            return row
        except ValidationError as exc:
            self.rejection_records.append(
                {"schema_id": schema_id, "reason_code": exc.code, "detail": exc.detail}
            )
            raise

    def record_execution_acknowledgement(self, acknowledgement: Mapping[str, Any]) -> None:
        self.execution_acknowledgements.append(dict(acknowledgement))
