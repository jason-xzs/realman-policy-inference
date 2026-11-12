"""Machine-readable schema constants and strict standard-library validation."""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Real
from typing import Any


ACTION_SCHEMA_ID = "realman.dual_arm.absolute.v1"
ACTION_DIMENSION = 16
OBSERVATION_STATE_DIMENSION = 28
ACTION_28_TO_16_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21)
DISCARDED_ACTION_28_INDICES = (8, 9, 10, 11, 12, 13, 22, 23, 24, 25, 26, 27)

ACTION_ORDER = (
    "left_q1_rad",
    "left_q2_rad",
    "left_q3_rad",
    "left_q4_rad",
    "left_q5_rad",
    "left_q6_rad",
    "left_q7_rad",
    "left_gripper_open",
    "right_q1_rad",
    "right_q2_rad",
    "right_q3_rad",
    "right_q4_rad",
    "right_q5_rad",
    "right_q6_rad",
    "right_q7_rad",
    "right_gripper_open",
)

OBSERVATION_STATE_ORDER = (
    "left_q1_rad",
    "left_q2_rad",
    "left_q3_rad",
    "left_q4_rad",
    "left_q5_rad",
    "left_q6_rad",
    "left_q7_rad",
    "left_gripper_open",
    "left_eef_x_m",
    "left_eef_y_m",
    "left_eef_z_m",
    "left_eef_rx_rad",
    "left_eef_ry_rad",
    "left_eef_rz_rad",
    "right_q1_rad",
    "right_q2_rad",
    "right_q3_rad",
    "right_q4_rad",
    "right_q5_rad",
    "right_q6_rad",
    "right_q7_rad",
    "right_gripper_open",
    "right_eef_x_m",
    "right_eef_y_m",
    "right_eef_z_m",
    "right_eef_rx_rad",
    "right_eef_ry_rad",
    "right_eef_rz_rad",
)

IMAGE_ROLE_TO_OPENPI_KEY = {
    "high_rgb": "base_0_rgb",
    "left_wrist_rgb": "left_wrist_0_rgb",
    "right_wrist_rgb": "right_wrist_0_rgb",
}


class ValidationError(ValueError):
    """A stable fail-closed schema validation error."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class IntegrationNotEnabledError(RuntimeError):
    """Raised before any live integration side effect is attempted."""

    def __init__(self, detail: str):
        self.code = "integration_not_enabled"
        super().__init__(f"{self.code}: {detail}")


def is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def coerce_finite_row(values: Any, expected_dimension: int, *, field: str) -> list[float]:
    if not is_sequence(values):
        raise ValidationError("not_a_sequence", f"{field} must be a sequence")
    if len(values) != expected_dimension:
        raise ValidationError(
            "wrong_dimension",
            f"{field} expected {expected_dimension} values, received {len(values)}",
        )

    row: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValidationError(
                "non_numeric_value",
                f"{field}[{index}] must be a real number and booleans are forbidden",
            )
        converted = float(value)
        if not math.isfinite(converted):
            raise ValidationError("nonfinite_value", f"{field}[{index}] must be finite")
        row.append(converted)
    return row


def require_chunk(values: Any, *, field: str) -> Sequence[Any]:
    if not is_sequence(values):
        raise ValidationError("not_a_sequence", f"{field} must be a sequence of rows")
    if len(values) == 0:
        raise ValidationError("empty_chunk", f"{field} must contain at least one row")
    return values
