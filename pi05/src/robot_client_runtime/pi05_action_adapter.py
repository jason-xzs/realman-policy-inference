"""Exact π0.5 action dimension adapter frozen by RR-DEP-001."""

from __future__ import annotations

from typing import Any

from .schemas import (
    ACTION_28_TO_16_INDICES,
    ValidationError,
    coerce_finite_row,
    is_sequence,
    require_chunk,
)


class Pi05ActionAdapter:
    def __init__(self, *, server_first_28_native_last_4_padding_confirmed: bool = False):
        if not isinstance(server_first_28_native_last_4_padding_confirmed, bool):
            raise ValidationError("invalid_confirmation", "32D confirmation must be boolean")
        self.server_first_28_native_last_4_padding_confirmed = (
            server_first_28_native_last_4_padding_confirmed
        )

    def adapt_action(self, action: Any) -> list[float]:
        if not is_sequence(action):
            raise ValidationError("not_a_sequence", "action must be a sequence")
        dimension = len(action)
        if dimension == 16:
            return coerce_finite_row(action, 16, field="action16")
        if dimension == 28:
            row = coerce_finite_row(action, 28, field="action28")
            return [row[index] for index in ACTION_28_TO_16_INDICES]
        if dimension == 32:
            if not self.server_first_28_native_last_4_padding_confirmed:
                raise ValidationError(
                    "unconfirmed_32d_layout",
                    "RR-DEP-002 has not confirmed first-28-native plus last-4-padding",
                )
            row = coerce_finite_row(action, 32, field="action32")
            return [row[index] for index in ACTION_28_TO_16_INDICES]
        raise ValidationError(
            "unknown_action_dimension",
            f"expected 16, 28, or explicitly confirmed 32 values; received {dimension}",
        )

    def adapt_chunk(self, chunk: Any) -> list[list[float]]:
        rows = require_chunk(chunk, field="action_chunk")
        adapted: list[list[float]] = []
        source_dimension: int | None = None
        for row_index, row in enumerate(rows):
            if not is_sequence(row):
                raise ValidationError("invalid_row", f"action_chunk[{row_index}] is not a row")
            if len(row) == 0:
                raise ValidationError("empty_row", f"action_chunk[{row_index}] is empty")
            if source_dimension is None:
                source_dimension = len(row)
            elif len(row) != source_dimension:
                raise ValidationError(
                    "ragged_chunk",
                    f"action_chunk[{row_index}] has {len(row)} values, expected {source_dimension}",
                )
            adapted.append(self.adapt_action(row))
        return adapted
