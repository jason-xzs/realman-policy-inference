"""Pure observation-key adapter with no camera or image operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schemas import (
    IMAGE_ROLE_TO_OPENPI_KEY,
    OBSERVATION_STATE_DIMENSION,
    ValidationError,
    coerce_finite_row,
)


class ObservationAdapter:
    """Maps existing Client roles to OpenPI observation keys without resizing."""

    def adapt(self, *, state: Any, images: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        state_row = coerce_finite_row(
            state,
            OBSERVATION_STATE_DIMENSION,
            field="observation.state",
        )
        if not isinstance(images, Mapping):
            raise ValidationError("invalid_images", "images must be a role-to-object mapping")
        if not isinstance(prompt, str):
            raise ValidationError("invalid_prompt", "prompt must be a string")

        result: dict[str, Any] = {"state": state_row, "prompt": prompt}
        for role, openpi_key in IMAGE_ROLE_TO_OPENPI_KEY.items():
            if role not in images:
                raise ValidationError("missing_image_role", f"missing required image role {role}")
            result[openpi_key] = images[role]
        return result
