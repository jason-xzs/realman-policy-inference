"""Transforms for RealMan LeRobot demonstrations."""

from __future__ import annotations

import torch


class RealManDualArmToPretrain:
    """Match 28-D LeRobot controls to the 16-D RealMan pretraining layout.

    The LeRobot recordings store left arm, left gripper, left EEF pose,
    right arm, right gripper, right EEF pose.  The RealMan pretraining data
    used right arm + gripper followed by left arm + gripper.  LeRobot stores
    gripper openness in [0, 1], while pretraining used approximately [0, 1000].
    """

    SOURCE_DIM = 28
    OUTPUT_INDICES = (14, 15, 16, 17, 18, 19, 20, 21, 0, 1, 2, 3, 4, 5, 6, 7)
    GRIPPER_OUTPUT_INDICES = (7, 15)

    def __init__(self, gripper_scale: float = 1000.0):
        self.gripper_scale = float(gripper_scale)

    def _convert(self, value: torch.Tensor, *, kind: str, key: str) -> torch.Tensor:
        if value.shape[-1] != self.SOURCE_DIM:
            raise ValueError(
                f"RealMan {kind} '{key}' must have {self.SOURCE_DIM} input dimensions, "
                f"got {value.shape[-1]}."
            )
        indices = torch.as_tensor(self.OUTPUT_INDICES, device=value.device, dtype=torch.long)
        converted = value.index_select(-1, indices).clone()
        converted[..., list(self.GRIPPER_OUTPUT_INDICES)] *= self.gripper_scale
        return converted

    def forward(self, batch):
        if "action" in batch:
            batch["action"] = {
                key: self._convert(value, kind="action", key=key)
                for key, value in batch["action"].items()
            }
        batch["state"] = {
            key: self._convert(value, kind="state", key=key)
            for key, value in batch["state"].items()
        }
        return batch

