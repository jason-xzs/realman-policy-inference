import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


def _resolve_backbone_name(backbone_name: str) -> str:
    name = os.path.expanduser(os.path.expandvars(str(backbone_name)))
    candidate = Path(name)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())

    project_candidate = Path(__file__).resolve().parents[6] / candidate
    if project_candidate.exists():
        return str(project_candidate.resolve())

    return name


class DINOv3(nn.Module):
    def __init__(self, backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        backbone_name = _resolve_backbone_name(backbone_name)
        self.processor = AutoImageProcessor.from_pretrained(
            backbone_name, do_resize=False, do_center_crop=False
        )
        self.backbone = AutoModel.from_pretrained(backbone_name)
        cfg = self.backbone.config
        self.patch_size = int(cfg.patch_size)
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_attention_heads)
        self.initializer_range = float(cfg.initializer_range)
        self.num_prefix_tokens = int(cfg.num_register_tokens) + 1

    @torch.no_grad
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _, h, w = frames.shape
        assert (
            h % self.patch_size == 0 and w % self.patch_size == 0
        ), f"Frame size ({h}, {w}) must be a multiple of patch size ({self.patch_size})"
        x = self.processor(
            frames.reshape(batch_size * num_frames, *frames.shape[2:]),
            return_tensors="pt",
        )["pixel_values"].to(frames.device)
        y = self.backbone(x).last_hidden_state[:, self.num_prefix_tokens :]
        return y.reshape(batch_size, num_frames, -1, self.hidden_size)
