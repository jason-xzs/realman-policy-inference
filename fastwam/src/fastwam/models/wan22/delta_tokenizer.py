from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger
from fastwam.models.wan22.lam.modules import LatentActionModel
from fastwam.models.wan22.lam.modules.blocks import patchify, unpatchify
from fastwam.models.wan22.deltatok.models.deltatok import DeltaTok as DeltaTokImpl
from fastwam.models.wan22.deltatok.models.dinov3 import DINOv3 as DINOv3Impl
from fastwam.models.wan22.deltatok.models.task_heads import RGBHead

logger = get_logger(__name__)
DeltaTok = DeltaTokImpl


def _load_state_dict_maybe_wrapped(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint format: {type(payload)}")


def _strip_prefix_if_present(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _resolve_project_relative_path(path_str: str, *, anchor_file: str) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = Path(anchor_file).resolve().parents[4] / path
    return path.resolve()


class DINOv3BackboneWithRGBHead(nn.Module):
    """
    DeltaTok-compatible backbone:
    - `forward`: RGB frames -> DINOv3 patch features [B,T,N,D]
    - `decode_features`: patch features [B,N,D] -> RGB frame [B,3,H,W]
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        rgb_head_path: str | None = None,
        device: str = "cuda",
        strict_rgb_head: bool = False,
        rgb_head_hidden_size: int = 1152,
        rgb_head_depth: int = 28,
        rgb_head_num_heads: int = 16,
        rgb_head_mlp_size: int = 4096,
        rgb_head_img_size: int = 512,
    ):
        super().__init__()
        self.dinov3 = DINOv3Impl(backbone_name=backbone_name)
        self.processor = self.dinov3.processor
        self.backbone = self.dinov3.backbone

        self.patch_size = int(self.dinov3.patch_size)
        self.hidden_size = int(self.dinov3.hidden_size)
        self.num_heads = int(self.dinov3.num_heads)
        self.initializer_range = float(self.dinov3.initializer_range)
        self.num_prefix_tokens = int(self.dinov3.num_prefix_tokens)

        norm_layer = getattr(self.backbone, "norm", None) or getattr(self.backbone, "layernorm", None)
        if norm_layer is None:
            raise ValueError("Cannot find DINO backbone norm layer (`norm` or `layernorm`).")

        self.rgb_head = RGBHead(
            in_size=self.hidden_size,
            norm_weight=norm_layer.weight.detach().clone(),
            norm_bias=norm_layer.bias.detach().clone(),
            img_mean=list(self.processor.image_mean),
            img_std=list(self.processor.image_std),
            hidden_size=int(rgb_head_hidden_size),
            depth=int(rgb_head_depth),
            num_heads=int(rgb_head_num_heads),
            mlp_size=int(rgb_head_mlp_size),
            patch=int(self.patch_size),
            img_size=int(rgb_head_img_size),
        )

        if rgb_head_path is not None and str(rgb_head_path).strip() != "":
            p = _resolve_project_relative_path(str(rgb_head_path), anchor_file=__file__)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"RGB head checkpoint not found: {p}")
            payload = torch.load(str(p), map_location=device)
            state_dict = _load_state_dict_maybe_wrapped(payload)
            # state_dict = _strip_prefix_if_present(state_dict, "rgb_head.")
            # state_dict = _strip_prefix_if_present(state_dict, "head.")
            incompatible = self.rgb_head.load_state_dict(state_dict, strict=bool(strict_rgb_head))
            if  not strict_rgb_head and (len(incompatible.missing_keys) > 0 or len(incompatible.unexpected_keys) > 0):
                logger.warning("RGB head state_dict mismatch: ")
                logger.warning("  missing keys: %s", incompatible.missing_keys)
                logger.warning("  unexpected keys: %s", incompatible.unexpected_keys)
            logger.info("Loaded RGB head checkpoint from %s", p)

    @staticmethod
    def _to_unit_range(frames: torch.Tensor) -> torch.Tensor:
        x = frames
        x_stats = x.float()
        if x_stats.min().item() < -0.05 or x_stats.max().item() > 1.05:
            x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        else:
            x = x.clamp(0.0, 1.0)
        return x

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"`frames` must be [B,T,C,H,W], got {tuple(frames.shape)}")
        frames = (self._to_unit_range(frames) * 255.0).round().clamp(0, 255).to(torch.uint8)
        return self.dinov3(frames)

    def _infer_grid(self, num_tokens: int, ref_frame: torch.Tensor | None = None) -> tuple[int, int]:
        if ref_frame is not None:
            h, w = int(ref_frame.shape[-2]), int(ref_frame.shape[-1])
            gh = max(h // self.patch_size, 1)
            gw = max(w // self.patch_size, 1)
            if gh * gw == num_tokens:
                return gh, gw
        side = int(round(num_tokens ** 0.5))
        if side * side != num_tokens:
            raise ValueError(f"Cannot infer token grid from num_tokens={num_tokens}.")
        return side, side

    @torch.no_grad()
    def decode_features(self, features: torch.Tensor, ref_frame: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim == 4 and features.shape[1] == 1:
            features = features[:, 0]
        if features.ndim != 3:
            raise ValueError(f"`features` must be [B,N,D] or [B,1,N,D], got {tuple(features.shape)}")
        if features.shape[-1] != self.hidden_size:
            raise ValueError(f"`features` hidden dim must be {self.hidden_size}, got {features.shape[-1]}")

        gh, gw = self._infer_grid(num_tokens=int(features.shape[1]), ref_frame=ref_frame)
        fmap = features.reshape(features.shape[0], gh, gw, self.hidden_size).permute(0, 3, 1, 2).contiguous()
        rgb = self.rgb_head(fmap).float()
        if ref_frame is not None and rgb.shape[-2:] != ref_frame.shape[-2:]:
            rgb = F.interpolate(rgb, size=ref_frame.shape[-2:], mode="bilinear", align_corners=False)
        rgb = rgb.clamp(0.0, 1.0)
        return rgb * 2.0 - 1.0


class BaseDeltaTokenizer(nn.Module, ABC):
    """Interface for tokenizing adjacent-frame pairs into delta tokens."""

    def __init__(self, token_dim: int):
        super().__init__()
        if int(token_dim) <= 0:
            raise ValueError(f"`token_dim` must be positive, got {token_dim}")
        self.token_dim = int(token_dim)

    @abstractmethod
    def encode_pair(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        """Encode adjacent frames to one token per pair. Return shape: [B, 1, token_dim]."""

    def decode_next(self, frame_t: torch.Tensor, delta_token: torch.Tensor) -> torch.Tensor:
        """
        Decode next frame from previous frame and delta token.
        Expected input shapes:
        - frame_t: [B, C, H, W]
        - delta_token: [B, D] or [B, 1, D]
        Return:
        - frame_t1: [B, C, H, W]
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `decode_next`. "
            "Use tokenizer impl `deltatok` for pixel reconstruction."
        )

    def forward(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        return self.encode_pair(frame_t=frame_t, frame_t1=frame_t1)

    def freeze_encoder(self) -> None:
        """Freeze parameters used by encoder path."""

    def save_checkpoint(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "token_dim": self.token_dim}, path)

    def load_checkpoint(self, path: str, device: str | torch.device) -> dict[str, Any]:
        payload = torch.load(path, map_location=device)
        state_dict = _load_state_dict_maybe_wrapped(payload)
        self.load_state_dict(state_dict, strict=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            return payload
        return {"state_dict": state_dict}


class DeltaTokEncoderTokenizer(BaseDeltaTokenizer):
    """Adapter that uses DeltaTok's encoder path only."""

    def __init__(self, deltatok: nn.Module, token_dim: int):
        super().__init__(token_dim=token_dim)
        self.deltatok = deltatok
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        if hasattr(self.deltatok, "backbone"):
            self.deltatok.backbone.requires_grad_(False).eval()
        for name in ["z_embed", "xy_embed", "encoder_blocks", "norm", "rope_embeddings"]:
            module = getattr(self.deltatok, name, None)
            if module is not None:
                module.requires_grad_(False)
                if isinstance(module, nn.Module):
                    module.eval()

    def _decode_backbone_features(self, y_hat: torch.Tensor, ref_frame: torch.Tensor) -> torch.Tensor:
        backbone = self.deltatok.backbone
        decode_features = getattr(backbone, "decode_features", None)
        if decode_features is None:
            raise RuntimeError(
                "DeltaTok backbone must implement `decode_features(features, ref_frame=None)` "
                "for pixel reconstruction."
            )
        out = decode_features(y_hat, ref_frame)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise RuntimeError(
                "DeltaTok backbone `decode_features` must return a torch.Tensor, "
                f"got {type(out)}"
            )
        if out.ndim == 5 and out.shape[1] == 1:
            out = out[:, 0]
        if out.ndim != 4:
            raise RuntimeError(
                "DeltaTok backbone `decode_features` must return [B,C,H,W], "
                f"got {tuple(out.shape)}"
            )
        return out

    @torch.no_grad()
    def encode_pair(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        if frame_t.ndim != 4 or frame_t1.ndim != 4:
            raise ValueError(
                "`frame_t` and `frame_t1` must be 4D [B,C,H,W], "
                f"got {tuple(frame_t.shape)} and {tuple(frame_t1.shape)}"
            )
        if frame_t.shape != frame_t1.shape:
            raise ValueError(f"Frame shape mismatch: {tuple(frame_t.shape)} vs {tuple(frame_t1.shape)}")

        frames = torch.stack([frame_t, frame_t1], dim=1)
        rope = self.deltatok._rope(frames)
        y = self.deltatok.backbone(frames)
        z = self.deltatok.encode(y[:, 0], y[:, 1], rope)

        if z.ndim != 3 or z.shape[1] != 1:
            raise ValueError(f"DeltaTok encoder output must be [B,1,D], got {tuple(z.shape)}")
        if z.shape[2] != self.token_dim:
            raise ValueError(f"DeltaTok token dim mismatch: expected {self.token_dim}, got {z.shape[2]}")
        return z

    @torch.no_grad()
    def decode_next(self, frame_t: torch.Tensor, delta_token: torch.Tensor) -> torch.Tensor:
        if frame_t.ndim != 4:
            raise ValueError(f"`frame_t` must be 4D [B,C,H,W], got {tuple(frame_t.shape)}")
        if delta_token.ndim == 2:
            delta_token = delta_token.unsqueeze(1)
        if delta_token.ndim != 3 or delta_token.shape[1] != 1:
            raise ValueError(f"`delta_token` must be [B,D] or [B,1,D], got {tuple(delta_token.shape)}")
        if delta_token.shape[0] != frame_t.shape[0]:
            raise ValueError(
                f"Batch mismatch: frame_t batch={frame_t.shape[0]} vs delta_token batch={delta_token.shape[0]}"
            )
        if delta_token.shape[2] != self.token_dim:
            raise ValueError(f"Delta token dim mismatch: expected {self.token_dim}, got {delta_token.shape[2]}")

        frames = frame_t.unsqueeze(1)
        x = self.deltatok.backbone(frames)[:, 0]
        rope = self.deltatok._rope(frames)
        y_hat = self.deltatok.decode(delta_token, x, rope)
        return self._decode_backbone_features(y_hat=y_hat, ref_frame=frame_t)

    @classmethod
    def from_pretrained(
        cls,
        token_dim: int,
        backbone: nn.Module,
        num_hidden_layers: int = 12,
        use_delta: bool = True,
        use_decoder_head: bool | None = None,
        use_decoder_xy_embed: bool | None = None,
        layer_scale_init: float = 1e-5,
        use_qk_norm: bool = True,
        use_gated_attn: bool = True,
        use_swiglu: bool = True,
        use_rope_aug: bool = True,
        pretrained_path: str | None = None,
        skip_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "DeltaTokEncoderTokenizer":
        p = None
        if pretrained_path is not None and str(pretrained_path).strip() != "":
            p = os.path.expanduser(os.path.expandvars(str(pretrained_path)))
            if not os.path.isabs(p):
                p = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", p)
                p = os.path.abspath(p)
            if not os.path.isfile(p) and not skip_load_from_pretrain:
                raise FileNotFoundError(f"DeltaTok checkpoint not found: {p}")

        deltatok = DeltaTokImpl(
            backbone=backbone,
            num_hidden_layers=int(num_hidden_layers),
            use_delta=bool(use_delta),
            use_decoder_head=use_decoder_head,
            use_decoder_xy_embed=use_decoder_xy_embed,
            layer_scale_init=float(layer_scale_init),
            use_qk_norm=bool(use_qk_norm),
            use_gated_attn=bool(use_gated_attn),
            use_swiglu=bool(use_swiglu),
            use_rope_aug=bool(use_rope_aug),
            ckpt_path=(None if skip_load_from_pretrain else p),
        ).to(device=device, dtype=torch_dtype)

        tokenizer = cls(deltatok=deltatok, token_dim=int(token_dim)).to(device=device, dtype=torch_dtype)
        if skip_load_from_pretrain:
            logger.info("Skipping DeltaTok ckpt load for tokenizer; using random DeltaTok init.")
        elif p is None:
            logger.info("No DeltaTok ckpt path provided; using random DeltaTok init.")
        else:
            logger.info("Loaded DeltaTok ckpt from %s", p)
        return tokenizer


class LAMDeltaTokenizer(BaseDeltaTokenizer):
    """Adapter that uses DreamDojo LAM encoder path only."""

    def __init__(self, lam: nn.Module, token_dim: int):
        super().__init__(token_dim=token_dim)
        self.lam = lam
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        self.lam.requires_grad_(False)
        self.lam.eval()

    @staticmethod
    def _to_unit_range(frames: torch.Tensor) -> torch.Tensor:
        x = frames
        x_stats = x.float()
        if x_stats.min().item() < -0.05 or x_stats.max().item() > 1.05:
            x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        else:
            x = x.clamp(0.0, 1.0)
        return x

    @torch.no_grad()
    def encode_pair(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        if frame_t.ndim != 4 or frame_t1.ndim != 4:
            raise ValueError(
                "`frame_t` and `frame_t1` must be 4D [B,C,H,W], "
                f"got {tuple(frame_t.shape)} and {tuple(frame_t1.shape)}"
            )
        if frame_t.shape != frame_t1.shape:
            raise ValueError(f"Frame shape mismatch: {tuple(frame_t.shape)} vs {tuple(frame_t1.shape)}")

        frames = torch.stack([frame_t, frame_t1], dim=1)  # [B,2,C,H,W]
        frames = self._to_unit_range(frames).permute(0, 1, 3, 4, 2).contiguous()  # [B,2,H,W,C]
        outputs = self.lam.encode(frames)
        z = outputs.get("z_rep", None)
        if z is None:
            raise RuntimeError("LAM encoder output missing `z_rep`.")
        if z.ndim != 4 or z.shape[1] != 1 or z.shape[2] != 1:
            raise ValueError(f"LAM encoder output must be [B,1,1,D], got {tuple(z.shape)}")
        z = z.squeeze(2)
        if z.shape[2] != self.token_dim:
            raise ValueError(f"LAM token dim mismatch: expected {self.token_dim}, got {z.shape[2]}")
        return z

    @torch.no_grad()
    def decode_next(self, frame_t: torch.Tensor, delta_token: torch.Tensor) -> torch.Tensor:
        if frame_t.ndim != 4:
            raise ValueError(f"`frame_t` must be 4D [B,C,H,W], got {tuple(frame_t.shape)}")
        if delta_token.ndim == 2:
            delta_token = delta_token.unsqueeze(1)
        if delta_token.ndim != 3 or delta_token.shape[1] != 1:
            raise ValueError(f"`delta_token` must be [B,D] or [B,1,D], got {tuple(delta_token.shape)}")
        if delta_token.shape[0] != frame_t.shape[0]:
            raise ValueError(
                f"Batch mismatch: frame_t batch={frame_t.shape[0]} vs delta_token batch={delta_token.shape[0]}"
            )
        if delta_token.shape[2] != self.token_dim:
            raise ValueError(f"LAM token dim mismatch: expected {self.token_dim}, got {delta_token.shape[2]}")

        frame_t_unit = self._to_unit_range(frame_t)
        videos = frame_t_unit.permute(0, 2, 3, 1).unsqueeze(1).contiguous()  # [B,1,H,W,C]
        patches = patchify(videos, self.lam.patch_size)  # [B,1,N,patch_dim]
        video_patches = self.lam.patch_up(patches)  # [B,1,N,E]
        action_patches = self.lam.action_up(delta_token.unsqueeze(2))  # [B,1,1,E]
        video_action_patches = video_patches + action_patches

        video_recon = self.lam.decoder(video_action_patches)
        video_recon = torch.sigmoid(video_recon)
        recon = unpatchify(video_recon, self.lam.patch_size, frame_t.shape[-2], frame_t.shape[-1])
        recon = recon[:, 0].permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
        recon = recon.clamp(0.0, 1.0)
        return recon * 2.0 - 1.0

    @classmethod
    def from_pretrained(
        cls,
        token_dim: int,
        image_channels: int = 3,
        lam_model_dim: int = 1024,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 24,
        lam_dec_blocks: int = 24,
        lam_num_heads: int = 16,
        lam_dropout: float = 0.0,
        pretrained_path: str | None = None,
        skip_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "LAMDeltaTokenizer":
        lam = LatentActionModel(
            in_dim=int(image_channels),
            model_dim=int(lam_model_dim),
            latent_dim=int(lam_latent_dim),
            patch_size=int(lam_patch_size),
            enc_blocks=int(lam_enc_blocks),
            dec_blocks=int(lam_dec_blocks),
            num_heads=int(lam_num_heads),
            dropout=float(lam_dropout),
        ).to(device=device, dtype=torch_dtype)

        tokenizer = cls(lam=lam, token_dim=int(token_dim)).to(device=device, dtype=torch_dtype)

        if int(lam_latent_dim) != int(token_dim):
            raise ValueError(
                "LAM token dim mismatch between config and model: "
                f"token_dim={token_dim}, lam_latent_dim={lam_latent_dim}."
            )

        if skip_load_from_pretrain:
            logger.info("Skipping LAM ckpt load for tokenizer; using random LAM init.")
            return tokenizer
        if pretrained_path is None or str(pretrained_path).strip() == "":
            logger.info("No LAM ckpt path provided; using random LAM init.")
            return tokenizer

        p = os.path.expanduser(os.path.expandvars(str(pretrained_path)))
        if not os.path.isabs(p):
            p = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", p)
            p = os.path.abspath(p)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"LAM checkpoint not found: {p}")

        payload = torch.load(p, map_location="cpu")
        state_dict = _load_state_dict_maybe_wrapped(payload)
        state_dict = _strip_prefix_if_present(state_dict, "lam.")
        incompatible = tokenizer.lam.load_state_dict(state_dict, strict=True)
        if len(incompatible.missing_keys) > 0 or len(incompatible.unexpected_keys) > 0:
            raise ValueError(
                f"LAM checkpoint mismatch for {p}: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        logger.info("Loaded LAM ckpt from %s", p)
        tokenizer.freeze_encoder()
        return tokenizer


class ConvDeltaTokenizer(BaseDeltaTokenizer):
    """Fallback lightweight tokenizer (kept for compatibility)."""

    def __init__(self, token_dim: int, in_channels: int = 3, hidden_dim: int = 512):
        super().__init__(token_dim=token_dim)
        if int(in_channels) <= 0:
            raise ValueError(f"`in_channels` must be positive, got {in_channels}")
        if int(hidden_dim) <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {hidden_dim}")

        in_channels = int(in_channels)
        hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(hidden_dim, self.token_dim)

    def freeze_encoder(self) -> None:
        self.requires_grad_(False)
        self.eval()

    def encode_pair(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        if frame_t.ndim != 4 or frame_t1.ndim != 4:
            raise ValueError(
                "`frame_t` and `frame_t1` must be 4D [B,C,H,W], "
                f"got {tuple(frame_t.shape)} and {tuple(frame_t1.shape)}"
            )
        if frame_t.shape != frame_t1.shape:
            raise ValueError(f"Frame shape mismatch: {tuple(frame_t.shape)} vs {tuple(frame_t1.shape)}")

        x = torch.cat([frame_t, frame_t1], dim=1)
        x = self.encoder(x).flatten(1)
        token = self.proj(x)
        return token.unsqueeze(1)

    def decode_next(self, frame_t: torch.Tensor, delta_token: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "ConvDeltaTokenizer does not support decoding to pixels. "
            "Use `delta_tokenizer_config.impl=deltatok` for decoder-based rollout visualization."
        )

    @classmethod
    def from_pretrained(
        cls,
        token_dim: int,
        in_channels: int = 3,
        hidden_dim: int = 512,
        pretrained_path: str | None = None,
        skip_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ConvDeltaTokenizer":
        tokenizer = cls(token_dim=token_dim, in_channels=in_channels, hidden_dim=hidden_dim)
        tokenizer = tokenizer.to(device=device, dtype=torch_dtype)

        if skip_load_from_pretrain:
            return tokenizer
        if pretrained_path is None or str(pretrained_path).strip() == "":
            return tokenizer

        p = os.path.expanduser(os.path.expandvars(str(pretrained_path)))
        if not os.path.isabs(p):
            p = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", p)
            p = os.path.abspath(p)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Delta tokenizer checkpoint not found: {p}")

        tokenizer.load_checkpoint(p, device=device)
        return tokenizer
