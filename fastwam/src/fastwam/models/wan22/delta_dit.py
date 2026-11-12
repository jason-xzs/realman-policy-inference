from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import DiTBlock, precompute_freqs_cis, sinusoidal_embedding_1d


class TaskPrivateAdapter(nn.Module):
    """Small residual adapter initialized as a no-op."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(int(hidden_dim), int(bottleneck_dim))
        self.act = nn.GELU(approximate="tanh")
        self.up = nn.Linear(int(bottleneck_dim), int(hidden_dim))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class DeltaDiT(nn.Module):
    """DiT for denoising a single delta token."""
    DELTA_BACKBONE_SKIP_PREFIXES = ("delta_projection.", "head.")
    DELTA_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
        "delta_token_dim",
    )

    def __init__(
        self,
        delta_token_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        delta_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.delta_token_dim = int(delta_token_dim)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.delta_attention_mask_mode = str(delta_attention_mask_mode).strip().lower()

        if self.delta_token_dim <= 0:
            raise ValueError(f"`delta_token_dim` must be > 0, got {delta_token_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if self.attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if self.attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")
        if int(num_layers) <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {num_layers}")
        if self.delta_attention_mask_mode not in {"bidirectional", "per_frame_causal"}:
            raise ValueError(
                f"Unsupported `delta_attention_mask_mode`: {self.delta_attention_mask_mode}. "
                "Expected one of ['bidirectional', 'per_frame_causal']."
            )

        self.delta_projection = nn.Linear(self.delta_token_dim, self.hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    eps=eps,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.head = nn.Linear(self.hidden_dim, self.delta_token_dim)
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=1024)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    def build_expert_self_mask(
        self,
        expert_seq_len: int,
        expert_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if expert_seq_len <= 0:
            raise ValueError(f"`expert_seq_len` must be positive, got {expert_seq_len}")
        if expert_tokens_per_frame <= 0:
            raise ValueError(f"`expert_tokens_per_frame` must be positive, got {expert_tokens_per_frame}")

        if self.delta_attention_mask_mode == "bidirectional":
            return torch.ones((expert_seq_len, expert_seq_len), dtype=torch.bool, device=device)

        if self.delta_attention_mask_mode == "per_frame_causal":
            if expert_seq_len % expert_tokens_per_frame != 0:
                raise ValueError(
                    "`expert_seq_len` must be divisible by `expert_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {expert_seq_len} and {expert_tokens_per_frame}"
                )
            num_video_frames = expert_seq_len // expert_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(expert_tokens_per_frame, dim=0).repeat_interleave(
                expert_tokens_per_frame, dim=1
            )

        raise ValueError(f"Unsupported delta attention mask mode: {self.delta_attention_mask_mode}")

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.DELTA_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        delta_dit_config: dict[str, Any],
        delta_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "DeltaDiT":
        if delta_dit_config is None:
            raise ValueError("`delta_dit_config` is required for DeltaDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            return cls(**delta_dit_config).to(device=device, dtype=torch_dtype)
        if not delta_dit_pretrained_path:
            return cls(**delta_dit_config).to(device=device, dtype=torch_dtype)

        p = os.path.expanduser(os.path.expandvars(str(delta_dit_pretrained_path)))
        if not os.path.isabs(p):
            p = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", p)
            p = os.path.abspath(p)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"`delta_dit_pretrained_path` does not exist: {p}")

        cfg = dict(delta_dit_config)
        delta_expert = cls(**cfg).to(device=device, dtype=torch_dtype)
        delta_state = delta_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(delta_state.keys())

        payload = torch.load(p, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid delta backbone payload type from {p}: {type(payload)}")

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"`meta` must be a dict in {p}, got {type(meta)}")
        expected_meta = {
            "hidden_dim": int(cfg["hidden_dim"]),
            "ffn_dim": int(cfg["ffn_dim"]),
            "num_layers": int(cfg["num_layers"]),
            "num_heads": int(cfg["num_heads"]),
            "attn_head_dim": int(cfg["attn_head_dim"]),
            "text_dim": int(cfg["text_dim"]),
            "freq_dim": int(cfg["freq_dim"]),
            "eps": float(cfg["eps"]),
            "delta_token_dim": int(cfg["delta_token_dim"]),
        }
        for key in cls.DELTA_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {p}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {p}: expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {p}: expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {p}, got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Delta backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(delta_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {p}, got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {p}: expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        delta_expert.load_state_dict(merged_state, strict=True)
        return delta_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        delta_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if delta_tokens.ndim != 3:
            raise ValueError(
                f"`delta_tokens` must be 3D [B, T, token_dim], got shape {tuple(delta_tokens.shape)}"
            )
        if delta_tokens.shape[2] != self.delta_token_dim:
            raise ValueError(
                f"`delta_tokens` last dim must be {self.delta_token_dim}, got {delta_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")

        batch_size = delta_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between delta tokens and context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, delta timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B,L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.delta_projection(delta_tokens)

        seq_len = int(tokens.shape[1])
        if seq_len <= 0:
            raise ValueError(f"`tokens` temporal dim must be positive, got {seq_len}")
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Delta token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        delta_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            delta_tokens=delta_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state)


class ActionDeltaDiT(nn.Module):
    """Shared DiT backbone for jointly denoising delta tokens and action tokens."""

    BACKBONE_SKIP_PREFIXES = (
        "delta_projection.",
        "action_encoder.",
        "delta_head.",
        "action_head.",
        "task_adapters.",
    )
    BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        delta_token_dim: int,
        action_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
        task_adapter_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.delta_token_dim = int(delta_token_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.num_layers = int(num_layers)

        if self.delta_token_dim <= 0:
            raise ValueError(f"`delta_token_dim` must be > 0, got {delta_token_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be > 0, got {action_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if self.attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if self.attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {num_layers}")

        self.delta_projection = nn.Linear(self.delta_token_dim, self.hidden_dim)
        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    eps=eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.delta_head = nn.Linear(self.hidden_dim, self.delta_token_dim)
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=1024)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self._init_task_adapters(task_adapter_config)

    @staticmethod
    def _parse_adapter_layers(layers: Any, num_layers: int) -> tuple[int, ...]:
        if not isinstance(layers, str):
            raise ValueError(
                "`task_adapter.layers` must be a string: 'all' or 'last_K' "
                f"(for example 'last_6'), got {type(layers)}."
            )
        key = layers.strip().lower()
        if key == "all":
            return tuple(range(num_layers))
        if key.startswith("last_"):
            count_text = key.removeprefix("last_")
            if not count_text.isdigit():
                raise ValueError(f"`task_adapter.layers` must be 'all' or 'last_K', got `{layers}`.")
            count = int(count_text)
            if count <= 0:
                raise ValueError(f"`task_adapter.layers` last_K must use K > 0, got `{layers}`.")
            if count > num_layers:
                raise ValueError(
                    f"`task_adapter.layers={layers}` exceeds num_layers={num_layers}."
                )
            return tuple(range(num_layers - count, num_layers))
        raise ValueError(f"`task_adapter.layers` must be 'all' or 'last_K', got `{layers}`.")

    def _init_task_adapters(self, task_adapter_config: Optional[dict[str, Any]]) -> None:
        cfg = dict(task_adapter_config or {})
        self.task_adapter_enabled = bool(cfg.get("enabled", False))
        if not self.task_adapter_enabled:
            self.task_adapter_layers = tuple()
            self.task_adapters = nn.ModuleDict({"delta": nn.ModuleDict(), "action": nn.ModuleDict()})
            return

        layers = self._parse_adapter_layers(cfg.get("layers", "last_6"), self.num_layers)
        if len(layers) == 0:
            raise ValueError("`task_adapter.enabled=true` requires at least one adapter layer.")
        bottleneck_dim = cfg.get("bottleneck_dim", None)
        if bottleneck_dim is None:
            bottleneck_ratio = int(cfg.get("bottleneck_ratio", 16))
            if bottleneck_ratio <= 0:
                raise ValueError(f"`task_adapter.bottleneck_ratio` must be > 0, got {bottleneck_ratio}")
            bottleneck_dim = max(self.hidden_dim // bottleneck_ratio, 1)
        bottleneck_dim = int(bottleneck_dim)
        if bottleneck_dim <= 0:
            raise ValueError(f"`task_adapter.bottleneck_dim` must be > 0, got {bottleneck_dim}")

        self.task_adapter_layers = tuple(layers)
        self.task_adapters = nn.ModuleDict(
            {
                "delta": nn.ModuleDict(
                    {str(layer_idx): TaskPrivateAdapter(self.hidden_dim, bottleneck_dim) for layer_idx in layers}
                ),
                "action": nn.ModuleDict(
                    {str(layer_idx): TaskPrivateAdapter(self.hidden_dim, bottleneck_dim) for layer_idx in layers}
                ),
            }
        )

    def apply_layer_adapters(
        self,
        tokens: torch.Tensor,
        *,
        layer_idx: int,
        adapter_context: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        if not self.task_adapter_enabled:
            return tokens
        layer_idx = int(layer_idx)
        if layer_idx not in self.task_adapter_layers:
            return tokens
        layer_key = str(layer_idx)
        if adapter_context is None:
            raise ValueError("Task adapters require adapter_context with `delta_len` and `action_len`.")

        delta_len = int(adapter_context.get("delta_len", 0))
        action_len = int(adapter_context.get("action_len", 0))
        if delta_len + action_len != int(tokens.shape[1]):
            raise ValueError(
                "Task adapter token length mismatch: "
                f"delta_len={delta_len}, action_len={action_len}, tokens={tokens.shape[1]}."
            )

        chunks = []
        if delta_len > 0:
            delta_tokens = tokens[:, :delta_len]
            chunks.append(delta_tokens + self.task_adapters["delta"][layer_key](delta_tokens))
        if action_len > 0:
            action_start = delta_len
            action_end = delta_len + action_len
            action_tokens = tokens[:, action_start:action_end]
            chunks.append(action_tokens + self.task_adapters["action"][layer_key](action_tokens))
        return torch.cat(chunks, dim=1)

    @classmethod
    def from_pretrained(
        cls,
        action_delta_dit_config: dict[str, Any],
        action_delta_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDeltaDiT":
        cfg = cls._normalize_config(action_delta_dit_config)
        model = cls(**cfg).to(device=device, dtype=torch_dtype)

        backbone_path = None if skip_dit_load_from_pretrain else action_delta_dit_pretrained_path
        if not backbone_path:
            return model

        model._load_preprocessed_backbone(
            backbone_path=backbone_path,
            expected_cfg=cfg,
            backbone_kind="action_delta",
        )
        return model.to(device=device, dtype=torch_dtype)

    @staticmethod
    def _normalize_config(action_delta_dit_config: dict[str, Any]) -> dict[str, Any]:
        if action_delta_dit_config is None:
            raise ValueError("`action_delta_dit_config` is required for ActionDeltaDiT.")
        cfg = dict(action_delta_dit_config)
        required_keys = (
            "delta_token_dim",
            "action_dim",
            "hidden_dim",
            "ffn_dim",
            "num_heads",
            "attn_head_dim",
            "num_layers",
            "text_dim",
            "freq_dim",
            "eps",
        )
        missing_keys = [key for key in required_keys if key not in cfg]
        if missing_keys:
            raise ValueError(f"`action_delta_dit_config` missing required keys: {missing_keys}")
        return {
            "delta_token_dim": int(cfg["delta_token_dim"]),
            "action_dim": int(cfg["action_dim"]),
            "hidden_dim": int(cfg["hidden_dim"]),
            "ffn_dim": int(cfg["ffn_dim"]),
            "num_heads": int(cfg["num_heads"]),
            "attn_head_dim": int(cfg["attn_head_dim"]),
            "num_layers": int(cfg["num_layers"]),
            "text_dim": int(cfg["text_dim"]),
            "freq_dim": int(cfg["freq_dim"]),
            "eps": float(cfg["eps"]),
            "use_gradient_checkpointing": bool(cfg.get("use_gradient_checkpointing", False)),
            "task_adapter_config": cfg.get("task_adapter", cfg.get("task_adapter_config", None)),
        }

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.BACKBONE_SKIP_PREFIXES)
        }

    @staticmethod
    def _resolve_pretrained_path(path: str) -> str:
        p = os.path.expanduser(os.path.expandvars(str(path)))
        if not os.path.isabs(p):
            p = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", p)
            p = os.path.abspath(p)
        return p

    def _load_preprocessed_backbone(
        self,
        *,
        backbone_path: str,
        expected_cfg: dict[str, Any],
        backbone_kind: str,
    ) -> None:
        p = self._resolve_pretrained_path(backbone_path)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"`{backbone_kind}_dit_pretrained_path` does not exist: {p}")

        payload = torch.load(p, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid {backbone_kind} backbone payload type from {p}: {type(payload)}")
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"`meta` must be a dict in {p}, got {type(meta)}")
        for key in self.BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {p}")
            expected_value = expected_cfg[key]
            got_value = meta[key]
            mismatch = (
                abs(float(got_value) - float(expected_value)) > 1e-12
                if key == "eps"
                else int(got_value) != int(expected_value)
            )
            if mismatch:
                raise ValueError(f"`meta.{key}` mismatch in {p}: expected {expected_value}, got {got_value}")

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(f"`backbone_state_dict` must be a dict in {p}, got {type(backbone_state_dict)}")

        state = self.state_dict()
        expected_keys = self.backbone_key_set(state.keys())
        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "ActionDeltaDiT backbone key mismatch. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(state)
        for key in expected_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"`backbone_state_dict[{key}]` must be torch.Tensor in {p}, got {type(value)}")
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {p}: expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)
        self.load_state_dict(merged_state, strict=True)

    def build_expert_self_mask(
        self,
        expert_seq_len: int,
        expert_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if expert_seq_len <= 0:
            raise ValueError(f"`expert_seq_len` must be positive, got {expert_seq_len}")
        del expert_tokens_per_frame
        return torch.ones((expert_seq_len, expert_seq_len), dtype=torch.bool, device=device)

    def _validate_branch(
        self,
        name: str,
        tokens: Optional[torch.Tensor],
        expected_dim: int,
    ) -> Optional[torch.Tensor]:
        if tokens is None:
            return None
        if tokens.ndim != 3:
            raise ValueError(f"`{name}` must be 3D [B,T,D], got shape {tuple(tokens.shape)}")
        if tokens.shape[2] != expected_dim:
            raise ValueError(f"`{name}` last dim must be {expected_dim}, got {tokens.shape[2]}")
        return tokens

    def _expand_timestep(self, name: str, timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError(f"`{name}` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`{name}` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError(f"During training, `{name}` length must match batch_size.")
            timestep = timestep.expand(batch_size)
        return timestep

    def pre_dit(
        self,
        *,
        delta_tokens: Optional[torch.Tensor],
        action_tokens: Optional[torch.Tensor],
        timestep_delta: Optional[torch.Tensor],
        timestep_action: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        delta_tokens = self._validate_branch("delta_tokens", delta_tokens, self.delta_token_dim)
        action_tokens = self._validate_branch("action_tokens", action_tokens, self.action_dim)
        if delta_tokens is None and action_tokens is None:
            raise ValueError("At least one of `delta_tokens` or `action_tokens` must be provided.")
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B,L,D], got shape {tuple(context.shape)}")

        ref_tokens = delta_tokens if delta_tokens is not None else action_tokens
        assert ref_tokens is not None
        batch_size = int(ref_tokens.shape[0])
        for name, branch in (("delta_tokens", delta_tokens), ("action_tokens", action_tokens)):
            if branch is not None and int(branch.shape[0]) != batch_size:
                raise ValueError(f"Batch mismatch between `{name}` and other branch: {branch.shape[0]} vs {batch_size}")
        if int(context.shape[0]) != batch_size:
            raise ValueError(f"Batch mismatch between joint tokens and context: {batch_size} vs {context.shape[0]}")

        if context_mask is None:
            context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B,L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B,L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        token_chunks = []
        freq_chunks = []
        t_mod_chunks = []
        delta_len = 0
        action_len = 0
        if delta_tokens is not None:
            if timestep_delta is None:
                raise ValueError("`timestep_delta` is required when `delta_tokens` is provided.")
            timestep_delta = self._expand_timestep("timestep_delta", timestep_delta, batch_size)
            delta_len = int(delta_tokens.shape[1])
            if delta_len > self.freqs.shape[0]:
                raise ValueError(f"Delta token length {delta_len} exceeds RoPE cache {self.freqs.shape[0]}.")
            delta_t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep_delta))
            delta_t_mod = self.time_projection(delta_t).unflatten(1, (6, self.hidden_dim))
            token_chunks.append(self.delta_projection(delta_tokens))
            freq_chunks.append(self.freqs[:delta_len].view(delta_len, 1, -1).to(delta_tokens.device))
            t_mod_chunks.append(delta_t_mod.unsqueeze(1).expand(-1, delta_len, -1, -1))
        if action_tokens is not None:
            if timestep_action is None:
                raise ValueError("`timestep_action` is required when `action_tokens` is provided.")
            timestep_action = self._expand_timestep("timestep_action", timestep_action, batch_size)
            action_len = int(action_tokens.shape[1])
            if action_len > self.freqs.shape[0]:
                raise ValueError(f"Action token length {action_len} exceeds RoPE cache {self.freqs.shape[0]}.")
            action_t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep_action))
            action_t_mod = self.time_projection(action_t).unflatten(1, (6, self.hidden_dim))
            token_chunks.append(self.action_encoder(action_tokens))
            freq_chunks.append(self.freqs[:action_len].view(action_len, 1, -1).to(action_tokens.device))
            t_mod_chunks.append(action_t_mod.unsqueeze(1).expand(-1, action_len, -1, -1))

        tokens = torch.cat(token_chunks, dim=1)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        return {
            "tokens": tokens,
            "freqs": torch.cat(freq_chunks, dim=0).to(tokens.device),
            "t_mod": torch.cat(t_mod_chunks, dim=1),
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "delta_len": delta_len,
                "action_len": action_len,
                "seq_len": int(tokens.shape[1]),
            },
        }

    def post_dit(
        self,
        tokens: torch.Tensor,
        pre_state: Dict[str, Any],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        delta_len = int(pre_state["meta"]["delta_len"])
        action_len = int(pre_state["meta"]["action_len"])
        offset = 0
        pred_delta = None
        pred_action = None
        if delta_len > 0:
            pred_delta = self.delta_head(tokens[:, offset : offset + delta_len])
            offset += delta_len
        if action_len > 0:
            pred_action = self.action_head(tokens[:, offset : offset + action_len])
        return pred_delta, pred_action

    def forward(
        self,
        delta_tokens: Optional[torch.Tensor],
        action_tokens: Optional[torch.Tensor],
        timestep_delta: Optional[torch.Tensor],
        timestep_action: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        pre_state = self.pre_dit(
            delta_tokens=delta_tokens,
            action_tokens=action_tokens,
            timestep_delta=timestep_delta,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]

        for layer_idx, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_attn_mask,
                )
            else:
                x = block(x, context_emb, t_mod, freqs, context_mask=context_attn_mask)
            x = self.apply_layer_adapters(
                x,
                layer_idx=layer_idx,
                adapter_context=pre_state["meta"],
            )

        return self.post_dit(x, pre_state)
