from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict


from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .delta_dit import DeltaDiT
from .delta_tokenizer import BaseDeltaTokenizer, ConvDeltaTokenizer, DeltaTokEncoderTokenizer, LAMDeltaTokenizer
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import WanVideoDiT, precompute_freqs_cis, sinusoidal_embedding_1d

logger = get_logger(__name__)


class WanLoRADeltaExpert(nn.Module):
    """Delta denoising head that reuses the Wan video blocks and their LoRA weights."""

    def __init__(
        self,
        wan_expert: nn.Module,
        delta_token_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        delta_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        base_wan = self._resolve_wan_base(wan_expert)
        object.__setattr__(self, "wan_expert", wan_expert)
        self.delta_token_dim = int(delta_token_dim)
        self.hidden_dim = int(base_wan.hidden_dim)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(base_wan.num_heads)
        self.attn_head_dim = int(base_wan.attn_head_dim)
        self.delta_attention_mask_mode = str(delta_attention_mask_mode).strip().lower()
        if self.delta_token_dim <= 0:
            raise ValueError(f"`delta_token_dim` must be > 0, got {delta_token_dim}")
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
        self.head = nn.Linear(self.hidden_dim, self.delta_token_dim)
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=1024)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    @staticmethod
    def _resolve_wan_base(wan_expert: nn.Module) -> nn.Module:
        get_base_model = getattr(wan_expert, "get_base_model", None)
        if callable(get_base_model):
            return get_base_model()
        return wan_expert

    @property
    def blocks(self):
        return self._resolve_wan_base(self.wan_expert).blocks

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

    def _expand_timestep(self, timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, `timestep` length must match batch_size.")
            timestep = timestep.expand(batch_size)
        return timestep

    def pre_dit(
        self,
        *,
        delta_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if delta_tokens.ndim != 3:
            raise ValueError(f"`delta_tokens` must be 3D [B,T,D], got shape {tuple(delta_tokens.shape)}")
        if delta_tokens.shape[2] != self.delta_token_dim:
            raise ValueError(
                f"`delta_tokens` last dim must be {self.delta_token_dim}, got {delta_tokens.shape[2]}"
            )
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B,L,D], got shape {tuple(context.shape)}")
        batch_size = int(delta_tokens.shape[0])
        if int(context.shape[0]) != batch_size:
            raise ValueError(f"Batch mismatch between delta tokens and context: {batch_size} vs {context.shape[0]}")
        if context_mask is None:
            context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B,L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B,L], got {tuple(context_mask.shape)}"
                )

        timestep = self._expand_timestep(timestep, batch_size)
        seq_len = int(delta_tokens.shape[1])
        if seq_len > self.freqs.shape[0]:
            raise ValueError(f"Delta token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}.")
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        tokens = self.delta_projection(delta_tokens)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        return {
            "tokens": tokens,
            "freqs": self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device),
            "t_mod": t_mod.unsqueeze(1).expand(-1, seq_len, -1, -1),
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        del pre_state
        return self.head(tokens)


class DeltaWAM(nn.Module):
    """DeltaWAM with two experts: DeltaDiT (single delta token) and ActionDiT."""

    def __init__(
        self,
        delta_expert: DeltaDiT,
        action_expert: ActionDiT,
        mot: MoT,
        delta_tokenizer: BaseDeltaTokenizer,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        delta_train_shift: float = 5.0,
        delta_infer_shift: float = 5.0,
        delta_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_delta: float = 1.0,
        loss_lambda_action: float = 1.0,
        delta_decode_default_mode: str = "gt_prev",
        delta_decode_eval_modes: Optional[Sequence[str]] = None,
        wan_expert: Optional[WanVideoDiT] = None,
        wan_lora_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.delta_expert = delta_expert
        self.action_expert = action_expert
        self.mot = mot
        self.delta_tokenizer = delta_tokenizer
        self.delta_tokenizer.freeze_encoder()
        self.delta_tokenizer.requires_grad_(False)
        if wan_expert is None:
            raise ValueError("`wan_expert` is required for DeltaWAM.")
        # `wan_expert` is only used to build first-frame anchor features.
        self.wan_expert = wan_expert
        self.wan_expert.requires_grad_(False)
        self.wan_expert.eval()
        self.wan_lora_config = dict(wan_lora_config or {})
        self.wan_train_full = bool(self.wan_lora_config.get("train_full", False))
        self.wan_lora_enabled = bool(self.wan_lora_config.get("enabled", False))
        if self.wan_train_full and self.wan_lora_enabled:
            raise ValueError("`wan_lora_config.train_full=true` is mutually exclusive with `enabled=true`.")
        self._set_wan_lora_trainable()
        self.vae = vae
        if self.vae is not None:
            self.vae.requires_grad_(False).eval()

        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)

        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_delta_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=delta_num_train_timesteps,
            shift=delta_train_shift,
        )
        self.infer_delta_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=delta_num_train_timesteps,
            shift=delta_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_delta = float(loss_lambda_delta)
        self.loss_lambda_action = float(loss_lambda_action)
        self.delta_decode_default_mode = str(delta_decode_default_mode).strip().lower()
        if delta_decode_eval_modes is None:
            delta_decode_eval_modes = ("gt_prev", "pred_prev")
        self.delta_decode_eval_modes = tuple(str(m).strip().lower() for m in delta_decode_eval_modes)
        valid_decode_modes = {"gt_prev", "pred_prev"}
        if self.delta_decode_default_mode not in valid_decode_modes:
            raise ValueError(
                f"`delta_decode_default_mode` must be one of {sorted(valid_decode_modes)}, "
                f"got `{self.delta_decode_default_mode}`"
            )
        for mode in self.delta_decode_eval_modes:
            if mode not in valid_decode_modes:
                raise ValueError(
                    f"Invalid decode mode `{mode}` in `delta_decode_eval_modes`; "
                    f"expected one of {sorted(valid_decode_modes)}"
                )

        self.to(self.device)

    def _wan_cache_expert(self):
        get_base_model = getattr(self.wan_expert, "get_base_model", None)
        if callable(get_base_model):
            return get_base_model()
        return self.wan_expert

    def _set_wan_lora_trainable(self) -> None:
        if self.wan_train_full:
            self.wan_expert.requires_grad_(True)
            return
        if not self.wan_lora_enabled:
            return
        for name, param in self.wan_expert.named_parameters():
            param.requires_grad_("lora_" in name)

    def extra_trainable_parameters(self):
        if self.wan_train_full:
            self.wan_expert.requires_grad_(True)
            return list(self.wan_expert.parameters())
        if not self.wan_lora_enabled:
            return []
        self._set_wan_lora_trainable()
        return [param for name, param in self.wan_expert.named_parameters() if "lora_" in name]

    def set_extra_trainable_modules_train(self) -> None:
        if self.wan_train_full:
            self.wan_expert.train()
            self.wan_expert.requires_grad_(True)
            return
        if not self.wan_lora_enabled:
            return
        self.wan_expert.eval()
        self._set_wan_lora_trainable()

    @classmethod
    def from_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        delta_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        delta_tokenizer_config: dict[str, Any] | None = None,
        delta_dit_pretrained_path: str | None = None,
        action_dit_pretrained_path: str | None = None,
        delta_tokenizer_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        skip_delta_dit_load_from_pretrain: bool = False,
        skip_delta_tokenizer_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        delta_train_shift: float = 5.0,
        delta_infer_shift: float = 5.0,
        delta_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_delta: float = 1.0,
        loss_lambda_action: float = 1.0,
        delta_decode_default_mode: str = "pred_prev",
        delta_decode_eval_modes: Optional[Sequence[str]] = None,
        wan_dit_config: Optional[dict[str, Any]] = None,
        wan_lora_config: Optional[dict[str, Any]] = None,
        delta_expert_impl: str = "delta_dit",
    ):
        if delta_dit_config is None:
            raise ValueError("`delta_dit_config` is required for DeltaWAM.from_pretrained().")
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for DeltaWAM.from_pretrained().")
        if delta_tokenizer_config is None:
            delta_tokenizer_config = {}

        # Avoid mutating caller-owned config dicts.
        delta_dit_config = dict(delta_dit_config)
        action_dit_config = dict(action_dit_config)
        delta_tokenizer_config = dict(delta_tokenizer_config)
        wan_lora_config = dict(wan_lora_config or {})
        delta_expert_impl = str(delta_expert_impl).strip().lower()
        if delta_expert_impl not in {"delta_dit", "wan_lora"}:
            raise ValueError(
                f"`delta_expert_impl` must be one of ['delta_dit', 'wan_lora'], got `{delta_expert_impl}`"
            )

        if "text_dim" not in delta_dit_config:
            raise ValueError("`delta_dit_config['text_dim']` is required.")

        tokenizer_impl = str(delta_tokenizer_config.get("impl", "conv")).strip().lower()
        tokenizer_token_dim = int(delta_dit_config["delta_token_dim"])
        tokenizer_backbone = None
        if tokenizer_impl == "deltatok":
            backbone = delta_tokenizer_config.get("backbone", None)
            if backbone is None:
                # Backward compatibility with older configs.
                backbone = delta_tokenizer_config.get("backbone_cfg", None)
            if backbone is not None:
                # Hydra may already recursively instantiate nested `_target_` nodes.
                if isinstance(backbone, nn.Module):
                    tokenizer_backbone = backbone
                else:
                    tokenizer_backbone = instantiate(backbone, device=device)
            if tokenizer_backbone is None:
                raise ValueError(
                    "`delta_tokenizer_config.impl=deltatok` requires `backbone` or `backbone_cfg`."
                )

            inferred_token_dim = getattr(tokenizer_backbone, "hidden_size", None)
            if inferred_token_dim is not None and inferred_token_dim != tokenizer_token_dim:
                raise ValueError(
                    "delta_token_dim mismatch between config and DeltaTok backbone: "
                    f"config={tokenizer_token_dim}, backbone={inferred_token_dim}. "
                    "Please update `delta_dit_config.delta_token_dim` to match the backbone output dim."
                )
        elif tokenizer_impl == "lam":
            lam_latent_dim = int(delta_tokenizer_config.get("lam_latent_dim", 32))
            if lam_latent_dim != tokenizer_token_dim:
                raise ValueError(
                    "delta_token_dim mismatch between config and LAM latent dim: "
                    f"config={tokenizer_token_dim}, lam_latent_dim={lam_latent_dim}. "
                    "Please update `delta_dit_config.delta_token_dim` to match the LAM output dim."
                )

        if wan_dit_config is None:
            raise ValueError("`wan_dit_config` is required for DeltaWAM.from_pretrained().")
        wan_dit_config = dict(wan_dit_config)

        # Always load VAE for first-frame multi-token anchor construction.
        # Text encoder/tokenizer still follow `load_text_encoder`.
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=wan_dit_config,
            skip_dit_load_from_pretrain=False,
            load_text_encoder=bool(load_text_encoder),
        )

        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        if bool(wan_lora_config.get("enabled", False)):
            target_modules = list(
                wan_lora_config.get(
                    "target_modules",
                    ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"],
                )
            )
            lora_cfg = LoraConfig(
                r=int(wan_lora_config.get("rank", 8)),
                lora_alpha=float(wan_lora_config.get("alpha", 16.0)),
                lora_dropout=float(wan_lora_config.get("dropout", 0.0)),
                target_modules=target_modules,
                bias=str(wan_lora_config.get("bias", "none")),
                init_lora_weights=bool(wan_lora_config.get("init_lora_weights", True)),
            )
            components.dit = get_peft_model(components.dit, lora_cfg)
            logger.info("Enabled PEFT Wan LoRA with target_modules=%s", target_modules)

        if delta_expert_impl == "wan_lora":
            if skip_delta_dit_load_from_pretrain:
                logger.info("Wan-LoRA delta expert has no DeltaDiT backbone to load.")
            elif delta_dit_pretrained_path:
                logger.info("Ignoring `delta_dit_pretrained_path=%s`; using Wan-LoRA delta expert.", delta_dit_pretrained_path)
            delta_expert = WanLoRADeltaExpert(
                wan_expert=components.dit,
                delta_token_dim=int(delta_dit_config["delta_token_dim"]),
                text_dim=int(delta_dit_config["text_dim"]),
                freq_dim=int(delta_dit_config["freq_dim"]),
                eps=float(delta_dit_config.get("eps", 1e-6)),
                delta_attention_mask_mode=str(delta_dit_config.get("delta_attention_mask_mode", "bidirectional")),
                use_gradient_checkpointing=bool(delta_dit_config.get("use_gradient_checkpointing", False)),
            ).to(device=device, dtype=torch_dtype)
        else:
            delta_expert = DeltaDiT.from_pretrained(
                delta_dit_config=delta_dit_config,
                delta_dit_pretrained_path=delta_dit_pretrained_path,
                skip_dit_load_from_pretrain=skip_delta_dit_load_from_pretrain,
                device=device,
                torch_dtype=torch_dtype,
            )

        if int(action_expert.num_heads) != int(delta_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match the delta expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(delta_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match the delta expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(delta_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")
        
        if tokenizer_impl == "deltatok":
            delta_tokenizer = DeltaTokEncoderTokenizer.from_pretrained(
                token_dim=tokenizer_token_dim,
                backbone=tokenizer_backbone,
                num_hidden_layers=int(delta_tokenizer_config.get("num_hidden_layers", 12)),
                use_delta=bool(delta_tokenizer_config.get("use_delta", True)),
                use_decoder_head=delta_tokenizer_config.get("use_decoder_head", None),
                use_decoder_xy_embed=delta_tokenizer_config.get("use_decoder_xy_embed", None),
                layer_scale_init=float(delta_tokenizer_config.get("layer_scale_init", 1e-5)),
                use_qk_norm=bool(delta_tokenizer_config.get("use_qk_norm", True)),
                use_gated_attn=bool(delta_tokenizer_config.get("use_gated_attn", True)),
                use_swiglu=bool(delta_tokenizer_config.get("use_swiglu", True)),
                use_rope_aug=bool(delta_tokenizer_config.get("use_rope_aug", True)),
                pretrained_path=delta_tokenizer_pretrained_path,
                skip_load_from_pretrain=skip_delta_tokenizer_load_from_pretrain,
                device=device,
                torch_dtype=torch_dtype,
            )
        elif tokenizer_impl == "lam":
            delta_tokenizer = LAMDeltaTokenizer.from_pretrained(
                token_dim=tokenizer_token_dim,
                image_channels=int(delta_tokenizer_config.get("image_channels", 3)),
                lam_model_dim=int(delta_tokenizer_config.get("lam_model_dim", 1024)),
                lam_latent_dim=int(delta_tokenizer_config.get("lam_latent_dim", 32)),
                lam_patch_size=int(delta_tokenizer_config.get("lam_patch_size", 16)),
                lam_enc_blocks=int(delta_tokenizer_config.get("lam_enc_blocks", 24)),
                lam_dec_blocks=int(delta_tokenizer_config.get("lam_dec_blocks", 24)),
                lam_num_heads=int(delta_tokenizer_config.get("lam_num_heads", 16)),
                lam_dropout=float(delta_tokenizer_config.get("lam_dropout", 0.0)),
                pretrained_path=delta_tokenizer_pretrained_path,
                skip_load_from_pretrain=skip_delta_tokenizer_load_from_pretrain,
                device=device,
                torch_dtype=torch_dtype,
            )
        elif tokenizer_impl == "conv":
            delta_tokenizer = ConvDeltaTokenizer.from_pretrained(
                token_dim=tokenizer_token_dim,
                in_channels=int(delta_tokenizer_config.get("in_channels", 3)),
                hidden_dim=int(delta_tokenizer_config.get("hidden_dim", 512)),
                pretrained_path=delta_tokenizer_pretrained_path,
                skip_load_from_pretrain=skip_delta_tokenizer_load_from_pretrain,
                device=device,
                torch_dtype=torch_dtype,
            )
        else:
            raise ValueError(
                f"Unsupported delta tokenizer impl: {tokenizer_impl}. Expected one of ['conv', 'deltatok', 'lam']."
            )

        mot = MoT(
            mixtures={"video": delta_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            delta_expert=delta_expert,
            action_expert=action_expert,
            mot=mot,
            delta_tokenizer=delta_tokenizer,
            wan_expert=components.dit,
            vae=components.vae,
            text_encoder=(components.text_encoder if components is not None else None),
            tokenizer=(components.tokenizer if components is not None else None),
            text_dim=int(delta_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            delta_train_shift=delta_train_shift,
            delta_infer_shift=delta_infer_shift,
            delta_num_train_timesteps=delta_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_delta=loss_lambda_delta,
            loss_lambda_action=loss_lambda_action,
            delta_decode_default_mode=delta_decode_default_mode,
            delta_decode_eval_modes=delta_decode_eval_modes,
            wan_lora_config=wan_lora_config,
        )
        model.model_paths = {
            "text_encoder": (components.text_encoder_path if components is not None else None),
            "tokenizer": (components.tokenizer_path if components is not None else None),
            "vae": (components.vae_path if components is not None else None),
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
            "delta_dit_backbone": (
                "WAN_LORA_DELTA"
                if delta_expert_impl == "wan_lora"
                else (
                "SKIPPED_PRETRAIN"
                if skip_delta_dit_load_from_pretrain
                else delta_dit_pretrained_path
                )
            ),
            "delta_tokenizer": delta_tokenizer_pretrained_path,
            "wan_dit": components.dit_path,
        }
        if model.wan_lora_enabled:
            model.model_paths["wan_lora"] = "peft"
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        self.delta_tokenizer.to(*args, **kwargs)
        if self.wan_expert is not None:
            self.wan_expert.to(*args, **kwargs)
        if self.vae is not None:
            self.vae.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def _encode_video_latents(
        self,
        video_tensor: torch.Tensor,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        if self.vae is None:
            raise ValueError("VAE is required for first-frame multi-token anchors, but `self.vae` is None.")
        return self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _build_delta_target(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError(f"`video` must be 5D [B,3,T,H,W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`video` channel dim must be 3, got {video.shape[1]}")
        if video.shape[2] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got T={video.shape[2]}")

        bsz, _, num_frames, height, width = video.shape
        # First frame latents from VAE
        first_frame = video[:, :, 0:1, :, :]
        first_frame_latents = self._encode_video_latents(first_frame, tiled=False)  # [B,Cz,1,Hz,Wz]

        # Subsequent frames: single delta token from tokenizer per transition.
        prev = video[:, :, :-1, :, :]
        curr = video[:, :, 1:, :, :]
        frame_t = prev.permute(0, 2, 1, 3, 4).reshape(-1, 3, height, width)
        frame_t1 = curr.permute(0, 2, 1, 3, 4).reshape(-1, 3, height, width)
        pair_tokens = self.delta_tokenizer.encode_pair(frame_t=frame_t, frame_t1=frame_t1)  # [B*(T-1),1,D]
        delta_tokens = pair_tokens.squeeze(1).reshape(
            bsz, num_frames - 1, self.delta_expert.delta_token_dim
        )  # [B,T-1,D]

        return delta_tokens, first_frame_latents

    def _build_wan_first_frame_pre_state(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, Any]:
        wan_expert = self._wan_cache_expert()
        if wan_expert is None:
            raise ValueError("WAN expert is not initialized.")
        if first_frame_latents.ndim != 5 or first_frame_latents.shape[2] != 1:
            raise ValueError(
                "`first_frame_latents` must be [B,C,1,H,W], got "
                f"{tuple(first_frame_latents.shape)}"
            )
        batch_size = int(first_frame_latents.shape[0])
        timestep = torch.zeros((batch_size,), dtype=first_frame_latents.dtype, device=first_frame_latents.device)
        pre = wan_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(wan_expert, "fuse_vae_embedding_in_latents", True)
            ),
        )
        tokens = pre["tokens"]
        if tokens.ndim != 3:
            raise ValueError(f"WAN pre_dit tokens must be [B,S,D], got {tuple(tokens.shape)}")
        return pre

    def build_inputs(self, sample):
        '''
        sample["video"]: [B, 3, T_video, H, W]
        sample["action"]: [B, T_action, action_dim]
        sample["proprio"]: [B, T_action, proprio_dim]
        sample["context"]: [B, L, D_text]
        sample["context_mask"]: [B, L]
        sample["action_is_pad"]: [B, T_action]
        sample["image_is_pad"]: [B, T_video]
        sample["proprio_is_pad"]: [B, T_action]
        sample["prompt"]: list[str]
        '''
        video = sample["video"]
        action = sample.get("action", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        proprio = sample.get("proprio", None)

        if action is None:
            raise ValueError("`sample['action']` is required for DeltaWAM training.")
        if context is None or context_mask is None:
            raise ValueError("DeltaWAM training requires `sample['context']` and `sample['context_mask']`.")
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B,T,D], got shape {tuple(action.shape)}")
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )

        delta_target, first_frame_latents = self._build_delta_target(
            video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        )

        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            expected_shape = (delta_target.shape[0], video.shape[2])
            if tuple(image_is_pad.shape) != expected_shape:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected {expected_shape}"
                )
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B,T,d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )

        return {
            "delta_target": delta_target,
            "first_frame_latents": first_frame_latents,
            "action": action,
            "context": context,
            "context_mask": context_mask,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        delta_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        *,
        first_frame_tokens: int = 1,
    ) -> torch.Tensor:
        total_seq_len = delta_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        first_frame_tokens = max(int(first_frame_tokens), 1)

        # delta -> delta
        mask[:delta_seq_len, :delta_seq_len] = self.delta_expert.build_video_to_video_mask(
            video_seq_len=delta_seq_len,
            video_tokens_per_frame=first_frame_tokens,
            device=device,
        )
        # action -> action
        mask[delta_seq_len:, delta_seq_len:] = True
        # action -> first-frame anchor tokens only
        first_frame_tokens = min(first_frame_tokens, delta_seq_len)
        mask[delta_seq_len:, :first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _build_expert_wan_attention_mask(
        self,
        *,
        wan_video_seq_len: int,
        wan_tokens_per_frame: int,
        device: torch.device,
        action_expert: nn.Module,
        action_seq_len: int,
        action_tokens_per_frame: int,
        delta_expert: nn.Module,
        delta_seq_len: int,
        delta_tokens_per_frame: int,
    ) -> dict[str, torch.Tensor]:
        if wan_video_seq_len <= 0:
            raise ValueError(f"`wan_video_seq_len` must be positive, got {wan_video_seq_len}")
        if action_seq_len <= 0:
            raise ValueError(f"`action_seq_len` must be positive, got {action_seq_len}")
        if delta_seq_len <= 0:
            raise ValueError(f"`delta_seq_len` must be positive, got {delta_seq_len}")
        wan_tokens_per_frame = max(int(wan_tokens_per_frame), 1)
        action_tokens_per_frame = max(int(action_tokens_per_frame), 1)
        delta_tokens_per_frame = max(int(delta_tokens_per_frame), 1)

        wan_video_mask = self._wan_cache_expert().build_video_to_video_mask(
            video_seq_len=int(wan_video_seq_len),
            video_tokens_per_frame=wan_tokens_per_frame,
            device=device,
        )

        action_self_mask = action_expert.build_expert_self_mask(
            expert_seq_len=int(action_seq_len),
            expert_tokens_per_frame=action_tokens_per_frame,
            device=device,
        )
        delta_self_mask = delta_expert.build_expert_self_mask(
            expert_seq_len=int(delta_seq_len),
            expert_tokens_per_frame=delta_tokens_per_frame,
            device=device,
        )

        visible_wan_tokens = min(wan_tokens_per_frame, wan_video_seq_len)

        def _compose_joint_mask(expert_self_mask: torch.Tensor) -> torch.Tensor:
            expert_seq_len = int(expert_self_mask.shape[0])
            total_seq_len = int(wan_video_seq_len) + expert_seq_len
            mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
            mask[:wan_video_seq_len, :wan_video_seq_len] = wan_video_mask
            mask[wan_video_seq_len:, wan_video_seq_len:] = expert_self_mask
            mask[wan_video_seq_len:, :visible_wan_tokens] = True
            return mask

        return {
            "wan_video_mask": wan_video_mask,
            "action_attention_mask": _compose_joint_mask(action_self_mask),
            "delta_attention_mask": _compose_joint_mask(delta_self_mask),
        }

    def training_loss(self, sample):
        inputs = self.build_inputs(sample)
        delta_target = inputs["delta_target"]
        action = inputs["action"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        first_frame_latents = inputs["first_frame_latents"]

        batch_size = action.shape[0]

        noise_delta = torch.randn_like(delta_target)
        timestep_delta = self.train_delta_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=delta_target.dtype,
        )
        noisy_delta = self.train_delta_scheduler.add_noise(delta_target, noise_delta, timestep_delta)
        target_delta = self.train_delta_scheduler.training_target(delta_target, noise_delta, timestep_delta)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        wan_first_frame_pre = self._build_wan_first_frame_pre_state(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
        )
        wan_video_seq_len = int(wan_first_frame_pre["tokens"].shape[1])
        wan_tokens_per_frame = int(wan_first_frame_pre["meta"]["tokens_per_frame"])
        mask_bundle = self._build_expert_wan_attention_mask(
            wan_video_seq_len=wan_video_seq_len,
            wan_tokens_per_frame=wan_tokens_per_frame,
            device=action.device,
            action_expert=self.action_expert,
            action_seq_len=int(action.shape[1]),
            action_tokens_per_frame=1,
            delta_expert=self.delta_expert,
            delta_seq_len=int(noisy_delta.shape[1]),
            delta_tokens_per_frame=1,
        )
        wan_attention_mask = mask_bundle["action_attention_mask"]
        delta_attention_mask = mask_bundle["delta_attention_mask"]
        wan_video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=wan_first_frame_pre["tokens"],
            video_freqs=wan_first_frame_pre["freqs"],
            video_t_mod=wan_first_frame_pre["t_mod"],
            video_context_payload={
                "context": wan_first_frame_pre["context"],
                "mask": wan_first_frame_pre["context_mask"],
            },
            video_attention_mask=mask_bundle["wan_video_mask"],
            video_kv_expert=self._wan_cache_expert(),
        )

        delta_pre = self.delta_expert.pre_dit(
            delta_tokens=noisy_delta,
            timestep=timestep_delta,
            context=context,
            context_mask=context_mask,
        )
        if int(delta_pre["tokens"].shape[1]) != int(noisy_delta.shape[1]):
            raise ValueError(
                "Delta token length changed after pre_dit unexpectedly: "
                f"pre_dit={delta_pre['tokens'].shape[1]} vs noisy={noisy_delta.shape[1]}"
            )
        delta_tokens = self.mot.forward_expert_with_video_cache(
            expert_name="video",
            expert_tokens=delta_pre["tokens"],
            expert_freqs=delta_pre["freqs"],
            expert_t_mod=delta_pre["t_mod"],
            expert_context_payload={
                "context": delta_pre["context"],
                "mask": delta_pre["context_mask"],
            },
            video_kv_cache=wan_video_kv_cache,
            attention_mask=delta_attention_mask,
            video_seq_len=wan_video_seq_len,
        )
        pred_delta = self.delta_expert.post_dit(delta_tokens, delta_pre)

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=wan_video_kv_cache,
            attention_mask=wan_attention_mask,
            video_seq_len=wan_video_seq_len,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)

        if pred_delta.shape[1] != target_delta.shape[1]:
            raise ValueError(
                "Delta prediction/target length mismatch: "
                f"pred={pred_delta.shape[1]} vs target={target_delta.shape[1]}"
            )
        delta_loss_token = F.mse_loss(pred_delta.float(), target_delta.float(), reduction="none").mean(dim=2)
        if image_is_pad is not None:
            tail_pad = image_is_pad[:, 1:]
            if tail_pad.shape[1] != delta_loss_token.shape[1]:
                raise ValueError(
                    "`image_is_pad` shape mismatch for delta loss: "
                    f"mask steps={tail_pad.shape[1]}, delta steps={delta_loss_token.shape[1]}."
                )
            valid = (~tail_pad).to(device=delta_loss_token.device, dtype=delta_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            delta_loss_per_sample = (delta_loss_token * valid).sum(dim=1) / valid_sum
        else:
            delta_loss_per_sample = delta_loss_token.mean(dim=1)
        delta_weight = self.train_delta_scheduler.training_weight(timestep_delta).to(
            delta_loss_per_sample.device, dtype=delta_loss_per_sample.dtype
        )
        loss_delta = (delta_loss_per_sample * delta_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_delta * loss_delta + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_delta": self.loss_lambda_delta * float(loss_delta.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        action_horizon: int,
        num_delta_tokens: int = 1,
        input_image: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        inference_mode: str = "action_only",
    ) -> dict[str, Any]:
        self.eval()
        del negative_prompt, text_cfg_scale, tiled
        inference_mode = str(inference_mode).strip().lower()
        if inference_mode not in {"joint", "action_only"}:
            raise ValueError(
                f"`inference_mode` must be one of ['joint', 'action_only'], got `{inference_mode}`"
            )

        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")
        if int(num_delta_tokens) <= 0:
            raise ValueError(f"`num_delta_tokens` must be positive, got {num_delta_tokens}")

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )

        if input_image is None:
            raise ValueError("`input_image` is required to build first-frame latent anchors for DeltaWAM inference.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_video = input_image.unsqueeze(2)  # [1,3,1,H,W]
        first_frame_latents = self._encode_video_latents(first_frame_video, tiled=False)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        wan_first_frame_pre = self._build_wan_first_frame_pre_state(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
        )
        wan_first_frame_tokens = wan_first_frame_pre["tokens"]
        wan_video_seq_len = int(wan_first_frame_tokens.shape[1])
        wan_tokens_per_frame = int(wan_first_frame_pre["meta"]["tokens_per_frame"])
        mask_bundle = self._build_expert_wan_attention_mask(
            wan_video_seq_len=wan_video_seq_len,
            wan_tokens_per_frame=wan_tokens_per_frame,
            device=latents_action.device,
            action_expert=self.action_expert,
            action_seq_len=int(latents_action.shape[1]),
            action_tokens_per_frame=1,
            delta_expert=self.delta_expert,
            delta_seq_len=int(num_delta_tokens),
            delta_tokens_per_frame=1,
        )
        wan_attention_mask = mask_bundle["action_attention_mask"]
        delta_attention_mask = mask_bundle["delta_attention_mask"]
        wan_video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=wan_first_frame_pre["tokens"],
            video_freqs=wan_first_frame_pre["freqs"],
            video_t_mod=wan_first_frame_pre["t_mod"],
            video_context_payload={
                "context": wan_first_frame_pre["context"],
                "mask": wan_first_frame_pre["context_mask"],
            },
            video_attention_mask=mask_bundle["wan_video_mask"],
            video_kv_expert=self._wan_cache_expert(),
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        latents_delta: Optional[torch.Tensor] = None
        if inference_mode == "joint":
            latents_delta = torch.randn(
                (1, int(num_delta_tokens), self.delta_expert.delta_token_dim),
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            infer_timesteps_delta, infer_deltas_delta = self.infer_delta_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_delta.dtype,
                shift_override=sigma_shift,
            )
            for step_t_delta, step_delta_delta, step_t_action, step_delta_action in zip(
                infer_timesteps_delta,
                infer_deltas_delta,
                infer_timesteps_action,
                infer_deltas_action,
            ):
                t_delta = step_t_delta.unsqueeze(0).to(dtype=latents_delta.dtype, device=self.device)
                t_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

                delta_pre = self.delta_expert.pre_dit(
                    delta_tokens=latents_delta,
                    timestep=t_delta,
                    context=context,
                    context_mask=context_mask,
                )
                if int(delta_pre["tokens"].shape[1]) != int(num_delta_tokens):
                    raise ValueError(
                        "Delta token length changed after pre_dit unexpectedly in joint inference: "
                        f"pre_dit={delta_pre['tokens'].shape[1]} vs requested={num_delta_tokens}"
                    )
                delta_tokens = self.mot.forward_expert_with_video_cache(
                    expert_name="video",
                    expert_tokens=delta_pre["tokens"],
                    expert_freqs=delta_pre["freqs"],
                    expert_t_mod=delta_pre["t_mod"],
                    expert_context_payload={
                        "context": delta_pre["context"],
                        "mask": delta_pre["context_mask"],
                    },
                    video_kv_cache=wan_video_kv_cache,
                    attention_mask=delta_attention_mask,
                    video_seq_len=wan_video_seq_len,
                )
                pred_delta = self.delta_expert.post_dit(delta_tokens, delta_pre)
                if pred_delta.shape[1] != latents_delta.shape[1]:
                    raise ValueError(
                        "Delta prediction length mismatch in joint inference: "
                        f"pred={pred_delta.shape[1]} vs latents={latents_delta.shape[1]}"
                    )

                action_pre = self.action_expert.pre_dit(
                    action_tokens=latents_action,
                    timestep=t_action,
                    context=context,
                    context_mask=context_mask,
                )
                action_tokens = self.mot.forward_action_with_video_cache(
                    action_tokens=action_pre["tokens"],
                    action_freqs=action_pre["freqs"],
                    action_t_mod=action_pre["t_mod"],
                    action_context_payload={
                        "context": action_pre["context"],
                        "mask": action_pre["context_mask"],
                    },
                    video_kv_cache=wan_video_kv_cache,
                    attention_mask=wan_attention_mask,
                    video_seq_len=wan_video_seq_len,
                )
                pred_action = self.action_expert.post_dit(action_tokens, action_pre)

                latents_delta = self.infer_delta_scheduler.step(pred_delta, step_delta_delta, latents_delta)
                latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        else:
            for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
                t_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                action_pre = self.action_expert.pre_dit(
                    action_tokens=latents_action,
                    timestep=t_action,
                    context=context,
                    context_mask=context_mask,
                )
                action_tokens = self.mot.forward_action_with_video_cache(
                    action_tokens=action_pre["tokens"],
                    action_freqs=action_pre["freqs"],
                    action_t_mod=action_pre["t_mod"],
                    action_context_payload={
                        "context": action_pre["context"],
                        "mask": action_pre["context_mask"],
                    },
                    video_kv_cache=wan_video_kv_cache,
                    attention_mask=wan_attention_mask,
                    video_seq_len=wan_video_seq_len,
                )
                pred_action = self.action_expert.post_dit(action_tokens, action_pre)
                latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "delta_token": latents_delta[0].detach().to(device="cpu", dtype=torch.float32)
            if inference_mode == "joint"
            else None,
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "inference_mode": inference_mode,
        }

    @torch.no_grad()
    def _decode_delta_video(
        self,
        delta_tokens: torch.Tensor,
        *,
        num_frames: int,
        mode: str,
        input_image: torch.Tensor,
        gt_video: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mode = str(mode).strip().lower()
        if mode not in {"gt_prev", "pred_prev"}:
            raise ValueError(f"Unsupported decode mode `{mode}`; expected one of ['gt_prev', 'pred_prev']")
        if mode == "gt_prev" and gt_video is None:
            raise ValueError("`gt_video` is required for decode mode `gt_prev`.")

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [B,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        if input_image.shape[0] != 1:
            raise ValueError(f"`input_image` batch size must be 1, got {input_image.shape[0]}")

        if gt_video is not None:
            if gt_video.ndim == 4:
                gt_video = gt_video.unsqueeze(0)
            if gt_video.ndim != 5 or gt_video.shape[1] != 3:
                raise ValueError(f"`gt_video` must be [B,3,T,H,W] or [3,T,H,W], got {tuple(gt_video.shape)}")
            if gt_video.shape[0] != 1:
                raise ValueError(f"`gt_video` batch size must be 1, got {gt_video.shape[0]}")
            gt_video = gt_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        delta_tokens = delta_tokens.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if delta_tokens.ndim == 2:
            delta_tokens = delta_tokens.unsqueeze(0)
        if delta_tokens.ndim != 3:
            raise ValueError(f"`delta_tokens` must be [B,T,D] or [T,D], got {tuple(delta_tokens.shape)}")
        if delta_tokens.shape[0] != 1:
            raise ValueError(f"`delta_tokens` batch size must be 1, got {delta_tokens.shape[0]}")
        if delta_tokens.shape[1] < (num_frames - 1):
            raise ValueError(
                f"`delta_tokens` length must be >= num_frames-1 ({num_frames - 1}), got {delta_tokens.shape[1]}"
            )

        bsz, _, c, h, w = 1, num_frames, 3, int(input_image.shape[2]), int(input_image.shape[3])
        decoded = torch.zeros((bsz, num_frames, c, h, w), device=self.device, dtype=self.torch_dtype)

        # Use the observed first frame as a clean anchor.
        # Delta tokens represent transitions from frame t-1 -> t, so step t uses token index t-1.
        first_frame = input_image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True).clamp(-1.0, 1.0)
        decoded[:, 0] = first_frame
        prev = first_frame
        for t in range(1, num_frames):
            z_t = delta_tokens[:, t - 1]
            pred_t = self.delta_tokenizer.decode_next(frame_t=prev, delta_token=z_t)
            if pred_t.ndim != 4 or pred_t.shape != prev.shape:
                raise ValueError(
                    "Decoded frame shape mismatch: "
                    f"expected {tuple(prev.shape)}, got {tuple(pred_t.shape)}"
                )
            decoded[:, t] = pred_t
            if mode == "gt_prev" and gt_video is not None and t + 1 < num_frames:
                prev = gt_video[:, :, t]
            else:
                prev = pred_t

        return decoded.clamp(-1.0, 1.0)

    @torch.no_grad()
    def _video_tensor_to_pil(self, video: torch.Tensor) -> list[Any]:
        if video.ndim != 5:
            raise ValueError(f"`video` must be [B,T,3,H,W], got {tuple(video.shape)}")
        if video.shape[0] != 1:
            raise ValueError(f"`video` batch size must be 1, got {video.shape[0]}")
        from PIL import Image

        frames = []
        x = video[0].detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)
        for t in range(x.shape[0]):
            arr = ((x[t] + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(arr))
        return frames

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        gt_video: Optional[torch.Tensor] = None,
        delta_decode_modes: Optional[Sequence[str]] = None,
    ):
        del negative_prompt, text_cfg_scale, action_cfg_scale, tiled
        if action_horizon is None:
            if action is None:
                raise ValueError("`action_horizon` is required when `action` is not provided.")
            action_horizon = int(action.shape[0] if action.ndim == 2 else action.shape[1])

        out = self.infer_action(
            prompt=prompt,
            action_horizon=int(action_horizon),
            num_delta_tokens=max(int(num_frames - 1), 1),
            input_image=input_image,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            inference_mode="joint",
        )
        if delta_decode_modes is None:
            decode_modes = list(self.delta_decode_eval_modes)
        else:
            decode_modes = [str(m).strip().lower() for m in delta_decode_modes]
        if len(decode_modes) == 0:
            decode_modes = [self.delta_decode_default_mode]
        if self.delta_decode_default_mode not in decode_modes:
            decode_modes.append(self.delta_decode_default_mode)

        video_modes: dict[str, list[Any]] = {}
        decode_errors: dict[str, str] = {}
        if out["delta_token"] is None:
            raise RuntimeError("DeltaWAM.infer() expects joint inference to return non-null `delta_token`.")
        for mode in decode_modes:
            try:
                decoded_tensor = self._decode_delta_video(
                    out["delta_token"],
                    num_frames=int(num_frames),
                    mode=mode,
                    input_image=input_image,
                    gt_video=gt_video,
                )
                video_modes[mode] = self._video_tensor_to_pil(decoded_tensor)
            except Exception as exc:
                decode_errors[mode] = repr(exc)

        if self.delta_decode_default_mode in video_modes:
            pred_video = video_modes[self.delta_decode_default_mode]
        elif "pred_prev" in video_modes:
            pred_video = video_modes["pred_prev"]
        elif len(video_modes) > 0:
            pred_video = next(iter(video_modes.values()))
        else:
            # Keep runtime usable even when decode path is not available.
            if input_image.ndim == 4:
                image = input_image[0]
            elif input_image.ndim == 3:
                image = input_image
            else:
                raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
            image = image.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)
            frame = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
            from PIL import Image

            pred_video = [Image.fromarray(frame) for _ in range(int(num_frames))]
            decode_errors["fallback"] = "All decode modes failed; used repeated input frame fallback."
            logger.warning(
                "Delta decode fallback triggered: all decode modes failed (%s). "
                "Using repeated input frame as evaluation video placeholder.",
                decode_errors,
            )

        return {
            "video": pred_video,
            "video_modes": video_modes,
            "action": out["action"],
            "delta_token": out["delta_token"],
            "decode_errors": decode_errors,
            "inference_mode": "joint",
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "delta_tokenizer": self.delta_tokenizer.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "wan_lora_config": self.wan_lora_config,
        }
        if self.wan_train_full:
            payload["wan_dit"] = self.wan_expert.state_dict()
        elif self.wan_lora_enabled:
            payload["wan_lora"] = get_peft_model_state_dict(self.wan_expert)
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing `mot` key: {path}")

        if "delta_tokenizer" in payload:
            self.delta_tokenizer.load_state_dict(payload["delta_tokenizer"], strict=False)
        else:
            logger.warning("Checkpoint has no `delta_tokenizer` weights; keeping current tokenizer params.")

        if self.wan_train_full:
            if "wan_dit" in payload:
                load_result = self.wan_expert.load_state_dict(payload["wan_dit"], strict=False)
                self.wan_expert.requires_grad_(True)
                logger.info(
                    "Loaded full Wan DiT weights from checkpoint: %s "
                    "(missing_keys=%d, unexpected_keys=%d)",
                    path,
                    len(load_result.missing_keys),
                    len(load_result.unexpected_keys),
                )
            else:
                logger.warning("Wan full fine-tuning is enabled but checkpoint has no `wan_dit` weights.")
        elif self.wan_lora_enabled:
            if "wan_lora" in payload:
                set_peft_model_state_dict(self.wan_expert, payload["wan_lora"])
                self._set_wan_lora_trainable()
            else:
                logger.warning("Wan LoRA is enabled but checkpoint has no `wan_lora` weights; keeping init LoRA.")
        elif "wan_dit" in payload:
            logger.warning("Checkpoint contains `wan_dit` weights but current config has Wan full fine-tuning disabled.")
        elif "wan_lora" in payload:
            logger.warning("Checkpoint contains `wan_lora` weights but current config has Wan LoRA disabled.")

        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current params.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
