from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from peft import LoraConfig, get_peft_model

from fastwam.utils.logging_config import get_logger

from .delta_dit import ActionDeltaDiT
from .delta_tokenizer import BaseDeltaTokenizer, ConvDeltaTokenizer, DeltaTokEncoderTokenizer, LAMDeltaTokenizer
from .deltawam import DeltaWAM
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import WanVideoDiT

logger = get_logger(__name__)


class DeltaWAMJoint(DeltaWAM):
    """DeltaWAM variant where action and delta are denoised by one shared DiT."""

    def __init__(
        self,
        joint_expert: ActionDeltaDiT,
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
        joint_timestep_mode: str = "independent",
        action_inference_mode: str = "action_only",
        joint_attention_mask_mode: str = "isolated",
        delta_decode_default_mode: str = "gt_prev",
        delta_decode_eval_modes: Optional[Sequence[str]] = None,
        wan_expert: Optional[WanVideoDiT] = None,
        wan_lora_config: Optional[dict[str, Any]] = None,
    ):
        nn.Module.__init__(self)
        if wan_expert is None:
            raise ValueError("`wan_expert` is required for DeltaWAMJoint.")

        self.mot = mot
        # Aliases keep inherited helpers compatible without registering the same module
        # outside the MoT container a second time.
        object.__setattr__(self, "joint_expert", joint_expert)
        object.__setattr__(self, "delta_expert", joint_expert)
        object.__setattr__(self, "action_expert", joint_expert)
        self.delta_tokenizer = delta_tokenizer
        self.delta_tokenizer.freeze_encoder()
        self.delta_tokenizer.requires_grad_(False)

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

        # Trainer compatibility: optimizer/freeze logic expects `model.dit`.
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
        self.joint_timestep_mode = str(joint_timestep_mode).strip().lower()
        if self.joint_timestep_mode not in {"independent", "shared"}:
            raise ValueError(
                "`joint_timestep_mode` must be one of ['independent', 'shared'], "
                f"got `{self.joint_timestep_mode}`"
            )
        self.action_inference_mode = str(action_inference_mode).strip().lower()
        if self.action_inference_mode not in {"action_only", "joint"}:
            raise ValueError(
                "`action_inference_mode` must be one of ['action_only', 'joint'], "
                f"got `{self.action_inference_mode}`"
            )
        self.joint_attention_mask_mode = str(joint_attention_mask_mode).strip().lower()
        if self.joint_attention_mask_mode not in {"isolated", "delta_attends_action_train"}:
            raise ValueError(
                "`joint_attention_mask_mode` must be one of ['isolated', 'delta_attends_action_train'], "
                f"got `{self.joint_attention_mask_mode}`"
            )
        logger.info("Using joint_attention_mask_mode=%s", self.joint_attention_mask_mode)
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
        action_delta_dit_config: dict[str, Any] | None = None,
        delta_tokenizer_config: dict[str, Any] | None = None,
        action_delta_dit_pretrained_path: str | None = None,
        delta_tokenizer_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
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
        joint_timestep_mode: str = "independent",
        action_inference_mode: str = "action_only",
        joint_attention_mask_mode: str = "isolated",
        delta_decode_default_mode: str = "pred_prev",
        delta_decode_eval_modes: Optional[Sequence[str]] = None,
        wan_dit_config: Optional[dict[str, Any]] = None,
        wan_lora_config: Optional[dict[str, Any]] = None,
    ):
        if action_delta_dit_config is None:
            raise ValueError("`action_delta_dit_config` is required for DeltaWAMJoint.from_pretrained().")
        if delta_tokenizer_config is None:
            delta_tokenizer_config = {}
        if wan_dit_config is None:
            raise ValueError("`wan_dit_config` is required for DeltaWAMJoint.from_pretrained().")

        action_delta_dit_config = dict(action_delta_dit_config)
        delta_tokenizer_config = dict(delta_tokenizer_config)
        wan_dit_config = dict(wan_dit_config)
        wan_lora_config = dict(wan_lora_config or {})

        tokenizer_impl = str(delta_tokenizer_config.get("impl", "conv")).strip().lower()
        tokenizer_token_dim = int(action_delta_dit_config["delta_token_dim"])
        tokenizer_backbone = None
        if tokenizer_impl == "deltatok":
            backbone = delta_tokenizer_config.get("backbone", None)
            if backbone is None:
                backbone = delta_tokenizer_config.get("backbone_cfg", None)
            if backbone is not None:
                tokenizer_backbone = backbone if isinstance(backbone, nn.Module) else instantiate(backbone, device=device)
            if tokenizer_backbone is None:
                raise ValueError("`delta_tokenizer_config.impl=deltatok` requires `backbone` or `backbone_cfg`.")
            inferred_token_dim = getattr(tokenizer_backbone, "hidden_size", None)
            if inferred_token_dim is not None and inferred_token_dim != tokenizer_token_dim:
                raise ValueError(
                    "delta_token_dim mismatch between config and DeltaTok backbone: "
                    f"config={tokenizer_token_dim}, backbone={inferred_token_dim}."
                )
        elif tokenizer_impl == "lam":
            lam_latent_dim = int(delta_tokenizer_config.get("lam_latent_dim", 32))
            if lam_latent_dim != tokenizer_token_dim:
                raise ValueError(
                    "delta_token_dim mismatch between config and LAM latent dim: "
                    f"config={tokenizer_token_dim}, lam_latent_dim={lam_latent_dim}."
                )

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

        joint_expert = ActionDeltaDiT.from_pretrained(
            action_delta_dit_config=action_delta_dit_config,
            action_delta_dit_pretrained_path=action_delta_dit_pretrained_path,
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

        # MoT still provides the Wan per-layer K/V interaction. Both slots point to
        # the same shared expert; only the `action` slot is used for joint tokens.
        mot = MoT(
            mixtures={"video": joint_expert, "action": joint_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            joint_expert=joint_expert,
            mot=mot,
            delta_tokenizer=delta_tokenizer,
            wan_expert=components.dit,
            vae=components.vae,
            text_encoder=(components.text_encoder if components is not None else None),
            tokenizer=(components.tokenizer if components is not None else None),
            text_dim=int(action_delta_dit_config["text_dim"]),
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
            joint_timestep_mode=joint_timestep_mode,
            action_inference_mode=action_inference_mode,
            joint_attention_mask_mode=joint_attention_mask_mode,
            delta_decode_default_mode=delta_decode_default_mode,
            delta_decode_eval_modes=delta_decode_eval_modes,
            wan_lora_config=wan_lora_config,
        )
        model.model_paths = {
            "text_encoder": (components.text_encoder_path if components is not None else None),
            "tokenizer": (components.tokenizer_path if components is not None else None),
            "vae": (components.vae_path if components is not None else None),
            "shared_action_delta_dit_backbone": (
                None if skip_dit_load_from_pretrain else action_delta_dit_pretrained_path
            ),
            "delta_tokenizer": delta_tokenizer_pretrained_path,
            "wan_dit": components.dit_path,
        }
        if model.wan_lora_enabled:
            model.model_paths["wan_lora"] = "peft"
        return model

    @torch.no_grad()
    def _build_joint_wan_attention_mask(
        self,
        *,
        wan_video_seq_len: int,
        wan_tokens_per_frame: int,
        delta_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        joint_attention_mask_mode: str = "isolated",
    ) -> dict[str, torch.Tensor]:
        if wan_video_seq_len <= 0:
            raise ValueError(f"`wan_video_seq_len` must be positive, got {wan_video_seq_len}")
        if delta_seq_len < 0:
            raise ValueError(f"`delta_seq_len` must be non-negative, got {delta_seq_len}")
        if action_seq_len < 0:
            raise ValueError(f"`action_seq_len` must be non-negative, got {action_seq_len}")
        joint_seq_len = int(delta_seq_len) + int(action_seq_len)
        if joint_seq_len <= 0:
            raise ValueError(f"`joint_seq_len` must be positive, got {joint_seq_len}")

        wan_tokens_per_frame = max(int(wan_tokens_per_frame), 1)
        wan_video_mask = self._wan_cache_expert().build_video_to_video_mask(
            video_seq_len=int(wan_video_seq_len),
            video_tokens_per_frame=wan_tokens_per_frame,
            device=device,
        )
        visible_wan_tokens = min(wan_tokens_per_frame, wan_video_seq_len)
        total_seq_len = int(wan_video_seq_len) + int(joint_seq_len)
        joint_attention_mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        joint_attention_mask[:wan_video_seq_len, :wan_video_seq_len] = wan_video_mask

        delta_start = wan_video_seq_len
        delta_end = delta_start + int(delta_seq_len)
        action_start = delta_end
        action_end = action_start + int(action_seq_len)
        if delta_seq_len > 0:
            joint_attention_mask[delta_start:delta_end, delta_start:delta_end] = True
        if action_seq_len > 0:
            joint_attention_mask[action_start:action_end, action_start:action_end] = True
        joint_attention_mask_mode = str(joint_attention_mask_mode).strip().lower()
        if joint_attention_mask_mode == "delta_attends_action_train":
            if delta_seq_len > 0 and action_seq_len > 0:
                joint_attention_mask[delta_start:delta_end, action_start:action_end] = True
        elif joint_attention_mask_mode != "isolated":
            raise ValueError(
                "`joint_attention_mask_mode` must be one of ['isolated', 'delta_attends_action_train'], "
                f"got `{joint_attention_mask_mode}`"
            )
        joint_attention_mask[wan_video_seq_len:, :visible_wan_tokens] = True
        return {
            "wan_video_mask": wan_video_mask,
            "joint_attention_mask": joint_attention_mask,
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
        if self.joint_timestep_mode == "shared":
            sigma_delta = timestep_delta.to(device=self.device, dtype=torch.float32) / float(
                self.train_delta_scheduler.num_train_timesteps
            )
            timestep_action = (
                sigma_delta * float(self.train_action_scheduler.num_train_timesteps)
            ).to(dtype=action.dtype)
        else:
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
        mask_bundle = self._build_joint_wan_attention_mask(
            wan_video_seq_len=wan_video_seq_len,
            wan_tokens_per_frame=wan_tokens_per_frame,
            delta_seq_len=int(noisy_delta.shape[1]),
            action_seq_len=int(noisy_action.shape[1]),
            device=action.device,
            joint_attention_mask_mode=self.joint_attention_mask_mode,
        )
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

        joint_pre = self.joint_expert.pre_dit(
            delta_tokens=noisy_delta,
            action_tokens=noisy_action,
            timestep_delta=timestep_delta,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        joint_tokens = self.mot.forward_expert_with_video_cache(
            expert_name="action",
            expert_tokens=joint_pre["tokens"],
            expert_freqs=joint_pre["freqs"],
            expert_t_mod=joint_pre["t_mod"],
            expert_context_payload={
                "context": joint_pre["context"],
                "mask": joint_pre["context_mask"],
            },
            video_kv_cache=wan_video_kv_cache,
            attention_mask=mask_bundle["joint_attention_mask"],
            video_seq_len=wan_video_seq_len,
            adapter_context=joint_pre["meta"],
        )
        pred_delta, pred_action = self.joint_expert.post_dit(joint_tokens, joint_pre)
        if pred_delta is None or pred_action is None:
            raise RuntimeError("DeltaWAMJoint must return both delta and action predictions during training.")

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
            delta_loss_per_sample.device,
            dtype=delta_loss_per_sample.dtype,
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
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_delta * loss_delta + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_delta": self.loss_lambda_delta * float(loss_delta.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        action_horizon: int,
        num_delta_tokens: Optional[int] = None,
        num_video_frames: Optional[int] = None,
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
        inference_mode: Optional[str] = None,
    ) -> dict[str, Any]:
        self.eval()
        del negative_prompt, text_cfg_scale, tiled
        if inference_mode is None:
            inference_mode = self.action_inference_mode
        inference_mode = str(inference_mode).strip().lower()
        if inference_mode not in {"joint", "action_only"}:
            raise ValueError(f"`inference_mode` must be one of ['joint', 'action_only'], got `{inference_mode}`")
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")
        if num_video_frames is not None:
            num_video_frames = int(num_video_frames)
            if num_video_frames <= 1:
                raise ValueError(f"`num_video_frames` must be > 1, got {num_video_frames}")
            num_delta_tokens = num_video_frames - 1
        elif num_delta_tokens is None:
            if inference_mode == "joint":
                raise ValueError(
                    "`num_video_frames` or `num_delta_tokens` is required when inference_mode='joint'."
                )
            num_delta_tokens = 0
        else:
            num_delta_tokens = int(num_delta_tokens)
            if num_delta_tokens <= 0:
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
            raise ValueError("`input_image` is required to build first-frame latent anchors for DeltaWAMJoint inference.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_video_latents(input_image.unsqueeze(2), tiled=False)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        latents_delta: Optional[torch.Tensor] = None
        if inference_mode == "joint":
            latents_delta = torch.randn(
                (1, int(num_delta_tokens), self.delta_expert.delta_token_dim),
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)

        wan_first_frame_pre = self._build_wan_first_frame_pre_state(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
        )
        wan_video_seq_len = int(wan_first_frame_pre["tokens"].shape[1])
        wan_tokens_per_frame = int(wan_first_frame_pre["meta"]["tokens_per_frame"])
        mask_bundle = self._build_joint_wan_attention_mask(
            wan_video_seq_len=wan_video_seq_len,
            wan_tokens_per_frame=wan_tokens_per_frame,
            delta_seq_len=0 if latents_delta is None else int(latents_delta.shape[1]),
            action_seq_len=int(latents_action.shape[1]),
            device=latents_action.device,
            joint_attention_mask_mode=self.joint_attention_mask_mode,
        )
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
        if inference_mode == "joint":
            assert latents_delta is not None
            infer_timesteps_delta, infer_deltas_delta = self.infer_delta_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_delta.dtype,
                shift_override=sigma_shift,
            )
            step_iter = zip(infer_timesteps_delta, infer_deltas_delta, infer_timesteps_action, infer_deltas_action)
            for step_t_delta, step_delta_delta, step_t_action, step_delta_action in step_iter:
                t_delta = step_t_delta.unsqueeze(0).to(dtype=latents_delta.dtype, device=self.device)
                t_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                joint_pre = self.joint_expert.pre_dit(
                    delta_tokens=latents_delta,
                    action_tokens=latents_action,
                    timestep_delta=t_delta,
                    timestep_action=t_action,
                    context=context,
                    context_mask=context_mask,
                )
                joint_tokens = self.mot.forward_expert_with_video_cache(
                    expert_name="action",
                    expert_tokens=joint_pre["tokens"],
                    expert_freqs=joint_pre["freqs"],
                    expert_t_mod=joint_pre["t_mod"],
                    expert_context_payload={
                        "context": joint_pre["context"],
                        "mask": joint_pre["context_mask"],
                    },
                    video_kv_cache=wan_video_kv_cache,
                    attention_mask=mask_bundle["joint_attention_mask"],
                    video_seq_len=wan_video_seq_len,
                    adapter_context=joint_pre["meta"],
                )
                pred_delta, pred_action = self.joint_expert.post_dit(joint_tokens, joint_pre)
                if pred_delta is None or pred_action is None:
                    raise RuntimeError("Joint inference must return both delta and action predictions.")
                latents_delta = self.infer_delta_scheduler.step(pred_delta, step_delta_delta, latents_delta)
                latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        else:
            for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
                t_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                joint_pre = self.joint_expert.pre_dit(
                    delta_tokens=None,
                    action_tokens=latents_action,
                    timestep_delta=None,
                    timestep_action=t_action,
                    context=context,
                    context_mask=context_mask,
                )
                joint_tokens = self.mot.forward_expert_with_video_cache(
                    expert_name="action",
                    expert_tokens=joint_pre["tokens"],
                    expert_freqs=joint_pre["freqs"],
                    expert_t_mod=joint_pre["t_mod"],
                    expert_context_payload={
                        "context": joint_pre["context"],
                        "mask": joint_pre["context_mask"],
                    },
                    video_kv_cache=wan_video_kv_cache,
                    attention_mask=mask_bundle["joint_attention_mask"],
                    video_seq_len=wan_video_seq_len,
                    adapter_context=joint_pre["meta"],
                )
                _, pred_action = self.joint_expert.post_dit(joint_tokens, joint_pre)
                if pred_action is None:
                    raise RuntimeError("Action-only inference did not return an action prediction.")
                latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "delta_token": latents_delta[0].detach().to(device="cpu", dtype=torch.float32)
            if inference_mode == "joint" and latents_delta is not None
            else None,
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "inference_mode": inference_mode,
        }
