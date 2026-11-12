"""FastWAM inference policy and a lightweight deterministic simulator."""

from __future__ import annotations

import base64
import io
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .protocol import ACTION_DIM, CAMERA_NAMES, ProtocolError, validate_infer_request

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path(
    "runs/realman_demo_organize_finetune_3cam_384_1e-5/"
    "2026-08-17_00-39-16/checkpoints/weights/step_005000.pt"
)


class MockPolicy:
    """Echoes current qpos as safe actions; used for network/full-loop simulation."""

    def __init__(self, action_horizon: int = 4, offset_rad: float = 0.001):
        self.action_horizon = int(action_horizon)
        self.offset_rad = float(offset_rad)
        self.calls = 0

    def infer(self, payload: dict[str, Any]) -> tuple[list[list[float]], dict[str, Any]]:
        request = validate_infer_request(payload)
        state = request["state"]
        # Raw data order is left(0:8), left EEF(8:14), right(14:22), right EEF(22:28).
        current = state[14:21] + [state[21] * 1000.0] + state[0:7] + [state[7] * 1000.0]
        actions = []
        for step in range(self.action_horizon):
            action = current.copy()
            action[0] += self.offset_rad * (step + 1)
            actions.append(action)
        self.calls += 1
        return actions, {"mock": True, "inference_ms": 0.0}

    def reset(self) -> None:
        self.calls = 0


class FastWAMPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        config: str | Path | None = None,
        stats: str | Path | None = None,
        device: str = "cuda",
        dtype: str = "bf16",
        action_horizon: int = 32,
        num_inference_steps: int = 10,
        sigma_shift: float | None = None,
        seed: int | None = 42,
        text_cfg_scale: float = 1.0,
        negative_prompt: str = "",
        rand_device: str = "cpu",
        tiled: bool = False,
        clip_to_training_range: bool = True,
    ) -> None:
        import numpy as np
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        self.np = np
        self.torch = torch
        self.default_prompt = DEFAULT_PROMPT
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint}")
        run_dir = self.checkpoint.parents[2]
        self.config_path = Path(config).expanduser().resolve() if config else run_dir / "config.yaml"
        self.stats_path = Path(stats).expanduser().resolve() if stats else run_dir / "dataset_stats.json"
        if not self.config_path.is_file():
            raise FileNotFoundError(f"training config not found: {self.config_path}")
        if not self.stats_path.is_file():
            raise FileNotFoundError(f"dataset stats not found: {self.stats_path}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; this 5B policy is not suitable for CPU serving")

        dtype_map = {"fp32": torch.float32, "float32": torch.float32,
                     "fp16": torch.float16, "float16": torch.float16,
                     "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
        if dtype.lower() not in dtype_map:
            raise ValueError(f"unsupported dtype {dtype!r}; choose fp32, fp16, or bf16")
        self.device = device
        self.model_dtype = dtype_map[dtype.lower()]

        cfg = OmegaConf.load(self.config_path)
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        # Training config stores this path relative to the project root.
        backbone = Path(str(model_cfg.action_dit_pretrained_path))
        if not backbone.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            model_cfg.action_dit_pretrained_path = str((project_root / backbone).resolve())
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(Path(__file__).resolve().parents[2] / "checkpoints"))

        logger.info("Loading FastWAM model from %s", self.checkpoint)
        self.model = instantiate(model_cfg, model_dtype=self.model_dtype, device=device)
        self.model.load_checkpoint(str(self.checkpoint))
        self.model = self.model.to(device).eval()
        self.processor = instantiate(cfg.data.train.processor).eval()
        dataset_stats = load_dataset_stats_from_json(str(self.stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)
        action_stats = json.loads(self.stats_path.read_text(encoding="utf-8"))["action"]["default"]
        self.action_min = np.asarray(action_stats["global_min"], dtype=np.float32)
        self.action_max = np.asarray(action_stats["global_max"], dtype=np.float32)

        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = negative_prompt
        self.rand_device = rand_device
        self.tiled = bool(tiled)
        self.clip_to_training_range = bool(clip_to_training_range)
        self.num_video_frames = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1

    def _decode_rgb(self, encoded: str):
        from PIL import Image

        try:
            raw = base64.b64decode(encoded, validate=True)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            image.load()
        except Exception as exc:
            raise ProtocolError(f"invalid JPEG image: {exc}") from exc
        return self.np.asarray(image, dtype=self.np.uint8)

    def _build_image_tensor(self, images: dict[str, str]):
        from PIL import Image

        def resize(array, size):
            return self.np.asarray(Image.fromarray(array, mode="RGB").resize(size, Image.Resampling.BILINEAR))

        high = resize(self._decode_rgb(images[CAMERA_NAMES[0]]), (320, 256))
        left = resize(self._decode_rgb(images[CAMERA_NAMES[1]]), (160, 128))
        right = resize(self._decode_rgb(images[CAMERA_NAMES[2]]), (160, 128))
        composite = self.np.concatenate([high, self.np.concatenate([left, right], axis=1)], axis=0)
        tensor = self.torch.from_numpy(composite.copy()).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.model_dtype)
        return tensor * (2.0 / 255.0) - 1.0

    def _normalize_state(self, raw_state: list[float]):
        key = self.processor.shape_meta["state"][0]["key"]
        batch = {"state": {key: self.torch.tensor(raw_state, dtype=self.torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][key]

    def _denormalize_action(self, action):
        if action.ndim == 2:
            action = action.unsqueeze(0)
        key = self.processor.shape_meta["action"][0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        return normalizer.backward(action.to(dtype=self.torch.float32, device="cpu")).numpy()[0]

    def infer(self, payload: dict[str, Any]) -> tuple[list[list[float]], dict[str, Any]]:
        request = validate_infer_request(payload)
        image = self._build_image_tensor(request["images"])
        proprio = self._normalize_state(request["state"])
        kwargs = {
            "prompt": self.default_prompt.format(task=request["instruction"]),
            "input_image": image,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            kwargs["num_video_frames"] = self.num_video_frames
        started = time.perf_counter()
        with self.torch.inference_mode():
            prediction = self.model.infer_action(**kwargs)
        actions = self._denormalize_action(prediction["action"])
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not self.np.isfinite(actions).all():
            raise RuntimeError(f"model returned invalid action shape/data: {actions.shape}")
        clipped = 0
        if self.clip_to_training_range:
            before = actions.copy()
            actions = self.np.clip(actions, self.action_min, self.action_max)
            clipped = int(self.np.count_nonzero(before != actions))
        return actions.tolist(), {
            "mock": False,
            "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "clipped_values": clipped,
        }

    def reset(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
