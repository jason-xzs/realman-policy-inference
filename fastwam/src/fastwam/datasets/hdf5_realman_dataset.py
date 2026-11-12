"""Direct reader for the raw RealMan RMC AIDA-L HDF5 dataset."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision.io import ImageReadMode, decode_jpeg

try:
    import h5py
except ImportError as exc:  # pragma: no cover - gives a useful error in old environments
    raise ImportError("RealManHDF5VideoDataset requires h5py; run `pip install -e .`.") from exc

from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


DEFAULT_CONTROL_INDICES = [0, 1, 2, 3, 4, 5, 6, 10, 50, 51, 52, 53, 54, 55, 56, 60]


class _RealManClipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str,
        dataset_dirs: Sequence[str],
        selected_task_names: Sequence[str] | None,
        num_frames: int,
        control_indices: Sequence[int],
        val_set_proportion: float,
        is_training_set: bool,
        seed: int,
    ):
        records = []
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            raise ValueError(f"The RealMan manifest is empty: {manifest_path}")

        dataset_roots = [Path(path).expanduser() for path in dataset_dirs]
        available_task_names = set()
        records_by_task = {}
        for record in records:
            # Manifest paths are already absolute. Avoid Path.resolve() here:
            # resolving every HDF5 path would issue tens of thousands of slow
            # metadata lookups on the shared dataset filesystem.
            record_path = Path(record["path"]).expanduser()
            task_name = record.get("task_name")
            if task_name is None:
                for dataset_root in dataset_roots:
                    try:
                        task_name = record_path.relative_to(dataset_root).parts[0]
                        break
                    except (ValueError, IndexError):
                        continue
            if task_name is None:
                raise ValueError(
                    f"Cannot determine the RealMan task directory for {record_path}; "
                    f"it is not under any configured dataset_dirs: {dataset_roots}"
                )
            task_name = str(task_name)
            available_task_names.add(task_name)
            records_by_task.setdefault(task_name, []).append(record)

        selected_names = [] if selected_task_names is None else [str(name) for name in selected_task_names]
        if selected_names:
            missing_names = sorted(set(selected_names) - available_task_names)
            if missing_names:
                raise ValueError(
                    f"Unknown selected_task_names: {missing_names}. "
                    f"Available RealMan tasks: {sorted(available_task_names)}"
                )
            selected_name_set = set(selected_names)
            records = [
                record
                for task_name, task_records in records_by_task.items()
                if task_name in selected_name_set
                for record in task_records
            ]
            if not records:
                raise ValueError(f"No RealMan episodes matched selected_task_names={selected_names}")

        rng = np.random.default_rng(seed)
        order = np.arange(len(records))
        rng.shuffle(order)
        split = int(len(records) * (1.0 - val_set_proportion))
        selected = order[:split] if is_training_set else order[split:]
        self.records = [records[int(i)] for i in selected]
        self.num_frames = int(num_frames)
        self.control_indices = list(map(int, control_indices))
        self.processor = None

        lengths = [int(record["length"]) for record in self.records]
        self.episode_ends = np.cumsum(lengths, dtype=np.int64).tolist()

    def __len__(self):
        return self.episode_ends[-1] if self.episode_ends else 0

    def set_processor(self, processor):
        self.processor = processor
        return self

    @staticmethod
    def _decode(encoded) -> torch.Tensor:
        encoded_tensor = torch.from_numpy(np.asarray(encoded, dtype=np.uint8).copy())
        image = decode_jpeg(encoded_tensor, mode=ImageReadMode.RGB)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected decoded JPEG in CHW RGB format, got {tuple(image.shape)}")
        return image

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self.episode_ends, index)
        episode_start = 0 if episode_index == 0 else self.episode_ends[episode_index - 1]
        frame_start = index - episode_start
        record = self.records[episode_index]
        episode_length = int(record["length"])
        frame_stop = min(frame_start + self.num_frames, episode_length)
        valid_length = frame_stop - frame_start
        pad_length = self.num_frames - valid_length

        with h5py.File(record["path"], "r") as handle:
            action = np.asarray(handle["action"][frame_start:frame_stop, self.control_indices])
            state = np.asarray(handle["observations/qpos"][frame_start:frame_stop, self.control_indices])
            images = {}
            image_group = handle["observations/images"]
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                encoded_frames = image_group[camera][frame_start:frame_stop]
                decoded = [self._decode(encoded) for encoded in encoded_frames]
                if not decoded:
                    raise ValueError(f"Empty episode slice in {record['path']}")
                decoded.extend([decoded[-1]] * pad_length)
                images[camera] = torch.stack(decoded)

        action_tensor = torch.from_numpy(action).float()
        state_tensor = torch.from_numpy(state).float()
        if pad_length:
            action_tensor = torch.cat([action_tensor, action_tensor[-1:].repeat(pad_length, 1)])
            state_tensor = torch.cat([state_tensor, state_tensor[-1:].repeat(pad_length, 1)])

        is_pad = torch.zeros(self.num_frames, dtype=torch.bool)
        if pad_length:
            is_pad[-pad_length:] = True
        sample = {
            "idx": index,
            "task": record["task"],
            "action": {"default": action_tensor[:-1]},
            "state": {"default": state_tensor},
            "images": images,
            "action_is_pad": is_pad[:-1],
            "state_is_pad": is_pad,
            "image_is_pad": is_pad,
        }
        return self.processor.preprocess(sample) if self.processor is not None else sample


class RealManHDF5VideoDataset(RobotVideoDataset):
    """FastWAM-compatible dataset without converting or duplicating the raw HDF5 files."""

    def __init__(
        self,
        dataset_dirs,
        manifest_path,
        shape_meta,
        num_frames=33,
        video_size=(384, 320),
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.01,
        is_training_set=True,
        action_video_freq_ratio=4,
        skip_padding_as_possible=False,
        max_padding_retry=3,
        concat_multi_camera="robotwin",
        override_instruction=None,
        control_indices=DEFAULT_CONTROL_INDICES,
        selected_task_names=None,
        seed=42,
    ):
        if not pretrained_norm_stats or not Path(pretrained_norm_stats).is_file():
            raise FileNotFoundError(
                f"Missing RealMan normalization stats: {pretrained_norm_stats}. "
                "Run scripts/tasks/prepare_realman_rmc_aidal.py first."
            )
        shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        self.lerobot_dataset = _RealManClipDataset(
            manifest_path=manifest_path,
            dataset_dirs=dataset_dirs,
            selected_task_names=selected_task_names,
            num_frames=num_frames,
            control_indices=control_indices,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            seed=seed,
        )

        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError("num_frames - 1 must be divisible by action_video_freq_ratio")
        if ((self.num_frames - 1) // self.action_video_freq_ratio) % 4 != 0:
            raise ValueError("The number of video transitions must be divisible by 4")
        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))

        self.camera_key = None
        self.video_size = list(video_size)
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)
        self.skip_padding_as_possible = bool(skip_padding_as_possible)
        self.max_padding_retry = int(max_padding_retry)
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        # Reuse FastWAM's final aspect-preserving resize/crop/normalization path.
        from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.crop_transform = CenterCrop(args={"img_w": self.video_size[1], "img_h": self.video_size[0]})
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        if isinstance(processor, DictConfig):
            processor = instantiate(processor)
        stats = load_dataset_stats_from_json(pretrained_norm_stats)
        processor.set_normalizer_from_stats(stats)
        processor.train() if is_training_set else processor.eval()
        self.lerobot_dataset.set_processor(processor)
