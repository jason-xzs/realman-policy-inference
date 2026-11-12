"""Reusable robot-side π0.5 inference loop for the RealMan dual-arm setup.

The active deployment path is independent of historical RR-DEP run IDs,
artifact directories, and Compute finalization. A named preset supplies the
camera, robot, policy, action, synchronization, and motion settings.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib
import json
import logging
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
import tomllib
from typing import Any, Mapping, Sequence

from .rr_dep_003 import RealSenseRig, RobotStatePoller
from .rr_dep_003_motion import MotionRejected
from .rr_dep_004 import evaluate_robot_gate_004
from .rr_dep_005 import latest_camera_observation
from .rr_dep_006 import RR006Backend, ensure_loopback_no_proxy


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "pi05_realman_presets.toml"
DEFAULT_PRESET = "pi05-sugar-15000"


class RuntimeEvents:
    """Thread-safe counters only; frames, state trajectories, and actions are not retained."""

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self._lock = threading.Lock()

    def emit(self, name: str, event: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self.counters[f"{name}.{event}"] += 1
        if name == "safety":
            LOGGER.warning("safety event %s: %s", event, payload.get("error", payload))


class UserExit(Exception):
    """Ctrl+C requested a normal, cleaned-up exit."""


class PresetBackend(RR006Backend):
    """RR006 sensing/cleanup backend with the SDK's full 1..100 speed range."""

    def __init__(
        self,
        config: dict[str, Any],
        events: RuntimeEvents,
        *,
        retries: int,
        backoff_sec: float,
    ) -> None:
        super().__init__(config, events, retries=retries, backoff_sec=backoff_sec)
        self.sdk_global_lock = threading.Lock()

    def _acquire_global_sdk_lock(self, operation: str) -> None:
        timeout = float(self.config["runtime"]["sdk_lock_timeout_sec"])
        if not self.sdk_global_lock.acquire(timeout=timeout):
            raise MotionRejected(f"robot_sdk_global_lock_timeout:{operation}:{timeout}")

    def _acquire_side_sdk_lock(self, side: str, operation: str) -> None:
        timeout = float(self.config["runtime"]["sdk_lock_timeout_sec"])
        if not self.locks[side].acquire(timeout=timeout):
            raise MotionRejected(f"robot_sdk_side_lock_timeout:{operation}:{side}:{timeout}")

    def _call(self, side: str, method: str) -> Any:
        self._acquire_global_sdk_lock(f"read:{method}:{side}")
        try:
            return super()._call(side, method)
        finally:
            self.sdk_global_lock.release()

    def command_absolute(self, side: str, target_rad: Sequence[float], speed_percent: int) -> int:
        if not 1 <= int(speed_percent) <= 100:
            raise MotionRejected(f"invalid_speed_percent:{speed_percent}")
        values = tuple(float(value) for value in target_rad)
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise MotionRejected(f"invalid_joint_target:{side}")
        self.motion_started = True
        target_deg = [math.degrees(value) for value in values]
        self._acquire_global_sdk_lock(f"joint_command:{side}")
        try:
            self._acquire_side_sdk_lock(side, "joint_command")
            try:
                motion = self.config["motion"]
                return int(self.arms[side].rm_movej_canfd(
                    target_deg,
                    bool(motion["follow"]),
                    trajectory_mode=int(motion["trajectory_mode"]),
                    radio=int(motion["smoothing"]),
                ))
            finally:
                self.locks[side].release()
        finally:
            self.sdk_global_lock.release()

    def command_dual_absolute(
        self,
        targets_rad: Mapping[str, Sequence[float]],
        speed_percent: int,
    ) -> dict[str, int]:
        if not 1 <= int(speed_percent) <= 100:
            raise MotionRejected(f"invalid_speed_percent:{speed_percent}")
        if set(targets_rad) != {"left", "right"}:
            raise MotionRejected(f"dual_joint_targets_invalid:{sorted(targets_rad)}")
        order = tuple(str(side) for side in self.config["action"]["command_order"])
        targets_deg = {}
        for side in ("left", "right"):
            values = tuple(float(value) for value in targets_rad[side])
            if len(values) != 7 or not all(math.isfinite(value) for value in values):
                raise MotionRejected(f"invalid_joint_target:{side}")
            targets_deg[side] = [math.degrees(value) for value in values]

        self._acquire_global_sdk_lock("dual_joint_command")
        acquired_sides = []
        try:
            for side in order:
                self._acquire_side_sdk_lock(side, "dual_joint_command")
                acquired_sides.append(side)
            self.motion_started = True
            motion = self.config["motion"]
            statuses = {}
            for side in order:
                statuses[side] = int(self.arms[side].rm_movej_canfd(
                    targets_deg[side],
                    bool(motion["follow"]),
                    trajectory_mode=int(motion["trajectory_mode"]),
                    radio=int(motion["smoothing"]),
                ))
                if statuses[side] != 0:
                    break
            return statuses
        finally:
            for side in reversed(acquired_sides):
                self.locks[side].release()
            self.sdk_global_lock.release()

    def command_gripper(self, side: str, position: int) -> int:
        if side not in ("left", "right") or not 1 <= int(position) <= 1000:
            raise MotionRejected(f"invalid_gripper_target:{side}:{position}")
        self._acquire_global_sdk_lock(f"gripper_command:{side}")
        try:
            self._acquire_side_sdk_lock(side, "gripper_command")
            try:
                mode_status = int(self.arms[side].rm_set_tool_rs485_mode(2, 115200))
                if mode_status != 0:
                    raise MotionRejected(f"gripper_protocol_mode_rejected:{side}:{mode_status}")
                self.motion_started = True
                return int(self.arms[side].rm_set_gripper_position(int(position), False, 0))
            finally:
                self.locks[side].release()
        finally:
            self.sdk_global_lock.release()

    def slow_stop(self, side: str) -> int:
        self._acquire_global_sdk_lock(f"slow_stop:{side}")
        try:
            self._acquire_side_sdk_lock(side, "slow_stop")
            try:
                status = int(self.arms[side].rm_set_arm_slow_stop())
            finally:
                self.locks[side].release()
        finally:
            self.sdk_global_lock.release()
        self.events.emit("safety", "slow_stop", {"side": side, "sdk_status": status})
        return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run π0.5 inference on the RealMan dual-arm client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preset", default=DEFAULT_PRESET, help="named preset inside the config registry")
    return parser


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise MotionRejected(f"config_section_missing:{key}")
    return value


def _merge_mappings(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge a preset over the registry defaults."""
    merged = copy.deepcopy(dict(defaults))
    for key, value in overrides.items():
        inherited = merged.get(key)
        if isinstance(inherited, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(inherited, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path, preset: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("rb") as stream:
        registry = tomllib.load(stream)
    if registry.get("schema_version") != 1:
        raise MotionRejected(f"preset_registry_schema_unsupported:{registry.get('schema_version')!r}")
    presets = _require_mapping(registry, "presets")
    profile = presets.get(preset)
    if not isinstance(profile, Mapping):
        available = ",".join(sorted(str(name) for name in presets))
        raise MotionRejected(f"preset_not_found:{preset}:available={available}")
    defaults = registry.get("default", {})
    if not isinstance(defaults, Mapping):
        raise MotionRejected("config_section_invalid:default")
    config = _merge_mappings(defaults, profile)
    cameras = _require_mapping(config, "cameras")
    robots = _require_mapping(config, "robots")
    policy = _require_mapping(config, "policy")
    execution = _require_mapping(config, "execution")
    initial_pose = _require_mapping(config, "initial_pose")
    observation = _require_mapping(config, "observation")
    action = _require_mapping(config, "action")
    runtime = _require_mapping(config, "runtime")
    motion = _require_mapping(config, "motion")
    gripper = _require_mapping(config, "gripper")
    recording = _require_mapping(config, "recording")
    sources = _require_mapping(config, "sources")
    left = _require_mapping(robots, "left")
    right = _require_mapping(robots, "right")

    serials = [str(cameras.get(role, "")) for role in ("head", "left_wrist", "right_wrist")]
    if any(not value for value in serials) or len(set(serials)) != 3:
        raise MotionRejected("three_distinct_camera_serials_required")
    if (
        int(cameras.get("width", 0)),
        int(cameras.get("height", 0)),
        int(cameras.get("fps", 0)),
        str(cameras.get("format", "")),
    ) != (640, 480, 30, "bgr8"):
        raise MotionRejected("camera_profile_must_be_640x480_30_bgr8")
    if not str(policy.get("host", "")):
        raise MotionRejected("policy_host_required")
    if not 1 <= int(policy.get("port", 0)) <= 65535:
        raise MotionRejected("policy_port_invalid")
    if not str(policy.get("prompt", "")):
        raise MotionRejected("policy_prompt_required")
    request_timeout_sec = float(policy.get("request_timeout_sec", math.nan))
    if not math.isfinite(request_timeout_sec) or request_timeout_sec <= 0.0:
        raise MotionRejected(f"policy_request_timeout_invalid:{request_timeout_sec}")
    if execution.get("mode") not in ("execute", "no_motion"):
        raise MotionRejected(f"execution_mode_invalid:{execution.get('mode')!r}")
    max_replans = execution.get("max_replans")
    if isinstance(max_replans, bool) or not isinstance(max_replans, int) or max_replans <= 0:
        raise MotionRejected(f"max_replans_invalid:{max_replans!r}")
    if execution.get("confirmation") != "enter":
        raise MotionRejected(f"confirmation_mode_invalid:{execution.get('confirmation')!r}")
    if execution.get("log_level") not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise MotionRejected(f"log_level_invalid:{execution.get('log_level')!r}")
    if initial_pose.get("command_api") != "rm_movej":
        raise MotionRejected(
            f"initial_pose_command_api_unsupported:{initial_pose.get('command_api')!r}"
        )
    if initial_pose.get("joint_order") != "q1_q7":
        raise MotionRejected("initial_pose_joint_order_unsupported")
    for side in ("left", "right"):
        joints = initial_pose.get(f"{side}_joint_deg")
        if (
            not isinstance(joints, list)
            or len(joints) != 7
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in joints
            )
        ):
            raise MotionRejected(f"initial_pose_joints_invalid:{side}:{joints!r}")
    for side in ("left", "right"):
        initial_speed = initial_pose.get(f"{side}_speed_percent")
        if (
            isinstance(initial_speed, bool)
            or not isinstance(initial_speed, int)
            or not 1 <= initial_speed <= 100
        ):
            raise MotionRejected(
                f"initial_pose_speed_percent_invalid:{side}:{initial_speed!r}"
            )
    initial_gripper_position = initial_pose.get("gripper_open_position")
    if (
        isinstance(initial_gripper_position, bool)
        or not isinstance(initial_gripper_position, int)
        or not 1 <= initial_gripper_position <= 1000
    ):
        raise MotionRejected(
            f"initial_pose_gripper_open_position_invalid:{initial_gripper_position!r}"
        )
    initial_gripper_tolerance = initial_pose.get("gripper_tolerance")
    if (
        isinstance(initial_gripper_tolerance, bool)
        or not isinstance(initial_gripper_tolerance, int)
        or not 0 <= initial_gripper_tolerance < 1000
    ):
        raise MotionRejected(
            f"initial_pose_gripper_tolerance_invalid:{initial_gripper_tolerance!r}"
        )
    initial_positive_values = {
        field: float(initial_pose.get(field, math.nan))
        for field in (
            "joint_tolerance_rad",
            "settle_timeout_sec",
            "stable_window_sec",
            "stable_span_threshold_rad",
            "poll_hz",
        )
    }
    for field, value in initial_positive_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise MotionRejected(f"initial_pose_value_must_be_positive:{field}:{value}")
    if initial_positive_values["stable_window_sec"] >= initial_positive_values["settle_timeout_sec"]:
        raise MotionRejected("initial_pose_stable_window_must_be_less_than_timeout")
    if int(observation.get("captured_state_dimension", 0)) != 28:
        raise MotionRejected("captured_state_dimension_must_be_28")
    if observation.get("captured_state_order") != "left_joints_gripper_eef_then_right_joints_gripper_eef":
        raise MotionRejected("captured_state_order_unsupported")
    policy_state_dimension = observation.get("state_dimension")
    if (
        isinstance(policy_state_dimension, bool)
        or not isinstance(policy_state_dimension, int)
        or policy_state_dimension not in (16, 28)
    ):
        raise MotionRejected(f"policy_state_dimension_invalid:{policy_state_dimension!r}")
    expected_policy_state_order = {
        16: "left_q1_q7_gripper_then_right_q1_q7_gripper",
        28: "left_joints_gripper_eef_then_right_joints_gripper_eef",
    }[policy_state_dimension]
    if observation.get("state_order") != expected_policy_state_order:
        raise MotionRejected("policy_state_order_unsupported")
    state_mapping = observation.get("mapping_28_to_16")
    if (
        not isinstance(state_mapping, list)
        or len(state_mapping) != 16
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 28
            for value in state_mapping
        )
        or len(set(state_mapping)) != 16
    ):
        raise MotionRejected(f"observation_state_mapping_invalid:{state_mapping!r}")
    if int(observation.get("model_image_size", 0)) != 224:
        raise MotionRejected("model_image_size_must_be_224")

    if int(action.get("canonical_dimension", 0)) != 16:
        raise MotionRejected("canonical_action_dimension_must_be_16")
    if action.get("canonical_order") != "left_q1_q7_gripper_then_right_q1_q7_gripper":
        raise MotionRejected("canonical_action_order_unsupported")
    if action.get("target_mode") != "direct_absolute":
        raise MotionRejected("action_target_mode_unsupported")
    allowed_dimensions = action.get("allowed_source_dimensions")
    if (
        not isinstance(allowed_dimensions, list)
        or not allowed_dimensions
        or any(isinstance(value, bool) or not isinstance(value, int) for value in allowed_dimensions)
        or not set(allowed_dimensions).issubset({16, 28})
    ):
        raise MotionRejected(f"allowed_action_dimensions_invalid:{allowed_dimensions!r}")
    mapping = action.get("mapping_28_to_16")
    if (
        not isinstance(mapping, list)
        or len(mapping) != 16
        or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 28 for value in mapping)
        or len(set(mapping)) != 16
    ):
        raise MotionRejected(f"action_mapping_invalid:{mapping!r}")
    row_index = action.get("row_index")
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise MotionRejected(f"action_row_index_invalid:{row_index!r}")
    actions_per_chunk = action.get("actions_per_chunk")
    if (
        isinstance(actions_per_chunk, bool)
        or not isinstance(actions_per_chunk, int)
        or actions_per_chunk <= 0
    ):
        raise MotionRejected(f"actions_per_chunk_invalid:{actions_per_chunk!r}")
    if row_index + actions_per_chunk > 50:
        raise MotionRejected(
            f"configured_action_window_exceeds_horizon:{row_index}:{actions_per_chunk}:50"
        )
    action_hz = float(action.get("action_hz", math.nan))
    if not math.isfinite(action_hz) or not 0.0 < action_hz <= 200.0:
        raise MotionRejected(f"action_hz_invalid:{action_hz}")
    if action.get("required_dtype") != "float32" or action.get("require_finite") is not True:
        raise MotionRejected("action_value_contract_must_be_float32_finite")
    command_order = action.get("command_order")
    if not isinstance(command_order, list) or sorted(command_order) != ["left", "right"]:
        raise MotionRejected(f"command_order_invalid:{command_order!r}")

    positive_values = {
        "state_stale_sec": float(runtime.get("state_stale_sec", math.nan)),
        "max_sync_skew_sec": float(runtime.get("max_sync_skew_sec", math.nan)),
        "response_to_command_sec": float(runtime.get("response_to_command_sec", math.nan)),
        "watchdog_hz": float(runtime.get("watchdog_hz", math.nan)),
        "sdk_lock_timeout_sec": float(runtime.get("sdk_lock_timeout_sec", math.nan)),
        "sync_timeout_sec": float(runtime.get("sync_timeout_sec", math.nan)),
        "startup_timeout_sec": float(runtime.get("startup_timeout_sec", math.nan)),
        "action_call_timeout_sec": float(runtime.get("action_call_timeout_sec", math.nan)),
        "chunk_settle_timeout_sec": float(runtime.get("chunk_settle_timeout_sec", math.nan)),
        "chunk_stable_window_sec": float(runtime.get("chunk_stable_window_sec", math.nan)),
        "chunk_stable_span_threshold_rad": float(
            runtime.get("chunk_stable_span_threshold_rad", math.nan)
        ),
        "chunk_settle_poll_hz": float(runtime.get("chunk_settle_poll_hz", math.nan)),
        "shutdown_grace_sec": float(runtime.get("shutdown_grace_sec", math.nan)),
        "emergency_stop_timeout_sec": float(runtime.get("emergency_stop_timeout_sec", math.nan)),
        "target_tolerance_rad": float(motion.get("target_tolerance_rad", math.nan)),
        "command_timeout_sec": float(motion.get("command_timeout_sec", math.nan)),
    }
    for field, value in positive_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise MotionRejected(f"config_value_must_be_positive:{field}:{value}")
    backoff = float(runtime.get("controller_query_backoff_sec", math.nan))
    if not math.isfinite(backoff) or backoff < 0.0:
        raise MotionRejected(f"config_value_must_be_nonnegative:controller_query_backoff_sec:{backoff}")
    retries = runtime.get("controller_query_retries")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise MotionRejected(f"controller_query_retries_invalid:{retries!r}")
    if not 1 <= int(motion.get("speed_percent", 0)) <= 100:
        raise MotionRejected("speed_percent_must_be_1_to_100")
    if motion.get("command_api") != "rm_movej_canfd":
        raise MotionRejected(f"motion_command_api_unsupported:{motion.get('command_api')!r}")
    if motion.get("follow") is not False:
        raise MotionRejected("canfd_high_follow_not_enabled")
    trajectory_mode = motion.get("trajectory_mode")
    smoothing = motion.get("smoothing")
    if (
        isinstance(trajectory_mode, bool)
        or not isinstance(trajectory_mode, int)
        or trajectory_mode not in (0, 1, 2)
    ):
        raise MotionRejected(f"canfd_trajectory_mode_invalid:{trajectory_mode!r}")
    if (
        isinstance(smoothing, bool)
        or not isinstance(smoothing, int)
        or not 0 <= smoothing <= 999
    ):
        raise MotionRejected(f"canfd_smoothing_invalid:{smoothing!r}")
    if motion.get("software_joint_step_limit_enabled") is not True:
        raise MotionRejected("software_joint_step_limit_must_be_enabled")
    max_joint_step_rad = float(motion.get("max_joint_step_rad", math.nan))
    if not math.isfinite(max_joint_step_rad) or not 0.0 < max_joint_step_rad :
        raise MotionRejected(f"max_joint_step_rad_invalid:{max_joint_step_rad}")
    if gripper.get("mode") != "binary_threshold":
        raise MotionRejected(f"gripper_mode_unsupported:{gripper.get('mode')!r}")
    if not isinstance(gripper.get("commands_enabled"), bool):
        raise MotionRejected("gripper_commands_enabled_must_be_boolean")
    close_threshold = gripper.get("close_threshold")
    open_threshold = gripper.get("open_threshold")
    if (
        isinstance(close_threshold, bool)
        or not isinstance(close_threshold, (int, float))
        or not math.isfinite(float(close_threshold))
        or isinstance(open_threshold, bool)
        or not isinstance(open_threshold, (int, float))
        or not math.isfinite(float(open_threshold))
        or not 0.0 <= float(close_threshold) < float(open_threshold) <= 1.0
    ):
        raise MotionRejected(
            f"gripper_thresholds_invalid:{close_threshold!r}:{open_threshold!r}"
        )
    value_tolerance = gripper.get("value_tolerance")
    if (
        isinstance(value_tolerance, bool)
        or not isinstance(value_tolerance, (int, float))
        or not math.isfinite(float(value_tolerance))
        or not 0.0 <= float(value_tolerance) <= 0.5
    ):
        raise MotionRejected(f"gripper_value_tolerance_invalid:{value_tolerance!r}")
    close_position = gripper.get("close_position")
    open_position = gripper.get("open_position")
    if (
        isinstance(close_position, bool)
        or not isinstance(close_position, int)
        or isinstance(open_position, bool)
        or not isinstance(open_position, int)
        or not 1 <= close_position < open_position <= 1000
    ):
        raise MotionRejected(
            f"gripper_sdk_positions_invalid:{close_position!r}:{open_position!r}"
        )
    if not isinstance(recording.get("enabled"), bool):
        raise MotionRejected("recording_enabled_must_be_boolean")
    configured_recording_root = Path(str(recording.get("root_dir", ""))).expanduser()
    configured_recording_cache = Path(str(recording.get("cache_dir", ""))).expanduser()
    if not configured_recording_root.is_absolute() or not configured_recording_cache.is_absolute():
        raise MotionRejected("recording_root_must_be_absolute")
    recording_root = configured_recording_root.resolve()
    recording_cache = configured_recording_cache.resolve()
    if recording_cache != recording_root / ".cache":
        raise MotionRejected("recording_cache_must_be_root_dot_cache")
    recording_name = str(recording.get("subdirectory", ""))
    if recording_name and (
        recording_name in (".", "..", ".cache")
        or Path(recording_name).name != recording_name
        or Path(recording_name).is_absolute()
        or "\x00" in recording_name
    ):
        raise MotionRejected(f"recording_subdirectory_invalid:{recording_name!r}")
    if recording["enabled"] and not recording_name:
        raise MotionRejected("recording_subdirectory_required_when_enabled")
    recording_fps = float(recording.get("fps", math.nan))
    if not math.isfinite(recording_fps) or recording_fps <= 0.0:
        raise MotionRejected(f"recording_fps_invalid:{recording_fps}")
    if recording.get("codec") != "h264_vaapi":
        raise MotionRejected(f"recording_codec_unsupported:{recording.get('codec')!r}")
    recording_queue_size = recording.get("queue_size")
    if (
        isinstance(recording_queue_size, bool)
        or not isinstance(recording_queue_size, int)
        or recording_queue_size <= 0
    ):
        raise MotionRejected(f"recording_queue_size_invalid:{recording_queue_size!r}")
    recording_shutdown_timeout = float(recording.get("shutdown_timeout_sec", math.nan))
    if not math.isfinite(recording_shutdown_timeout) or recording_shutdown_timeout <= 0.0:
        raise MotionRejected(
            f"recording_shutdown_timeout_invalid:{recording_shutdown_timeout}"
        )
    for side, item in (("left", left), ("right", right)):
        if not str(item.get("ip", "")) or not 1 <= int(item.get("port", 0)) <= 65535:
            raise MotionRejected(f"robot_address_invalid:{side}")
        if not str(item.get("local_ip", "")) or not 1 <= int(item.get("udp_port", 0)) <= 65535:
            raise MotionRejected(f"robot_capture_address_invalid:{side}")
    if not Path(str(sources.get("deltawam_src", ""))).expanduser().resolve().is_dir():
        raise MotionRejected("deltawam_source_directory_missing")
    config["_path"] = str(resolved)
    config["_preset"] = preset
    config["_available_presets"] = sorted(str(name) for name in presets)
    return config


def runtime_args(config: Mapping[str, Any]) -> argparse.Namespace:
    observation = config["observation"]
    runtime = config["runtime"]
    motion = config["motion"]
    return argparse.Namespace(
        model_image_size=int(observation["model_image_size"]),
        state_stale_sec=float(runtime["state_stale_sec"]),
        response_to_command_sec=float(runtime["response_to_command_sec"]),
        watchdog_hz=float(runtime["watchdog_hz"]),
        controller_query_retries=int(runtime["controller_query_retries"]),
        controller_query_backoff_sec=float(runtime["controller_query_backoff_sec"]),
        target_tolerance_rad=float(motion["target_tolerance_rad"]),
        command_timeout_sec=float(motion["command_timeout_sec"]),
        speed_percent=int(motion["speed_percent"]),
        max_joint_step_rad=float(motion["max_joint_step_rad"]),
    )


def adapt_action_chunk(actions: Any, config: Mapping[str, Any]) -> Any:
    import numpy as np

    array = np.asarray(actions)
    allowed_dimensions = tuple(int(value) for value in config["allowed_source_dimensions"])
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] not in allowed_dimensions:
        raise MotionRejected(f"policy_action_shape_invalid:{array.shape}")
    if str(array.dtype) != str(config["required_dtype"]):
        raise MotionRejected(f"policy_action_values_invalid:{array.dtype}")
    if config["require_finite"] and not np.isfinite(array).all():
        raise MotionRejected(f"policy_action_values_invalid:{array.dtype}")
    if array.shape[1] == 28:
        array = array[:, tuple(int(value) for value in config["mapping_28_to_16"])]
    return np.ascontiguousarray(array, dtype=np.float32)


def adapt_observation_state(state: Any, config: Mapping[str, Any]) -> Any:
    """Map the captured 28D state to the preset's exact policy input contract."""
    import numpy as np

    array = np.asarray(state)
    captured_dimension = int(config["captured_state_dimension"])
    if array.shape != (captured_dimension,) or str(array.dtype) != "float32":
        raise MotionRejected(f"observation_state_invalid:{array.shape}:{array.dtype}")
    if not np.isfinite(array).all():
        raise MotionRejected(f"observation_state_invalid:{array.shape}:{array.dtype}")
    policy_dimension = int(config["state_dimension"])
    if policy_dimension == 16:
        array = array[[int(value) for value in config["mapping_28_to_16"]]]
    if array.shape != (policy_dimension,):
        raise MotionRejected(f"policy_state_shape_invalid:{array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def joint_targets_from_row(row: Sequence[float]) -> dict[str, list[float]]:
    values = [float(value) for value in row]
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise MotionRejected("canonical_action_row_invalid")
    return {"left": values[0:7], "right": values[8:15]}


def gripper_targets_from_row(
    row: Sequence[float],
    config: Mapping[str, Any],
    previous_targets: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> dict[str, dict[str, Any]]:
    import numpy as np

    values = [float(value) for value in row]
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise MotionRejected("canonical_action_row_invalid")
    close_threshold = np.float32(config["close_threshold"])
    open_threshold = np.float32(config["open_threshold"])
    tolerance = np.float32(config["value_tolerance"])
    result = {}
    for side, value in (("left", values[7]), ("right", values[15])):
        sample = np.float32(value)
        if sample < -tolerance or sample > np.float32(1.0) + tolerance:
            raise MotionRejected(f"gripper_value_out_of_normalized_range:{side}:{value}")
        effective = np.clip(sample, np.float32(0.0), np.float32(1.0))
        if effective <= close_threshold:
            result[side] = {"state": "closed", "position": int(config["close_position"])}
        elif effective >= open_threshold:
            result[side] = {"state": "open", "position": int(config["open_position"])}
        else:
            previous = None if previous_targets is None else previous_targets.get(side)
            if previous is None:
                result[side] = {"state": "hold", "position": None}
            else:
                position = previous.get("position")
                if isinstance(position, bool) or not isinstance(position, (int, float)):
                    raise MotionRejected(f"gripper_previous_target_invalid:{side}:{position!r}")
                result[side] = {
                    "state": str(previous.get("state", "hold")),
                    "position": int(position),
                }
    return result


def execute_gripper_target(
    *,
    backend: PresetBackend,
    events: RuntimeEvents,
    side: str,
    position: int,
) -> dict[str, Any]:
    events.emit("command", "gripper_submitting", {"side": side, "position": position})
    try:
        status = backend.command_gripper(side, position)
    except MotionRejected:
        raise
    except BaseException as exc:
        raise MotionRejected(f"gripper_command_acceptance_unknown:{side}") from exc
    events.emit("command", "gripper_sdk_acceptance", {"side": side, "sdk_status": status})
    if status != 0:
        raise MotionRejected(f"gripper_command_sdk_rejected:{side}:{status}")
    return {"side": side, "position": position, "sdk_status": status}


def validate_service_metadata(
    metadata: Any,
    config: Mapping[str, Any],
    *,
    request_budget: int,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise MotionRejected("server_metadata_must_be_mapping")
    policy = config["policy"]
    expected = {
        "config": str(policy.get("expected_config", "")),
        "checkpoint": str(policy.get("expected_checkpoint", "")),
    }
    for field, value in expected.items():
        if value and metadata.get(field) != value:
            raise MotionRejected(f"server_metadata_mismatch:{field}:{metadata.get(field)!r}")
    maximum = metadata.get("maximum_unique_matching_Client_requests")
    if maximum is not None and (not isinstance(maximum, int) or maximum < request_budget):
        raise MotionRejected(f"server_request_budget_too_small:{maximum}:{request_budget}")
    state_dimension = metadata.get("state_dimension", metadata.get("state_dim"))
    if state_dimension is not None:
        try:
            state_dimension = int(state_dimension)
        except (TypeError, ValueError) as exc:
            raise MotionRejected(f"server_state_dimension_invalid:{state_dimension}") from exc
        expected_state_dimension = int(config["observation"]["state_dimension"])
        if state_dimension != expected_state_dimension:
            raise MotionRejected(
                f"server_state_dimension_mismatch:{state_dimension}:{expected_state_dimension}"
            )
    action_dimension = metadata.get("action_dimension", metadata.get("action_dim"))
    if action_dimension is not None:
        try:
            action_dimension = int(action_dimension)
        except (TypeError, ValueError) as exc:
            raise MotionRejected(f"server_action_dimension_invalid:{action_dimension}") from exc
        if action_dimension not in set(int(value) for value in config["action"]["allowed_source_dimensions"]):
            raise MotionRejected(f"server_action_dimension_unsupported:{action_dimension}")
    return {
        "config": metadata.get("config"),
        "checkpoint": metadata.get("checkpoint"),
        "service_session_id": metadata.get("service_session_id"),
        "stable_runtime_commit": metadata.get("stable_runtime_commit"),
        "state_dimension": state_dimension,
        "action_dimension": action_dimension,
        "maximum_requests": maximum,
    }


def _structured_policy_error(exc: BaseException) -> tuple[str, str, str] | None:
    """Extract a bounded server rejection without treating arbitrary errors as known."""
    prefix = "Error in inference server:\n"
    message = str(exc)
    if not isinstance(exc, RuntimeError) or not message.startswith(prefix):
        return None
    try:
        payload = json.loads(message[len(prefix):])
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    status = str(payload.get("status", ""))
    error_type = str(payload.get("error_type", ""))
    reason = str(payload.get("reason", ""))
    if status not in ("rejected", "failed") or not error_type or not reason:
        return None
    return status, error_type[:80], reason.replace("\n", " ")[:400]


def _ancestor_pids() -> set[int]:
    result = {os.getpid()}
    current = os.getppid()
    while current > 1 and current not in result:
        result.add(current)
        try:
            stat = Path(f"/proc/{current}/stat").read_text(encoding="utf-8")
            current = int(stat.rsplit(")", 1)[1].split()[1])
        except (FileNotFoundError, IndexError, ValueError):
            break
    return result


def other_robot_command_processes() -> list[str]:
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    patterns = (
        "realman_deployment.client",
        "run_realman_fastwam_client",
        "spacemouse_control_arm",
        "run_rr_dep_003.py",
        "run_rr_dep_004.py",
        "run_rr_dep_006.py",
        "run_pi05_realman.py",
    )
    ignored = _ancestor_pids()
    conflicts = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid not in ignored and any(pattern in fields[1] for pattern in patterns):
            conflicts.append(line.strip())
    return conflicts


def require_fresh_safe(
    poller: RobotStatePoller,
    rig: RealSenseRig,
    args: argparse.Namespace,
    stage: str,
) -> dict[str, Any]:
    if not rig.healthy(args.state_stale_sec):
        raise MotionRejected(f"camera_unhealthy:{stage}")
    with poller._condition:
        if poller._errors:
            raise MotionRejected(f"robot_state_poll_errors:{stage}:{poller._errors}")
    now_ns = time.monotonic_ns()
    samples = {}
    for side in ("left", "right"):
        sample = poller.latest(side)
        if sample is None:
            raise MotionRejected(f"feedback_missing:{stage}:{side}")
        age_sec = (now_ns - sample.receive_monotonic_ns) / 1e9
        if age_sec < 0 or age_sec > args.state_stale_sec:
            raise MotionRejected(f"feedback_stale:{stage}:{side}:{age_sec:.6f}")
        if sample.sdk_status or sample.arm_errors or sample.controller_error:
            raise MotionRejected(f"robot_alarm:{stage}:{side}")
        samples[side] = sample
    return samples


def open_live_stack(config: dict[str, Any], args: argparse.Namespace, events: RuntimeEvents):
    conflicts = other_robot_command_processes()
    if conflicts:
        raise MotionRejected(f"other_robot_command_source:{conflicts}")
    backend = PresetBackend(
        config,
        events,
        retries=args.controller_query_retries,
        backoff_sec=args.controller_query_backoff_sec,
    )
    rig: RealSenseRig | None = None
    poller: RobotStatePoller | None = None
    try:
        rig = RealSenseRig(config, events)
        rig.enumerate_and_start()
        rig.wait_ready()
        backend.open()
        bindings = {side: backend.gather_binding(side) for side in ("left", "right")}
        poller = RobotStatePoller(backend, events, hz=args.watchdog_hz)
        poller.start()
        stability = poller.wait_stable(seconds=2.0)
        gate = evaluate_robot_gate_004(bindings, stability)
        backend.start_capture_state()
        require_fresh_safe(poller, rig, args, "live_stack_ready")
        return backend, rig, poller, gate
    except BaseException:
        if poller is not None:
            poller.close()
        if rig is not None:
            rig.close()
        backend.close()
        raise


def close_live_stack(backend: PresetBackend, rig: RealSenseRig, poller: RobotStatePoller) -> None:
    errors = []
    for close in (rig.close, poller.close, backend.close):
        try:
            close()
        except BaseException as exc:
            errors.append(repr(exc))
    if errors:
        raise MotionRejected("live_stack_close_failed:" + ";".join(errors))


def open_policy(config: Mapping[str, Any], request_budget: int):
    policy_config = config["policy"]
    if str(policy_config["host"]) in ("127.0.0.1", "localhost"):
        ensure_loopback_no_proxy()
    module = importlib.import_module("openpi_client.websocket_client_policy")
    policy = module.WebsocketClientPolicy(
        host=str(policy_config["host"]),
        port=int(policy_config["port"]),
    )
    try:
        metadata = validate_service_metadata(
            policy.get_server_metadata(), config, request_budget=request_budget
        )
    except BaseException:
        close = getattr(policy, "close", None)
        if close is not None:
            close()
        raise
    return policy, metadata


def capture_observation(
    rig: RealSenseRig,
    backend: PresetBackend,
    config: Mapping[str, Any],
    args: argparse.Namespace,
    prompt: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    import numpy as np

    state, _, state_metadata = backend.capture_observation_state()
    images, _, camera_metadata = latest_camera_observation(rig, args)
    policy_state = adapt_observation_state(state, config["observation"])
    timestamps = {
        **{
            f"camera:{role}": int(value["receive_monotonic_ns"])
            for role, value in camera_metadata.items()
        },
        **{
            f"robot:{side}": int(state_metadata[side]["receive_monotonic_ns"])
            for side in ("left", "right")
        },
    }
    oldest_source = min(timestamps, key=timestamps.get)
    newest_source = max(timestamps, key=timestamps.get)
    skew_sec = (timestamps[newest_source] - timestamps[oldest_source]) / 1e9
    if skew_sec > float(config["runtime"]["max_sync_skew_sec"]):
        raise MotionRejected(
            f"observation_sync_skew_exceeded:{skew_sec:.6f}:"
            f"oldest={oldest_source}:newest={newest_source}"
        )
    age_sec = (time.monotonic_ns() - timestamps[oldest_source]) / 1e9
    if age_sec < 0 or age_sec > args.state_stale_sec:
        raise MotionRejected(f"observation_stale:{age_sec:.6f}")
    return {**images, "state": policy_state, "prompt": prompt}, {
        "age_sec": age_sec,
        "skew_sec": skew_sec,
    }


def stop_both(backend: PresetBackend, events: RuntimeEvents) -> dict[str, Any]:
    statuses = {}
    for side in ("left", "right"):
        try:
            statuses[side] = backend.slow_stop(side)
        except BaseException as exc:
            statuses[side] = f"unknown:{type(exc).__name__}"
            events.emit("safety", "slow_stop_failure", {"side": side, "error": repr(exc)})
    return statuses


def execute_joint_target(
    *,
    backend: PresetBackend,
    poller: RobotStatePoller,
    rig: RealSenseRig,
    events: RuntimeEvents,
    args: argparse.Namespace,
    stop_event: threading.Event,
    side: str,
    target: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> dict[str, Any]:
    values = tuple(float(value) for value in target)
    bounds = tuple(float(value) for value in lower), tuple(float(value) for value in upper)
    if side not in ("left", "right"):
        raise MotionRejected(f"invalid_robot_side:{side}")
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        raise MotionRejected(f"invalid_joint_target:{side}")
    if any(len(items) != 7 or not all(math.isfinite(value) for value in items) for items in bounds):
        raise MotionRejected(f"invalid_joint_limits:{side}")
    lower, upper = bounds
    before = require_fresh_safe(poller, rig, args, f"before_command:{side}")[side]
    for index, (value, lo, hi) in enumerate(zip(values, lower, upper), start=1):
        if value < float(lo) or value > float(hi):
            raise MotionRejected(f"joint_target_out_of_limit:{side}:q{index}")
    initial_error = max(abs(actual - expected) for actual, expected in zip(before.joints_rad, values))
    if initial_error <= args.target_tolerance_rad:
        return {"side": side, "status": "already_within_tolerance", "target_error_rad": initial_error}
    if initial_error > args.max_joint_step_rad:
        raise MotionRejected(
            f"pre_send_joint_step_exceeded:{side}:{initial_error:.6f}:{args.max_joint_step_rad:.6f}"
        )
    if stop_event.is_set():
        raise UserExit("ctrl_c_before_command")
    events.emit("command", "submitting", {"side": side})
    try:
        status = backend.command_absolute(side, values, args.speed_percent)
    except MotionRejected:
        raise
    except BaseException as exc:
        backend.slow_stop(side)
        raise MotionRejected(f"command_acceptance_unknown:{side}") from exc
    events.emit("command", "sdk_acceptance", {"side": side, "sdk_status": status})
    if status != 0:
        backend.slow_stop(side)
        raise MotionRejected(f"command_sdk_rejected:{side}:{status}")
    started_ns = time.monotonic_ns()
    deadline = time.monotonic() + args.command_timeout_sec
    while True:
        if stop_event.is_set():
            backend.slow_stop(side)
            raise UserExit("ctrl_c_during_command")
        sample = require_fresh_safe(poller, rig, args, f"during_command:{side}")[side]
        for index, (actual, lo, hi) in enumerate(zip(sample.joints_rad, lower, upper), start=1):
            if actual < float(lo) or actual > float(hi):
                backend.slow_stop(side)
                raise MotionRejected(f"observed_joint_out_of_limit:{side}:q{index}")
        error = max(abs(actual - expected) for actual, expected in zip(sample.joints_rad, values))
        if error <= args.target_tolerance_rad and time.monotonic_ns() - started_ns >= 250_000_000:
            events.emit("command", "observed_target", {"side": side, "target_error_rad": error})
            return {"side": side, "status": "observed", "sdk_status": status, "target_error_rad": error}
        if time.monotonic() >= deadline:
            backend.slow_stop(side)
            raise MotionRejected(f"command_timeout:{side}")
        time.sleep(0.01)


def execute_dual_joint_targets(
    *,
    backend: PresetBackend,
    poller: RobotStatePoller,
    rig: RealSenseRig,
    events: RuntimeEvents,
    args: argparse.Namespace,
    stop_event: threading.Event,
    targets: Mapping[str, Sequence[float]],
    limits: Mapping[str, Mapping[str, Sequence[float]]],
    command_order: Sequence[str],
) -> list[dict[str, Any]]:
    order = tuple(str(side) for side in command_order)
    if sorted(order) != ["left", "right"] or set(targets) != {"left", "right"}:
        raise MotionRejected("dual_command_contract_invalid")
    values = {}
    bounds = {}
    for side in ("left", "right"):
        side_values = tuple(float(value) for value in targets[side])
        lower = tuple(float(value) for value in limits[side]["lower"])
        upper = tuple(float(value) for value in limits[side]["upper"])
        if len(side_values) != 7 or not all(math.isfinite(value) for value in side_values):
            raise MotionRejected(f"invalid_joint_target:{side}")
        if any(len(items) != 7 or not all(math.isfinite(value) for value in items) for items in (lower, upper)):
            raise MotionRejected(f"invalid_joint_limits:{side}")
        values[side] = side_values
        bounds[side] = (lower, upper)

    before = require_fresh_safe(poller, rig, args, "before_dual_command")
    initial_errors = {}
    for side in ("left", "right"):
        lower, upper = bounds[side]
        for index, (value, lo, hi) in enumerate(zip(values[side], lower, upper), start=1):
            if value < lo or value > hi:
                raise MotionRejected(f"joint_target_out_of_limit:{side}:q{index}")
        initial_errors[side] = max(
            abs(actual - expected)
            for actual, expected in zip(before[side].joints_rad, values[side])
        )
        if initial_errors[side] > args.max_joint_step_rad:
            raise MotionRejected(
                f"pre_send_joint_step_exceeded:{side}:{initial_errors[side]:.6f}:"
                f"{args.max_joint_step_rad:.6f}"
            )
    if all(error <= args.target_tolerance_rad for error in initial_errors.values()):
        return [
            {
                "side": side,
                "status": "already_within_tolerance",
                "target_error_rad": initial_errors[side],
            }
            for side in order
        ]
    if stop_event.is_set():
        raise UserExit("ctrl_c_before_dual_command")

    for side in order:
        events.emit("command", "submitting", {"side": side})
    try:
        statuses = backend.command_dual_absolute(values, args.speed_percent)
    except MotionRejected:
        raise
    except BaseException as exc:
        raise MotionRejected("dual_command_acceptance_unknown") from exc
    for side in order:
        if side not in statuses:
            raise MotionRejected(f"dual_command_not_submitted:{side}:{statuses}")
        status = int(statuses[side])
        events.emit("command", "sdk_acceptance", {"side": side, "sdk_status": status})
        if status != 0:
            raise MotionRejected(f"dual_command_sdk_rejected:{side}:{status}")

    started_ns = time.monotonic_ns()
    deadline = time.monotonic() + args.command_timeout_sec
    while True:
        if stop_event.is_set():
            stop_both(backend, events)
            raise UserExit("ctrl_c_during_dual_command")
        samples = require_fresh_safe(poller, rig, args, "during_dual_command")
        errors = {}
        for side in ("left", "right"):
            lower, upper = bounds[side]
            for index, (actual, lo, hi) in enumerate(
                zip(samples[side].joints_rad, lower, upper), start=1
            ):
                if actual < lo or actual > hi:
                    stop_both(backend, events)
                    raise MotionRejected(f"observed_joint_out_of_limit:{side}:q{index}")
            errors[side] = max(
                abs(actual - expected)
                for actual, expected in zip(samples[side].joints_rad, values[side])
            )
        if (
            all(error <= args.target_tolerance_rad for error in errors.values())
            and time.monotonic_ns() - started_ns >= 250_000_000
        ):
            for side in order:
                events.emit(
                    "command", "observed_target", {"side": side, "target_error_rad": errors[side]}
                )
            return [
                {
                    "side": side,
                    "status": "observed",
                    "sdk_status": int(statuses[side]),
                    "target_error_rad": errors[side],
                }
                for side in order
            ]
        if time.monotonic() >= deadline:
            stop_both(backend, events)
            raise MotionRejected(
                f"dual_command_timeout:left={errors['left']:.6f}:right={errors['right']:.6f}"
            )
        time.sleep(0.01)


def operator_gate(
    *,
    stop_event: threading.Event,
    safety_check: Any,
) -> None:
    print("请回车确认，Ctrl+C 退出：", end="", flush=True)
    while True:
        if stop_event.is_set():
            raise UserExit("ctrl_c_before_confirmation")
        safety_check()
        readable, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not readable:
            continue
        line = sys.stdin.readline()
        if line == "":
            raise MotionRejected("operator_input_closed")
        safety_check()
        return


def run_live(
    config: dict[str, Any],
    *,
    stop_event: threading.Event,
) -> dict[str, Any]:
    args = runtime_args(config)
    execution = config["execution"]
    execute = execution["mode"] == "execute"
    max_replans = int(execution["max_replans"])
    prompt = str(config["policy"]["prompt"])
    events = RuntimeEvents()
    backend = rig = poller = policy = None
    failed = False
    try:
        backend, rig, poller, gate = open_live_stack(config, args, events)
        policy, service = open_policy(config, request_budget=max_replans if execute else 1)
        LOGGER.info("Validated policy service: %s", service)
        if execute:
            operator_gate(
                stop_event=stop_event,
                safety_check=lambda: require_fresh_safe(poller, rig, args, "awaiting_authorization"),
            )
            LOGGER.info("Operator confirmation accepted; starting inference")
        command_results = []
        gripper_results = []
        commanded_gripper_targets = {"left": None, "right": None}
        round_trips_ms = []
        sync_skews_ms = []
        for ordinal in range(1, max_replans + 1 if execute else 2):
            if stop_event.is_set():
                raise UserExit("ctrl_c_before_inference")
            require_fresh_safe(poller, rig, args, f"before_inference:{ordinal}")
            observation, timing = capture_observation(rig, backend, config, args, prompt)
            started_ns = time.monotonic_ns()
            events.emit("policy", "request_sending", {"ordinal": ordinal})
            LOGGER.info("Requesting inference for replan %d/%d", ordinal, max_replans)
            try:
                response = policy.infer(observation)
            except BaseException as exc:
                structured = _structured_policy_error(exc)
                if structured is not None:
                    status, error_type, reason = structured
                    events.emit(
                        "safety",
                        f"policy_request_{status}",
                        {"error": f"{error_type}:{reason}"},
                    )
                    raise MotionRejected(
                        f"policy_request_{status}:{ordinal}:{error_type}:{reason}"
                    ) from exc
                events.emit("safety", "policy_request_outcome_unknown", {"error": type(exc).__name__})
                raise MotionRejected(f"policy_request_outcome_unknown:{ordinal}") from exc
            received_ns = time.monotonic_ns()
            if stop_event.is_set():
                raise UserExit("ctrl_c_after_inference")
            if not isinstance(response, Mapping) or "actions" not in response:
                raise MotionRejected("policy_response_missing_actions")
            canonical = adapt_action_chunk(response["actions"], config["action"])
            events.emit("policy", "response_validated", {"ordinal": ordinal, "shape": list(canonical.shape)})
            round_trips_ms.append((received_ns - started_ns) / 1e6)
            sync_skews_ms.append(timing["skew_sec"] * 1000.0)
            if not execute:
                return {
                    "status": "no_motion_pass",
                    "policy_requests": 1,
                    "action_shape": list(canonical.shape),
                    "robot_commands": 0,
                    "gripper_commands": 0,
                    "service": service,
                }
            response_age = (time.monotonic_ns() - received_ns) / 1e9
            if response_age > args.response_to_command_sec:
                raise MotionRejected(f"response_stale_before_command:{response_age:.6f}")
            require_fresh_safe(poller, rig, args, f"before_action:{ordinal}")
            limits = {
                side: {
                    "lower": gate["arms"][side]["lower_rad"],
                    "upper": gate["arms"][side]["upper_rad"],
                }
                for side in ("left", "right")
            }
            first_row = int(config["action"]["row_index"])
            actions_per_chunk = int(config["action"]["actions_per_chunk"])
            stop_row = first_row + actions_per_chunk
            if stop_row > canonical.shape[0]:
                raise MotionRejected(
                    f"configured_action_rows_missing:{first_row}:{stop_row}:{canonical.shape[0]}"
                )
            for action_row in range(first_row, stop_row):
                LOGGER.info(
                    "Executing replan %d/%d action row %d (%d/%d)",
                    ordinal,
                    max_replans,
                    action_row,
                    action_row - first_row + 1,
                    actions_per_chunk,
                )
                require_fresh_safe(
                    poller,
                    rig,
                    args,
                    f"before_action:{ordinal}:row:{action_row}",
                )
                targets = joint_targets_from_row(canonical[action_row])
                LOGGER.info(
                    "Submitting replan %d/%d action row %d dual CANFD pair",
                    ordinal,
                    max_replans,
                    action_row,
                )
                results = execute_dual_joint_targets(
                    backend=backend,
                    poller=poller,
                    rig=rig,
                    events=events,
                    args=args,
                    stop_event=stop_event,
                    targets=targets,
                    limits=limits,
                    command_order=config["action"]["command_order"],
                )
                command_results.extend(
                    {"action_row": action_row, **result} for result in results
                )
                if config["gripper"]["commands_enabled"]:
                    gripper_targets = gripper_targets_from_row(
                        canonical[action_row],
                        config["gripper"],
                        commanded_gripper_targets,
                    )
                    for side in config["action"]["command_order"]:
                        target = gripper_targets[side]
                        if target["position"] is None:
                            continue
                        previous = commanded_gripper_targets[side]
                        if previous is not None and previous["position"] == target["position"]:
                            commanded_gripper_targets[side] = target
                            continue
                        result = execute_gripper_target(
                            backend=backend,
                            events=events,
                            side=side,
                            position=int(target["position"]),
                        )
                        commanded_gripper_targets[side] = target
                        gripper_results.append({
                            "action_row": action_row,
                            "state": target["state"],
                            **result,
                        })
                LOGGER.info(
                    "Completed replan %d/%d action row %d",
                    ordinal,
                    max_replans,
                    action_row,
                )
            LOGGER.info(
                "Completed replan %d/%d using action rows %d..%d",
                ordinal,
                max_replans,
                first_row,
                stop_row - 1,
            )
        return {
            "status": "completed_bounded_inference",
            "replans": max_replans,
            "policy_requests": max_replans,
            "joint_commands_submitted": int(events.counters.get("command.submitting", 0)),
            "joint_targets_observed": int(events.counters.get("command.observed_target", 0)),
            "gripper_commands": len(gripper_results),
            "preset": config["_preset"],
            "action_row": int(config["action"]["row_index"]),
            "actions_per_chunk": int(config["action"]["actions_per_chunk"]),
            "mean_round_trip_ms": sum(round_trips_ms) / len(round_trips_ms),
            "max_sync_skew_ms": max(sync_skews_ms),
            "commands": command_results,
            "grippers": gripper_results,
            "service": service,
        }
    except BaseException:
        failed = True
        if backend is not None:
            stop_both(backend, events)
        raise
    finally:
        close_errors = []
        close = getattr(policy, "close", None)
        if close is not None:
            try:
                close()
            except BaseException as exc:
                close_errors.append(f"policy:{exc!r}")
        if backend is not None and rig is not None and poller is not None:
            try:
                close_live_stack(backend, rig, poller)
            except BaseException as exc:
                close_errors.append(f"live_stack:{exc!r}")
        if close_errors:
            if not failed:
                raise MotionRejected("runtime_close_failed:" + ";".join(close_errors))
            LOGGER.error("Runtime close failed after terminal error: %s", ";".join(close_errors))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.preset)
        logging.basicConfig(
            level=getattr(logging, str(config["execution"]["log_level"])),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        stop_event = threading.Event()
        try:
            result = run_live(config, stop_event=stop_event)
        except (UserExit, KeyboardInterrupt):
            result = {"status": "stopped", "preset": config["_preset"]}
        result["preset"] = config["_preset"]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        print(json.dumps({
            "status": "blocked",
            "reason": f"{type(exc).__name__}:{exc}",
        }, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
