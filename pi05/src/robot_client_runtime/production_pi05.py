"""Supervised production runner for OpenPI and RealMan dual-arm hardware.

The parent process owns the terminal, deadlines, and cancellation. A spawned
worker exclusively owns OpenPI, cameras, and the RealMan Python SDK so a stuck
native SDK call cannot make Ctrl+C unresponsive in the parent.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
from pathlib import Path
import queue
import shutil
import signal
import sys
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence
import uuid

from .realman_pi05 import (
    DEFAULT_CONFIG,
    MotionRejected,
    adapt_action_chunk,
    adapt_observation_state,
    gripper_targets_from_row,
    load_config,
    open_policy,
)
from .rr_dep_003 import resize_with_pad_rgb


LOGGER = logging.getLogger(__name__)
DEFAULT_PRESET = "pi05-sugar-15000"
CAMERA_TO_OPENPI = {
    "high_rgb": "base_0_rgb",
    "left_wrist_rgb": "left_wrist_0_rgb",
    "right_wrist_rgb": "right_wrist_0_rgb",
}
RECORDING_OUTPUT_NAMES = {
    "high_rgb": "high",
    "left_wrist_rgb": "left",
    "right_wrist_rgb": "right",
}


class WorkerTimeout(MotionRejected):
    pass


class WorkerStopped(Exception):
    pass


class _ArmStateRetryProxy:
    """Retry only read-only arm-state receives that time out with SDK status -2."""

    def __init__(self, arm: Any, *, side: str, retries: int, backoff_sec: float) -> None:
        self._arm = arm
        self._side = str(side)
        self._retries = int(retries)
        self._backoff_sec = float(backoff_sec)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._arm, name)

    def rm_get_current_arm_state(self):
        retries_used = 0
        while True:
            code, state = self._arm.rm_get_current_arm_state()
            if int(code) != -2 or retries_used >= self._retries:
                if int(code) == 0 and retries_used:
                    LOGGER.info(
                        "Recovered %s arm state after %d SDK -2 retries",
                        self._side,
                        retries_used,
                    )
                return code, state
            retries_used += 1
            LOGGER.warning(
                "Read %s arm state returned SDK -2; retrying %d/%d after %.3f sec",
                self._side,
                retries_used,
                self._retries,
                self._backoff_sec,
            )
            if self._backoff_sec > 0.0:
                time.sleep(self._backoff_sec)


def _install_arm_state_retries(hardware: Any, config: Mapping[str, Any]) -> None:
    """Make transient SDK -2 state reads recoverable across every runner phase."""
    runtime = config["runtime"]
    retries = int(runtime["controller_query_retries"])
    backoff_sec = float(runtime["controller_query_backoff_sec"])
    for side in ("left", "right"):
        arm = getattr(hardware, f"{side}_arm")
        if not isinstance(arm, _ArmStateRetryProxy):
            setattr(
                hardware,
                f"{side}_arm",
                _ArmStateRetryProxy(
                    arm,
                    side=side,
                    retries=retries,
                    backoff_sec=backoff_sec,
                ),
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run supervised OpenPI inference on RealMan dual-arm hardware",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help="named preset inside the config registry",
    )
    return parser


def _emit(emit: Callable[[dict[str, Any]], None], event: str, **payload: Any) -> None:
    emit({"event": event, "monotonic_ns": time.monotonic_ns(), **payload})


def _format_worker_message(message: Mapping[str, Any]) -> str:
    """Render compact, stable worker progress lines for the terminal."""
    def display_gripper(value: Any) -> str:
        numeric = float(value)
        if not math.isfinite(numeric):
            return str(value)
        clamped = min(1.0, max(0.0, numeric))
        if clamped == 0.0:
            clamped = 0.0
        return f"{clamped:.2f}"

    fields = []
    for key in (
        "event",
        "phase",
        "replan",
        "action_row",
        "gripper",
        "config",
        "checkpoint",
        "service_session_id",
        "stable_runtime_commit",
        "state_dimension",
        "action_dimension",
        "maximum_requests",
    ):
        if key not in message:
            continue
        value = message[key]
        if key == "gripper" and isinstance(value, Mapping):
            value = " & ".join(
                display_gripper(value[side])
                for side in ("left", "right")
                if side in value
            )
        fields.append(f"{key}: {value}")
    if message.get("event") == "inference_done" and "action_shape" in message:
        fields.append(f"action_shape: {message['action_shape']}")
    if message.get("event") == "chunk_settle_done":
        for key in ("target_error_rad", "feedback_span_rad", "stable_duration_sec"):
            if key in message:
                fields.append(f"{key}: {message[key]:.4f}")
    if message.get("event") == "initial_pose_done":
        for key in ("joint_error_rad", "gripper_error", "feedback_span_rad"):
            if key in message:
                fields.append(f"{key}: {message[key]:.4f}")
    return ", ".join(fields)


class FlatCameraVideoRecorder:
    """Stage native camera recording in cache, then publish three flat MP4 files."""

    def __init__(
        self,
        *,
        target_dir: Path,
        cache_dir: Path,
        basename: str,
        fps: float,
        codec: str,
        queue_size: int,
        recorder_type: Any,
        start_paused: bool = False,
    ) -> None:
        self.target_dir = target_dir
        self.basename = basename
        self.cache_session_dir = cache_dir / f"{basename}_{uuid.uuid4().hex}"
        self.cache_session_dir.mkdir(parents=True, exist_ok=False)
        self._delegate = recorder_type(
            str(self.cache_session_dir),
            fps=fps,
            codec=codec,
            queue_size=queue_size,
        )
        self._delegate_closed = False
        self._published = False
        self._aborted = False
        self._output_files: list[str] = []
        self._start_lock = threading.Lock()
        self._recording_started_ns = None if start_paused else 0

    @property
    def output_files(self) -> list[str]:
        return list(self._output_files)

    @property
    def started(self) -> bool:
        with self._start_lock:
            return self._recording_started_ns is not None

    def start(self) -> int:
        """Allow only frames captured at or after this monotonic timestamp."""
        with self._start_lock:
            if self._aborted or self._published:
                raise RuntimeError("recording_already_closed")
            if self._recording_started_ns is None:
                self._recording_started_ns = time.monotonic_ns()
            return int(self._recording_started_ns)

    def submit(self, name: str, image: Any, *, sequence: int, timestamp_ns: int) -> None:
        with self._start_lock:
            started_ns = self._recording_started_ns
        if started_ns is None or int(timestamp_ns) < started_ns:
            return
        self._delegate.submit(name, image, sequence=sequence, timestamp_ns=timestamp_ns)

    def close(self) -> None:
        if self._published or self._aborted:
            return
        if not self.started:
            self.abort()
            return
        if not self._delegate_closed:
            self._delegate.close()
            self._delegate_closed = True
        alive_writers = [
            thread.name
            for thread in getattr(self._delegate, "_threads", ())
            if thread.is_alive()
        ]
        if alive_writers:
            raise RuntimeError(f"recording_writer_threads_alive:{alive_writers}")
        writer_errors = dict(getattr(self._delegate, "_errors", {}))
        if writer_errors:
            raise RuntimeError(f"recording_writer_errors:{writer_errors}")
        sources = {
            camera: Path(self._delegate.run_dir) / f"{camera}.mp4"
            for camera in RECORDING_OUTPUT_NAMES
        }
        missing = [
            camera
            for camera, path in sources.items()
            if not path.is_file() or path.stat().st_size <= 0
        ]
        if missing:
            raise RuntimeError(f"recording_outputs_missing_or_empty:{missing}")
        destinations = {
            camera: self.target_dir / f"{self.basename}_{suffix}.mp4"
            for camera, suffix in RECORDING_OUTPUT_NAMES.items()
        }
        for camera in RECORDING_OUTPUT_NAMES:
            sources[camera].replace(destinations[camera])
        shutil.rmtree(self.cache_session_dir)
        self._output_files = [
            str(destinations[camera]) for camera in RECORDING_OUTPUT_NAMES
        ]
        self._published = True

    def abort(self) -> None:
        if self._published or self._aborted:
            return
        if not self._delegate_closed:
            try:
                self._delegate.close()
            except BaseException:
                pass
            self._delegate_closed = True
        if self.cache_session_dir.exists():
            shutil.rmtree(self.cache_session_dir)
        self._aborted = True


def prepare_recording_destination(
    config: dict[str, Any], *, input_fn: Callable[[str], str] | None = None
) -> bool:
    """Prepare the exact recording directory before any worker or inference starts."""
    recording = config["recording"]
    if not recording["enabled"]:
        return True
    root_dir = Path(str(recording["root_dir"])).expanduser().resolve()
    cache_dir = Path(str(recording["cache_dir"])).expanduser().resolve()
    basename = str(recording["subdirectory"])
    target_dir = root_dir / basename
    if target_dir.parent != root_dir or target_dir.is_symlink():
        raise MotionRejected("recording_target_must_be_direct_non_symlink_child")
    root_dir.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and not target_dir.is_dir():
        raise MotionRejected("recording_target_exists_and_is_not_directory")
    nonempty = target_dir.is_dir() and next(target_dir.iterdir(), None) is not None
    if nonempty:
        prompt = input if input_fn is None else input_fn
        try:
            answer = prompt("目录非空，覆盖[y/N]:")
        except EOFError:
            answer = ""
        if answer.strip().lower() != "y":
            return False
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=False, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    config["_recording_target_dir"] = str(target_dir)
    return True


def _create_frame_recorder(config: Mapping[str, Any]) -> FlatCameraVideoRecorder | None:
    recording = config["recording"]
    if not recording["enabled"]:
        return None
    target_dir_value = config.get("_recording_target_dir")
    if not target_dir_value:
        raise MotionRejected("recording_destination_not_prepared")
    from realman_deployment.client import BackgroundCameraVideoRecorder

    return FlatCameraVideoRecorder(
        target_dir=Path(str(target_dir_value)),
        cache_dir=Path(str(recording["cache_dir"])).expanduser().resolve(),
        basename=str(recording["subdirectory"]),
        fps=float(recording["fps"]),
        codec=str(recording["codec"]),
        queue_size=int(recording["queue_size"]),
        recorder_type=BackgroundCameraVideoRecorder,
        start_paused=True,
    )


def _load_fastwam_hardware(config: Mapping[str, Any]):
    source_root = str(Path(str(config["sources"]["deltawam_src"])).resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from realman_deployment.hardware import RealManHardware

    cameras = config["cameras"]
    robots = config["robots"]
    motion = config["motion"]
    runtime = config["runtime"]
    commands_enabled = bool(config["gripper"]["commands_enabled"])
    recorder = _create_frame_recorder(config)
    try:
        hardware = RealManHardware(
            left_ip=str(robots["left"]["ip"]),
            right_ip=str(robots["right"]["ip"]),
            arm_port=int(robots["left"]["port"]),
            high_camera_id=str(cameras["head"]),
            left_camera_id=str(cameras["left_wrist"]),
            right_camera_id=str(cameras["right_wrist"]),
            max_joint_step_rad=float(motion["max_joint_step_rad"]),
            trajectory_mode=int(motion["trajectory_mode"]),
            smoothing=int(motion["smoothing"]),
            gripper_mode="position",
            gripper_threshold=5 if commands_enabled else 1001,
            max_sync_skew_ms=float(runtime["max_sync_skew_sec"]) * 1000.0,
            sync_timeout_s=float(runtime["sync_timeout_sec"]),
            frame_recorder=recorder,
        )
        _install_arm_state_retries(hardware, config)
    except BaseException:
        if recorder is not None:
            recorder.abort()
        raise
    hardware._production_frame_recorder = recorder
    if not commands_enabled:
        # Position mode sends on first use when the cache is None. A midpoint
        # cache plus an impossible delta threshold suppresses all gripper calls.
        hardware._last_grippers = {"left": 500, "right": 500}
    return hardware


def _validate_hardware_ready(hardware: Any) -> dict[str, Any]:
    """Read-only alarm/enable gate using the handles owned by RealManHardware."""
    result = {}
    with hardware._lock:
        for side, arm in (("left", hardware.left_arm), ("right", hardware.right_arm)):
            state_code, state = arm.rm_get_current_arm_state()
            controller = arm.rm_get_controller_state()
            enable_code, enables = arm.rm_get_joint_en_state()
            errors = arm.rm_get_joint_err_flag()
            raw_errors = state.get("err", {}).get("err", []) if isinstance(state, dict) else []
            if int(state_code) != 0 or not isinstance(state, dict):
                raise MotionRejected(f"hardware_state_gate_failed:{side}:{state_code}")
            if any(str(value).strip() not in ("0", "0.0") for value in raw_errors):
                raise MotionRejected(f"hardware_arm_alarm:{side}:{raw_errors}")
            if int(controller.get("return_code", -1)) != 0 or int(controller.get("system_error", -1)) != 0:
                raise MotionRejected(f"hardware_controller_alarm:{side}:{controller}")
            if int(enable_code) != 0 or list(enables) != [1] * 7:
                raise MotionRejected(f"hardware_joint_enable_gate_failed:{side}:{enable_code}:{enables}")
            if (
                int(errors.get("return_code", -1)) != 0
                or any(int(value) != 0 for value in errors.get("err_flag", []))
                or any(int(value) != 0 for value in errors.get("brake_state", []))
            ):
                raise MotionRejected(f"hardware_joint_error_gate_failed:{side}:{errors}")
            result[side] = {"state": "ready", "controller_error": 0, "joints_enabled": True}
    return result


def _read_current_grippers(hardware: Any) -> dict[str, float]:
    reader = getattr(hardware, "current_grippers", None)
    if callable(reader):
        values = reader()
        if isinstance(values, Mapping):
            result = {side: float(values[side]) for side in ("left", "right")}
        else:
            left, right = values
            result = {"left": float(left), "right": float(right)}
    else:
        result = {}
        with hardware._lock:
            for side, arm in (("left", hardware.left_arm), ("right", hardware.right_arm)):
                code, state = arm.rm_get_gripper_state()
                if int(code) != 0 or not isinstance(state, Mapping) or "actpos" not in state:
                    raise MotionRejected(f"initial_pose_gripper_feedback_failed:{side}:{code}")
                result[side] = float(state["actpos"])
    if set(result) != {"left", "right"} or not all(
        math.isfinite(value) for value in result.values()
    ):
        raise MotionRejected(f"initial_pose_gripper_feedback_invalid:{result}")
    return result


def wait_for_initial_pose_completion(
    *,
    hardware: Any,
    config: Mapping[str, Any],
    stop_event: Any,
) -> dict[str, float]:
    """Require fresh dual-arm and dual-gripper feedback before inference/recording."""
    import numpy as np

    initial = config["initial_pose"]
    targets = {
        side: np.deg2rad(np.asarray(initial[f"{side}_joint_deg"], dtype=np.float64))
        for side in ("left", "right")
    }
    open_position = float(initial["gripper_open_position"])
    joint_tolerance = float(initial["joint_tolerance_rad"])
    gripper_tolerance = float(initial["gripper_tolerance"])
    stable_window = float(initial["stable_window_sec"])
    stable_span_threshold = float(initial["stable_span_threshold_rad"])
    poll_period = 1.0 / float(initial["poll_hz"])
    deadline = time.monotonic() + float(initial["settle_timeout_sec"])
    previous = None
    stable_since = None
    last_joint_error = math.inf
    last_gripper_error = math.inf
    last_span = math.inf

    while True:
        if stop_event.is_set():
            raise WorkerStopped("stop_during_initial_pose_settle")
        left, right = hardware.current_joints()
        current = {
            "left": np.asarray(left, dtype=np.float64),
            "right": np.asarray(right, dtype=np.float64),
        }
        if any(values.shape != (7,) or not np.isfinite(values).all() for values in current.values()):
            raise MotionRejected("initial_pose_joint_feedback_invalid")
        grippers = _read_current_grippers(hardware)
        last_joint_error = max(
            float(np.max(np.abs(current[side] - targets[side])))
            for side in ("left", "right")
        )
        last_gripper_error = max(
            abs(grippers[side] - open_position) for side in ("left", "right")
        )
        current_vector = np.concatenate((current["left"], current["right"]))
        last_span = (
            0.0
            if previous is None
            else float(np.max(np.abs(current_vector - previous)))
        )
        now = time.monotonic()
        if (
            last_joint_error <= joint_tolerance
            and last_gripper_error <= gripper_tolerance
            and last_span <= stable_span_threshold
        ):
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= stable_window:
                return {
                    "joint_error_rad": last_joint_error,
                    "gripper_error": last_gripper_error,
                    "feedback_span_rad": last_span,
                    "stable_duration_sec": now - stable_since,
                }
        else:
            stable_since = None
        previous = current_vector
        remaining = deadline - now
        if remaining <= 0.0:
            raise MotionRejected(
                "initial_pose_settle_timeout:"
                f"joint_error={last_joint_error:.6f}:"
                f"gripper_error={last_gripper_error:.1f}:"
                f"span={last_span:.6f}"
            )
        if stop_event.wait(min(poll_period, remaining)):
            raise WorkerStopped("stop_during_initial_pose_settle")


def move_to_initial_pose(
    *,
    hardware: Any,
    config: Mapping[str, Any],
    stop_event: Any,
) -> dict[str, Any]:
    """Submit one nonblocking rm_movej per arm, open both grippers, then verify feedback."""
    initial = config["initial_pose"]
    targets_deg = {
        side: [float(value) for value in initial[f"{side}_joint_deg"]]
        for side in ("left", "right")
    }
    speeds = {
        side: int(initial[f"{side}_speed_percent"])
        for side in ("left", "right")
    }
    open_position = int(initial["gripper_open_position"])
    command_order = tuple(str(side) for side in config["action"]["command_order"])
    statuses = {"joints": {}, "grippers": {}}
    with hardware._lock:
        for side in command_order:
            arm = hardware.left_arm if side == "left" else hardware.right_arm
            status = int(arm.rm_movej(targets_deg[side], speeds[side], 0, 0, 0))
            statuses["joints"][side] = status
            if status != 0:
                raise MotionRejected(f"initial_pose_joint_command_rejected:{side}:{status}")
        for side in command_order:
            arm = hardware.left_arm if side == "left" else hardware.right_arm
            status = int(arm.rm_set_gripper_position(open_position, False, 0))
            statuses["grippers"][side] = status
            if status != 0:
                raise MotionRejected(f"initial_pose_gripper_command_rejected:{side}:{status}")
        cached = getattr(hardware, "_last_grippers", None)
        if isinstance(cached, dict):
            cached.update({"left": open_position, "right": open_position})
    settled = wait_for_initial_pose_completion(
        hardware=hardware,
        config=config,
        stop_event=stop_event,
    )
    return {
        "joint_sdk_status": statuses["joints"],
        "gripper_sdk_status": statuses["grippers"],
        "speed_percent_by_side": speeds,
        **settled,
    }


def prepare_openpi_observation(
    images: Mapping[str, Any], state: Sequence[float], config: Mapping[str, Any]
) -> dict[str, Any]:
    import numpy as np

    if set(images) != set(CAMERA_TO_OPENPI):
        raise MotionRejected(f"hardware_camera_roles_invalid:{sorted(images)}")
    converted = {
        CAMERA_TO_OPENPI[name]: np.ascontiguousarray(
            resize_with_pad_rgb(image, int(config["observation"]["model_image_size"])),
            dtype=np.uint8,
        )
        for name, image in images.items()
    }
    raw_state = np.ascontiguousarray(np.asarray(state, dtype=np.float32))
    policy_state = adapt_observation_state(raw_state, config["observation"])
    return {**converted, "state": policy_state, "prompt": str(config["policy"]["prompt"])}


def canonical_row_to_hardware_action(
    row: Sequence[float],
    config: Mapping[str, Any],
    gripper_targets: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[float]:
    """Convert canonical left-first normalized 16D to FastWAM right-first SDK action."""
    values = [float(value) for value in row]
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise MotionRejected("canonical_action_row_invalid")
    grippers = (
        gripper_targets_from_row(values, config["gripper"])
        if gripper_targets is None
        else gripper_targets
    )
    if any(grippers[side].get("position") is None for side in ("left", "right")):
        raise MotionRejected("gripper_hold_without_previous_target")
    return (
        values[8:15]
        + [float(grippers["right"]["position"])]
        + values[0:7]
        + [float(grippers["left"]["position"])]
    )


def wait_for_chunk_completion(
    *,
    hardware: Any,
    target: Sequence[float],
    tolerance_rad: float,
    stable_window_sec: float,
    stable_span_threshold_rad: float,
    timeout_sec: float,
    poll_hz: float,
    stop_event: Any,
) -> dict[str, float]:
    """Wait for both arms to reach and hold the final target of a chunk.

    CANFD returns after accepting a target, not after the arm reaches it.  The
    completion gate therefore uses fresh joint feedback and requires the final
    target to remain within tolerance for a short stability window.
    """
    import numpy as np

    target_array = np.asarray(target, dtype=np.float64)
    if target_array.shape != (16,) or not np.isfinite(target_array).all():
        raise MotionRejected("chunk_completion_target_invalid")
    if tolerance_rad <= 0.0 or stable_window_sec <= 0.0 or stable_span_threshold_rad <= 0.0:
        raise MotionRejected("chunk_completion_settings_invalid")
    if timeout_sec <= 0.0 or poll_hz <= 0.0:
        raise MotionRejected("chunk_completion_timeout_or_poll_invalid")

    deadline = time.monotonic() + float(timeout_sec)
    poll_period = 1.0 / float(poll_hz)
    previous = None
    stable_since = None
    last_error = math.inf
    last_span = math.inf

    while True:
        if stop_event.is_set():
            raise WorkerStopped("stop_during_chunk_settle")
        left, right = hardware.current_joints()
        current = np.concatenate(
            (np.asarray(right, dtype=np.float64), np.asarray(left, dtype=np.float64))
        )
        if current.shape != (14,) or not np.isfinite(current).all():
            raise MotionRejected("chunk_completion_feedback_invalid")
        target_joints = np.concatenate((target_array[0:7], target_array[8:15]))
        last_error = float(np.max(np.abs(current - target_joints)))
        last_span = 0.0 if previous is None else float(np.max(np.abs(current - previous)))
        now = time.monotonic()
        if last_error <= tolerance_rad and last_span <= stable_span_threshold_rad:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= stable_window_sec:
                return {
                    "target_error_rad": last_error,
                    "feedback_span_rad": last_span,
                    "stable_duration_sec": now - stable_since,
                }
        else:
            stable_since = None
        previous = current
        remaining = deadline - now
        if remaining <= 0.0:
            raise MotionRejected(
                f"chunk_settle_timeout:error={last_error:.6f}:span={last_span:.6f}"
            )
        if stop_event.wait(min(poll_period, remaining)):
            raise WorkerStopped("stop_during_chunk_settle")


def run_session(
    *,
    config: Mapping[str, Any],
    hardware: Any,
    policy: Any,
    stop_event: Any,
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    execution = config["execution"]
    action_config = config["action"]
    execute = execution["mode"] == "execute"
    max_replans = int(execution["max_replans"])
    first_row = int(action_config["row_index"])
    actions_per_chunk = int(action_config["actions_per_chunk"])
    action_hz = float(action_config["action_hz"])
    runtime = config["runtime"]
    motion = config["motion"]
    settle_timeout_sec = float(runtime["chunk_settle_timeout_sec"])
    stable_window_sec = float(runtime["chunk_stable_window_sec"])
    stable_span_threshold_rad = float(runtime["chunk_stable_span_threshold_rad"])
    settle_poll_hz = float(runtime["chunk_settle_poll_hz"])
    target_tolerance_rad = float(motion["target_tolerance_rad"])
    command_count = 0
    gripper_transitions = 0
    settled_chunks = 0
    prior_gripper_targets = None

    for ordinal in range(1, max_replans + 1 if execute else 2):
        if stop_event.is_set():
            raise WorkerStopped("stop_before_observation")
        _emit(emit, "phase", phase="observe", timeout_sec=float(config["runtime"]["sync_timeout_sec"]) + 5.0)
        images, state = hardware.observe()
        if prior_gripper_targets is None:
            actual_grippers = {"left": float(state[7]), "right": float(state[21])}
            prior_gripper_targets = {}
            for side, value in actual_grippers.items():
                if not math.isfinite(value):
                    raise MotionRejected(f"initial_gripper_feedback_invalid:{side}:{value}")
                position = min(1000, max(1, int(round(min(1.0, max(0.0, value)) * 1000.0))))
                prior_gripper_targets[side] = {"state": "hold", "position": position}
            cached = getattr(hardware, "_last_grippers", None)
            if isinstance(cached, dict):
                for side in ("left", "right"):
                    if cached.get(side) is None:
                        cached[side] = prior_gripper_targets[side]["position"]
        observation = prepare_openpi_observation(images, state, config)
        _emit(
            emit,
            "phase",
            phase="inference",
            timeout_sec=float(config["policy"].get("request_timeout_sec", 300.0)) + 5.0,
            replan=ordinal,
        )
        response = policy.infer(observation)
        if not isinstance(response, Mapping) or "actions" not in response:
            raise MotionRejected("policy_response_missing_actions")
        canonical = adapt_action_chunk(response["actions"], action_config)
        _emit(emit, "inference_done", replan=ordinal, action_shape=list(canonical.shape))
        if not execute:
            return {
                "status": "no_motion_pass",
                "replans": 1,
                "policy_requests": 1,
                "robot_commands": 0,
                "gripper_commands": 0,
                "action_shape": list(canonical.shape),
            }

        stop_row = first_row + actions_per_chunk
        if stop_row > canonical.shape[0]:
            raise MotionRejected(
                f"configured_action_rows_missing:{first_row}:{stop_row}:{canonical.shape[0]}"
            )

        next_send_at = time.monotonic()
        last_hardware_action = None
        for action_row in range(first_row, stop_row):
            if stop_event.is_set():
                raise WorkerStopped("stop_before_action")
            row = canonical[action_row]
            raw_grippers = {"left": row[7], "right": row[15]}
            gripper_targets = gripper_targets_from_row(
                row,
                config["gripper"],
                prior_gripper_targets,
            )
            hardware_action = canonical_row_to_hardware_action(
                row,
                config,
                gripper_targets,
            )
            last_hardware_action = hardware_action
            transitions = sum(
                prior_gripper_targets[side]["position"] != gripper_targets[side]["position"]
                for side in ("left", "right")
            )
            _emit(
                emit,
                "phase",
                phase="action",
                timeout_sec=(
                    float(config["runtime"]["action_call_timeout_sec"])
                    + 1.0 / action_hz
                ),
                replan=ordinal,
                action_row=action_row,
                gripper=raw_grippers,
            )
            if stop_event.wait(max(0.0, next_send_at - time.monotonic())):
                raise WorkerStopped("stop_during_action_pacing")
            hardware.execute_action(hardware_action, dry_run=False)
            command_count += 1
            gripper_transitions += transitions
            prior_gripper_targets = gripper_targets
            _emit(
                emit,
                "action_done",
                replan=ordinal,
                action_row=action_row,
                gripper=raw_grippers,
            )
            # Do not burst commands to catch up after a slow SDK call.  The
            # next command is scheduled one period after this call returns.
            next_send_at = time.monotonic() + 1.0 / action_hz

        assert last_hardware_action is not None
        _emit(
            emit,
            "phase",
            phase="chunk_settle",
            timeout_sec=settle_timeout_sec,
            replan=ordinal,
            action_row=stop_row - 1,
        )
        settle_result = wait_for_chunk_completion(
            hardware=hardware,
            target=last_hardware_action,
            tolerance_rad=target_tolerance_rad,
            stable_window_sec=stable_window_sec,
            stable_span_threshold_rad=stable_span_threshold_rad,
            timeout_sec=settle_timeout_sec,
            poll_hz=settle_poll_hz,
            stop_event=stop_event,
        )
        settled_chunks += 1
        _emit(emit, "chunk_settle_done", replan=ordinal, **settle_result)
    return {
        "status": "completed",
        "replans": max_replans,
        "policy_requests": max_replans,
        "robot_commands": command_count,
        "gripper_commands": gripper_transitions,
        "settled_chunks": settled_chunks,
    }


def _safe_close(policy: Any, hardware: Any) -> list[str]:
    errors = []
    recorder = getattr(hardware, "_production_frame_recorder", None)
    if hardware is not None:
        try:
            hardware.close()
        except BaseException as exc:
            errors.append(f"hardware:{type(exc).__name__}:{exc}")
    if recorder is not None:
        try:
            recorder.close()
        except BaseException as exc:
            errors.append(f"recording:{type(exc).__name__}:{exc}")
    close = getattr(policy, "close", None)
    if close is not None:
        try:
            close()
        except BaseException as exc:
            errors.append(f"policy:{type(exc).__name__}:{exc}")
    return errors


def worker_entry(
    config: dict[str, Any],
    authorize_event: Any,
    return_home_event: Any,
    stop_event: Any,
    status_queue: Any,
) -> None:
    # The parent owns terminal Ctrl+C.  The worker observes stop_event so its
    # recorder can drain and close cleanly instead of being interrupted mid-file.
    if multiprocessing.current_process().name != "MainProcess":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    hardware = policy = None
    motion_started = False
    emit = status_queue.put
    try:
        _emit(emit, "phase", phase="startup", timeout_sec=float(config["runtime"]["startup_timeout_sec"]))
        hardware = _load_fastwam_hardware(config)
        gate = _validate_hardware_ready(hardware)
        policy, service = open_policy(
            config,
            request_budget=int(config["execution"]["max_replans"])
            if config["execution"]["mode"] == "execute"
            else 1,
        )
        _emit(emit, "ready", service=service, gate=gate)
        while not authorize_event.wait(0.1):
            if stop_event.is_set():
                raise WorkerStopped("stop_before_authorization")

        def tracking_emit(message: dict[str, Any]) -> None:
            nonlocal motion_started
            if (
                message.get("event") == "phase"
                and message.get("phase") in ("initial_pose", "action")
            ):
                motion_started = True
            status_queue.put(message)

        initial_pose_status = "not_applicable"
        if config["execution"]["mode"] == "execute":
            if return_home_event.is_set():
                tracking_emit(
                    {
                        "event": "phase",
                        "monotonic_ns": time.monotonic_ns(),
                        "phase": "initial_pose",
                        "timeout_sec": float(config["initial_pose"]["settle_timeout_sec"]) + 5.0,
                    }
                )
                initial_result = move_to_initial_pose(
                    hardware=hardware,
                    config=config,
                    stop_event=stop_event,
                )
                _emit(emit, "initial_pose_done", **initial_result)
                initial_pose_status = "reached"
            else:
                initial_pose_status = "skipped"

        recorder = getattr(hardware, "_production_frame_recorder", None)
        if recorder is not None:
            started_ns = recorder.start()
            _emit(emit, "recording_started", start_monotonic_ns=started_ns)

        result = run_session(
            config=config,
            hardware=hardware,
            policy=policy,
            stop_event=stop_event,
            emit=tracking_emit,
        )
        result["initial_pose"] = initial_pose_status
        close_errors = _safe_close(policy, hardware)
        policy = hardware = None
        if close_errors:
            raise MotionRejected(f"worker_cleanup_failed:{close_errors}")
        if recorder is not None:
            result["recording_outputs"] = recorder.output_files
        _emit(emit, "result", result=result)
    except WorkerStopped as exc:
        if motion_started and hardware is not None:
            try:
                hardware.stop()
            except BaseException:
                pass
        recorder = getattr(hardware, "_production_frame_recorder", None)
        close_errors = _safe_close(policy, hardware)
        policy = hardware = None
        if close_errors:
            _emit(
                emit,
                "error",
                error_type="WorkerCleanupError",
                reason=str(close_errors)[:800],
            )
        else:
            stopped_result = {"status": "stopped", "reason": str(exc)}
            if recorder is not None:
                stopped_result["recording_outputs"] = recorder.output_files
            _emit(emit, "result", result=stopped_result)
    except BaseException as exc:
        if motion_started and hardware is not None:
            try:
                hardware.stop()
            except BaseException:
                pass
        _emit(
            emit,
            "error",
            error_type=type(exc).__name__,
            reason=str(exc)[:800],
            traceback="".join(traceback.format_exception(exc))[-4000:],
        )
    finally:
        close_errors = _safe_close(policy, hardware)
        if close_errors:
            _emit(emit, "close_errors", errors=close_errors)


def emergency_stop_entry(config: dict[str, Any], status_queue: Any) -> None:
    source_root = str(Path(str(config["sources"]["deltawam_src"])).resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from realman_deployment.Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    left = RoboticArm(rm_thread_mode_e.RM_DUAL_MODE_E)
    right = RoboticArm()
    arms = {"left": left, "right": right}
    statuses = {}
    try:
        for side in ("left", "right"):
            item = config["robots"][side]
            handle = arms[side].rm_create_robot_arm(str(item["ip"]), int(item["port"]))
            if handle.id == -1:
                statuses[side] = "connection_failed"
            else:
                statuses[side] = int(arms[side].rm_set_arm_slow_stop())
    except BaseException as exc:
        statuses["error"] = f"{type(exc).__name__}:{exc}"
    finally:
        for arm in (right, left):
            try:
                if hasattr(arm, "handle"):
                    arm.rm_delete_robot_arm()
            except BaseException:
                pass
        try:
            RoboticArm.rm_destroy()
        except BaseException:
            pass
        status_queue.put(statuses)


def _run_emergency_stop(ctx: Any, config: dict[str, Any]) -> dict[str, Any]:
    result_queue = ctx.Queue()
    process = ctx.Process(target=emergency_stop_entry, args=(config, result_queue), daemon=True)
    process.start()
    process.join(timeout=float(config["runtime"]["emergency_stop_timeout_sec"]))
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)
        return {"status": "timeout"}
    try:
        return {"status": "completed", "sdk": result_queue.get_nowait()}
    except queue.Empty:
        return {"status": "completed_without_result"}


def prompt_return_to_initial_pose(
    input_fn: Callable[[str], str] | None = None,
) -> bool:
    prompt = input if input_fn is None else input_fn
    while True:
        answer = prompt("是否回到初始位姿？回车同意，n/N拒绝：").strip()
        if answer == "":
            return True
        if answer.lower() == "n":
            return False
        print("请输入回车同意，或 n/N 拒绝。")


def supervise(
    config: dict[str, Any],
    *,
    worker_target: Callable[[dict[str, Any], Any, Any, Any, Any], None] = worker_entry,
    input_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    ctx = multiprocessing.get_context("spawn")
    authorize_event = ctx.Event()
    return_home_event = ctx.Event()
    stop_event = ctx.Event()
    status_queue = ctx.Queue()
    worker = ctx.Process(
        target=worker_target,
        args=(config, authorize_event, return_home_event, stop_event, status_queue),
        name="pi05-realman-worker",
    )
    worker.start()
    phase = "startup"
    phase_deadline = time.monotonic() + float(config["runtime"]["startup_timeout_sec"])
    motion_started = False
    result = None
    terminal_error = None
    operator_stop = False
    emergency = None

    try:
        while worker.is_alive() or not status_queue.empty():
            try:
                message = status_queue.get(timeout=0.1)
            except queue.Empty:
                if time.monotonic() > phase_deadline:
                    raise WorkerTimeout(f"worker_phase_timeout:{phase}")
                continue
            event = message.get("event")
            if event == "phase":
                phase = str(message["phase"])
                phase_deadline = time.monotonic() + float(message["timeout_sec"])
                if phase in ("initial_pose", "action"):
                    motion_started = True
                LOGGER.info(_format_worker_message(message))
            elif event == "ready":
                LOGGER.info(
                    _format_worker_message({"event": "ready", **dict(message["service"])})
                )
                if config["execution"]["mode"] == "execute":
                    prompt = input if input_fn is None else input_fn
                    prompt("请回车确认，Ctrl+C 退出：")
                    if prompt_return_to_initial_pose(prompt):
                        return_home_event.set()
                authorize_event.set()
                phase = "authorized"
                phase_deadline = time.monotonic() + float(config["runtime"]["sync_timeout_sec"]) + 5.0
            elif event in (
                "initial_pose_done",
                "recording_started",
                "inference_done",
                "action_done",
                "chunk_settle_done",
            ):
                LOGGER.info(_format_worker_message(message))
            elif event == "result":
                result = dict(message["result"])
                break
            elif event == "error":
                terminal_error = message
                break
            elif event == "close_errors":
                LOGGER.error("Worker cleanup errors: %s", message["errors"])
        if result is None and terminal_error is None:
            terminal_error = {
                "error_type": "WorkerExit",
                "reason": f"worker exited with code {worker.exitcode}",
            }
    except KeyboardInterrupt:
        stop_event.set()
        terminal_error = None
        operator_stop = True
        if motion_started:
            emergency = _run_emergency_stop(ctx, config)
            sdk_result = emergency.get("sdk", {})
            if (
                emergency.get("status") != "completed"
                or sdk_result.get("left") != 0
                or sdk_result.get("right") != 0
            ):
                terminal_error = {
                    "error_type": "OperatorStopFailed",
                    "reason": str(emergency),
                }
    except WorkerTimeout as exc:
        stop_event.set()
        terminal_error = {"error_type": type(exc).__name__, "reason": str(exc) or "operator_stop"}
    finally:
        if terminal_error is not None:
            stop_event.set()
        shutdown_timeout = float(config["runtime"]["shutdown_grace_sec"])
        if config["recording"]["enabled"]:
            shutdown_timeout = max(
                shutdown_timeout,
                float(config["recording"]["shutdown_timeout_sec"]),
            )
        worker.join(timeout=shutdown_timeout)
        forced = worker.is_alive()
        if not forced and terminal_error is None:
            while True:
                try:
                    message = status_queue.get(timeout=0.1)
                except queue.Empty:
                    break
                if message.get("event") == "error":
                    terminal_error = message
                    break
                if message.get("event") == "result":
                    result = dict(message["result"])
                if message.get("event") == "close_errors":
                    terminal_error = {
                        "error_type": "WorkerCleanupError",
                        "reason": str(message.get("errors", "")),
                    }
                    break
        if forced:
            worker.terminate()
            worker.join(timeout=2.0)
            if terminal_error is None:
                terminal_error = {
                    "error_type": "WorkerCleanupTimeout",
                    "reason": f"hardware worker did not close within {shutdown_timeout:.1f} sec",
                }
        if emergency is None and motion_started and (forced or terminal_error is not None):
            emergency = _run_emergency_stop(ctx, config)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=1.0)
        if emergency is not None:
            LOGGER.warning("Emergency slow-stop result: %s", emergency)

    if terminal_error is not None:
        return {
            "status": "blocked",
            "reason": f"{terminal_error.get('error_type')}:{terminal_error.get('reason')}",
            "motion_started": motion_started,
        }
    if operator_stop:
        stopped_result = {} if result is None else dict(result)
        stopped_result.update(
            {
                "status": "stopped",
                "reason": "operator_stop",
                "motion_started": motion_started,
                "preset": config["_preset"],
            }
        )
        return stopped_result
    assert result is not None
    result["preset"] = config["_preset"]
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, args.preset)
        if not prepare_recording_destination(config):
            print(
                json.dumps(
                    {"status": "stopped", "reason": "recording_overwrite_declined"},
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        logging.basicConfig(
            level=getattr(logging, str(config["execution"]["log_level"])),
            format="%(message)s",
        )
        result = supervise(config)
        stream = sys.stderr if result.get("status") == "blocked" else sys.stdout
        print(json.dumps(result, indent=2, ensure_ascii=False), file=stream)
        return 1 if result.get("status") == "blocked" else 0
    except BaseException as exc:
        print(
            json.dumps(
                {"status": "blocked", "reason": f"{type(exc).__name__}:{exc}"},
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
