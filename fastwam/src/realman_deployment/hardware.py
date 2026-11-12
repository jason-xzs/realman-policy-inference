"""RealMan/r3kit adapter plus a deterministic hardware simulator."""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from .protocol import ACTION_DIM, CAMERA_NAMES, RAW_STATE_DIM, ProtocolError

logger = logging.getLogger(__name__)


class HardwareError(RuntimeError):
    pass


class SafetyError(HardwareError):
    pass


@dataclass(frozen=True)
class TimedFrame:
    sequence: int
    timestamp_ns: int
    image: Any


def orient_deployment_images(images: dict[str, Any]) -> dict[str, Any]:
    """Match the camera orientation used by the RealMan training recorder."""
    if set(images) != set(CAMERA_NAMES):
        raise HardwareError(f"images must contain exactly {list(CAMERA_NAMES)}")
    return {
        "high_rgb": images["high_rgb"],
        "left_wrist_rgb": images["left_wrist_rgb"][::-1, ::-1].copy(),
        "right_wrist_rgb": images["right_wrist_rgb"][::-1, ::-1].copy(),
    }


def select_synchronized_frames(
    buffers: dict[str, list[TimedFrame]],
    target_ns: int,
    max_skew_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose the frame tuple whose timestamps best match the robot-state time."""
    missing = [name for name in CAMERA_NAMES if not buffers.get(name)]
    if missing:
        raise HardwareError(f"camera buffers are empty: {missing}")

    best = min(
        product(*(buffers[name] for name in CAMERA_NAMES)),
        key=lambda frames: (
            max(abs(frame.timestamp_ns - target_ns) for frame in frames),
            max([target_ns, *(frame.timestamp_ns for frame in frames)])
            - min([target_ns, *(frame.timestamp_ns for frame in frames)]),
            -min(frame.timestamp_ns for frame in frames),
        ),
    )
    timestamps = {name: frame.timestamp_ns for name, frame in zip(CAMERA_NAMES, best)}
    all_timestamps = [target_ns, *timestamps.values()]
    skew_ns = max(all_timestamps) - min(all_timestamps)
    if skew_ns > max_skew_ns:
        offsets_ms = {
            name: round((timestamp - target_ns) / 1_000_000.0, 3)
            for name, timestamp in timestamps.items()
        }
        raise HardwareError(
            f"camera/state synchronization skew {skew_ns / 1_000_000.0:.3f} ms exceeds "
            f"limit {max_skew_ns / 1_000_000.0:.3f} ms; offsets={offsets_ms}"
        )
    return (
        {name: frame.image for name, frame in zip(CAMERA_NAMES, best)},
        {
            "target_timestamp_ns": int(target_ns),
            "camera_timestamp_ns": timestamps,
            "camera_sequence": {
                name: frame.sequence for name, frame in zip(CAMERA_NAMES, best)
            },
            "skew_ms": skew_ns / 1_000_000.0,
        },
    )


class BufferedCameraRig:
    """Continuously capture independent cameras and select a time-aligned tuple."""

    def __init__(
        self,
        cameras: dict[str, Any],
        *,
        max_skew_ms: float = 50.0,
        timeout_s: float = 8.0,
        buffer_size: int = 16,
        frame_recorder=None,
    ) -> None:
        if set(cameras) != set(CAMERA_NAMES):
            raise ValueError(f"cameras must contain exactly {list(CAMERA_NAMES)}")
        if max_skew_ms <= 0 or timeout_s <= 0 or buffer_size < 2:
            raise ValueError("camera sync skew/timeout must be positive and buffer_size must be >= 2")
        self.cameras = cameras
        self.max_skew_ns = int(max_skew_ms * 1_000_000)
        self.timeout_s = float(timeout_s)
        self.frame_recorder = frame_recorder
        self._buffers = {name: deque(maxlen=buffer_size) for name in CAMERA_NAMES}
        self._last_sequences = {name: -1 for name in CAMERA_NAMES}
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._errors: dict[str, BaseException] = {}
        self._threads = []
        for name in CAMERA_NAMES:
            thread = threading.Thread(
                target=self._capture_loop,
                args=(name, cameras[name]),
                name=f"camera-{name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _capture_loop(self, name: str, camera: Any) -> None:
        sequence = 0
        while not self._stop_event.is_set():
            try:
                image = camera.get()[0]
                captured_ns = time.monotonic_ns()
                frame = TimedFrame(sequence, captured_ns, image.copy())
                if self.frame_recorder is not None:
                    self.frame_recorder.submit(name, frame.image, sequence=sequence, timestamp_ns=captured_ns)
                sequence += 1
                with self._condition:
                    self._buffers[name].append(frame)
                    self._condition.notify_all()
            except BaseException as exc:
                if self._stop_event.is_set():
                    break
                logger.exception("Camera capture failed: %s", name)
                with self._condition:
                    self._errors[name] = exc
                    self._condition.notify_all()
                break

    def wait_until_ready(self, timeout_s: float = 5.0) -> None:
        """Wait for every capture thread to produce a frame before sampling robot state."""
        if timeout_s <= 0:
            raise ValueError("camera warm-up timeout must be positive")
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while True:
                if self._errors:
                    name, error = next(iter(self._errors.items()))
                    raise HardwareError(f"camera {name} failed during warm-up: {error}") from error
                if all(self._buffers[name] for name in CAMERA_NAMES):
                    logger.info(
                        "Camera rig warm-up complete: sequences=%s",
                        {name: self._buffers[name][-1].sequence for name in CAMERA_NAMES},
                    )
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HardwareError("timed out warming up all three cameras")
                self._condition.wait(timeout=remaining)

    def capture_near(self, target_ns: int) -> tuple[dict[str, Any], dict[str, Any]]:
        deadline = time.monotonic() + self.timeout_s
        last_error: HardwareError | None = None
        with self._condition:
            while True:
                if self._errors:
                    name, error = next(iter(self._errors.items()))
                    raise HardwareError(f"camera {name} capture thread failed: {error}") from error
                eligible = {
                    name: [
                        frame for frame in self._buffers[name]
                        if frame.sequence > self._last_sequences[name]
                    ]
                    for name in CAMERA_NAMES
                }
                # Wait for at least one post-state frame from every camera. Keeping
                # pre-state frames in the buffer lets the selector choose either side.
                has_post_state = all(
                    eligible[name] and eligible[name][-1].timestamp_ns >= target_ns
                    for name in CAMERA_NAMES
                )
                if has_post_state:
                    try:
                        images, metadata = select_synchronized_frames(
                            eligible, target_ns, self.max_skew_ns
                        )
                        for name in CAMERA_NAMES:
                            self._last_sequences[name] = metadata["camera_sequence"][name]
                        return images, metadata
                    except HardwareError as exc:
                        last_error = exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if last_error is not None:
                        raise HardwareError(f"timed out waiting for synchronized frames: {last_error}")
                    raise HardwareError("timed out waiting for frames from all three cameras")
                self._condition.wait(timeout=remaining)

    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        for thread in self._threads:
            thread.join(timeout=1.0)


def validate_joint_step(current_left, current_right, action, max_joint_step_rad: float) -> float:
    import numpy as np

    target = np.asarray(action, dtype=np.float64)
    left = np.asarray(current_left, dtype=np.float64)
    right = np.asarray(current_right, dtype=np.float64)
    if target.shape != (ACTION_DIM,) or left.shape != (7,) or right.shape != (7,):
        raise SafetyError("invalid current/target joint dimensions")
    if not np.isfinite(target).all() or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise SafetyError("current/target joint data contains NaN or infinity")
    max_delta = max(float(np.max(np.abs(target[0:7] - right))),
                    float(np.max(np.abs(target[8:15] - left))))
    if max_delta > float(max_joint_step_rad):
        raise SafetyError(
            f"joint target jump {max_delta:.4f} rad exceeds --max-joint-step-rad "
            f"{float(max_joint_step_rad):.4f}; action rejected"
        )
    return max_delta

class BinaryGripperHysteresis:
    """Convert continuous 0-1000 policy values into one-shot open/close commands."""

    def __init__(
        self,
        *,
        close_threshold: float,
        open_threshold: float,
        confirm_actions: int,
        confirm_seconds: float,
    ) -> None:
        def per_side(value, cast, name):
            if isinstance(value, dict):
                if set(value) != {"left", "right"}:
                    raise ValueError(f"{name} dict must contain left and right")
                return {side: cast(value[side]) for side in ("left", "right")}
            return {side: cast(value) for side in ("left", "right")}

        self.close_threshold = per_side(close_threshold, float, "close_threshold")
        self.open_threshold = per_side(open_threshold, float, "open_threshold")
        self.confirm_actions = per_side(confirm_actions, int, "confirm_actions")
        self.confirm_seconds = per_side(confirm_seconds, float, "confirm_seconds")
        for side in ("left", "right"):
            if not 0.0 <= self.close_threshold[side] < self.open_threshold[side] <= 1000.0:
                raise ValueError(f"{side} gripper thresholds must satisfy 0 <= close < open <= 1000")
            if self.confirm_actions[side] < 0 or self.confirm_seconds[side] < 0:
                raise ValueError(f"{side} gripper confirmation actions/seconds must be non-negative")
            if self.confirm_actions[side] == 0 and self.confirm_seconds[side] == 0:
                raise ValueError(f"enable {side} gripper confirmation by actions, seconds, or both")
        self._states = {
            side: {"commanded": None, "candidate": None, "count": 0, "since": None}
            for side in ("left", "right")
        }

    def update(self, side: str, value: float, *, now: float | None = None) -> int | None:
        if side not in self._states or not math.isfinite(float(value)):
            raise ValueError(f"invalid gripper sample side={side!r} value={value!r}")
        now = time.monotonic() if now is None else float(now)
        desired = "closed" if value <= self.close_threshold[side] else (
            "open" if value >= self.open_threshold[side] else None
        )
        state = self._states[side]
        if desired is None or desired == state["commanded"]:
            state.update(candidate=None, count=0, since=None)
            return None
        if desired != state["candidate"]:
            state.update(candidate=desired, count=1, since=now)
        else:
            state["count"] += 1
        actions_ready = self.confirm_actions[side] > 0 and state["count"] >= self.confirm_actions[side]
        time_ready = self.confirm_seconds[side] > 0 and now - state["since"] >= self.confirm_seconds[side]
        if not (actions_ready or time_ready):
            return None
        state.update(commanded=desired, candidate=None, count=0, since=None)
        return 1 if desired == "closed" else 1000

class MockRealManHardware:
    def __init__(self, height: int = 480, width: int = 640):
        import numpy as np

        self.np = np
        self.images = {
            name: np.full((height, width, 3), 30 + index * 60, dtype=np.uint8)
            for index, name in enumerate(CAMERA_NAMES)
        }
        self.left_joints = np.zeros(7, dtype=np.float64)
        self.right_joints = np.zeros(7, dtype=np.float64)
        self.left_gripper = 1.0
        self.right_gripper = 1.0
        self.executed: list[list[float]] = []
        self.stopped = False
        self.closed = False

    def observe(self) -> tuple[dict[str, Any], list[float]]:
        left_pose = [0.0] * 6
        right_pose = [0.0] * 6
        state = (self.left_joints.tolist() + [self.left_gripper] + left_pose
                 + self.right_joints.tolist() + [self.right_gripper] + right_pose)
        return {key: value.copy() for key, value in self.images.items()}, state

    def current_joints(self):
        return self.left_joints.copy(), self.right_joints.copy()

    def wait_for_joint_target(
        self, action, *, tolerance_rad: float, timeout_s: float,
        resend_hz: float, stop_event=None,
    ):
        if tolerance_rad <= 0 or timeout_s <= 0 or resend_hz <= 0:
            raise ValueError("tracking gate tolerance, timeout, and resend_hz must be positive")
        return {
            "final_error_rad": 0.0,
            "elapsed_s": 0.0,
            "checks": 1,
            "resends": 0,
        }

    def execute_action(self, action: list[float], **_: Any) -> None:
        if len(action) != ACTION_DIM:
            raise ProtocolError(f"action must have {ACTION_DIM} values")
        self.right_joints[:] = action[0:7]
        self.right_gripper = action[7] / 1000.0
        self.left_joints[:] = action[8:15]
        self.left_gripper = action[15] / 1000.0
        self.executed.append(list(action))

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class RealManHardware:
    """Three D435 cameras and two 7-DoF RealMan arms.

    Model/raw-state order follows the training data: left arm first in the 28-D
    observation, while model actions are right arm first in the 16-D output.
    """

    def __init__(
        self,
        *,
        left_ip: str,
        right_ip: str,
        arm_port: int,
        high_camera_id: str,
        left_camera_id: str,
        right_camera_id: str,
        max_joint_step_rad: float = 0.15,
        trajectory_mode: int = 2,
        smoothing: int = 50,
        gripper_threshold: int = 5,
        gripper_mode: str = "position",
        gripper_close_threshold: float = 300.0,
        gripper_open_threshold: float = 700.0,
        left_gripper_close_threshold: float | None = None,
        right_gripper_close_threshold: float | None = None,
        left_gripper_confirm_actions: int | None = None,
        right_gripper_confirm_actions: int | None = None,
        frame_recorder=None,
        gripper_before_arm: bool = False,
        gripper_blocking_timeout_s: float = 5.0,
        gripper_confirm_actions: int = 3,
        gripper_confirm_seconds: float = 0.0,
        max_sync_skew_ms: float = 50.0,
        sync_timeout_s: float = 8.0,
    ) -> None:
        deployment_dir = Path(__file__).resolve().parent
        r3kit_root = deployment_dir / "r3kit"
        if str(r3kit_root) not in sys.path:
            sys.path.insert(0, str(r3kit_root))
        try:
            from .Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
            from r3kit.devices.camera.realsense.d435 import D435
        except (ImportError, OSError) as exc:
            raise HardwareError(
                "RealMan/r3kit dependencies failed to load. Install pyrealsense2 and use the matching "
                f"Robotic_Arm native library: {exc}"
            ) from exc

        self._RoboticArm = RoboticArm
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.trajectory_mode = int(trajectory_mode)
        self.smoothing = int(smoothing)
        self.gripper_threshold = int(gripper_threshold)
        if gripper_mode not in ("position", "binary-hysteresis"):
            raise ValueError("gripper_mode must be position or binary-hysteresis")
        self.gripper_mode = gripper_mode
        self.binary_grippers = BinaryGripperHysteresis(
            close_threshold={
                "left": gripper_close_threshold if left_gripper_close_threshold is None
                else left_gripper_close_threshold,
                "right": gripper_close_threshold if right_gripper_close_threshold is None
                else right_gripper_close_threshold,
            },
            open_threshold=gripper_open_threshold,
            confirm_actions={
                "left": gripper_confirm_actions if left_gripper_confirm_actions is None
                else left_gripper_confirm_actions,
                "right": gripper_confirm_actions if right_gripper_confirm_actions is None
                else right_gripper_confirm_actions,
            },
            confirm_seconds=gripper_confirm_seconds,
        ) if gripper_mode == "binary-hysteresis" else None

        if gripper_blocking_timeout_s <= 0:
            raise ValueError("gripper_blocking_timeout_s must be positive")
        self.gripper_before_arm = bool(gripper_before_arm)
        self.gripper_blocking_timeout_s = float(gripper_blocking_timeout_s)
        self._last_grippers: dict[str, int | None] = {"left": None, "right": None}
        self._lock = threading.Lock()
        self._closed = False
        self._arm_state_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="arm-state")
        self.last_sync_metadata: dict[str, Any] | None = None
        self._observation_count = 0
        self._executed_action_count = 0

        self.left_arm = RoboticArm(rm_thread_mode_e.RM_DUAL_MODE_E)
        self.right_arm = RoboticArm()
        left_handle = self.left_arm.rm_create_robot_arm(left_ip, arm_port)
        right_handle = self.right_arm.rm_create_robot_arm(right_ip, arm_port)
        if left_handle.id == -1 or right_handle.id == -1:
            self.close()
            raise HardwareError(f"arm connection failed (left id={left_handle.id}, right id={right_handle.id})")

        # Explicit serials are mandatory so views cannot silently swap between boots.
        serials = [high_camera_id, left_camera_id, right_camera_id]
        if any(not value for value in serials) or len(set(serials)) != 3:
            self.close()
            raise HardwareError("three distinct RealSense serial numbers are required")
        self.cameras = {}
        try:
            for name, camera_id in zip(CAMERA_NAMES, serials):
                self.cameras[name] = D435(
                    id=camera_id, depth=False, name=name,
                    color_width=640, color_height=480, color_fps=30,
                )
            self.camera_rig = BufferedCameraRig(
                self.cameras,
                frame_recorder=frame_recorder,
                max_skew_ms=max_sync_skew_ms,
                timeout_s=sync_timeout_s,
            )
            self.camera_rig.wait_until_ready(timeout_s=max(5.0, sync_timeout_s))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if code != 0:
            raise HardwareError(f"{operation} failed with SDK status {code}")

    def _arm_state(self, arm, side: str) -> tuple[list[float], list[float], float]:
        import numpy as np

        code, state = arm.rm_get_current_arm_state()
        self._check(code, f"read {side} arm state")
        code, gripper = arm.rm_get_gripper_state()
        self._check(code, f"read {side} gripper state")
        joints_rad = np.deg2rad(np.asarray(state["joint"], dtype=np.float64)).tolist()
        if len(joints_rad) != 7 or len(state["pose"]) != 6:
            raise HardwareError(f"{side} arm returned unexpected state dimensions")
        return joints_rad, [float(x) for x in state["pose"]], float(gripper["actpos"]) / 1000.0

    def _timed_arm_state(self, arm, side: str):
        started_ns = time.monotonic_ns()
        state = self._arm_state(arm, side)
        finished_ns = time.monotonic_ns()
        return state, (started_ns + finished_ns) // 2

    def observe(self) -> tuple[dict[str, Any], list[float]]:
        with self._lock:
            left_future = self._arm_state_executor.submit(self._timed_arm_state, self.left_arm, "left")
            right_future = self._arm_state_executor.submit(self._timed_arm_state, self.right_arm, "right")
            (left_joints, left_pose, left_gripper), left_timestamp_ns = left_future.result()
            (right_joints, right_pose, right_gripper), right_timestamp_ns = right_future.result()
        state_timestamp_ns = (left_timestamp_ns + right_timestamp_ns) // 2
        images, sync_metadata = self.camera_rig.capture_near(state_timestamp_ns)
        images = orient_deployment_images(images)
        sync_metadata["left_arm_offset_ms"] = (left_timestamp_ns - state_timestamp_ns) / 1_000_000.0
        sync_metadata["right_arm_offset_ms"] = (right_timestamp_ns - state_timestamp_ns) / 1_000_000.0
        all_timestamps = [
            left_timestamp_ns,
            right_timestamp_ns,
            *sync_metadata["camera_timestamp_ns"].values(),
        ]
        overall_skew_ns = max(all_timestamps) - min(all_timestamps)
        if overall_skew_ns > self.camera_rig.max_skew_ns:
            raise HardwareError(
                f"camera/dual-arm synchronization skew {overall_skew_ns / 1_000_000.0:.3f} ms "
                f"exceeds limit {self.camera_rig.max_skew_ns / 1_000_000.0:.3f} ms"
            )
        sync_metadata["overall_skew_ms"] = overall_skew_ns / 1_000_000.0
        self.last_sync_metadata = sync_metadata
        self._observation_count += 1
        logger.info(
            "Synchronized observation #%d: overall_skew=%.3f ms camera_offsets_ms=%s",
            self._observation_count,
            sync_metadata["overall_skew_ms"],
            {
                name: round((timestamp - state_timestamp_ns) / 1_000_000.0, 3)
                for name, timestamp in sync_metadata["camera_timestamp_ns"].items()
            },
        )
        state = left_joints + [left_gripper] + left_pose + right_joints + [right_gripper] + right_pose
        if len(state) != RAW_STATE_DIM or not all(math.isfinite(x) for x in state):
            raise HardwareError("hardware observation contains invalid state values")
        return images, state

    def _current_joints(self) -> tuple[Any, Any]:
        """Read only qpos for the per-action safety check (no camera/gripper I/O)."""
        import numpy as np

        values = []
        with self._lock:
            for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
                code, state = arm.rm_get_current_arm_state()
                self._check(code, f"read {side} arm state")
                joints = np.deg2rad(np.asarray(state["joint"], dtype=np.float64))
                if joints.shape != (7,) or not np.isfinite(joints).all():
                    raise HardwareError(f"{side} arm returned invalid joint data")
                values.append(joints)
        return values[0], values[1]

    def current_joints(self):
        return self._current_joints()

    def wait_for_joint_target(
        self, action, *, tolerance_rad: float, timeout_s: float,
        resend_hz: float, stop_event=None,
    ):
        import numpy as np

        if tolerance_rad <= 0 or timeout_s <= 0 or resend_hz <= 0:
            raise ValueError("tracking gate tolerance, timeout, and resend_hz must be positive")
        target = np.asarray(action, dtype=np.float64)
        if target.shape != (ACTION_DIM,) or not np.isfinite(target).all():
            raise SafetyError("invalid target for tracking gate")
        target_right = target[0:7]
        target_left = target[8:15]
        right_deg = np.rad2deg(target_right).tolist()
        left_deg = np.rad2deg(target_left).tolist()
        period = 1.0 / float(resend_hz)
        started = time.monotonic()
        next_check = started + period
        checks = 0
        resends = 0
        while True:
            wait_s = max(0.0, next_check - time.monotonic())
            if stop_event is not None:
                if stop_event.wait(wait_s):
                    raise HardwareError("tracking gate interrupted by stop request")
            elif wait_s > 0:
                time.sleep(wait_s)
            current_left, current_right = self._current_joints()
            right_error = np.abs(target_right - current_right)
            left_error = np.abs(target_left - current_left)
            max_error = max(float(np.max(right_error)), float(np.max(left_error)))
            checks += 1
            elapsed = time.monotonic() - started
            if max_error <= tolerance_rad:
                return {
                    "final_error_rad": max_error,
                    "elapsed_s": elapsed,
                    "checks": checks,
                    "resends": resends,
                }
            if elapsed >= timeout_s:
                right_index = int(np.argmax(right_error))
                left_index = int(np.argmax(left_error))
                if right_error[right_index] >= left_error[left_index]:
                    detail = f"right joint {right_index + 1}"
                else:
                    detail = f"left joint {left_index + 1}"
                raise SafetyError(
                    f"tracking gate timeout after {elapsed:.3f}s: max error "
                    f"{max_error:.4f} rad at {detail} exceeds {tolerance_rad:.4f}"
                )
            with self._lock:
                self._check(self.right_arm.rm_movej_canfd(
                    right_deg, False, trajectory_mode=self.trajectory_mode, radio=self.smoothing
                ), "resend right arm tracking target")
                self._check(self.left_arm.rm_movej_canfd(
                    left_deg, False, trajectory_mode=self.trajectory_mode, radio=self.smoothing
                ), "resend left arm tracking target")
            resends += 1
            next_check = max(next_check + period, time.monotonic())

    def execute_action(self, action: list[float], *, dry_run: bool = False) -> None:
        import numpy as np

        target = np.asarray(action, dtype=np.float64)
        current_left, current_right = self._current_joints()
        max_delta = validate_joint_step(
            current_left, current_right, target, self.max_joint_step_rad
        )
        target_right = target[0:7]
        target_left = target[8:15]
        binary_commands = []
        if self.binary_grippers is not None:
            sample_time = time.monotonic()
            for side, value in (("right", target[7]), ("left", target[15])):
                position = self.binary_grippers.update(side, float(value), now=sample_time)
                if position is not None:
                    binary_commands.append((side, position))
        if dry_run:
            preview = ",".join(f"{side}={position}" for side, position in binary_commands)
            logger.info("dry-run action accepted; max joint delta %.4f rad; grippers=%s",
                        max_delta, preview or "unchanged")
            return

        right_deg = np.rad2deg(target_right).tolist()
        left_deg = np.rad2deg(target_left).tolist()
        if self.binary_grippers is not None:
            gripper_targets = [
                (side, self.right_arm if side == "right" else self.left_arm, position)
                for side, position in binary_commands
            ]
        else:
            gripper_targets = []
            for side, arm, value in (
                ("right", self.right_arm, target[7]), ("left", self.left_arm, target[15])
            ):
                position = int(round(float(np.clip(value, 1.0, 1000.0))))
                previous = self._last_grippers[side]
                if previous is None or abs(position - previous) >= self.gripper_threshold:
                    gripper_targets.append((side, arm, position))
        gripper_updates = []

        def send_grippers(blocking: bool) -> None:
            timeout = max(1, int(math.ceil(self.gripper_blocking_timeout_s))) if blocking else 0
            for side, arm, position in gripper_targets:
                self._check(
                    arm.rm_set_gripper_position(position, blocking, timeout),
                    f"move {side} gripper",
                )
                self._last_grippers[side] = position
                gripper_updates.append(f"{side}={position}")

        with self._lock:
            if self.gripper_before_arm and gripper_targets:
                send_grippers(blocking=True)
            self._check(self.right_arm.rm_movej_canfd(
                right_deg, False, trajectory_mode=self.trajectory_mode, radio=self.smoothing
            ), "move right arm")
            self._check(self.left_arm.rm_movej_canfd(
                left_deg, False, trajectory_mode=self.trajectory_mode, radio=self.smoothing
            ), "move left arm")
            if not self.gripper_before_arm and gripper_targets:
                send_grippers(blocking=False)
        self._executed_action_count += 1
        logger.info(
            "executed hardware action #%d; max joint delta %.4f rad; grippers=%s; order=%s",
            self._executed_action_count, max_delta,
            ",".join(gripper_updates) if gripper_updates else "unchanged",
            "gripper-before-arm" if self.gripper_before_arm else "arm-before-gripper",
        )

    def stop(self) -> None:
        for side, arm in (("left", getattr(self, "left_arm", None)),
                          ("right", getattr(self, "right_arm", None))):
            if arm is not None and hasattr(arm, "handle"):
                try:
                    arm.rm_set_arm_slow_stop()
                except Exception:
                    logger.exception("Failed to stop %s arm", side)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        camera_rig = getattr(self, "camera_rig", None)
        if camera_rig is not None:
            camera_rig.close()
        executor = getattr(self, "_arm_state_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for camera in getattr(self, "cameras", {}).values():
            try:
                if hasattr(camera, "close"):
                    camera.close()
                else:
                    camera.pipeline.stop()
            except Exception:
                logger.exception("Failed to close camera")
        for arm in (getattr(self, "left_arm", None), getattr(self, "right_arm", None)):
            if arm is not None and hasattr(arm, "handle"):
                try:
                    arm.rm_delete_robot_arm()
                except Exception:
                    logger.exception("Failed to disconnect arm")
        if hasattr(self, "left_arm"):
            try:
                self._RoboticArm.rm_destroy()
            except Exception:
                logger.exception("Failed to destroy RealMan SDK")
