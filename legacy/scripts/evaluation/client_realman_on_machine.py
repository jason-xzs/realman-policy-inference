#!/usr/bin/env python3
"""RealMan client for LingBot-VA real-robot inference.

Default mode is dry-run. The robot only moves when --enable-robot is provided.
"""

import argparse
import inspect
import json
import logging
import math
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import msgpack
import numpy as np

try:
    from websockets.sync.client import connect
except Exception as exc:  # pragma: no cover
    connect = None
    WEBSOCKETS_IMPORT_ERROR = exc
else:
    WEBSOCKETS_IMPORT_ERROR = None


PROTOCOL_VERSION = "realman_va_socket_v1"
ACTION_SPACE = "realman8_abs_joint_deg_gripper"
CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_wrist",
]
DEFAULT_CAMERA_SPECS = {
    "observation.images.cam_high": os.getenv("REAL_HIGH_CAMERA_SERIAL", "036522071395"),
    "observation.images.cam_wrist": os.getenv("REAL_WRIST_CAMERA_SERIAL", "323622272614"),
}
DEFAULT_START_POS_DEG = np.asarray(
    [3.15, -2.35, 7.65, 69.75, 175.52, -89.94, 96.94],
    dtype=np.float32)
DEFAULT_STOP_FILE = "/tmp/realman_va_stop"
DEFAULT_DATASET_PATH = Path(os.getenv(
    "REAL_DATASET_PATH",
    "/data1/public_data/xzs_data/real_data/lerobot",
))
DEFAULT_ACTION_STATS_PATH = DEFAULT_DATASET_PATH / "meta" / "action_norm_stats.json"


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def websocket_timeout_value(value):
    value = float(value)
    return None if value <= 0 else value


def pack_message(obj):
    return msgpack.packb(obj, use_bin_type=True)


def unpack_message(data):
    return msgpack.unpackb(data, raw=False)


def setup_logging(save_rollout_dir):
    log_dir = Path(save_rollout_dir or Path(__file__).resolve().parent / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"client_realman_{stamp}.log"
    events_path = log_dir / f"client_realman_{stamp}.jsonl"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path, events_path


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.fp = self.path.open("a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, event):
        event = dict(event)
        event.setdefault("wall_time", time.time())
        with self.lock:
            self.fp.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.fp.flush()

    def close(self):
        self.fp.close()


def finite_array(value, shape, name):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != tuple(shape):
        raise ValueError(f"{name} shape mismatch: got {arr.shape}, expected {shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def encode_bgr_to_jpeg(bgr, quality=85):
    bgr = np.asarray(bgr)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 BGR, got {bgr.shape}")
    ok, encoded = cv2.imencode(".jpg", bgr.astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


class MockObservationSource:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.state = np.zeros(8, dtype=np.float32)
        self.state[:7] = DEFAULT_START_POS_DEG
        self.state[7] = 1.0

    def read(self):
        images = {}
        for index, key in enumerate(CAMERA_KEYS):
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            img[..., index] = 80
            cv2.putText(img, key.split(".")[-1], (24, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                        cv2.LINE_AA)
            images[key] = img
        return self.state.copy(), images

    def close(self):
        return None


@dataclass
class CameraReadConfig:
    height: int
    width: int
    fps: float
    read_retries: int
    retry_sleep: float
    read_timeout_ms: int
    max_stale_seconds: float
    unavailable_timeout: float
    max_consecutive_failures: int
    reconnect_backoff: float
    reconnect_max_backoff: float
    start_timeout: float
    status_log_interval: float
    allow_stale_frame: bool

    @property
    def period(self):
        return 1.0 / self.fps if self.fps > 0 else 0.0


def classify_camera_error(exc):
    message = str(exc)
    lowered = message.lower()
    if "frame didn't arrive" in lowered or "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "empty frame" in lowered or "empty" in lowered or "none" in lowered:
        return "empty"
    return "exception"


def close_camera_object(camera):
    close_fn = getattr(camera, "close", None) or getattr(camera, "release", None)
    if close_fn:
        close_fn()


class CameraCaptureWorker:
    def __init__(self, key, camera_factory, frame_reader, close_camera, config):
        self.key = key
        self.camera_factory = camera_factory
        self.frame_reader = frame_reader
        self.close_camera = close_camera
        self.config = config
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.camera_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reconnect_event = threading.Event()
        self.thread = None
        self.camera = None
        self.frame = None
        self.frame_time = 0.0
        self.frame_id = 0
        self.delivered_frame_id = -1
        self.capture_times = deque(maxlen=200)
        self.total_failures = 0
        self.timeout_failures = 0
        self.empty_failures = 0
        self.exception_failures = 0
        self.consecutive_failures = 0
        self.stale_fallbacks = 0
        self.cache_reuses = 0
        self.reconnect_count = 0
        self.open_failures = 0
        self.last_error = ""
        self.last_error_kind = ""
        self.last_status_log = 0.0
        self.last_stale_log = 0.0
        self.next_backoff = max(0.0, self.config.reconnect_backoff)

    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name=f"camera-capture-{self.key.split('.')[-1]}",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.reconnect_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.config.reconnect_max_backoff + 1.0))
        self._close_camera()

    def _normalize_image(self, image):
        image = np.asarray(image)
        if image.size == 0:
            raise RuntimeError("empty frame")
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"frame must be HxWx3, got {image.shape}")
        image = image.astype(np.uint8, copy=False)
        if image.shape[:2] != (self.config.height, self.config.width):
            image = cv2.resize(image, (self.config.width, self.config.height))
        return image

    def _open_camera(self):
        with self.camera_lock:
            self.camera = self.camera_factory()
        logging.info("camera opened key=%s target_fps=%s timeout_ms=%s", self.key,
                     self.config.fps, self.config.read_timeout_ms)

    def _close_camera(self):
        acquired = self.camera_lock.acquire(timeout=2.0)
        if not acquired:
            logging.warning("camera close skipped key=%s reason=camera_lock_timeout",
                            self.key)
            return
        try:
            camera = self.camera
            self.camera = None
            if camera is not None:
                try:
                    self.close_camera(camera)
                except Exception as exc:
                    logging.warning("camera close failed key=%s error=%s", self.key,
                                    exc)
        finally:
            self.camera_lock.release()

    def _request_reconnect(self, reason):
        if not self.reconnect_event.is_set():
            logging.warning("camera reconnect requested key=%s reason=%s", self.key,
                            reason)
        self.reconnect_event.set()

    def _reconnect(self, reason, initial=False):
        self.reconnect_event.clear()
        backoff = 0.0 if initial else self.next_backoff
        with self.condition:
            self.reconnect_count += 1
            last_age = time.time() - self.frame_time if self.frame is not None else None
        logging.warning(
            "camera reconnect key=%s reason=%s backoff=%.2fs last_valid_age=%s",
            self.key, reason, backoff,
            "none" if last_age is None else f"{last_age:.3f}s")
        if backoff > 0 and self.stop_event.wait(backoff):
            return
        self._close_camera()
        try:
            self._open_camera()
            self.next_backoff = max(0.0, self.config.reconnect_backoff)
        except Exception as exc:
            with self.condition:
                self.open_failures += 1
                self.consecutive_failures += 1
                self.total_failures += 1
                self.exception_failures += 1
                self.last_error = str(exc)
                self.last_error_kind = "open"
                self.condition.notify_all()
            logging.warning("camera open failed key=%s error=%s", self.key, exc)
            base = max(0.1, self.config.reconnect_backoff)
            self.next_backoff = min(
                max(base, self.next_backoff * 2 if self.next_backoff else base),
                self.config.reconnect_max_backoff,
            )
            self.reconnect_event.set()

    def _record_failure(self, kind, exc, attempt):
        with self.condition:
            self.total_failures += 1
            self.consecutive_failures += 1
            if kind == "timeout":
                self.timeout_failures += 1
            elif kind == "empty":
                self.empty_failures += 1
            else:
                self.exception_failures += 1
            self.last_error = str(exc)
            self.last_error_kind = kind
            last_age = time.time() - self.frame_time if self.frame is not None else None
            failures = self.consecutive_failures
            self.condition.notify_all()
        if failures == 1 or failures % max(1, self.config.max_consecutive_failures) == 0:
            logging.warning(
                "camera read failed key=%s kind=%s attempt=%s consecutive=%s "
                "timeout_ms=%s last_valid_age=%s error=%s",
                self.key, kind, attempt, failures, self.config.read_timeout_ms,
                "none" if last_age is None else f"{last_age:.3f}s", exc)
        if failures >= self.config.max_consecutive_failures:
            self._request_reconnect(f"consecutive_failures={failures}")

    def _record_success(self, image):
        now = time.time()
        with self.condition:
            had_failures = self.consecutive_failures
            self.frame = image.copy()
            self.frame_time = now
            self.frame_id += 1
            self.capture_times.append(now)
            self.consecutive_failures = 0
            self.last_error = ""
            self.last_error_kind = ""
            self.condition.notify_all()
        if had_failures:
            logging.info("camera recovered key=%s previous_failures=%s frame_id=%s",
                         self.key, had_failures, self.frame_id)

    def _capture_once(self):
        for attempt in range(1, self.config.read_retries + 1):
            try:
                with self.camera_lock:
                    if self.camera is None:
                        raise RuntimeError("camera is not open")
                    image = self.frame_reader(self.key, self.camera,
                                              self.config.read_timeout_ms)
                if image is None:
                    raise RuntimeError("empty frame")
                return self._normalize_image(image)
            except Exception as exc:
                self._record_failure(classify_camera_error(exc), exc, attempt)
                if attempt < self.config.read_retries and self.config.retry_sleep:
                    if self.stop_event.wait(self.config.retry_sleep):
                        return None
        return None

    def _run(self):
        logging.info("camera worker started key=%s target_fps=%s", self.key,
                     self.config.fps)
        self._reconnect("initial_open", initial=True)
        while not self.stop_event.is_set():
            loop_started = time.monotonic()
            if self.camera is None:
                self._reconnect("camera_not_open")
                continue
            if self.reconnect_event.is_set():
                self._reconnect("requested")
                continue
            image = self._capture_once()
            if image is not None:
                self._record_success(image)
                self.next_backoff = max(0.0, self.config.reconnect_backoff)
            wait_time = self.config.period - (time.monotonic() - loop_started)
            if wait_time > 0:
                self.stop_event.wait(wait_time)
        logging.info("camera worker stopped key=%s", self.key)

    def _actual_fps_locked(self):
        if len(self.capture_times) < 2:
            return 0.0
        elapsed = self.capture_times[-1] - self.capture_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.capture_times) - 1) / elapsed

    def get_latest(self):
        deadline = time.time() + self.config.start_timeout
        with self.condition:
            while self.frame is None and not self.stop_event.is_set():
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=min(0.1, remaining))
            if self.frame is None:
                raise RuntimeError(
                    f"camera {self.key} unavailable: no valid frame within "
                    f"{self.config.start_timeout:.1f}s; consecutive_failures="
                    f"{self.consecutive_failures}; last_error_kind="
                    f"{self.last_error_kind}; last_error={self.last_error}"
                )
            now = time.time()
            age = now - self.frame_time
            frame_id = self.frame_id
            image = self.frame.copy()
            if frame_id == self.delivered_frame_id:
                self.cache_reuses += 1
            else:
                self.delivered_frame_id = frame_id
            consecutive_failures = self.consecutive_failures
            last_error = self.last_error
            if age > self.config.max_stale_seconds:
                self.stale_fallbacks += 1
                stale_fallbacks = self.stale_fallbacks
            else:
                stale_fallbacks = self.stale_fallbacks
        if age > self.config.max_stale_seconds:
            self._request_reconnect(f"stale_frame_age={age:.3f}s")
            if (not self.config.allow_stale_frame
                    or age > self.config.unavailable_timeout):
                raise RuntimeError(
                    f"camera {self.key} stale for {age:.3f}s; max_stale_seconds="
                    f"{self.config.max_stale_seconds}; unavailable_timeout="
                    f"{self.config.unavailable_timeout}; frame_id={frame_id}; "
                    f"consecutive_failures={consecutive_failures}; "
                    f"last_error={last_error}"
                )
            now = time.time()
            with self.condition:
                should_log = now - self.last_stale_log >= 1.0
                if should_log:
                    self.last_stale_log = now
            if should_log:
                logging.warning(
                    "camera stale fallback key=%s age=%.3fs frame_id=%s "
                    "consecutive_failures=%s stale_fallbacks=%s last_error=%s",
                    self.key, age, frame_id, consecutive_failures,
                    stale_fallbacks, last_error)
        return image

    def maybe_log_status(self):
        interval = self.config.status_log_interval
        if interval <= 0:
            return
        now = time.time()
        with self.condition:
            if now - self.last_status_log < interval:
                return
            self.last_status_log = now
            age = now - self.frame_time if self.frame is not None else None
            fps = self._actual_fps_locked()
            logging.info(
                "camera status key=%s actual_fps=%.2f frame_id=%s "
                "last_valid_age=%s cache_reuses=%s stale_fallbacks=%s "
                "consecutive_failures=%s total_failures=%s timeouts=%s "
                "empty=%s exceptions=%s reconnects=%s open_failures=%s "
                "last_error_kind=%s last_error=%s",
                self.key, fps, self.frame_id,
                "none" if age is None else f"{age:.3f}s", self.cache_reuses,
                self.stale_fallbacks, self.consecutive_failures,
                self.total_failures, self.timeout_failures, self.empty_failures,
                self.exception_failures, self.reconnect_count,
                self.open_failures, self.last_error_kind, self.last_error)

    def status_snapshot(self):
        with self.condition:
            age = time.time() - self.frame_time if self.frame is not None else None
            return {
                "key": self.key,
                "frame_id": self.frame_id,
                "last_valid_age": age,
                "actual_fps": self._actual_fps_locked(),
                "cache_reuses": self.cache_reuses,
                "stale_fallbacks": self.stale_fallbacks,
                "consecutive_failures": self.consecutive_failures,
                "total_failures": self.total_failures,
                "timeout_failures": self.timeout_failures,
                "empty_failures": self.empty_failures,
                "exception_failures": self.exception_failures,
                "reconnect_count": self.reconnect_count,
                "open_failures": self.open_failures,
                "last_error_kind": self.last_error_kind,
                "last_error": self.last_error,
            }


class ThreadedCameraSource:
    def __init__(self, camera_factories, frame_reader, config,
                 close_camera=close_camera_object):
        self.config = config
        self.workers = {
            key: CameraCaptureWorker(key, factory, frame_reader, close_camera, config)
            for key, factory in camera_factories.items()
        }
        for worker in self.workers.values():
            worker.start()

    def read_images(self):
        images = {}
        for key in CAMERA_KEYS:
            worker = self.workers[key]
            images[key] = worker.get_latest()
            worker.maybe_log_status()
        return images

    def status_snapshot(self):
        return {key: worker.status_snapshot() for key, worker in self.workers.items()}

    def close(self):
        for worker in self.workers.values():
            worker.stop()


class OpenCVCameraSource(ThreadedCameraSource):
    def __init__(self, cam_high_id, cam_wrist_id, config):
        def make_factory(dev, key):
            def factory():
                cap = cv2.VideoCapture(dev)
                if not cap.isOpened():
                    raise RuntimeError(f"failed to open camera {key} device={dev}")
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
                return cap
            return factory

        camera_factories = {
            CAMERA_KEYS[0]: make_factory(cam_high_id, CAMERA_KEYS[0]),
            CAMERA_KEYS[1]: make_factory(cam_wrist_id, CAMERA_KEYS[1]),
        }

        def read_frame(key, cap, read_timeout_ms):
            del key, read_timeout_ms
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError("empty frame")
            return bgr

        super().__init__(camera_factories, read_frame, config)


class SDKCameraSource(ThreadedCameraSource):
    """Camera source following record_data.py's serial/resolution/fps convention."""

    def __init__(self, sdk_root, config):
        sdk_root = Path(sdk_root).expanduser() if sdk_root else None
        if sdk_root and str(sdk_root) not in sys.path:
            sys.path.insert(0, str(sdk_root))
        try:
            from camera import Camera
        except Exception as exc:
            raise RuntimeError(
                "failed to import Camera SDK. Use --mock-observation for tests "
                "or provide --camera-sdk-root matching record_data.py."
            ) from exc

        def make_factory(serial):
            return lambda: Camera(serial_number=serial,
                                  resolution=(config.height, config.width),
                                  fps=int(config.fps))

        camera_factories = {
            key: make_factory(serial)
            for key, serial in DEFAULT_CAMERA_SPECS.items()
        }

        def read_frame(key, cam, read_timeout_ms):
            del key
            capture_frame = cam.capture_frame
            try:
                params = inspect.signature(capture_frame).parameters
            except (TypeError, ValueError):
                params = {}
            if "timeout_ms" in params:
                frame = capture_frame(timeout_ms=read_timeout_ms)
            elif "timeout" in params:
                frame = capture_frame(timeout=read_timeout_ms / 1000.0)
            else:
                frame = capture_frame()
            image = frame.get("rgb") if isinstance(frame, dict) else frame
            if image is None:
                raise RuntimeError("empty frame")
            return image

        super().__init__(camera_factories, read_frame, config)


class RealManRobot:
    def __init__(self, robot_ip, robot_port, speed, use_canfd):
        self.robot_ip = robot_ip
        self.robot_port = int(robot_port)
        self.speed = int(speed)
        self.use_canfd = bool(use_canfd)
        self.arm = None
        self.last_gripper_command = None
        self.warned_canfd_blocking_fallback = False

    def connect(self):
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        except Exception:
            try:
                from rm_robot_interface import RoboticArm, rm_thread_mode_e
            except Exception as exc:
                raise RuntimeError(
                    "failed to import RealMan SDK. Run on the robot client with "
                    "rm_robot_interface.py and the vendor SDK on PYTHONPATH."
                ) from exc
        logging.info("connecting RealMan robot %s:%s", self.robot_ip,
                     self.robot_port)
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.arm.rm_create_robot_arm(self.robot_ip, self.robot_port)
        if getattr(handle, "id", -1) < 0:
            raise ConnectionError(
                f"RealMan robot connection failed: {self.robot_ip}:{self.robot_port}"
            )
        logging.info("robot connected handle_id=%s", handle.id)

    def close(self):
        if self.arm is not None:
            try:
                self.arm.rm_delete_robot_arm()
            except Exception:
                logging.exception("failed to disconnect RealMan robot")
        self.arm = None

    def read_state(self):
        ret, state = self.arm.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state returned {ret}")
        joints = np.asarray(state.get("joint", [])[:7], dtype=np.float32)
        if joints.shape != (7,):
            raise RuntimeError(f"robot joint state must be 7-dim, got {joints}")
        try:
            grip_ret, grip = self.arm.rm_get_gripper_state()
            if grip_ret == 0:
                gripper_actpos = float(grip.get("actpos", 0.0))
            else:
                gripper_actpos = 0.0
        except Exception:
            logging.exception("failed to read gripper state; using 0")
            gripper_actpos = 0.0
        return np.concatenate([joints, [gripper_actpos]]).astype(np.float32)

    def hold_current(self):
        if self.arm is None:
            return
        current = self.read_state()
        self.execute_action(current, current, send_gripper=False)

    def move_home(self,
                  home_joints,
                  speed=30,
                  open_gripper=True,
                  gripper_timeout_s=10.0):
        home_joints = np.asarray(home_joints, dtype=np.float32)
        if home_joints.shape != (7,):
            raise ValueError(f"home_joints must be 7-dim, got {home_joints}")
        started = time.perf_counter()
        before = self.read_state()
        joints = [float(v) for v in home_joints]
        ret = self.arm.rm_movej(joints, int(speed), 0, 0, 1)
        if ret != 0:
            raise RuntimeError(f"RealMan home movej returned {ret}: {joints}")
        after_move = self.read_state()
        gripper_command_sent = False
        if open_gripper:
            self._send_gripper(1, blocking=True, timeout_s=gripper_timeout_s)
            gripper_command_sent = True
        after = self.read_state()
        return {
            "home_joints": home_joints.tolist(),
            "home_speed": int(speed),
            "open_gripper": bool(open_gripper),
            "gripper_command_sent": gripper_command_sent,
            "state_before": before.tolist(),
            "state_after_move": after_move.tolist(),
            "state_after": after.tolist(),
            "joint_error_after_move_max":
            float(np.max(np.abs(after_move[:7] - home_joints))),
            "joint_error_after_max":
            float(np.max(np.abs(after[:7] - home_joints))),
            "elapsed_s": float(time.perf_counter() - started),
        }

    @staticmethod
    def _target_open_from_action(action):
        return 1 if float(action[7]) >= 0.5 else 0

    @staticmethod
    def _target_open_from_state(state):
        return 1 if float(state[7]) >= 0.5 else 0

    def _maybe_initialize_gripper_command(self, current_state):
        if self.last_gripper_command is not None or current_state is None:
            return
        try:
            current_state = np.asarray(current_state, dtype=np.float32)
            if current_state.shape[0] >= 8 and np.isfinite(current_state[7]):
                self.last_gripper_command = self._target_open_from_state(
                    current_state)
        except Exception:
            logging.exception("failed to initialize gripper command state")

    def _send_gripper(self, target_open, blocking, timeout_s):
        block = bool(blocking)
        timeout = max(1, int(math.ceil(float(timeout_s))))
        if target_open:
            ret = self.arm.rm_set_gripper_release(500, block, timeout)
        else:
            ret = self.arm.rm_set_gripper_pick_on(250, 200, block, timeout)
        if ret != 0:
            raise RuntimeError(
                f"RealMan gripper command returned {ret}, target={target_open}"
            )
        self.last_gripper_command = int(target_open)

    def _wait_until_joints_reached(self, target_joints, tolerance_deg,
                                   timeout_s, poll_s):
        target_joints = np.asarray(target_joints, dtype=np.float32)
        tolerance = float(tolerance_deg)
        timeout = max(0.0, float(timeout_s))
        poll = max(0.01, float(poll_s))
        started = time.time()
        polls = 0
        current = self.read_state()
        error = float(np.max(np.abs(current[:7] - target_joints)))
        reached = error <= tolerance
        deadline = started + timeout
        while not reached and time.time() < deadline:
            time.sleep(min(poll, max(0.0, deadline - time.time())))
            current = self.read_state()
            polls += 1
            error = float(np.max(np.abs(current[:7] - target_joints)))
            reached = error <= tolerance
        return {
            "waited": True,
            "reached": bool(reached),
            "target_tolerance_deg": tolerance,
            "timeout_s": timeout,
            "poll_s": poll,
            "polls": int(polls),
            "elapsed_s": float(time.time() - started),
            "final_error_max": float(error),
        }

    def execute_action(self,
                       action,
                       current_state,
                       send_gripper=True,
                       joint_blocking=False,
                       gripper_after_motion=False,
                       gripper_blocking=False,
                       blocking_timeout_s=3.0,
                       blocking_tolerance_deg=2.0,
                       blocking_poll_s=0.05,
                       gripper_timeout_policy="send",
                       gripper_timeout_s=10.0):
        joints = [float(v) for v in action[:7]]
        self._maybe_initialize_gripper_command(current_state)
        target_open = self._target_open_from_action(action)
        gripper_change = (
            bool(send_gripper) and target_open != self.last_gripper_command)
        wait_before_gripper = bool(gripper_after_motion and gripper_change)
        block_joint_command = bool(joint_blocking or wait_before_gripper)
        command_info = {
            "joint_blocking": bool(joint_blocking),
            "gripper_after_motion": bool(gripper_after_motion),
            "gripper_blocking": bool(gripper_blocking),
            "gripper_timeout_policy": str(gripper_timeout_policy),
            "send_gripper": bool(send_gripper),
            "gripper_target_open": int(target_open),
            "gripper_command_changed": bool(gripper_change),
            "gripper_command_sent": False,
            "gripper_command_skipped": False,
            "joint_wait": None,
            "used_canfd": False,
        }

        if (self.use_canfd and hasattr(self.arm, "rm_movej_canfd")
                and not block_joint_command):
            ret = self.arm.rm_movej_canfd(joints, False)
            command_info["used_canfd"] = True
        else:
            if (self.use_canfd and block_joint_command
                    and not self.warned_canfd_blocking_fallback):
                logging.warning(
                    "action/gripper blocking requested with CANFD; falling back "
                    "to rm_movej(..., block=1) for blocking joint commands")
                self.warned_canfd_blocking_fallback = True
            ret = self.arm.rm_movej(joints, self.speed, 0, 0,
                                    1 if block_joint_command else 0)
        if ret != 0:
            raise RuntimeError(f"RealMan joint command returned {ret}: {joints}")

        if wait_before_gripper and block_joint_command:
            current = self.read_state()
            final_error = float(np.max(np.abs(current[:7] -
                                              np.asarray(joints,
                                                         dtype=np.float32))))
            command_info["joint_wait"] = {
                "waited": True,
                "reached": bool(final_error <= float(blocking_tolerance_deg)),
                "target_tolerance_deg": float(blocking_tolerance_deg),
                "timeout_s": float(blocking_timeout_s),
                "poll_s": float(blocking_poll_s),
                "polls": 0,
                "elapsed_s": 0.0,
                "final_error_max": final_error,
            }
        elif wait_before_gripper:
            command_info["joint_wait"] = self._wait_until_joints_reached(
                joints, blocking_tolerance_deg, blocking_timeout_s,
                blocking_poll_s)

        if send_gripper:
            if wait_before_gripper and not command_info["joint_wait"]["reached"]:
                message = (
                    "gripper-blocking target was not within tolerance after "
                    "joint motion finished: final_error_max="
                    f"{command_info['joint_wait']['final_error_max']:.3f} deg, "
                    f"tolerance={blocking_tolerance_deg:.3f} deg")
                if gripper_timeout_policy == "abort":
                    raise RuntimeError(message)
                if gripper_timeout_policy == "skip":
                    logging.warning("%s; skipping gripper command", message)
                    command_info["gripper_command_skipped"] = True
                    return command_info
                logging.warning("%s; sending gripper command anyway", message)
            if target_open != self.last_gripper_command:
                self._send_gripper(target_open, gripper_blocking,
                                   gripper_timeout_s)
                command_info["gripper_command_sent"] = True
        return command_info


class RealObservationSource:
    def __init__(self, robot, camera_source):
        self.robot = robot
        self.camera_source = camera_source

    def read(self):
        state = self.robot.read_state()
        images = self.camera_source.read_images()
        return state, images

    def close(self):
        self.camera_source.close()


class CameraOnlyObservationSource:
    def __init__(self, camera_source):
        self.camera_source = camera_source

    def read(self):
        state = np.zeros(8, dtype=np.float32)
        images = self.camera_source.read_images()
        return state, images

    def close(self):
        self.camera_source.close()


@dataclass
class SafetyConfig:
    max_joint_delta_deg_per_step: float
    target_delta_limit_mode: str
    soft_joint_limits_deg: np.ndarray
    gripper_open_threshold: float
    allow_out_of_distribution_start: bool
    start_tolerance_deg: float


class SafetyFilter:
    def __init__(self, config):
        self.config = config
        if self.config.target_delta_limit_mode not in ("off", "scale"):
            raise ValueError(
                "target_delta_limit_mode must be one of: off, scale")
        if (self.config.target_delta_limit_mode == "scale"
                and self.config.max_joint_delta_deg_per_step <= 0):
            raise ValueError(
                "max_joint_delta_deg_per_step must be positive when "
                "target_delta_limit_mode=scale")

    def check_start(self, state):
        diff = np.max(np.abs(state[:7] - DEFAULT_START_POS_DEG))
        if diff > self.config.start_tolerance_deg and not self.config.allow_out_of_distribution_start:
            raise RuntimeError(
                f"current start joints differ from training start by {diff:.2f} deg; "
                "reset robot or pass --allow-out-of-distribution-start"
            )

    def filter_action(self, action, current_state):
        action = finite_array(action, (8,), "action").copy()
        current_state = finite_array(current_state, (8,), "current_state")
        raw = action.copy()
        limits = self.config.soft_joint_limits_deg
        action[:7] = np.minimum(np.maximum(action[:7], limits[:, 0]),
                                limits[:, 1])
        action[7] = (
            1.0 if float(np.clip(action[7], 0.0, 1.0))
            >= self.config.gripper_open_threshold else 0.0)
        soft_limited = action.copy()

        delta = action[:7] - current_state[:7]
        max_delta = float(self.config.max_joint_delta_deg_per_step)
        max_abs = float(np.max(np.abs(delta)))
        scaled = False
        if (self.config.target_delta_limit_mode == "scale"
                and max_abs > max_delta):
            scale = max_delta / (max_abs + 1e-6)
            action[:7] = current_state[:7] + delta * scale
            scaled = True
        soft_limit_delta_max = float(np.max(np.abs(raw[:7] - soft_limited[:7])))
        raw_to_sent_delta_max = float(np.max(np.abs(raw[:7] - action[:7])))
        return action, {
            "raw_action": raw.tolist(),
            "soft_limited_action": soft_limited.tolist(),
            "clipped_action": action.tolist(),
            "max_delta_before": max_abs,
            "delta_scaled": scaled,
            "target_delta_limit_mode": self.config.target_delta_limit_mode,
            "gripper_open_threshold": self.config.gripper_open_threshold,
            "joint_soft_limited": soft_limit_delta_max > 1e-5,
            "soft_limit_delta_max": soft_limit_delta_max,
            "raw_to_sent_joint_delta_max": raw_to_sent_delta_max,
        }


def build_default_soft_limits(stats_path=DEFAULT_ACTION_STATS_PATH,
                              margin_deg=5.0):
    with Path(stats_path).expanduser().open("r", encoding="utf-8") as fp:
        stats = json.load(fp)

    q01 = np.asarray(stats["q01"][:7], dtype=np.float32)
    q99 = np.asarray(stats["q99"][:7], dtype=np.float32)
    if q01.shape != (7,) or q99.shape != (7,):
        raise ValueError(
            f"action q01/q99 must contain at least 7 joint values: {stats_path}"
        )
    margin = float(margin_deg)
    return np.stack([q01 - margin, q99 + margin], axis=1)


class RealManVAClient:
    def __init__(self, args, observation_source, robot, event_logger):
        self.args = args
        self.observation_source = observation_source
        self.robot = robot
        self.event_logger = event_logger
        self.seq_id = 0
        self.websocket_lock = threading.Lock()
        self.stop_requested = False
        self.safety = SafetyFilter(
            SafetyConfig(
                max_joint_delta_deg_per_step=args.max_joint_delta_deg_per_step,
                target_delta_limit_mode=args.target_delta_limit_mode,
                soft_joint_limits_deg=build_default_soft_limits(
                    args.action_stats_path,
                    args.soft_limit_margin_deg,
                ),
                gripper_open_threshold=args.gripper_open_threshold,
                allow_out_of_distribution_start=args.allow_out_of_distribution_start,
                start_tolerance_deg=args.start_tolerance_deg,
            ))
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logging.warning("received signal %s, stopping after current command",
                        signum)
        self.stop_requested = True

    def should_stop(self, start_time, visual_step, check_time=True):
        if self.stop_requested:
            return True
        if self.args.stop_file and Path(self.args.stop_file).exists():
            logging.warning("emergency stop file exists: %s", self.args.stop_file)
            return True
        if self.args.max_steps and visual_step >= self.args.max_steps:
            return True
        if (check_time and self.args.max_seconds and start_time is not None
                and (time.time() - start_time) >= self.args.max_seconds):
            return True
        return False

    def home_on_start(self):
        if not self.args.home_on_start:
            logging.info("home-on-start disabled")
            self.event_logger.write({
                "event": "home_on_start",
                "enabled": False,
                "executed": False,
                "reason": "disabled_by_argument",
            })
            return
        if self.args.test_camera:
            logging.info("skip home-on-start during test-camera mode")
            self.event_logger.write({
                "event": "home_on_start",
                "enabled": True,
                "executed": False,
                "reason": "test_camera",
            })
            return
        if not self.args.enable_robot or self.robot is None:
            logging.info("skip home-on-start because robot motion is disabled")
            self.event_logger.write({
                "event": "home_on_start",
                "enabled": True,
                "executed": False,
                "reason": "robot_motion_disabled",
                "home_joints": DEFAULT_START_POS_DEG.tolist(),
                "home_speed": int(self.args.home_speed),
                "open_gripper": bool(self.args.home_open_gripper),
            })
            return

        logging.info(
            "home-on-start moving to initial joints speed=%s open_gripper=%s "
            "target=%s", self.args.home_speed, self.args.home_open_gripper,
            np.round(DEFAULT_START_POS_DEG, 4).tolist())
        info = self.robot.move_home(
            DEFAULT_START_POS_DEG,
            speed=self.args.home_speed,
            open_gripper=self.args.home_open_gripper,
            gripper_timeout_s=self.args.gripper_command_timeout_seconds,
        )
        self.event_logger.write({
            "event": "home_on_start",
            "enabled": True,
            "executed": True,
            **info,
        })
        logging.info(
            "home-on-start done elapsed=%.2fs joint_error_after=%.2f",
            info["elapsed_s"], info["joint_error_after_max"])

    def build_request(self, state, images):
        encoded_images = {
            key: encode_bgr_to_jpeg(images[key], self.args.jpeg_quality)
            for key in CAMERA_KEYS
        }
        request = {
            "version": PROTOCOL_VERSION,
            "type": "predict",
            "seq_id": self.seq_id,
            "task_text": self.args.task_text,
            "timestamp": time.time(),
            "observation_state": state.astype(np.float32).tolist(),
            "images": encoded_images,
            "camera_order": list(CAMERA_KEYS),
            "control_hz": 20,
            "visual_hz": 5,
        }
        return request

    def build_test_camera_request(self, images):
        encoded_images = {
            key: encode_bgr_to_jpeg(images[key], self.args.jpeg_quality)
            for key in CAMERA_KEYS
        }
        return {
            "version": PROTOCOL_VERSION,
            "type": "test_camera",
            "seq_id": self.seq_id,
            "timestamp": time.time(),
            "images": encoded_images,
            "camera_order": list(CAMERA_KEYS),
        }

    def build_kv_cache_request(self, image_groups, executed_actions):
        encoded_groups = []
        for images in image_groups:
            encoded_groups.append({
                "images": {
                    key: encode_bgr_to_jpeg(images[key], self.args.jpeg_quality)
                    for key in CAMERA_KEYS
                }
            })
        return {
            "version": PROTOCOL_VERSION,
            "type": "update_kv_cache",
            "seq_id": self.seq_id,
            "timestamp": time.time(),
            "image_groups": encoded_groups,
            "executed_action_chunk": np.asarray(
                executed_actions, dtype=np.float32).tolist(),
            "camera_order": list(CAMERA_KEYS),
        }

    def send_test_camera_sample(self, websocket, images):
        with self.websocket_lock:
            request = self.build_test_camera_request(images)
            seq_id = self.seq_id
            websocket.send(pack_message(request))
            response = unpack_message(
                websocket.recv(timeout=self.args.server_timeout))
            if response.get("type") == "error":
                raise RuntimeError(f"server error: {response.get('error')}")
            if response.get("type") != "test_camera_ack":
                raise RuntimeError(
                    f"unexpected response type: {response.get('type')}")
            if response.get("seq_id") != seq_id:
                raise RuntimeError(
                    "seq_id mismatch: got "
                    f"{response.get('seq_id')} expected {seq_id}")
            logging.info("test-camera seq=%s saved_images=%s", seq_id,
                         response.get("saved_images"))
            self.event_logger.write({
                "event": "test_camera_ack",
                "seq_id": seq_id,
                "saved_images": response.get("saved_images", {}),
            })
            self.seq_id += 1

    def update_kv_cache(self, websocket, image_groups, executed_actions):
        if not image_groups:
            logging.warning("skip kv-cache update: no keyframe images collected")
            return None
        if not executed_actions:
            logging.warning("skip kv-cache update: no executed actions collected")
            return None
        with self.websocket_lock:
            request = self.build_kv_cache_request(image_groups, executed_actions)
            seq_id = self.seq_id
            started = time.perf_counter()
            websocket.send(pack_message(request))
            response = unpack_message(
                websocket.recv(timeout=self.args.server_timeout))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if response.get("type") == "error":
                raise RuntimeError(f"server error: {response.get('error')}")
            if response.get("type") != "kv_cache_ack":
                raise RuntimeError(
                    f"unexpected response type: {response.get('type')}")
            if response.get("seq_id") != seq_id:
                raise RuntimeError(
                    "seq_id mismatch: got "
                    f"{response.get('seq_id')} expected {seq_id}")
            info = response.get("kv_cache", {})
            logging.info(
                "kv-cache seq=%s image_groups=%s executed_actions=%s info=%s "
                "latency=%.1fms", seq_id, len(image_groups),
                len(executed_actions), info, elapsed_ms)
            self.event_logger.write({
                "event": "kv_cache_ack",
                "seq_id": seq_id,
                "latency_ms_client": elapsed_ms,
                "image_groups": len(image_groups),
                "executed_actions": len(executed_actions),
                "kv_cache": info,
            })
            self.seq_id += 1
            return info

    def request_action(self, websocket, state, images):
        with self.websocket_lock:
            request = self.build_request(state, images)
            seq_id = self.seq_id
            started = time.perf_counter()
            websocket.send(pack_message(request))
            response = unpack_message(
                websocket.recv(timeout=self.args.server_timeout))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if response.get("type") == "error":
                raise RuntimeError(f"server error: {response.get('error')}")
            if response.get("version") != PROTOCOL_VERSION:
                raise RuntimeError("server protocol version mismatch")
            if response.get("seq_id") != seq_id:
                raise RuntimeError(
                    "seq_id mismatch: got "
                    f"{response.get('seq_id')} expected {seq_id}")
            if response.get("action_space") != ACTION_SPACE:
                raise RuntimeError(
                    f"action_space mismatch: got {response.get('action_space')}")
            chunk = np.asarray(response.get("action_chunk"), dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != 8:
                raise RuntimeError(
                    f"action_chunk must be [N,8], got {chunk.shape}")
            if not np.isfinite(chunk).all():
                raise RuntimeError("action_chunk contains NaN or Inf")
            state_to_first_error = self._joint_target_error_max(state, chunk[0])
            state_to_last_error = self._joint_target_error_max(state, chunk[-1])
            self.event_logger.write({
                "event": "server_response",
                "seq_id": seq_id,
                "latency_ms_client": elapsed_ms,
                "latency_ms_server": response.get("model_latency_ms"),
                "action_chunk": chunk.tolist(),
                "server_safety": response.get("safety", {}),
                "state_to_first_action_error_max": state_to_first_error,
                "state_to_last_action_error_max": state_to_last_error,
            })
            logging.info(
                "server_response seq=%s actions=%s latency=%.1fms "
                "state_to_first_error=%.2f state_to_last_error=%.2f",
                seq_id, len(chunk), elapsed_ms, state_to_first_error,
                state_to_last_error)
            self.seq_id += 1
            return chunk, response

    def execute_chunk(self, chunk, state, action_start_time, visual_step,
                      prefetch_hook=None):
        if self.args.execution_mode == "waypoint_hold":
            return self.execute_chunk_waypoint_hold(
                chunk, state, action_start_time, visual_step, prefetch_hook)
        return self.execute_chunk_streaming(
            chunk, state, action_start_time, visual_step, prefetch_hook)

    def _select_waypoint_target(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 8 or len(actions) == 0:
            raise ValueError(f"waypoint action group must be [N,8], got {actions.shape}")
        source = self.args.waypoint_target_source
        if source == "last":
            return actions[-1].copy()
        if source == "mean":
            return np.mean(actions, axis=0).astype(np.float32)
        if source == "median":
            return np.median(actions, axis=0).astype(np.float32)
        raise ValueError(f"unsupported waypoint target source: {source}")

    def _wait_until(self, deadline, action_start_time, visual_step):
        while True:
            if self.should_stop(action_start_time, visual_step):
                return False
            remaining = deadline - time.time()
            if remaining <= 0:
                return True
            time.sleep(min(0.02, remaining))

    def _planned_hold_seconds(self, action_count):
        if self.args.waypoint_hold_seconds > 0:
            return float(self.args.waypoint_hold_seconds)
        return float(action_count) / 20.0

    def _actions_to_execute(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32)
        if self.args.execute_actions <= 0:
            return chunk
        return chunk[:self.args.execute_actions]

    def _execute_robot_action(self, action, current_state, send_gripper=True):
        return self.robot.execute_action(
            action,
            current_state,
            send_gripper=send_gripper,
            joint_blocking=self.args.action_blocking,
            gripper_after_motion=self.args.gripper_blocking,
            gripper_blocking=self.args.gripper_blocking,
            blocking_timeout_s=self.args.blocking_timeout_seconds,
            blocking_tolerance_deg=self.args.blocking_target_tolerance_deg,
            blocking_poll_s=self.args.blocking_poll_seconds,
            gripper_timeout_policy=self.args.gripper_blocking_timeout_policy,
            gripper_timeout_s=self.args.gripper_command_timeout_seconds,
        )

    @staticmethod
    def _joint_target_error_max(state, target):
        return float(np.max(np.abs(
            np.asarray(state, dtype=np.float32)[:7] -
            np.asarray(target, dtype=np.float32)[:7])))

    def _settle_streaming_final_target(self, final_target, current_state,
                                       action_start_time, visual_step):
        final_target = np.asarray(final_target, dtype=np.float32)
        current_state = np.asarray(current_state, dtype=np.float32).copy()
        timeout = float(self.args.streaming_final_settle_seconds)
        tolerance = float(self.args.streaming_final_target_tolerance_deg)
        poll = float(self.args.streaming_final_settle_poll_seconds)
        poll = max(0.01, poll)
        start_error = self._joint_target_error_max(current_state, final_target)
        end_error = start_error
        polls = 0
        reached = start_error <= tolerance

        if self.args.enable_robot and timeout > 0 and not reached:
            deadline = time.time() + timeout
            try:
                self._execute_robot_action(final_target,
                                           current_state,
                                           send_gripper=False)
            except Exception:
                logging.exception("failed to resend final streaming target")
                raise
            while time.time() < deadline:
                if self.should_stop(action_start_time, visual_step):
                    break
                time.sleep(min(poll, max(0.0, deadline - time.time())))
                current_state = self.robot.read_state()
                polls += 1
                end_error = self._joint_target_error_max(current_state,
                                                         final_target)
                if end_error <= tolerance:
                    reached = True
                    break
        elif not self.args.enable_robot:
            current_state[:7] = final_target[:7]
            current_state[7] = final_target[7]
            end_error = 0.0
            reached = True

        info = {
            "event": "streaming_final_settle",
            "dry_run": not self.args.enable_robot,
            "visual_step": visual_step,
            "target": final_target.tolist(),
            "timeout_seconds": timeout,
            "tolerance_deg": tolerance,
            "start_error_max": start_error,
            "end_error_max": end_error,
            "polls": polls,
            "reached": reached,
            "current_state_after": current_state.tolist(),
        }
        self.event_logger.write(info)
        logging.info(
            "streaming final settle visual_step=%s reached=%s "
            "error %.2f->%.2f deg timeout=%.2fs polls=%s",
            visual_step, reached, start_error, end_error, timeout, polls)
        return current_state, info

    def execute_chunk_streaming(self, chunk, state, action_start_time, visual_step,
                                prefetch_hook=None):
        current_state = state.astype(np.float32).copy()
        period = 1.0 / 20.0
        executed = 0
        keyframe_images = []
        executed_action_history = []
        delta_scaled_count = 0
        soft_limited_count = 0
        max_delta_before_chunk = 0.0
        max_raw_to_sent_delta = 0.0
        settle_info = None
        actions_to_execute = self._actions_to_execute(chunk)
        for action_index, action in enumerate(actions_to_execute):
            if self.should_stop(action_start_time, visual_step):
                break
            filtered, info = self.safety.filter_action(action, current_state)
            delta_scaled_count += int(bool(info["delta_scaled"]))
            soft_limited_count += int(bool(info["joint_soft_limited"]))
            max_delta_before_chunk = max(max_delta_before_chunk,
                                         float(info["max_delta_before"]))
            max_raw_to_sent_delta = max(
                max_raw_to_sent_delta,
                float(info["raw_to_sent_joint_delta_max"]))
            self.event_logger.write({
                "event": "execute_action",
                "dry_run": not self.args.enable_robot,
                "visual_step": visual_step,
                "action_index": action_index,
                "current_state": current_state.tolist(),
                "action_blocking": self.args.action_blocking,
                "gripper_blocking": self.args.gripper_blocking,
                **info,
            })
            command_started = time.perf_counter()
            command_info = None
            if self.args.enable_robot:
                logging.info("execute action visual_step=%s action_index=%s target=%s",
                             visual_step, action_index,
                             np.round(filtered, 4).tolist())
                command_info = self._execute_robot_action(filtered,
                                                          current_state)
                current_state = self.robot.read_state()
            else:
                logging.info("dry-run action: %s", np.round(filtered, 4).tolist())
                current_state[:7] = filtered[:7]
                current_state[7] = filtered[7]
                command_info = {
                    "joint_blocking": bool(self.args.action_blocking),
                    "gripper_after_motion": bool(self.args.gripper_blocking),
                    "gripper_blocking": bool(self.args.gripper_blocking),
                    "send_gripper": True,
                    "dry_run": True,
                }
            executed_action_history.append(filtered.astype(np.float32).copy())
            executed += 1
            command_elapsed = time.perf_counter() - command_started
            self.event_logger.write({
                "event": "execute_action_result",
                "dry_run": not self.args.enable_robot,
                "visual_step": visual_step,
                "action_index": action_index,
                "state_after": current_state.tolist(),
                "command_elapsed_s": float(command_elapsed),
                "robot_command": command_info,
            })
            time.sleep(max(0.0, period - command_elapsed))
            if (self.args.enable_kv_cache
                    and executed % self.args.kv_cache_action_stride == 0):
                _, images = self.observation_source.read()
                keyframe_images.append(images)
            if (prefetch_hook is not None
                    and executed >= self.args.async_prefetch_after_actions):
                prefetch_state, prefetch_images = self.observation_source.read()
                prefetch_hook(prefetch_state, prefetch_images)
                prefetch_hook = None
        if executed_action_history:
            final_target = executed_action_history[-1]
            current_state, settle_info = self._settle_streaming_final_target(
                final_target, current_state, action_start_time, visual_step)
            if (self.args.enable_kv_cache and keyframe_images
                    and self.args.streaming_final_settle_seconds > 0
                    and self.args.streaming_settle_updates_kv_keyframe):
                _, settled_images = self.observation_source.read()
                keyframe_images[-1] = settled_images
                stride = min(self.args.kv_cache_action_stride,
                             len(executed_action_history))
                for i in range(1, stride + 1):
                    executed_action_history[-i] = final_target.astype(
                        np.float32).copy()
                logging.info(
                    "replaced last KV keyframe with settled image and last %s "
                    "executed action tokens with final target", stride)
                self.event_logger.write({
                    "event": "streaming_settled_kv_keyframe",
                    "visual_step": visual_step,
                    "replaced_last_keyframe": True,
                    "replaced_action_tokens": int(stride),
                    "final_target": final_target.tolist(),
                })
        final_target_error_after = (
            None if settle_info is None else settle_info["end_error_max"])
        self.event_logger.write({
            "event": "chunk_execution_summary",
            "dry_run": not self.args.enable_robot,
            "execution_mode": self.args.execution_mode,
            "visual_step": visual_step,
            "actions_requested": int(len(chunk)),
            "actions_planned": int(len(actions_to_execute)),
            "actions_executed": executed,
            "action_blocking": self.args.action_blocking,
            "gripper_blocking": self.args.gripper_blocking,
            "target_delta_limit_mode": self.args.target_delta_limit_mode,
            "delta_scaled_count": delta_scaled_count,
            "soft_limited_count": soft_limited_count,
            "max_delta_before_chunk": max_delta_before_chunk,
            "max_raw_to_sent_joint_delta": max_raw_to_sent_delta,
            "final_target_error_after": final_target_error_after,
            "streaming_final_settle": settle_info,
        })
        logging.info(
            "chunk execution mode=%s visual_step=%s executed=%s/%s delta_mode=%s "
            "scaled=%s soft_limited=%s max_delta_before=%.2f "
            "max_raw_to_sent_delta=%.2f final_target_error_after=%s",
            self.args.execution_mode, visual_step, executed,
            len(actions_to_execute), self.args.target_delta_limit_mode,
            delta_scaled_count, soft_limited_count, max_delta_before_chunk,
            max_raw_to_sent_delta, final_target_error_after)
        return executed, keyframe_images, executed_action_history

    def execute_chunk_waypoint_hold(self, chunk, state, action_start_time,
                                    visual_step, prefetch_hook=None):
        current_state = state.astype(np.float32).copy()
        executed = 0
        keyframe_images = []
        executed_action_history = []
        delta_scaled_count = 0
        soft_limited_count = 0
        max_delta_before_chunk = 0.0
        max_raw_to_sent_delta = 0.0
        max_target_error_after = 0.0
        waypoint_count = 0
        actions_to_execute = self._actions_to_execute(chunk)
        frame_size = int(self.args.action_frame_size)
        stopped_during_hold = False

        for group_start in range(0, len(actions_to_execute), frame_size):
            if self.should_stop(action_start_time, visual_step):
                break
            group = np.asarray(
                actions_to_execute[group_start:group_start + frame_size],
                dtype=np.float32,
            )
            if len(group) == 0:
                continue
            raw_target = self._select_waypoint_target(group)
            filtered, info = self.safety.filter_action(raw_target, current_state)
            delta_scaled_count += int(bool(info["delta_scaled"]))
            soft_limited_count += int(bool(info["joint_soft_limited"]))
            max_delta_before_chunk = max(max_delta_before_chunk,
                                         float(info["max_delta_before"]))
            max_raw_to_sent_delta = max(
                max_raw_to_sent_delta,
                float(info["raw_to_sent_joint_delta_max"]))
            hold_seconds = self._planned_hold_seconds(len(group))
            sample_count = 0
            if self.args.enable_kv_cache:
                sample_count = max(
                    1,
                    int(math.ceil(len(group) / self.args.kv_cache_action_stride)),
                )
            self.event_logger.write({
                "event": "execute_waypoint",
                "dry_run": not self.args.enable_robot,
                "execution_mode": self.args.execution_mode,
                "visual_step": visual_step,
                "waypoint_index": waypoint_count,
                "action_index_start": int(group_start),
                "action_count": int(len(group)),
                "waypoint_target_source": self.args.waypoint_target_source,
                "hold_seconds": hold_seconds,
                "kv_samples_planned": sample_count,
                "current_state": current_state.tolist(),
                "action_blocking": self.args.action_blocking,
                "gripper_blocking": self.args.gripper_blocking,
                "group_first_action": group[0].tolist(),
                "group_last_action": group[-1].tolist(),
                **info,
            })

            command_started = time.perf_counter()
            command_info = None
            if self.args.enable_robot:
                logging.info(
                    "execute waypoint visual_step=%s waypoint=%s actions=%s:%s "
                    "hold=%.3fs target=%s",
                    visual_step,
                    waypoint_count,
                    group_start,
                    group_start + len(group) - 1,
                    hold_seconds,
                    np.round(filtered, 4).tolist(),
                )
                command_info = self._execute_robot_action(filtered,
                                                          current_state)
                if self.args.action_blocking or self.args.gripper_blocking:
                    current_state = self.robot.read_state()
            else:
                logging.info(
                    "dry-run waypoint actions=%s:%s hold=%.3fs target=%s",
                    group_start,
                    group_start + len(group) - 1,
                    hold_seconds,
                    np.round(filtered, 4).tolist(),
                )
                current_state[:7] = filtered[:7]
                current_state[7] = filtered[7]
                command_info = {
                    "joint_blocking": bool(self.args.action_blocking),
                    "gripper_after_motion": bool(self.args.gripper_blocking),
                    "gripper_blocking": bool(self.args.gripper_blocking),
                    "send_gripper": True,
                    "dry_run": True,
                }

            group_started = time.time()
            command_elapsed = time.perf_counter() - command_started
            executed += len(group)
            waypoint_count += 1
            for _ in range(len(group)):
                executed_action_history.append(filtered.astype(np.float32).copy())
            self.event_logger.write({
                "event": "execute_waypoint_result",
                "dry_run": not self.args.enable_robot,
                "visual_step": visual_step,
                "waypoint_index": waypoint_count - 1,
                "state_after_command": current_state.tolist(),
                "command_elapsed_s": float(command_elapsed),
                "robot_command": command_info,
            })

            for sample_index in range(sample_count):
                deadline = group_started + hold_seconds * (
                    float(sample_index + 1) / float(sample_count))
                if not self._wait_until(deadline, action_start_time, visual_step):
                    stopped_during_hold = True
                    break
                sample_state, sample_images = self.observation_source.read()
                if self.args.enable_robot:
                    current_state = sample_state.astype(np.float32)
                keyframe_images.append(sample_images)
                self.event_logger.write({
                    "event": "waypoint_keyframe",
                    "visual_step": visual_step,
                    "waypoint_index": waypoint_count - 1,
                    "sample_index": sample_index,
                    "sample_count": sample_count,
                    "elapsed_since_waypoint_start": time.time() - group_started,
                    "current_state": current_state.tolist(),
                })

            if not self.args.enable_kv_cache:
                deadline = group_started + hold_seconds
                if not self._wait_until(deadline, action_start_time, visual_step):
                    stopped_during_hold = True
                elif self.args.enable_robot:
                    current_state = self.robot.read_state()

            if self.args.enable_robot and sample_count == 0 and not stopped_during_hold:
                current_state = self.robot.read_state()
            if not self.args.enable_robot:
                current_state[:7] = filtered[:7]
                current_state[7] = filtered[7]

            target_error_after = float(np.max(np.abs(current_state[:7] - filtered[:7])))
            max_target_error_after = max(max_target_error_after, target_error_after)
            self.event_logger.write({
                "event": "waypoint_done",
                "dry_run": not self.args.enable_robot,
                "visual_step": visual_step,
                "waypoint_index": waypoint_count - 1,
                "action_index_start": int(group_start),
                "action_count": int(len(group)),
                "target": filtered.tolist(),
                "state_after": current_state.tolist(),
                "target_error_after_max": target_error_after,
                "kv_images_collected_total": len(keyframe_images),
                "stopped_during_hold": stopped_during_hold,
            })

            if (prefetch_hook is not None
                    and executed >= self.args.async_prefetch_after_actions):
                prefetch_state, prefetch_images = self.observation_source.read()
                prefetch_hook(prefetch_state, prefetch_images)
                prefetch_hook = None
            if stopped_during_hold:
                break

        self.event_logger.write({
            "event": "chunk_execution_summary",
            "dry_run": not self.args.enable_robot,
            "execution_mode": self.args.execution_mode,
            "visual_step": visual_step,
            "actions_requested": int(len(chunk)),
            "actions_planned": int(len(actions_to_execute)),
            "actions_executed": executed,
            "waypoints_executed": waypoint_count,
            "action_frame_size": frame_size,
            "waypoint_hold_seconds": self.args.waypoint_hold_seconds,
            "action_blocking": self.args.action_blocking,
            "gripper_blocking": self.args.gripper_blocking,
            "target_delta_limit_mode": self.args.target_delta_limit_mode,
            "delta_scaled_count": delta_scaled_count,
            "soft_limited_count": soft_limited_count,
            "max_delta_before_chunk": max_delta_before_chunk,
            "max_raw_to_sent_joint_delta": max_raw_to_sent_delta,
            "max_target_error_after": max_target_error_after,
            "kv_cache_image_groups": len(keyframe_images),
        })
        logging.info(
            "chunk execution mode=%s visual_step=%s waypoints=%s "
            "executed_actions=%s/%s kv_images=%s delta_mode=%s scaled=%s "
            "soft_limited=%s max_delta_before=%.2f max_target_error_after=%.2f",
            self.args.execution_mode,
            visual_step,
            waypoint_count,
            executed,
            len(actions_to_execute),
            len(keyframe_images),
            self.args.target_delta_limit_mode,
            delta_scaled_count,
            soft_limited_count,
            max_delta_before_chunk,
            max_target_error_after,
        )
        return executed, keyframe_images, executed_action_history

    def run_safe_sync(self, websocket):
        state, images = self.observation_source.read()
        self.safety.check_start(state)
        action_start_time = None
        visual_step = 0
        while not self.should_stop(action_start_time, visual_step):
            state, images = self.observation_source.read()
            logging.info("requesting action chunk seq=%s visual_step=%s",
                         self.seq_id, visual_step)
            chunk, _ = self.request_action(websocket, state, images)
            if action_start_time is None:
                action_start_time = time.time()
            executed, keyframe_images, executed_actions = self.execute_chunk(
                chunk, state, action_start_time, visual_step)
            if (self.args.enable_kv_cache and len(keyframe_images) >= 4
                    and not self.stop_requested):
                self.update_kv_cache(websocket, keyframe_images,
                                     executed_actions)
            elif self.args.enable_kv_cache and executed > 0:
                logging.warning("skip kv-cache update: collected only %s image groups",
                                len(keyframe_images))
            self.event_logger.write({
                "event": "visual_step_done",
                "mode": "safe_sync",
                "execution_mode": self.args.execution_mode,
                "visual_step": visual_step,
                "executed_actions": executed,
                "kv_cache_image_groups": len(keyframe_images),
            })
            logging.info("visual_step=%s done executed_actions=%s", visual_step,
                         executed)
            visual_step += 1

    def run_async(self, websocket):
        state, images = self.observation_source.read()
        self.safety.check_start(state)
        action_start_time = None
        visual_step = 0
        logging.info(
            "async mode uses speculative next-chunk requests; kv-cache updates "
            "must be disabled for predictions to overlap execution; "
            "prefetch_after_actions=%s", self.args.async_prefetch_after_actions)
        with ThreadPoolExecutor(max_workers=1) as executor:
            logging.info("requesting action chunk seq=%s visual_step=%s",
                         self.seq_id, visual_step)
            future = executor.submit(self.request_action, websocket, state, images)
            while not self.should_stop(action_start_time, visual_step):
                try:
                    chunk, _ = future.result(timeout=self.args.server_timeout)
                except FutureTimeout:
                    logging.warning("server timeout; holding current position")
                    if self.args.enable_robot:
                        self.robot.hold_current()
                    break
                if action_start_time is None:
                    action_start_time = time.time()
                execute_state, execute_images = self.observation_source.read()
                prefetch_future = None

                def start_prefetch(prefetch_state, prefetch_images):
                    nonlocal prefetch_future
                    if prefetch_future is not None or self.stop_requested:
                        return
                    logging.info(
                        "requesting speculative action chunk seq=%s "
                        "visual_step=%s after_executed_actions=%s",
                        self.seq_id, visual_step + 1,
                        self.args.async_prefetch_after_actions)
                    prefetch_future = executor.submit(
                        self.request_action, websocket, prefetch_state,
                        prefetch_images)

                if self.args.async_prefetch_after_actions <= 0:
                    start_prefetch(execute_state, execute_images)
                executed, keyframe_images, executed_actions = self.execute_chunk(
                    chunk,
                    execute_state,
                    action_start_time,
                    visual_step,
                    prefetch_hook=(
                        start_prefetch
                        if self.args.async_prefetch_after_actions > 0 else None),
                )
                if prefetch_future is None and not self.stop_requested:
                    prefetch_state, prefetch_images = self.observation_source.read()
                    start_prefetch(prefetch_state, prefetch_images)
                if (self.args.enable_kv_cache and len(keyframe_images) >= 4
                        and not self.stop_requested):
                    self.update_kv_cache(websocket, keyframe_images,
                                         executed_actions)
                elif self.args.enable_kv_cache and executed > 0:
                    logging.warning(
                        "skip kv-cache update: collected only %s image groups",
                        len(keyframe_images))
                self.event_logger.write({
                    "event": "visual_step_done",
                    "mode": "async",
                    "execution_mode": self.args.execution_mode,
                    "visual_step": visual_step,
                    "executed_actions": executed,
                    "kv_cache_image_groups": len(keyframe_images),
                })
                logging.info("visual_step=%s done executed_actions=%s", visual_step,
                             executed)
                visual_step += 1
                if prefetch_future is None:
                    logging.warning("no next async request was scheduled; stopping")
                    break
                future = prefetch_future

    def run_test_camera(self, websocket):
        for _ in range(10):
            _, images = self.observation_source.read()
            self.send_test_camera_sample(websocket, images)
        logging.info("test-camera finished after 10 image pairs")

    def run(self):
        if connect is None:
            raise RuntimeError(
                f"websockets.sync.client import failed: {WEBSOCKETS_IMPORT_ERROR}")
        uri = f"ws://{self.args.server_host}:{self.args.server_port}"
        logging.info("connecting server %s", uri)
        with connect(uri,
                     open_timeout=self.args.server_timeout,
                     proxy=None,
                     max_size=64 * 1024 * 1024,
                     ping_interval=websocket_timeout_value(
                         self.args.websocket_ping_interval),
                     ping_timeout=websocket_timeout_value(
                         self.args.websocket_ping_timeout),
                     close_timeout=websocket_timeout_value(
                         self.args.websocket_close_timeout)) as websocket:
            metadata = unpack_message(websocket.recv(timeout=self.args.server_timeout))
            logging.info("server metadata: %s", metadata)
            if metadata.get("action_space") != ACTION_SPACE:
                raise RuntimeError(
                    f"server action_space mismatch: {metadata.get('action_space')}")
            if self.args.test_camera:
                self.run_test_camera(websocket)
                return
            if self.args.mode == "async" and self.args.enable_kv_cache:
                logging.warning(
                    "async mode requested with kv-cache enabled; running "
                    "safe_sync so each prediction sees the executed-action "
                    "kv-cache update. Use --disable-kv-cache or "
                    "--async-kv-policy disable_kv to overlap server inference "
                    "with chunk execution.")
                self.run_safe_sync(websocket)
                return
            if self.args.mode == "async":
                self.run_async(websocket)
            else:
                self.run_safe_sync(websocket)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", default=os.getenv("REAL_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--server-port", type=int, default=int(os.getenv("REAL_SERVER_PORT", "18080")))
    parser.add_argument("--task-text", default="")
    parser.add_argument("--dry-run", action="store_true", help="accepted for clarity; dry-run is default")
    parser.add_argument("--enable-robot", action="store_true", help="explicitly allow robot motion")
    parser.add_argument("--test-camera", action="store_true")
    parser.add_argument("--mock-observation", action="store_true")
    parser.add_argument("--mode", choices=["safe_sync", "async"], default="safe_sync")
    parser.add_argument(
        "--async-kv-policy",
        choices=["safe_sync", "disable_kv"],
        default=os.getenv("REAL_ASYNC_KV_POLICY", "safe_sync"),
        help=("policy when --mode async is requested while kv-cache is enabled: "
              "safe_sync keeps ordered kv-cache semantics; disable_kv turns off "
              "client kv-cache updates so next prediction can overlap execution"))
    parser.add_argument(
        "--async-prefetch-after-actions",
        type=int,
        default=int(os.getenv("REAL_ASYNC_PREFETCH_AFTER_ACTIONS", "0")),
        help=("when --mode async is active, schedule the next prediction after "
              "this many executed actions; 0 schedules it before executing the "
              "current chunk for maximum overlap"))
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--execute-actions",
                        type=int,
                        default=int(os.getenv("REAL_EXECUTE_ACTIONS", "0")),
                        help=("execute only the first N actions from each "
                              "action_chunk; 0 means execute the full chunk"))
    parser.add_argument(
        "--execution-mode",
        choices=["waypoint_hold", "streaming"],
        default=os.getenv("REAL_EXECUTION_MODE", "streaming"),
        help=("waypoint_hold sends one representative absolute target per "
              "action frame and holds it while collecting KV keyframes; "
              "streaming sends every returned action at 20Hz"))
    parser.add_argument(
        "--action-frame-size",
        type=int,
        default=int(os.getenv("REAL_ACTION_FRAME_SIZE", "16")),
        help="number of high-frequency actions in one Wan action frame")
    parser.add_argument(
        "--waypoint-target-source",
        choices=["last", "mean", "median"],
        default=os.getenv("REAL_WAYPOINT_TARGET_SOURCE", "last"),
        help="representative target to send for each waypoint action frame")
    parser.add_argument(
        "--waypoint-hold-seconds",
        type=float,
        default=float(os.getenv("REAL_WAYPOINT_HOLD_SECONDS", "0")),
        help="0 uses action_count / 20 seconds; positive overrides each waypoint hold")
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--kv-cache-action-stride", type=int, default=4)
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=float(os.getenv("REAL_CLIENT_SERVER_TIMEOUT", "120.0")))
    parser.add_argument(
        "--websocket-ping-interval",
        type=float,
        default=float(os.getenv("REAL_WEBSOCKET_PING_INTERVAL", "0")),
        help="websocket keepalive ping interval; <=0 disables ping keepalive")
    parser.add_argument(
        "--websocket-ping-timeout",
        type=float,
        default=float(os.getenv("REAL_WEBSOCKET_PING_TIMEOUT", "600")),
        help="websocket keepalive ping timeout in seconds")
    parser.add_argument(
        "--websocket-close-timeout",
        type=float,
        default=float(os.getenv("REAL_WEBSOCKET_CLOSE_TIMEOUT", "120")),
        help="websocket close timeout in seconds")
    parser.add_argument("--save-rollout-dir", default=None)
    parser.add_argument("--stop-file", default=DEFAULT_STOP_FILE)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--camera-backend", choices=["sdk", "opencv"], default="sdk")
    parser.add_argument("--camera-sdk-root", default=os.getenv("REAL_CAMERA_SDK_ROOT", ""))
    parser.add_argument("--cam-high-id", default=0)
    parser.add_argument("--cam-wrist-id", default=1)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-read-retries", type=int, default=1)
    parser.add_argument("--camera-retry-sleep", type=float, default=0.05)
    parser.add_argument("--camera-read-timeout-ms", type=int, default=1000)
    parser.add_argument("--max-stale-seconds", type=float, default=0.5)
    parser.add_argument("--camera-unavailable-timeout", type=float, default=10.0)
    parser.add_argument("--max-consecutive-camera-failures", type=int, default=3)
    parser.add_argument("--camera-reconnect-backoff", type=float, default=0.5)
    parser.add_argument("--camera-reconnect-max-backoff", type=float, default=5.0)
    parser.add_argument("--camera-start-timeout", type=float, default=5.0)
    parser.add_argument("--camera-status-log-interval", type=float, default=10.0)
    parser.add_argument("--max-stale-camera-age", type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-stale-camera-frame", action="store_true")
    parser.add_argument("--robot-ip", default=os.getenv("REALMAN_ROBOT_IP", "192.168.1.18"))
    parser.add_argument("--robot-port", type=int, default=int(os.getenv("REALMAN_ROBOT_PORT", "8080")))
    parser.add_argument("--robot-speed", type=int, default=10)
    parser.add_argument("--use-canfd", action="store_true")
    parser.add_argument(
        "--home-on-start",
        dest="home_on_start",
        action="store_true",
        help="move to the training initial pose before the first prediction")
    parser.add_argument(
        "--no-home-on-start",
        dest="home_on_start",
        action="store_false",
        help="do not move to the initial pose when the client starts")
    parser.add_argument(
        "--home-speed",
        type=int,
        default=int(os.getenv("REAL_HOME_SPEED", "30")),
        help="RealMan movej speed for home-on-start")
    parser.add_argument(
        "--home-open-gripper",
        dest="home_open_gripper",
        action="store_true",
        help="open the gripper after home-on-start")
    parser.add_argument(
        "--no-home-open-gripper",
        dest="home_open_gripper",
        action="store_false",
        help="do not open the gripper after home-on-start")
    parser.add_argument(
        "--action-blocking",
        dest="action_blocking",
        action="store_true",
        help=("send every joint action as a blocking RealMan movej command; "
              "this is slower than 20Hz streaming but waits for each target"))
    parser.add_argument(
        "--no-action-blocking",
        dest="action_blocking",
        action="store_false",
        help="disable action-blocking even if REAL_ACTION_BLOCKING is set")
    parser.add_argument(
        "--gripper-blocking",
        dest="gripper_blocking",
        action="store_true",
        help=("when the target gripper state changes, move joints to the "
              "current action target first, then issue a blocking gripper "
              "open/close command"))
    parser.add_argument(
        "--no-gripper-blocking",
        dest="gripper_blocking",
        action="store_false",
        help="disable gripper-blocking even if REAL_GRIPPER_BLOCKING is set")
    parser.add_argument(
        "--blocking-target-tolerance-deg",
        type=float,
        default=float(os.getenv("REAL_BLOCKING_TARGET_TOLERANCE_DEG", "2.0")),
        help="joint tolerance used by blocking diagnostics and waits")
    parser.add_argument(
        "--blocking-timeout-seconds",
        type=float,
        default=float(os.getenv("REAL_BLOCKING_TIMEOUT_SECONDS", "3.0")),
        help="maximum time to wait for a joint target before gripper changes")
    parser.add_argument(
        "--blocking-poll-seconds",
        type=float,
        default=float(os.getenv("REAL_BLOCKING_POLL_SECONDS", "0.05")),
        help="poll period while waiting for blocking joint targets")
    parser.add_argument(
        "--gripper-blocking-timeout-policy",
        choices=["send", "skip", "abort"],
        default=os.getenv("REAL_GRIPPER_BLOCKING_TIMEOUT_POLICY", "send"),
        help=("what to do when gripper-blocking finishes joint motion but the "
              "measured joint error is still above tolerance: send logs a "
              "warning and sends the gripper command; skip suppresses the "
              "gripper command; abort raises an error"))
    parser.add_argument(
        "--gripper-command-timeout-seconds",
        type=float,
        default=float(os.getenv("REAL_GRIPPER_COMMAND_TIMEOUT_SECONDS", "10")),
        help="timeout passed to RealMan gripper commands")
    parser.add_argument("--max-joint-delta-deg-per-step", type=float, default=2.0)
    parser.add_argument(
        "--target-delta-limit-mode",
        choices=["off", "scale"],
        default=os.getenv("REAL_TARGET_DELTA_LIMIT_MODE", "off"),
        help=("off sends absolute target joints after soft-limit clipping; scale "
              "keeps the old per-command delta scaling for cautious debug runs"))
    parser.add_argument("--action-stats-path", default=str(DEFAULT_ACTION_STATS_PATH))
    parser.add_argument("--soft-limit-margin-deg", type=float, default=5.0)
    parser.add_argument(
        "--gripper-open-threshold",
        type=float,
        default=float(os.getenv("REAL_GRIPPER_OPEN_THRESHOLD", "0.5")),
        help=("binarize target_gripper: values >= threshold open the gripper, "
              "values below threshold close it"))
    parser.add_argument(
        "--streaming-final-settle-seconds",
        type=float,
        default=float(os.getenv("REAL_STREAMING_FINAL_SETTLE_SECONDS", "1.2")),
        help=("after a streaming chunk, hold the last target for up to this "
              "many seconds before updating KV cache"))
    parser.add_argument(
        "--streaming-final-target-tolerance-deg",
        type=float,
        default=float(os.getenv("REAL_STREAMING_FINAL_TARGET_TOLERANCE_DEG",
                                "2.0")),
        help="stop streaming final settle once max joint error is below this")
    parser.add_argument(
        "--streaming-final-settle-poll-seconds",
        type=float,
        default=float(os.getenv("REAL_STREAMING_FINAL_SETTLE_POLL_SECONDS",
                                "0.05")),
        help="poll period while waiting for the final streaming target")
    parser.add_argument(
        "--no-streaming-settle-updates-kv-keyframe",
        dest="streaming_settle_updates_kv_keyframe",
        action="store_false",
        default=env_bool("REAL_STREAMING_SETTLE_UPDATES_KV_KEYFRAME", True),
        help=("do not replace the last KV keyframe with the settled image "
              "after streaming final-target settle"))
    parser.add_argument("--start-tolerance-deg", type=float, default=20.0)
    parser.add_argument("--allow-out-of-distribution-start", action="store_true")
    parser.set_defaults(
        home_on_start=env_bool("REAL_HOME_ON_START", True),
        home_open_gripper=env_bool("REAL_HOME_OPEN_GRIPPER", True),
        action_blocking=env_bool("REAL_ACTION_BLOCKING", False),
        gripper_blocking=env_bool("REAL_GRIPPER_BLOCKING", False),
    )
    args = parser.parse_args()
    args.enable_kv_cache = not args.disable_kv_cache
    args.async_disabled_kv_cache = False
    if (args.mode == "async" and args.enable_kv_cache
            and args.async_kv_policy == "disable_kv"):
        args.enable_kv_cache = False
        args.async_disabled_kv_cache = True
    if args.kv_cache_action_stride <= 0:
        raise ValueError("--kv-cache-action-stride must be positive")
    if args.server_timeout <= 0:
        raise ValueError("--server-timeout must be positive")
    if args.websocket_close_timeout <= 0:
        raise ValueError("--websocket-close-timeout must be positive")
    if args.websocket_ping_timeout < 0:
        raise ValueError("--websocket-ping-timeout must be non-negative")
    if args.async_prefetch_after_actions < 0:
        raise ValueError("--async-prefetch-after-actions must be non-negative")
    if args.execute_actions < 0:
        raise ValueError("--execute-actions must be non-negative")
    if args.action_frame_size <= 0:
        raise ValueError("--action-frame-size must be positive")
    if args.waypoint_hold_seconds < 0:
        raise ValueError("--waypoint-hold-seconds must be non-negative")
    if (args.target_delta_limit_mode == "scale"
            and args.max_joint_delta_deg_per_step <= 0):
        raise ValueError(
            "--max-joint-delta-deg-per-step must be positive when "
            "--target-delta-limit-mode scale")
    if not 0.0 <= args.gripper_open_threshold <= 1.0:
        raise ValueError("--gripper-open-threshold must be in [0, 1]")
    if args.streaming_final_settle_seconds < 0:
        raise ValueError("--streaming-final-settle-seconds must be non-negative")
    if args.streaming_final_target_tolerance_deg < 0:
        raise ValueError(
            "--streaming-final-target-tolerance-deg must be non-negative")
    if args.streaming_final_settle_poll_seconds <= 0:
        raise ValueError(
            "--streaming-final-settle-poll-seconds must be positive")
    if args.camera_fps <= 0:
        raise ValueError("--camera-fps must be positive")
    if args.camera_read_retries <= 0:
        raise ValueError("--camera-read-retries must be positive")
    if args.camera_retry_sleep < 0:
        raise ValueError("--camera-retry-sleep must be non-negative")
    if args.camera_read_timeout_ms <= 0:
        raise ValueError("--camera-read-timeout-ms must be positive")
    if args.max_stale_seconds < 0:
        raise ValueError("--max-stale-seconds must be non-negative")
    if args.camera_unavailable_timeout <= 0:
        raise ValueError("--camera-unavailable-timeout must be positive")
    if args.max_consecutive_camera_failures <= 0:
        raise ValueError("--max-consecutive-camera-failures must be positive")
    if args.camera_reconnect_backoff < 0:
        raise ValueError("--camera-reconnect-backoff must be non-negative")
    if args.camera_reconnect_max_backoff < args.camera_reconnect_backoff:
        raise ValueError(
            "--camera-reconnect-max-backoff must be >= --camera-reconnect-backoff"
        )
    if args.camera_start_timeout <= 0:
        raise ValueError("--camera-start-timeout must be positive")
    if args.camera_status_log_interval < 0:
        raise ValueError("--camera-status-log-interval must be non-negative")
    if args.home_speed <= 0:
        raise ValueError("--home-speed must be positive")
    if args.blocking_target_tolerance_deg < 0:
        raise ValueError("--blocking-target-tolerance-deg must be non-negative")
    if args.blocking_timeout_seconds < 0:
        raise ValueError("--blocking-timeout-seconds must be non-negative")
    if args.blocking_poll_seconds <= 0:
        raise ValueError("--blocking-poll-seconds must be positive")
    if args.gripper_command_timeout_seconds <= 0:
        raise ValueError("--gripper-command-timeout-seconds must be positive")
    if args.max_stale_camera_age is not None:
        args.camera_unavailable_timeout = args.max_stale_camera_age
    args.allow_stale_camera_frame = not args.no_stale_camera_frame
    return args


def build_camera_config(args):
    return CameraReadConfig(
        height=args.image_height,
        width=args.image_width,
        fps=args.camera_fps,
        read_retries=args.camera_read_retries,
        retry_sleep=args.camera_retry_sleep,
        read_timeout_ms=args.camera_read_timeout_ms,
        max_stale_seconds=args.max_stale_seconds,
        unavailable_timeout=args.camera_unavailable_timeout,
        max_consecutive_failures=args.max_consecutive_camera_failures,
        reconnect_backoff=args.camera_reconnect_backoff,
        reconnect_max_backoff=args.camera_reconnect_max_backoff,
        start_timeout=args.camera_start_timeout,
        status_log_interval=args.camera_status_log_interval,
        allow_stale_frame=args.allow_stale_camera_frame,
    )


def build_sources(args):
    if args.mock_observation:
        return MockObservationSource(args.image_height, args.image_width), None

    camera_config = build_camera_config(args)
    if args.test_camera:
        if args.camera_backend == "opencv":
            camera_source = OpenCVCameraSource(args.cam_high_id, args.cam_wrist_id,
                                               camera_config)
        else:
            camera_source = SDKCameraSource(args.camera_sdk_root, camera_config)
        return CameraOnlyObservationSource(camera_source), None

    robot = RealManRobot(args.robot_ip, args.robot_port, args.robot_speed,
                         args.use_canfd)
    robot.connect()

    if args.camera_backend == "opencv":
        camera_source = OpenCVCameraSource(args.cam_high_id, args.cam_wrist_id,
                                           camera_config)
    else:
        camera_source = SDKCameraSource(args.camera_sdk_root, camera_config)
    return RealObservationSource(robot, camera_source), robot


def main():
    args = parse_args()
    if not args.test_camera and not args.task_text.strip():
        raise ValueError("--task-text is required unless --test-camera is set")
    log_path, events_path = setup_logging(args.save_rollout_dir)
    logging.info("log_path=%s events_path=%s", log_path, events_path)
    logging.info("dry_run=%s enable_robot=%s", not args.enable_robot,
                 args.enable_robot)
    logging.info(
        "mode=%s async_kv_policy=%s enable_kv_cache=%s execute_actions=%s "
        "execution_mode=%s action_frame_size=%s waypoint_target_source=%s "
        "waypoint_hold_seconds=%s kv_cache_action_stride=%s "
        "async_prefetch_after_actions=%s",
        args.mode, args.async_kv_policy, args.enable_kv_cache,
        args.execute_actions, args.execution_mode, args.action_frame_size,
        args.waypoint_target_source, args.waypoint_hold_seconds,
        args.kv_cache_action_stride,
        args.async_prefetch_after_actions)
    if args.async_disabled_kv_cache:
        logging.warning(
            "async-kv-policy=disable_kv turned off kv-cache updates; "
            "async predictions are speculative and use state/images sampled "
            "before the current chunk finishes")
    elif not args.enable_kv_cache:
        logging.warning(
            "kv-cache is disabled; the server will keep frame_st_id=0 and each "
            "prediction restarts from the first executable action frame")
    logging.info(
        "robot_speed=%s use_canfd=%s target_delta_limit_mode=%s "
        "home_on_start=%s home_speed=%s home_open_gripper=%s "
        "max_joint_delta_deg_per_step=%s gripper_open_threshold=%s "
        "action_blocking=%s gripper_blocking=%s "
        "blocking_target_tolerance_deg=%s blocking_timeout_seconds=%s "
        "blocking_poll_seconds=%s gripper_blocking_timeout_policy=%s "
        "gripper_command_timeout_seconds=%s",
        args.robot_speed, args.use_canfd, args.target_delta_limit_mode,
        args.home_on_start, args.home_speed, args.home_open_gripper,
        args.max_joint_delta_deg_per_step, args.gripper_open_threshold,
        args.action_blocking, args.gripper_blocking,
        args.blocking_target_tolerance_deg, args.blocking_timeout_seconds,
        args.blocking_poll_seconds, args.gripper_blocking_timeout_policy,
        args.gripper_command_timeout_seconds)
    if (args.enable_kv_cache and args.execute_actions > 0
            and args.execute_actions < args.kv_cache_action_stride * 4):
        logging.warning(
            "execute_actions=%s collects fewer than 4 KV image groups with "
            "kv_cache_action_stride=%s; the client will skip KV update for "
            "that chunk", args.execute_actions, args.kv_cache_action_stride)
    logging.info(
        "streaming_final_settle_seconds=%s "
        "streaming_final_target_tolerance_deg=%s "
        "streaming_final_settle_poll_seconds=%s "
        "streaming_settle_updates_kv_keyframe=%s server_timeout=%s "
        "websocket_ping_interval=%s websocket_ping_timeout=%s "
        "websocket_close_timeout=%s",
        args.streaming_final_settle_seconds,
        args.streaming_final_target_tolerance_deg,
        args.streaming_final_settle_poll_seconds,
        args.streaming_settle_updates_kv_keyframe,
        args.server_timeout,
        args.websocket_ping_interval,
        args.websocket_ping_timeout,
        args.websocket_close_timeout)
    logging.info(
        "camera_fps=%s camera_read_timeout_ms=%s camera_read_retries=%s "
        "max_stale_seconds=%s camera_unavailable_timeout=%s "
        "max_consecutive_camera_failures=%s allow_stale_camera_frame=%s",
        args.camera_fps, args.camera_read_timeout_ms, args.camera_read_retries,
        args.max_stale_seconds, args.camera_unavailable_timeout,
        args.max_consecutive_camera_failures, args.allow_stale_camera_frame)
    if not args.enable_robot:
        logging.info("dry-run is active; no robot motion commands will be sent")
    event_logger = JsonlLogger(events_path)
    observation_source = None
    robot = None
    try:
        observation_source, robot = build_sources(args)
        client = RealManVAClient(args, observation_source, robot, event_logger)
        client.home_on_start()
        client.run()
    finally:
        if robot is not None and args.enable_robot:
            try:
                robot.hold_current()
            except Exception:
                logging.exception("failed to hold current position on shutdown")
            robot.close()
        if observation_source is not None:
            observation_source.close()
        event_logger.close()


if __name__ == "__main__":
    main()
