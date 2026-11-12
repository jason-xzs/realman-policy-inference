#!/usr/bin/env python3


# conda activate realman-pi
# export OPENPI_CLIENT_SRC=/home/robot2/PhyWAM/openpi/openpi-client/src
# python /home/robot2/PhyWAM/scripts/evaluation/client_pi05_real.py \
#   --server-host 127.0.0.1 \
#   --server-port 18080 \
#   --task-text "Place the brown cup in the white basin." \
#   --enable-robot \
#   --robot-ip 192.168.1.18 \
#   --robot-speed 30 \
#   --replan-steps 1 \
#   --max-seconds 600 \
#   --gripper-open-threshold 0.7

# 默认行为：
# - 真机模式启动后先回到起始位姿 [3.15, -2.35, 7.65, 69.75, 175.52, -89.94, 96.94]，再打开夹爪。
# - 每个 policy action 都先阻塞移动到目标关节，再阻塞执行夹爪开/关。
# - 阻塞模式下每次只执行 action chunk 的第一个 action，动作完成后重新采集图像和状态再推理下一步。

# 使用方法：
# 1. 在 RealMan 客户端电脑上运行，使用 realman conda 环境：
#      conda activate realman
# 2. 安装或暴露 OpenPI 轻量客户端。任选一种：
#      pip install -e /path/to/openpi/packages/openpi-client
#      export OPENPI_CLIENT_SRC=/path/to/openpi/packages/openpi-client/src
# 3. dry-run，不连接真机/相机，只验证 websocket 和 observation/action 格式：
#      python client_pi05_real_realman.py --server-host <server-ip> --server-port 8000 \
#        --task-text "perform the realman manipulation task" --mock-observation --max-steps 2
# 4. 只检查双相机并向 pi05 server 请求动作，不连接机械臂：
#      python client_pi05_real_realman.py --server-host <server-ip> --server-port 8000 \
#        --task-text "perform the realman manipulation task" --test-camera --test-camera-samples 3
# 5. 真机执行必须显式加 --enable-robot：
#      python client_pi05_real_realman.py --server-host <server-ip> --server-port 8000 \
#        --task-text "perform the realman manipulation task" --enable-robot \
#        --robot-ip 192.168.1.18 --robot-speed 10 --replan-steps 1 --max-seconds 60
# 6. 急停方式：Ctrl+C 停止后续动作；或在客户端创建 /tmp/pi05_realman_stop。

import argparse
import contextlib
import importlib.util
import inspect
import json
import logging
import os
import signal
import sys
import time
import types
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


OPENPI_CLIENT_SRC = os.getenv("OPENPI_CLIENT_SRC")
if OPENPI_CLIENT_SRC:
    sys.path.insert(0, OPENPI_CLIENT_SRC)
else:
    local_client_src = Path(__file__).resolve().parents[2] / "openpi" / "openpi-client" / "src"
    if local_client_src.exists():
        sys.path.insert(0, str(local_client_src))

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy
except Exception as exc:  # pragma: no cover - depends on client machine env
    raise RuntimeError(
        "无法导入 openpi_client。请先执行 "
        "`pip install -e /home/robot2/PhyWAM/openpi/openpi-client`，"
        "或设置 OPENPI_CLIENT_SRC=/home/robot2/PhyWAM/openpi/openpi-client/src。"
    ) from exc


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_wrist",
)
DEFAULT_CAMERA_SERIALS = {
    CAMERA_KEYS[0]: "036522071395",
    CAMERA_KEYS[1]: "323622272614",
}
DEFAULT_PROMPT = "perform the realman manipulation task"
DEFAULT_STOP_FILE = "/tmp/pi05_realman_stop"
DEFAULT_NORM_STATS_PATH = Path(__file__).resolve().parents[1] / "data" / "lerobot" / "meta" / "action_norm_stats.json"
DEFAULT_START_POS_DEG = np.asarray([3.15, -2.35, 7.65, 69.75, 175.52, -89.94, 96.94], dtype=np.float32)
DEFAULT_OPEN_GRIPPER_ACTPOS = 1000.0
DEFAULT_START_STATE = np.concatenate([DEFAULT_START_POS_DEG, [DEFAULT_OPEN_GRIPPER_ACTPOS]]).astype(np.float32)
DEFAULT_STATE = np.asarray(
    [-2.5842106, 23.14384, 6.8027787, 66.873405, 173.00708, -75.80059, 99.64512, 595.44043],
    dtype=np.float32,
)
DEFAULT_STATE_Q01 = np.asarray(
    [-71.84082, -2.4598272, -25.98421, 28.488684, 157.57788, -105.37709, 73.8805],
    dtype=np.float32,
)
DEFAULT_STATE_Q99 = np.asarray(
    [25.798397, 47.588573, 39.091503, 80.41829, 175.55426, -47.548714, 128.12186],
    dtype=np.float32,
)
FALLBACK_HARD_JOINT_LIMITS_DEG = np.stack(
    [np.full(7, -180.0, dtype=np.float32), np.full(7, 180.0, dtype=np.float32)],
    axis=1,
)


def setup_logging(save_rollout_dir):
    log_dir = Path(save_rollout_dir or Path(__file__).resolve().parent / "logs" / "pi05_real_client")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / ("client_pi05_realman_%s.log" % stamp)
    events_path = log_dir / ("client_pi05_realman_%s.jsonl" % stamp)
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
        self._fp = Path(path).open("a", encoding="utf-8")

    def write(self, event):
        payload = dict(event)
        payload.setdefault("wall_time", time.time())
        self._fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self):
        self._fp.close()


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = {"value": False}
    old_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        interrupted["value"] = True
        if callable(old_handler):
            old_handler(signum, frame)

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if interrupted["value"]:
            raise KeyboardInterrupt


def finite_array(value, shape, name):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != tuple(shape):
        raise ValueError("%s shape mismatch: got %s, expected %s" % (name, arr.shape, shape))
    if not np.isfinite(arr).all():
        raise ValueError("%s contains NaN or Inf" % name)
    return arr


def parse_joint_list(text):
    values = [item for item in str(text).replace(",", " ").split() if item]
    if len(values) != 7:
        raise ValueError("start joints must contain 7 numbers, got %s: %r" % (len(values), text))
    return finite_array([float(item) for item in values], (7,), "start_joints")


def normalize_rgb(image, height, width, *, bgr=False):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError("camera frame must be HxWx3, got %s" % (image.shape,))
    image = image.astype(np.uint8, copy=False)
    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


class MockObservationSource:
    def __init__(self, height, width):
        self.height = int(height)
        self.width = int(width)
        self.state = DEFAULT_START_STATE.copy()

    def read(self):
        images = {}
        for idx, key in enumerate(CAMERA_KEYS):
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            image[..., idx] = 80
            cv2.putText(
                image,
                key.split(".")[-1],
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            images[key] = image
        return self.state.copy(), images

    def set_state(self, state):
        self.state = np.asarray(state, dtype=np.float32).copy()

    def close(self):
        pass


class OpenCVCameraSource:
    def __init__(self, cam_high_id, cam_wrist_id, height, width):
        self.height = int(height)
        self.width = int(width)
        self.caps = {}
        for key, dev in ((CAMERA_KEYS[0], cam_high_id), (CAMERA_KEYS[1], cam_wrist_id)):
            cap = cv2.VideoCapture(dev)
            if not cap.isOpened():
                raise RuntimeError("failed to open OpenCV camera %s device=%s" % (key, dev))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.caps[key] = cap
            logging.info("opened OpenCV camera key=%s device=%s", key, dev)

    def read_images(self):
        images = {}
        for key, cap in self.caps.items():
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError("failed to read OpenCV camera %s" % key)
            images[key] = normalize_rgb(bgr, self.height, self.width, bgr=True)
        return images

    def close(self):
        for cap in self.caps.values():
            cap.release()


class SDKCameraSource:
    def __init__(self, sdk_root, height, width, fps, timeout_ms, read_retries, serials):
        if sdk_root:
            sys.path.insert(0, str(Path(sdk_root).expanduser()))
        try:
            from camera import Camera
        except Exception as exc:
            raise RuntimeError(
                "无法导入 camera.Camera。请设置 --camera-sdk-root 或 REAL_CAMERA_SDK_ROOT，"
                "或改用 --camera-backend realsense/opencv。"
            ) from exc

        self.height = int(height)
        self.width = int(width)
        self.timeout_ms = int(timeout_ms)
        self.read_retries = int(read_retries)
        self.cameras = {}
        for key, serial in serials.items():
            self.cameras[key] = self._create_camera(Camera, serial, fps)
            logging.info("opened SDK camera key=%s serial=%s", key, serial)

    def _create_camera(self, Camera, serial, fps):
        attempts = (
            lambda: Camera(serial_number=serial, resolution=(self.height, self.width), fps=int(fps)),
            lambda: Camera(serial=serial, resolution=(self.height, self.width), fps=int(fps)),
            lambda: Camera(serial, (self.height, self.width), int(fps)),
        )
        last_error = None
        for factory in attempts:
            try:
                return factory()
            except Exception as exc:
                last_error = exc
        raise RuntimeError("failed to create SDK camera serial=%s: %s" % (serial, last_error))

    def _capture_frame(self, camera):
        capture_frame = camera.capture_frame
        try:
            params = inspect.signature(capture_frame).parameters
        except (TypeError, ValueError):
            params = {}
        if "timeout_ms" in params:
            return capture_frame(timeout_ms=self.timeout_ms)
        if "timeout" in params:
            return capture_frame(timeout=self.timeout_ms / 1000.0)
        return capture_frame()

    def read_images(self):
        images = {}
        for key, camera in self.cameras.items():
            last_error = None
            for _ in range(self.read_retries):
                try:
                    frame = self._capture_frame(camera)
                    rgb = frame.get("rgb") if isinstance(frame, dict) else frame
                    if rgb is None:
                        raise RuntimeError("empty frame")
                    images[key] = normalize_rgb(rgb, self.height, self.width, bgr=False)
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(0.02)
            if key not in images:
                raise RuntimeError("failed to read SDK camera %s: %s" % (key, last_error))
        return images

    def close(self):
        for camera in self.cameras.values():
            close_fn = getattr(camera, "close", None) or getattr(camera, "release", None) or getattr(camera, "stop", None)
            if close_fn:
                close_fn()


class RealSenseCameraSource:
    def __init__(self, height, width, fps, timeout_ms, read_retries, serials):
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise RuntimeError("无法导入 pyrealsense2；请安装 RealSense SDK 或改用 --camera-backend sdk/opencv。") from exc
        self.rs = rs
        self.height = int(height)
        self.width = int(width)
        self.timeout_ms = int(timeout_ms)
        self.read_retries = int(read_retries)
        self.pipelines = {}
        for key, serial in serials.items():
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(str(serial))
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, int(fps))
            pipeline.start(config)
            self.pipelines[key] = pipeline
            logging.info("opened RealSense camera key=%s serial=%s", key, serial)

    def read_images(self):
        images = {}
        for key, pipeline in self.pipelines.items():
            last_error = None
            for _ in range(self.read_retries):
                try:
                    frames = pipeline.wait_for_frames(self.timeout_ms)
                    color = frames.get_color_frame()
                    if not color:
                        raise RuntimeError("empty color frame")
                    images[key] = normalize_rgb(np.asanyarray(color.get_data()), self.height, self.width, bgr=False)
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(0.02)
            if key not in images:
                raise RuntimeError("failed to read RealSense camera %s: %s" % (key, last_error))
        return images

    def close(self):
        for pipeline in self.pipelines.values():
            pipeline.stop()


def build_camera_source(args):
    serials = {
        CAMERA_KEYS[0]: args.head_camera_serial,
        CAMERA_KEYS[1]: args.wrist_camera_serial,
    }
    if args.camera_backend == "opencv":
        return OpenCVCameraSource(args.cam_high_id, args.cam_wrist_id, args.image_height, args.image_width)
    if args.camera_backend == "sdk":
        return SDKCameraSource(
            args.camera_sdk_root,
            args.image_height,
            args.image_width,
            args.camera_fps,
            args.camera_timeout_ms,
            args.camera_read_retries,
            serials,
        )
    if args.camera_backend == "realsense":
        return RealSenseCameraSource(
            args.image_height,
            args.image_width,
            args.camera_fps,
            args.camera_timeout_ms,
            args.camera_read_retries,
            serials,
        )

    errors = []
    for backend in ("sdk", "realsense"):
        try:
            args.camera_backend = backend
            return build_camera_source(args)
        except Exception as exc:
            errors.append("%s: %s" % (backend, exc))
    raise RuntimeError("auto camera backend failed: " + "; ".join(errors))


class RealManRobot:
    def __init__(self, args):
        self.robot_ip = args.robot_ip
        self.robot_port = int(args.robot_port)
        self.speed = int(args.robot_speed)
        self.reset_speed = int(args.reset_speed)
        self.use_canfd = bool(args.use_canfd)
        self.blocking_actions = bool(args.blocking_actions)
        self.gripper_blocking = bool(args.gripper_blocking)
        self.gripper_timeout = int(args.gripper_timeout)
        self.rm_sdk_root = args.rm_sdk_root
        self.arm = None
        self.last_gripper_command = None
        self.joint_limits_deg = None
        self.stop_sent = False

    def _load_local_rm_interface(self):
        if not self.rm_sdk_root:
            raise ImportError("rm_sdk_root is empty")
        root = Path(self.rm_sdk_root).expanduser()
        module_path = root if root.is_file() else root / "rm_robot_interface.py"
        if not module_path.exists():
            raise FileNotFoundError("rm_robot_interface.py not found under %s" % root)

        package_name = "_pi05_realman_sdk"
        package = types.ModuleType(package_name)
        package.__path__ = [str(module_path.parent)]
        sys.modules[package_name] = package

        spec = importlib.util.spec_from_file_location(package_name + ".rm_robot_interface", str(module_path))
        if spec is None or spec.loader is None:
            raise ImportError("failed to create import spec for %s" % module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def connect(self):
        if self.rm_sdk_root:
            root = Path(self.rm_sdk_root).expanduser()
            sys.path.insert(0, str(root if root.is_dir() else root.parent))
            sys.path.insert(0, str((root if root.is_dir() else root.parent).parent))
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        except Exception:
            try:
                from rm_robot_interface import RoboticArm, rm_thread_mode_e
            except Exception as exc:
                try:
                    module = self._load_local_rm_interface()
                    RoboticArm = module.RoboticArm
                    rm_thread_mode_e = module.rm_thread_mode_e
                except Exception as local_exc:
                    raise RuntimeError(
                        "无法导入 RealMan SDK。请在 realman 环境中设置 PYTHONPATH，"
                        "或通过 --rm-sdk-root 指向包含 Robotic_Arm/rm_robot_interface.py 的目录、"
                        "rm_robot_interface.py 文件，或其 vendor SDK 根目录。"
                    ) from local_exc
        logging.info("connecting RealMan robot %s:%s", self.robot_ip, self.robot_port)
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.arm.rm_create_robot_arm(self.robot_ip, self.robot_port)
        if getattr(handle, "id", -1) < 0:
            raise ConnectionError("RealMan robot connection failed: %s:%s" % (self.robot_ip, self.robot_port))
        logging.info("robot connected handle_id=%s", handle.id)
        self.joint_limits_deg = self.read_joint_limits()

    def _query_joint_limit_pair(self, min_name, max_name):
        min_fn = getattr(self.arm, min_name, None)
        max_fn = getattr(self.arm, max_name, None)
        if min_fn is None or max_fn is None:
            return None
        min_ret, min_pos = min_fn()
        max_ret, max_pos = max_fn()
        if min_ret != 0 or max_ret != 0:
            logging.warning("%s/%s returned %s/%s", min_name, max_name, min_ret, max_ret)
            return None
        low = np.asarray(min_pos[:7], dtype=np.float32)
        high = np.asarray(max_pos[:7], dtype=np.float32)
        if low.shape != (7,) or high.shape != (7,) or not np.isfinite(low).all() or not np.isfinite(high).all():
            logging.warning("invalid controller joint limits from %s/%s: min=%s max=%s", min_name, max_name, min_pos, max_pos)
            return None
        if not np.all(low < high):
            logging.warning("non-increasing controller joint limits from %s/%s: min=%s max=%s", min_name, max_name, low.tolist(), high.tolist())
            return None
        return np.stack([low, high], axis=1)

    def read_joint_limits(self):
        limits = self._query_joint_limit_pair("rm_get_joint_min_pos", "rm_get_joint_max_pos")
        if limits is None:
            limits = self._query_joint_limit_pair("rm_get_joint_drive_min_pos", "rm_get_joint_drive_max_pos")
        if limits is None:
            limits = FALLBACK_HARD_JOINT_LIMITS_DEG.copy()
            logging.warning("controller joint limits unavailable; using fallback hard limits: %s", limits.tolist())
        else:
            logging.info("controller joint limits deg: %s", limits.tolist())
        return limits.astype(np.float32)

    def close(self):
        if self.arm is not None:
            try:
                self.arm.rm_delete_robot_arm()
            except Exception:
                logging.exception("failed to disconnect RealMan robot")
        self.arm = None

    def stop_motion(self, reason="shutdown", *, clear_trajectory=True):
        if self.arm is None:
            return
        logging.warning("stopping RealMan motion reason=%s", reason)
        for name in ("rm_set_arm_stop", "rm_set_arm_slow_stop"):
            fn = getattr(self.arm, name, None)
            if fn is None:
                continue
            try:
                ret = fn()
                logging.warning("%s returned %s", name, ret)
                self.stop_sent = True
                break
            except Exception:
                logging.exception("%s failed", name)
        if clear_trajectory:
            for name in ("rm_set_delete_current_trajectory", "rm_set_arm_delete_trajectory"):
                fn = getattr(self.arm, name, None)
                if fn is None:
                    continue
                try:
                    ret = fn()
                    logging.info("%s returned %s", name, ret)
                except Exception:
                    logging.exception("%s failed", name)

    def read_state(self):
        ret, state = self.arm.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError("rm_get_current_arm_state returned %s" % ret)
        joints = np.asarray(state.get("joint", [])[:7], dtype=np.float32)
        if joints.shape != (7,):
            raise RuntimeError("robot joint state must be 7D, got %s" % (joints,))
        gripper_actpos = 0.0
        try:
            grip_ret, grip = self.arm.rm_get_gripper_state()
            if grip_ret == 0:
                gripper_actpos = float(grip.get("actpos", 0.0))
            else:
                logging.warning("rm_get_gripper_state returned %s; using 0", grip_ret)
        except Exception:
            logging.exception("failed to read gripper state; using 0")
        return np.concatenate([joints, [gripper_actpos]]).astype(np.float32)

    def hold_current(self):
        if self.arm is None:
            return
        current = self.read_state()
        self.execute_action(current, send_gripper=False)

    def _move_joints(self, joints, *, speed=None, blocking=None, context="joint_command"):
        joints = [float(v) for v in joints]
        speed = self.speed if speed is None else int(speed)
        blocking = self.blocking_actions if blocking is None else bool(blocking)
        if self.use_canfd and hasattr(self.arm, "rm_movej_canfd"):
            if blocking:
                raise RuntimeError("--use-canfd is non-blocking and cannot be used with blocking joint execution.")
            ret = self.arm.rm_movej_canfd(joints, False)
        else:
            ret = self.arm.rm_movej(joints, speed, 0, 0, 1 if blocking else 0)
        if ret != 0:
            if self.stop_sent:
                logging.warning("RealMan joint command returned %s after stop request; ignoring target=%s", ret, joints)
                return
            with contextlib.suppress(Exception):
                logging.error(
                    "RealMan rejected %s ret=%s target=%s current_state=%s joint_limits=%s",
                    context,
                    ret,
                    joints,
                    self.read_state().tolist(),
                    None if self.joint_limits_deg is None else self.joint_limits_deg.tolist(),
                )
            if hasattr(self.arm, "rm_clear_system_err"):
                try:
                    self.arm.rm_clear_system_err()
                except Exception:
                    logging.exception("failed to clear RealMan system error")
            raise RuntimeError("RealMan joint command returned %s: %s" % (ret, joints))

    def set_gripper_open(self, target_open, *, force=False, blocking=None):
        target_open = 1 if target_open else 0
        if not force and target_open == self.last_gripper_command:
            return
        blocking = self.gripper_blocking if blocking is None else bool(blocking)
        if target_open:
            ret = self.arm.rm_set_gripper_release(500, blocking, self.gripper_timeout)
        else:
            ret = self.arm.rm_set_gripper_pick_on(250, 200, blocking, self.gripper_timeout)
        if ret != 0:
            raise RuntimeError("RealMan gripper command returned %s target=%s" % (ret, target_open))
        self.last_gripper_command = target_open

    def reset_to_start(self, start_joints, *, settle_seconds=0.5):
        self.stop_sent = False
        joints = finite_array(start_joints, (7,), "start_joints")
        logging.info("reset robot to start joints=%s speed=%s blocking=True", np.round(joints, 4).tolist(), self.reset_speed)
        self._move_joints(joints, speed=self.reset_speed, blocking=True, context="reset_to_start")
        logging.info("open gripper after reset")
        self.set_gripper_open(1, force=True, blocking=True)
        if settle_seconds > 0:
            time.sleep(float(settle_seconds))
        logging.info("reset complete state=%s", np.round(self.read_state(), 4).tolist())

    def execute_action(self, action, send_gripper=True):
        # Keep execution ordered: first wait for the arm target, then update the gripper.
        self._move_joints(action[:7], blocking=self.blocking_actions, context="policy_action")
        if send_gripper:
            target_open = 1 if float(action[7]) >= 0.5 else 0
            self.set_gripper_open(target_open)


class RealObservationSource:
    def __init__(self, robot, camera_source):
        self.robot = robot
        self.camera_source = camera_source

    def read(self):
        return self.robot.read_state(), self.camera_source.read_images()

    def close(self):
        self.camera_source.close()


class CameraOnlyObservationSource:
    def __init__(self, camera_source):
        self.camera_source = camera_source

    def read(self):
        return DEFAULT_STATE.copy(), self.camera_source.read_images()

    def close(self):
        self.camera_source.close()


def load_soft_limits(norm_stats_path, margin_deg, disabled, hard_limits_deg=None, hard_limit_margin_deg=1.0):
    if disabled:
        limits = np.stack([np.full(7, -360.0, dtype=np.float32), np.full(7, 360.0, dtype=np.float32)], axis=1)
    else:
        q01 = DEFAULT_STATE_Q01.copy()
        q99 = DEFAULT_STATE_Q99.copy()
        if norm_stats_path:
            path = Path(norm_stats_path).expanduser()
            if path.exists():
                with path.open("r", encoding="utf-8") as fp:
                    payload = json.load(fp)
                stats = payload.get("norm_stats", payload)
                state_stats = stats.get("state", stats)
                if "q01" not in state_stats or "q99" not in state_stats:
                    raise ValueError("norm_stats missing q01/q99: %s" % path)
                q01 = np.asarray(state_stats["q01"][:7], dtype=np.float32)
                q99 = np.asarray(state_stats["q99"][:7], dtype=np.float32)
                logging.info("loaded q01/q99 soft limits from %s", path)
            else:
                logging.warning("norm_stats path not found, using embedded RealMan defaults: %s", path)

        margin = float(margin_deg)
        limits = np.stack([q01 - margin, q99 + margin], axis=1).astype(np.float32)

    if hard_limits_deg is not None:
        hard_limits = np.asarray(hard_limits_deg, dtype=np.float32)
        if hard_limits.shape != (7, 2) or not np.isfinite(hard_limits).all():
            raise ValueError("hard joint limits must be finite [7,2], got %s" % (hard_limits.shape,))
        margin = float(hard_limit_margin_deg)
        hard_low = hard_limits[:, 0] + margin
        hard_high = hard_limits[:, 1] - margin
        if not np.all(hard_low < hard_high):
            raise ValueError("hard joint limit margin too large: limits=%s margin=%s" % (hard_limits.tolist(), margin))
        limits[:, 0] = np.maximum(limits[:, 0], hard_low)
        limits[:, 1] = np.minimum(limits[:, 1], hard_high)

    if not np.all(limits[:, 0] < limits[:, 1]):
        raise ValueError("invalid effective joint limits: %s" % limits.tolist())
    logging.info("effective joint limits deg: %s", limits.tolist())
    return limits.astype(np.float32)


@dataclass
class SafetyConfig:
    soft_joint_limits_deg: np.ndarray
    gripper_open_threshold: float
    target_delta_limit_mode: str
    max_joint_delta_deg_per_step: float


class SafetyFilter:
    def __init__(self, config):
        self.config = config
        if config.target_delta_limit_mode not in ("off", "scale"):
            raise ValueError("target_delta_limit_mode must be off or scale")
        if config.target_delta_limit_mode == "scale" and config.max_joint_delta_deg_per_step <= 0:
            raise ValueError("max_joint_delta_deg_per_step must be positive when target_delta_limit_mode=scale")

    def filter_action(self, action, current_state):
        action = finite_array(action, (8,), "action").copy()
        current_state = finite_array(current_state, (8,), "current_state")
        raw = action.copy()
        limits = self.config.soft_joint_limits_deg
        action[:7] = np.minimum(np.maximum(action[:7], limits[:, 0]), limits[:, 1])
        action[7] = 1.0 if float(np.clip(action[7], 0.0, 1.0)) >= self.config.gripper_open_threshold else 0.0
        soft_limited = action.copy()
        joint_limit_delta_max = float(np.max(np.abs(raw[:7] - soft_limited[:7])))

        delta = action[:7] - current_state[:7]
        max_abs = float(np.max(np.abs(delta)))
        scaled = False
        if self.config.target_delta_limit_mode == "scale" and max_abs > self.config.max_joint_delta_deg_per_step:
            scale = self.config.max_joint_delta_deg_per_step / (max_abs + 1e-6)
            action[:7] = current_state[:7] + delta * scale
            scaled = True

        return action.astype(np.float32), {
            "raw_action": raw.tolist(),
            "soft_limited_action": soft_limited.tolist(),
            "clipped_action": action.tolist(),
            "max_delta_before": max_abs,
            "delta_scaled": scaled,
            "joint_limited": joint_limit_delta_max > 1e-5,
            "joint_limit_delta_max": joint_limit_delta_max,
            "effective_joint_limits": limits.tolist(),
            "target_delta_limit_mode": self.config.target_delta_limit_mode,
            "gripper_open_threshold": self.config.gripper_open_threshold,
        }


class Pi05RealManClient:
    def __init__(self, args, observation_source, robot, event_logger):
        self.args = args
        self.observation_source = observation_source
        self.robot = robot
        self.event_logger = event_logger
        self.stop_requested = False
        self.stop_reason = None
        self.action_plan = deque()
        hard_limits = getattr(robot, "joint_limits_deg", None)
        if hard_limits is None and args.enable_robot:
            hard_limits = FALLBACK_HARD_JOINT_LIMITS_DEG
        self.safety = SafetyFilter(
            SafetyConfig(
                soft_joint_limits_deg=load_soft_limits(
                    args.norm_stats_path,
                    args.soft_limit_margin_deg,
                    args.disable_soft_limits,
                    hard_limits,
                    args.hard_limit_margin_deg,
                ),
                gripper_open_threshold=args.gripper_open_threshold,
                target_delta_limit_mode=args.target_delta_limit_mode,
                max_joint_delta_deg_per_step=args.max_joint_delta_deg_per_step,
            )
        )
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        del frame
        self.request_stop("signal:%s" % signum)

    def request_stop(self, reason):
        if self.stop_requested:
            return
        self.stop_requested = True
        self.stop_reason = reason
        self.action_plan.clear()
        logging.warning("stop requested reason=%s; clearing action plan and stopping robot motion", reason)
        if self.args.enable_robot and self.robot is not None:
            self.robot.stop_motion(reason)

    def should_stop(self, start_time, step):
        if self.stop_requested:
            return True
        if self.args.stop_file and Path(self.args.stop_file).exists():
            logging.warning("stop file exists: %s", self.args.stop_file)
            self.request_stop("stop_file:%s" % self.args.stop_file)
            with contextlib.suppress(Exception):
                Path(self.args.stop_file).unlink()
                logging.info("removed consumed stop file: %s", self.args.stop_file)
            return True
        if self.args.max_steps and step >= self.args.max_steps:
            self.request_stop("max_steps:%s" % self.args.max_steps)
            return True
        if self.args.max_seconds and time.time() - start_time >= self.args.max_seconds:
            self.request_stop("max_seconds:%s" % self.args.max_seconds)
            return True
        return False

    def make_observation(self, state, images):
        head = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(images[CAMERA_KEYS[0]], self.args.resize_size, self.args.resize_size)
        )
        wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(images[CAMERA_KEYS[1]], self.args.resize_size, self.args.resize_size)
        )
        return {
            "observation/image": np.ascontiguousarray(head),
            "observation/wrist_image": np.ascontiguousarray(wrist),
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": self.args.task_text,
        }

    def infer_actions(self, policy_client, state, images, step):
        obs = self.make_observation(state, images)
        started = time.perf_counter()
        with prevent_keyboard_interrupt():
            result = policy_client.infer(obs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        chunk = np.asarray(result.get("actions"), dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 8:
            raise RuntimeError("OpenPI action chunk must have shape [N,8], got %s" % (chunk.shape,))
        if not np.isfinite(chunk).all():
            raise RuntimeError("OpenPI action chunk contains NaN or Inf")
        timing = result.get("server_timing", {})
        self.event_logger.write(
            {
                "event": "server_response",
                "step": step,
                "latency_ms_client": elapsed_ms,
                "server_timing": timing,
                "action_chunk": chunk.tolist(),
            }
        )
        logging.info("policy response step=%s chunk_shape=%s latency=%.1fms server_timing=%s", step, chunk.shape, elapsed_ms, timing)
        return chunk

    def reset_robot_before_inference(self):
        if not self.args.enable_robot or self.robot is None:
            return
        if not self.args.reset_to_start:
            logging.warning("reset-to-start disabled by --no-reset-to-start")
            return
        self.event_logger.write(
            {
                "event": "reset_to_start_begin",
                "start_joints": self.args.start_joints.tolist(),
                "open_gripper": True,
                "blocking": True,
            }
        )
        self.robot.reset_to_start(self.args.start_joints, settle_seconds=self.args.reset_settle_seconds)
        state = self.robot.read_state()
        self.event_logger.write(
            {
                "event": "reset_to_start_done",
                "state": state.tolist(),
            }
        )

    def refill_action_plan(self, policy_client, state, images, step):
        chunk = self.infer_actions(policy_client, state, images, step)
        if self.args.blocking_actions:
            # Blocking execution has variable duration. Replan from a fresh image/state after each completed action.
            take = 1
        elif self.args.replan_steps <= 0:
            take = len(chunk)
        else:
            take = min(self.args.replan_steps, len(chunk))
        self.action_plan.clear()
        self.action_plan.extend(chunk[:take])

    def execute_one_action(self, action, current_state, step):
        if self.stop_requested:
            raise KeyboardInterrupt("stop requested before action execution: %s" % self.stop_reason)
        filtered, info = self.safety.filter_action(action, current_state)
        self.event_logger.write(
            {
                "event": "execute_action",
                "dry_run": not self.args.enable_robot,
                "step": step,
                "current_state": np.asarray(current_state, dtype=np.float32).tolist(),
                **info,
            }
        )
        if self.args.enable_robot:
            if self.stop_requested:
                raise KeyboardInterrupt("stop requested before robot command: %s" % self.stop_reason)
            logging.info("execute step=%s target=%s", step, np.round(filtered, 4).tolist())
            self.robot.execute_action(filtered, send_gripper=True)
            if self.stop_requested:
                self.robot.stop_motion("stop_after_robot_command:%s" % self.stop_reason)
                raise KeyboardInterrupt("stop requested after robot command: %s" % self.stop_reason)
            return self.robot.read_state()

        logging.info("dry-run step=%s target=%s", step, np.round(filtered, 4).tolist())
        next_state = np.asarray(current_state, dtype=np.float32).copy()
        next_state[:7] = filtered[:7]
        next_state[7] = filtered[7]
        set_state = getattr(self.observation_source, "set_state", None)
        if set_state:
            set_state(next_state)
        return next_state

    def run_test_camera(self, policy_client):
        for idx in range(self.args.test_camera_samples):
            state, images = self.observation_source.read()
            chunk = self.infer_actions(policy_client, state, images, idx)
            self.event_logger.write(
                {
                    "event": "test_camera_sample",
                    "sample": idx,
                    "state": state.tolist(),
                    "image_shapes": {key: list(value.shape) for key, value in images.items()},
                    "action_shape": list(chunk.shape),
                }
            )
            logging.info("test-camera sample=%s image_shapes=%s action_shape=%s", idx, {k: v.shape for k, v in images.items()}, chunk.shape)

    def run_rollout(self, policy_client):
        start_time = time.time()
        step = 0
        period = 1.0 / float(self.args.control_hz)
        while not self.should_stop(start_time, step):
            loop_started = time.perf_counter()
            state, images = self.observation_source.read()
            if not self.action_plan:
                self.refill_action_plan(policy_client, state, images, step)
            action = self.action_plan.popleft()
            next_state = self.execute_one_action(action, state, step)
            self.event_logger.write({"event": "step_done", "step": step, "next_state": next_state.tolist()})
            step += 1
            elapsed = time.perf_counter() - loop_started
            if elapsed < period:
                time.sleep(period - elapsed)

    def run(self):
        self.reset_robot_before_inference()
        logging.info("connecting OpenPI policy server host=%s port=%s", self.args.server_host, self.args.server_port)
        policy_client = websocket_client_policy.WebsocketClientPolicy(self.args.server_host, self.args.server_port)
        metadata = policy_client.get_server_metadata()
        logging.info("server metadata: %s", metadata)
        if metadata.get("debug_train_data", {}).get("enabled"):
            logging.warning("server training-data debug mode is enabled: %s", metadata["debug_train_data"])
        if metadata.get("robot") not in (None, "realman_rm75"):
            raise RuntimeError("server robot metadata mismatch: %s" % metadata.get("robot"))
        if metadata.get("action_dim") not in (None, 8):
            raise RuntimeError("server action_dim metadata mismatch: %s" % metadata.get("action_dim"))
        self.event_logger.write({"event": "server_metadata", "metadata": metadata})

        if self.args.test_camera:
            self.run_test_camera(policy_client)
        else:
            self.run_rollout(policy_client)


def build_sources(args):
    if args.mock_observation:
        return MockObservationSource(args.image_height, args.image_width), None

    camera_source = build_camera_source(args)
    if args.test_camera:
        return CameraOnlyObservationSource(camera_source), None

    robot = RealManRobot(args)
    robot.connect()
    return RealObservationSource(robot, camera_source), robot


def parse_args():
    parser = argparse.ArgumentParser(description="RealMan RM75 client for OpenPI pi0.5 websocket policy inference.")
    parser.add_argument("--server-host", default=os.getenv("PI05_SERVER_HOST", os.getenv("REAL_SERVER_HOST", "127.0.0.1")))
    parser.add_argument("--server-port", type=int, default=int(os.getenv("PI05_SERVER_PORT", os.getenv("REAL_SERVER_PORT", "8000"))))
    parser.add_argument("--task-text", default=os.getenv("PI05_REAL_TASK_TEXT", DEFAULT_PROMPT))
    parser.add_argument("--enable-robot", action="store_true", help="显式允许向机械臂发送运动命令；默认 dry-run。")
    parser.add_argument("--mock-observation", action="store_true", help="不连接相机/机械臂，用假 observation 测 websocket。")
    parser.add_argument("--test-camera", action="store_true", help="只连接相机并请求动作，不连接机械臂。")
    parser.add_argument("--test-camera-samples", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="0 表示不按步数停止。")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 表示不按时间停止。")
    parser.add_argument("--replan-steps", type=int, default=1, help="每次 server 返回 action chunk 后执行几步；0 表示执行完整 chunk。")
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--save-rollout-dir", default=None)
    parser.add_argument("--stop-file", default=DEFAULT_STOP_FILE)

    parser.add_argument("--camera-backend", choices=["auto", "sdk", "realsense", "opencv"], default=os.getenv("PI05_CAMERA_BACKEND", "auto"))
    parser.add_argument("--camera-sdk-root", default=os.getenv("REAL_CAMERA_SDK_ROOT", ""))
    parser.add_argument("--head-camera-serial", default=os.getenv("PI05_HEAD_CAMERA_SERIAL", DEFAULT_CAMERA_SERIALS[CAMERA_KEYS[0]]))
    parser.add_argument("--wrist-camera-serial", default=os.getenv("PI05_WRIST_CAMERA_SERIAL", DEFAULT_CAMERA_SERIALS[CAMERA_KEYS[1]]))
    parser.add_argument("--cam-high-id", default=os.getenv("PI05_CAM_HIGH_ID", "0"))
    parser.add_argument("--cam-wrist-id", default=os.getenv("PI05_CAM_WRIST_ID", "1"))
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--camera-read-retries", type=int, default=2)

    parser.add_argument("--rm-sdk-root", default=os.getenv("REALMAN_RM_SDK_ROOT", ""))
    parser.add_argument("--robot-ip", default=os.getenv("REALMAN_ROBOT_IP", "192.168.1.18"))
    parser.add_argument("--robot-port", type=int, default=int(os.getenv("REALMAN_ROBOT_PORT", "8080")))
    parser.add_argument("--robot-speed", type=int, default=int(os.getenv("REALMAN_ROBOT_SPEED", "10")))
    parser.add_argument("--use-canfd", action="store_true")
    parser.add_argument("--reset-to-start", action=argparse.BooleanOptionalAction, default=os.getenv("PI05_RESET_TO_START", "1").lower() not in {"0", "false", "no", "off"})
    parser.add_argument(
        "--start-joints",
        default=os.getenv("PI05_START_JOINTS", "3.15,-2.35,7.65,69.75,175.52,-89.94,96.94"),
        help="启动真机推理前恢复的 7D 起始关节角，逗号或空格分隔。",
    )
    parser.add_argument("--reset-speed", type=int, default=int(os.getenv("PI05_RESET_SPEED", os.getenv("REALMAN_ROBOT_SPEED", "30"))))
    parser.add_argument("--reset-settle-seconds", type=float, default=float(os.getenv("PI05_RESET_SETTLE_SECONDS", "0.5")))
    parser.add_argument("--blocking-actions", action=argparse.BooleanOptionalAction, default=os.getenv("PI05_BLOCKING_ACTIONS", "1").lower() not in {"0", "false", "no", "off"})
    parser.add_argument("--gripper-blocking", action=argparse.BooleanOptionalAction, default=os.getenv("PI05_GRIPPER_BLOCKING", "1").lower() not in {"0", "false", "no", "off"})
    parser.add_argument("--gripper-timeout", type=int, default=int(os.getenv("PI05_GRIPPER_TIMEOUT", "10")))

    parser.add_argument("--norm-stats-path", default=os.getenv("PI05_REAL_NORM_STATS_PATH", str(DEFAULT_NORM_STATS_PATH)))
    parser.add_argument("--soft-limit-margin-deg", type=float, default=5.0)
    parser.add_argument("--hard-limit-margin-deg", type=float, default=float(os.getenv("PI05_HARD_LIMIT_MARGIN_DEG", "1.0")))
    parser.add_argument("--disable-soft-limits", action="store_true")
    parser.add_argument("--target-delta-limit-mode", choices=["off", "scale"], default=os.getenv("PI05_TARGET_DELTA_LIMIT_MODE", "off"))
    parser.add_argument("--max-joint-delta-deg-per-step", type=float, default=2.0)
    parser.add_argument("--gripper-open-threshold", type=float, default=float(os.getenv("PI05_GRIPPER_OPEN_THRESHOLD", "0.5")))

    args = parser.parse_args()
    if args.test_camera_samples <= 0:
        raise ValueError("--test-camera-samples must be positive")
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be positive")
    if args.replan_steps < 0:
        raise ValueError("--replan-steps must be >= 0")
    if not 0.0 <= args.gripper_open_threshold <= 1.0:
        raise ValueError("--gripper-open-threshold must be in [0, 1]")
    if args.camera_read_retries <= 0:
        raise ValueError("--camera-read-retries must be positive")
    if args.camera_timeout_ms <= 0:
        raise ValueError("--camera-timeout-ms must be positive")
    if args.hard_limit_margin_deg < 0:
        raise ValueError("--hard-limit-margin-deg must be >= 0")
    if args.reset_speed <= 0:
        raise ValueError("--reset-speed must be positive")
    if args.reset_settle_seconds < 0:
        raise ValueError("--reset-settle-seconds must be >= 0")
    if args.gripper_timeout <= 0:
        raise ValueError("--gripper-timeout must be positive")
    args.start_joints = parse_joint_list(args.start_joints)
    if args.blocking_actions and args.use_canfd:
        raise ValueError("--use-canfd is non-blocking follow mode and cannot be combined with --blocking-actions.")
    return args


def main():
    args = parse_args()
    log_path, events_path = setup_logging(args.save_rollout_dir)
    logging.info("log_path=%s events_path=%s", log_path, events_path)
    logging.info("dry_run=%s enable_robot=%s mock_observation=%s test_camera=%s", not args.enable_robot, args.enable_robot, args.mock_observation, args.test_camera)
    logging.info("camera_keys=%s serials=%s", CAMERA_KEYS, {
        CAMERA_KEYS[0]: args.head_camera_serial,
        CAMERA_KEYS[1]: args.wrist_camera_serial,
    })
    logging.info(
        "policy_input_keys=observation/image, observation/wrist_image, observation/state, prompt; action_space=8D absolute joints + gripper"
    )
    logging.info(
        "reset_to_start=%s start_joints=%s reset_speed=%s reset_settle_seconds=%s",
        args.reset_to_start,
        np.round(args.start_joints, 4).tolist(),
        args.reset_speed,
        args.reset_settle_seconds,
    )
    logging.info(
        "blocking_actions=%s gripper_blocking=%s gripper_timeout=%s replan_steps=%s",
        args.blocking_actions,
        args.gripper_blocking,
        args.gripper_timeout,
        args.replan_steps,
    )
    if args.blocking_actions and args.replan_steps != 1:
        logging.warning(
            "blocking action mode ignores --replan-steps=%s; it executes only the first action from each chunk, "
            "then captures fresh state/images before the next policy call.",
            args.replan_steps,
        )
    if not args.enable_robot:
        logging.info("dry-run active: no robot motion commands will be sent")
    if args.stop_file and Path(args.stop_file).exists():
        try:
            Path(args.stop_file).unlink()
            logging.warning("removed stale stop file before startup: %s", args.stop_file)
        except Exception:
            logging.exception("failed to remove stale stop file before startup: %s", args.stop_file)

    event_logger = JsonlLogger(events_path)
    observation_source = None
    robot = None
    client = None
    interrupted = False
    failed = False
    try:
        observation_source, robot = build_sources(args)
        client = Pi05RealManClient(args, observation_source, robot, event_logger)
        client.run()
    except KeyboardInterrupt:
        interrupted = True
        logging.warning("KeyboardInterrupt received; stopping rollout")
    except Exception:
        failed = True
        logging.exception("client failed; stopping rollout")
        raise
    finally:
        if robot is not None and args.enable_robot:
            should_stop_robot = interrupted or failed or (client is not None and client.stop_requested)
            if should_stop_robot:
                try:
                    robot.stop_motion("shutdown")
                except Exception:
                    logging.exception("failed to stop robot motion on shutdown")
            else:
                try:
                    robot.hold_current()
                except Exception:
                    logging.exception("failed to hold current position on shutdown")
            robot.close()
        elif robot is not None:
            robot.close()
        if observation_source is not None:
            observation_source.close()
        if args.stop_file and Path(args.stop_file).exists():
            with contextlib.suppress(Exception):
                Path(args.stop_file).unlink()
                logging.info("removed stop file on shutdown: %s", args.stop_file)
        event_logger.close()
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
