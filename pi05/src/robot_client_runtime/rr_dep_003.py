"""RR-DEP-003 live camera and operator-gated RealMan micro-motion runner."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tomllib
from typing import Any, Sequence
import uuid

from .rr_dep_003_motion import (
    BoundedCommandExecutor,
    MotionRejected,
    MotionThresholds,
    StateSample,
    build_arm_targets,
    select_q7_direction,
)


TASK_ID = "RR-DEP-003"
RUN_ID = "RR-DEP-003-RUN-001"
WORKSPACE_PHRASE = "RR-DEP-003 WORKSPACE CLEAR E-STOP READY"
LEFT_PHRASE = "AUTHORIZE RR-DEP-003 LEFT Q7"
RIGHT_PHRASE = "AUTHORIZE RR-DEP-003 RIGHT Q7"
ROLE_ORDER = ("head", "left_wrist", "right_wrist")
SOURCE_KEYS = {"head": "high_rgb", "left_wrist": "left_wrist_rgb", "right_wrist": "right_wrist_rgb"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(json_safe(payload), stream, sort_keys=False, allow_unicode=True)


class EventBundle:
    FILES = {
        "operator": "operator-authorizations.jsonl",
        "command": "command-ledger.jsonl",
        "state": "robot-state-trajectory.jsonl",
        "safety": "safety-events.jsonl",
        "camera": "camera-frames.jsonl",
        "ack": "acknowledgements.jsonl",
    }

    def __init__(self, root: Path) -> None:
        self.root = root
        self._locks = {name: threading.Lock() for name in self.FILES}
        self._streams = {}
        for name, relative in self.FILES.items():
            path = root / relative
            self._streams[name] = path.open("a", encoding="utf-8", buffering=1)

    def emit(self, name: str, event: str, payload: dict[str, Any]) -> None:
        record = {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "event": event,
            "utc": utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            **json_safe(payload),
        }
        with self._locks[name]:
            self._streams[name].write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def close(self) -> None:
        for stream in self._streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_sources(manifest_path: Path) -> dict[str, Any]:
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split("\t") != [
        "source_path", "snapshot_path", "size_bytes", "mtime_utc", "sha256"
    ]:
        raise MotionRejected("invalid_source_manifest_header")
    checked = []
    for line in lines[1:]:
        source_text, snapshot_text, size_text, _mtime, expected = line.split("\t")
        source = Path(source_text)
        snapshot = Path(snapshot_text)
        if not source.is_file() or not snapshot.is_file():
            raise MotionRejected(f"frozen_source_missing:{source}")
        source_digest = sha256_file(source)
        snapshot_digest = sha256_file(snapshot)
        if source_digest != expected or snapshot_digest != expected:
            raise MotionRejected(f"safety_critical_source_changed_after_freeze:{source}")
        if snapshot.stat().st_size != int(size_text):
            raise MotionRejected(f"frozen_source_size_mismatch:{snapshot}")
        checked.append({"source": str(source), "sha256": expected, "size_bytes": int(size_text)})
    return {
        "status": "pass",
        "checked_files": len(checked),
        "manifest_sha256": sha256_file(manifest_path),
        "files": checked,
    }


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    camera = config.get("cameras", {})
    robots = config.get("robots", {})
    serials = [str(camera.get(role, "")) for role in ROLE_ORDER]
    if any(not serial for serial in serials) or len(set(serials)) != 3:
        raise MotionRejected("config_requires_three_distinct_camera_serials")
    for side in ("left", "right"):
        arm = robots.get(side, {})
        if not arm.get("ip") or int(arm.get("port", 0)) <= 0:
            raise MotionRejected(f"missing_robot_address:{side}")
    if [int(camera.get("width", 0)), int(camera.get("height", 0))] != [640, 480]:
        raise MotionRejected("camera_config_not_640x480")
    return config


def resize_with_pad_rgb(image_bgr: Any, size: int = 224) -> Any:
    import numpy as np
    from PIL import Image

    array = np.asarray(image_bgr)
    if array.shape != (480, 640, 3) or array.dtype != np.uint8:
        raise MotionRejected(f"invalid_camera_sample:{array.shape}:{array.dtype}")
    rgb = np.ascontiguousarray(array[:, :, ::-1])
    source = Image.fromarray(rgb, mode="RGB")
    scale = min(size / source.width, size / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (size, size))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    result = np.asarray(canvas, dtype=np.uint8)
    if result.shape != (size, size, 3) or result.dtype != np.uint8:
        raise MotionRejected("resize_with_pad_output_invalid")
    return result


def validate_camera_binding(
    configured: dict[str, str], enumerated: list[dict[str, str]]
) -> dict[str, str]:
    if set(configured) != set(ROLE_ORDER):
        raise MotionRejected("camera_binding_roles_must_be_exact")
    serials = [str(configured[role]) for role in ROLE_ORDER]
    if any(not serial for serial in serials) or len(set(serials)) != 3:
        raise MotionRejected("camera_binding_serials_must_be_distinct")
    present = {str(item.get("serial", "")) for item in enumerated}
    missing = {role: serial for role, serial in configured.items() if serial not in present}
    if missing:
        raise MotionRejected(f"configured_camera_serials_not_enumerated:{missing}")
    return {role: str(configured[role]) for role in ROLE_ORDER}


def validate_camera_frame(
    role: str,
    frame: Any,
    *,
    frame_number: int,
    sdk_timestamp_ms: float,
    receive_monotonic_ns: int,
    previous_frame_number: int | None = None,
    previous_receive_monotonic_ns: int | None = None,
    require_contiguous: bool = False,
) -> None:
    import numpy as np

    array = np.asarray(frame)
    if role not in ROLE_ORDER:
        raise MotionRejected(f"unknown_camera_role:{role}")
    if array.shape != (480, 640, 3) or array.dtype != np.uint8:
        raise MotionRejected(f"invalid_live_frame:{role}:{array.shape}:{array.dtype}")
    if frame_number < 0 or not math.isfinite(float(sdk_timestamp_ms)):
        raise MotionRejected(f"invalid_camera_timestamp_or_number:{role}")
    if previous_receive_monotonic_ns is not None and receive_monotonic_ns <= previous_receive_monotonic_ns:
        raise MotionRejected(f"nonmonotonic_camera_receive_timestamp:{role}")
    if (
        require_contiguous
        and previous_frame_number is not None
        and frame_number != previous_frame_number + 1
    ):
        raise MotionRejected(
            f"native_frame_gap:{role}:previous={previous_frame_number}:current={frame_number}"
        )


class _VideoWriter:
    def __init__(self, path: Path, *, width: int, height: int, fps: int) -> None:
        self.path = path
        self.stderr_path = path.with_suffix(".ffmpeg.stderr.log")
        self._stderr = self.stderr_path.open("xb")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-vaapi_device", "/dev/dri/renderD128",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0", "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi", "-qp", "18", "-bf", "0",
            "-y", str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self._stderr)
        self.frames = 0

    def write(self, frame: Any) -> None:
        if self.process.poll() is not None:
            raise MotionRejected(f"ffmpeg_exited_early:{self.path}:{self.process.returncode}")
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            return_code = self.process.wait(timeout=10.0)
        except subprocess.TimeoutExpired as exc:
            raise MotionRejected(f"ffmpeg_did_not_close_naturally:{self.path}") from exc
        finally:
            self._stderr.flush()
            os.fsync(self._stderr.fileno())
            self._stderr.close()
        if return_code != 0:
            raise MotionRejected(f"ffmpeg_failed:{self.path}:{return_code}")


class CameraRecorder:
    def __init__(self, root: Path, label: str, *, width: int, height: int, fps: int) -> None:
        self.root = root / label
        self.root.mkdir(parents=True, exist_ok=False)
        self.started_monotonic_ns = time.monotonic_ns()
        self.stopped_monotonic_ns: int | None = None
        self._writers = {
            role: _VideoWriter(self.root / f"{SOURCE_KEYS[role]}.mp4", width=width, height=height, fps=fps)
            for role in ROLE_ORDER
        }
        self._queues = {role: queue.Queue(maxsize=fps * 10) for role in ROLE_ORDER}
        self._worker_errors: list[str] = []
        self._workers = {
            role: threading.Thread(target=self._writer_loop, args=(role,), name=f"camera-writer-{role}")
            for role in ROLE_ORDER
        }
        for worker in self._workers.values():
            worker.start()
        self._metadata = (self.root / "camera-frames.jsonl").open("x", encoding="utf-8", buffering=1)
        self._metadata_lock = threading.Lock()
        self._write_condition = threading.Condition()
        self._active_writes = 0
        self._closing = False
        self._previous_frame_number: dict[str, int | None] = {role: None for role in ROLE_ORDER}
        self._first_receive_ns: dict[str, int | None] = {role: None for role in ROLE_ORDER}
        self._last_receive_ns: dict[str, int | None] = {role: None for role in ROLE_ORDER}
        self._errors: list[str] = []

    def _writer_loop(self, role: str) -> None:
        work_queue = self._queues[role]
        while True:
            item = work_queue.get()
            try:
                if item is None:
                    return
                self._writers[role].write(item)
            except Exception as exc:
                self._worker_errors.append(f"{role}:{exc!r}")
                return
            finally:
                work_queue.task_done()

    def write(
        self,
        role: str,
        frame: Any,
        *,
        frame_number: int,
        sdk_timestamp_ms: float,
        timestamp_domain: str,
        receive_monotonic_ns: int,
    ) -> None:
        with self._write_condition:
            if self._closing:
                return
            self._active_writes += 1
        previous = self._previous_frame_number[role]
        try:
            if self._worker_errors:
                raise MotionRejected("camera_encoder_worker_failed:" + ";".join(self._worker_errors))
            validate_camera_frame(
                role,
                frame,
                frame_number=frame_number,
                sdk_timestamp_ms=sdk_timestamp_ms,
                receive_monotonic_ns=receive_monotonic_ns,
                previous_frame_number=previous,
                previous_receive_monotonic_ns=self._last_receive_ns[role],
                require_contiguous=True,
            )
            try:
                self._queues[role].put_nowait(frame.copy())
            except queue.Full as exc:
                raise MotionRejected(f"camera_encoder_queue_full:{role}") from exc
            self._previous_frame_number[role] = frame_number
            self._first_receive_ns[role] = self._first_receive_ns[role] or receive_monotonic_ns
            self._last_receive_ns[role] = receive_monotonic_ns
            record = {
                "role": role,
                "source_key": SOURCE_KEYS[role],
                "frame_number": frame_number,
                "sdk_timestamp_ms": sdk_timestamp_ms,
                "timestamp_domain": timestamp_domain,
                "receive_monotonic_ns": receive_monotonic_ns,
            }
            with self._metadata_lock:
                self._metadata.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        except MotionRejected as exc:
            self._errors.append(str(exc))
            raise
        finally:
            with self._write_condition:
                self._active_writes -= 1
                self._write_condition.notify_all()

    def close(self) -> dict[str, Any]:
        with self._write_condition:
            self._closing = True
            while self._active_writes:
                self._write_condition.wait()
        for role, worker in self._workers.items():
            if worker.is_alive():
                self._queues[role].put(None)
            worker.join(timeout=30.0)
            if worker.is_alive():
                self._worker_errors.append(f"{role}:encoder_worker_did_not_drain")
        errors = []
        for role, writer in self._writers.items():
            try:
                writer.close()
            except Exception as exc:
                errors.append(f"{role}:{exc}")
        self.stopped_monotonic_ns = time.monotonic_ns()
        self._metadata.flush()
        os.fsync(self._metadata.fileno())
        self._metadata.close()
        if errors or self._errors or self._worker_errors:
            raise MotionRejected("camera_recording_failure:" + ";".join(errors + self._errors + self._worker_errors))
        roles = {}
        for role, writer in self._writers.items():
            first = self._first_receive_ns[role]
            last = self._last_receive_ns[role]
            if first is None or last is None or writer.frames < 2:
                raise MotionRejected(f"insufficient_recorded_frames:{role}")
            span_s = (last - first) / 1_000_000_000.0
            roles[role] = {
                "frames": writer.frames,
                "frame_span_sec": span_s,
                "measured_fps": (writer.frames - 1) / span_s if span_s > 0 else 0.0,
                "video": str(writer.path),
                "ffmpeg_stderr": str(writer.stderr_path),
            }
        return {
            "root": str(self.root),
            "wall_elapsed_sec": (self.stopped_monotonic_ns - self.started_monotonic_ns) / 1_000_000_000.0,
            "roles": roles,
        }


class RealSenseRig:
    def __init__(self, config: dict[str, Any], events: EventBundle) -> None:
        self.config = config
        self.events = events
        self.rs = None
        self.pipelines: dict[str, Any] = {}
        self.profiles: dict[str, Any] = {}
        self.binding: dict[str, Any] = {}
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest: dict[str, tuple[Any, dict[str, Any]]] = {}
        self._errors: dict[str, str] = {}
        self._recorder_lock = threading.Lock()
        self._recorder: CameraRecorder | None = None

    def enumerate_and_start(self) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        context = rs.context()
        enumerated = []
        by_serial = {}
        for device in context.query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            firmware = device.get_info(rs.camera_info.firmware_version)
            item = {"serial": serial, "name": name, "firmware": firmware}
            enumerated.append(item)
            by_serial[serial] = item
        expected = {role: str(self.config["cameras"][role]) for role in ROLE_ORDER}
        missing = {role: serial for role, serial in expected.items() if serial not in by_serial}
        self.binding = {"configured": expected, "enumerated": enumerated, "missing": missing}
        validate_camera_binding(expected, enumerated)
        width = int(self.config["cameras"]["width"])
        height = int(self.config["cameras"]["height"])
        fps = int(self.config["cameras"]["fps"])
        try:
            for role in ROLE_ORDER:
                pipeline = rs.pipeline()
                stream_config = rs.config()
                stream_config.enable_device(expected[role])
                stream_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                profile = pipeline.start(stream_config)
                video = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intrinsics = video.get_intrinsics()
                active = {
                    "width": video.width(),
                    "height": video.height(),
                    "fps": video.fps(),
                    "format": str(video.format()),
                    "intrinsics": {
                        "width": intrinsics.width,
                        "height": intrinsics.height,
                        "ppx": intrinsics.ppx,
                        "ppy": intrinsics.ppy,
                        "fx": intrinsics.fx,
                        "fy": intrinsics.fy,
                        "model": str(intrinsics.model),
                        "coeffs": list(intrinsics.coeffs),
                    },
                    **by_serial[expected[role]],
                }
                if [video.width(), video.height(), video.fps()] != [640, 480, fps]:
                    raise MotionRejected(f"active_camera_profile_mismatch:{role}:{active}")
                if video.format() != rs.format.bgr8:
                    raise MotionRejected(f"active_camera_format_mismatch:{role}:{video.format()}")
                self.pipelines[role] = pipeline
                self.profiles[role] = active
            for role in ROLE_ORDER:
                thread = threading.Thread(
                    target=self._capture_loop,
                    args=(role,),
                    name=f"rr-dep-003-camera-{role}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
        except Exception:
            self.close()
            raise

    def _capture_loop(self, role: str) -> None:
        import numpy as np

        pipeline = self.pipelines[role]
        while not self._stop.is_set():
            try:
                frames = pipeline.wait_for_frames(1000)
                color = frames.get_color_frame()
                if not color:
                    raise MotionRejected(f"missing_color_frame:{role}")
                frame = np.asanyarray(color.get_data())
                frame = np.ascontiguousarray(frame)
                metadata = {
                    "frame_number": int(color.get_frame_number()),
                    "sdk_timestamp_ms": float(color.get_timestamp()),
                    "timestamp_domain": str(color.get_frame_timestamp_domain()),
                    "receive_monotonic_ns": time.monotonic_ns(),
                }
                previous = self._latest.get(role)
                validate_camera_frame(
                    role,
                    frame,
                    frame_number=metadata["frame_number"],
                    sdk_timestamp_ms=metadata["sdk_timestamp_ms"],
                    receive_monotonic_ns=metadata["receive_monotonic_ns"],
                    previous_receive_monotonic_ns=(
                        previous[1]["receive_monotonic_ns"] if previous is not None else None
                    ),
                )
                with self._recorder_lock:
                    recorder = self._recorder
                if recorder is not None:
                    recorder.write(role, frame, **metadata)
                self.events.emit("camera", "frame", {"role": role, **metadata})
                with self._condition:
                    self._latest[role] = (frame.copy(), metadata)
                    self._condition.notify_all()
            except Exception as exc:
                if not self._stop.is_set():
                    with self._condition:
                        self._errors[role] = repr(exc)
                        self._condition.notify_all()
                    self.events.emit("safety", "camera_capture_failure", {"role": role, "error": repr(exc)})
                return

    def wait_ready(self, timeout_s: float = 8.0) -> None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._errors:
                    raise MotionRejected(f"camera_capture_errors:{self._errors}")
                if all(role in self._latest for role in ROLE_ORDER):
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MotionRejected("camera_warmup_timeout")
                self._condition.wait(remaining)

    def latest_images(self) -> dict[str, Any]:
        with self._condition:
            if not all(role in self._latest for role in ROLE_ORDER):
                raise MotionRejected("camera_frames_not_ready")
            return {role: self._latest[role][0].copy() for role in ROLE_ORDER}

    def healthy(self, stale_s: float = 0.25) -> bool:
        with self._condition:
            if self._errors or not all(role in self._latest for role in ROLE_ORDER):
                return False
            now = time.monotonic_ns()
            return all(
                0 <= now - self._latest[role][1]["receive_monotonic_ns"] <= int(stale_s * 1e9)
                for role in ROLE_ORDER
            )

    def start_recording(self, root: Path, label: str) -> None:
        recorder = CameraRecorder(
            root,
            label,
            width=int(self.config["cameras"]["width"]),
            height=int(self.config["cameras"]["height"]),
            fps=int(self.config["cameras"]["fps"]),
        )
        with self._recorder_lock:
            if self._recorder is not None:
                raise MotionRejected("camera_recorder_already_active")
            self._recorder = recorder

    def stop_recording(self) -> dict[str, Any]:
        with self._recorder_lock:
            recorder = self._recorder
            self._recorder = None
        if recorder is None:
            raise MotionRejected("camera_recorder_not_active")
        return recorder.close()

    def close(self) -> None:
        with self._recorder_lock:
            recorder = self._recorder
            self._recorder = None
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                self.events.emit("safety", "camera_recorder_close_failure", {"error": repr(exc)})
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.25)

        def stop_pipeline(role: str, pipeline: Any) -> None:
            try:
                pipeline.stop()
            except Exception as exc:
                self.events.emit(
                    "safety", "camera_close_failure", {"role": role, "error": repr(exc)}
                )

        stoppers = []
        for role, pipeline in self.pipelines.items():
            stopper = threading.Thread(
                target=stop_pipeline,
                args=(role, pipeline),
                name=f"rr-dep-003-camera-stop-{role}",
                daemon=True,
            )
            stopper.start()
            stoppers.append((role, stopper))
        deadline = time.monotonic() + 2.0
        for role, stopper in stoppers:
            stopper.join(timeout=max(0.0, deadline - time.monotonic()))
            if stopper.is_alive():
                self.events.emit("safety", "camera_close_timeout", {"role": role})
        for thread in self._threads:
            thread.join(timeout=0.25)
            if thread.is_alive():
                self.events.emit(
                    "safety", "camera_thread_close_timeout", {"thread": thread.name}
                )


class RealManBackend:
    def __init__(self, config: dict[str, Any], events: EventBundle) -> None:
        self.config = config
        self.events = events
        self.module = None
        self.arms: dict[str, Any] = {}
        self.locks = {side: threading.Lock() for side in ("left", "right")}
        self._closed = False

    def open(self) -> None:
        sdk_parent = str(Path(self.config["sources"]["deltawam_src"]).resolve())
        if sdk_parent not in sys.path:
            sys.path.insert(0, sdk_parent)
        module = importlib.import_module("realman_deployment.Robotic_Arm.rm_robot_interface")
        self.module = module
        left = module.RoboticArm(module.rm_thread_mode_e.RM_DUAL_MODE_E)
        right = module.RoboticArm()
        self.arms = {"left": left, "right": right}
        try:
            for side in ("left", "right"):
                arm_config = self.config["robots"][side]
                handle = self.arms[side].rm_create_robot_arm(str(arm_config["ip"]), int(arm_config["port"]))
                if handle.id == -1:
                    raise MotionRejected(f"robot_connection_failed:{side}")
        except Exception:
            self.close()
            raise

    def _call(self, side: str, method: str) -> Any:
        arm = self.arms[side]
        with self.locks[side]:
            return getattr(arm, method)()

    def gather_binding(self, side: str) -> dict[str, Any]:
        methods = (
            "rm_get_robot_info",
            "rm_get_arm_software_info",
            "rm_get_joint_software_version",
            "rm_get_joint_min_pos",
            "rm_get_joint_max_pos",
            "rm_get_joint_max_speed",
            "rm_get_joint_max_acc",
            "rm_get_joint_drive_min_pos",
            "rm_get_joint_drive_max_pos",
            "rm_get_joint_drive_max_speed",
            "rm_get_joint_drive_max_acc",
            "rm_get_joint_en_state",
            "rm_get_joint_err_flag",
            "rm_get_controller_state",
            "rm_get_arm_power_state",
            "rm_get_collision_stage",
            "rm_get_collision_detection",
            "rm_get_self_collision_enable",
            "rm_get_current_arm_state",
        )
        result = {
            "address": json_safe(self.config["robots"][side]),
            "sdk_api_version": json_safe(self.module.rm_api_version()),
            "joint_order": [f"q{index}" for index in range(1, 8)],
            "canonical_unit": "radian",
            "sdk_joint_unit": "degree",
            "command": "rm_movej_absolute_degrees_v_percent_nonblocking_plus_Client_feedback_block",
            "slow_stop": "rm_set_arm_slow_stop",
        }
        for method in methods:
            try:
                result[method] = json_safe(self._call(side, method))
            except Exception as exc:
                result[method] = {"call_error": repr(exc)}
        return result

    def read_state(self, side: str, controller_error: int = 0) -> StateSample:
        code, state = self._call(side, "rm_get_current_arm_state")
        joints_deg = state.get("joint", []) if isinstance(state, dict) else []
        joints_rad = tuple(math.radians(float(value)) for value in joints_deg)
        errors = tuple(str(value) for value in state.get("err", {}).get("err", [])) if isinstance(state, dict) else ()
        return StateSample(
            side=side,
            joints_rad=joints_rad,
            receive_monotonic_ns=time.monotonic_ns(),
            sdk_status=int(code),
            arm_errors=errors,
            controller_error=int(controller_error),
        )

    def controller_error(self, side: str) -> int:
        state = self._call(side, "rm_get_controller_state")
        if int(state.get("return_code", -1)) != 0:
            raise MotionRejected(f"controller_state_call_failed:{side}:{state}")
        return int(state.get("system_error", -1))

    def command_absolute(self, side: str, target_rad: Sequence[float], speed_percent: int) -> int:
        if speed_percent < 1 or speed_percent > 5:
            raise MotionRejected(f"invalid_speed_percent:{speed_percent}")
        target_deg = [math.degrees(float(value)) for value in target_rad]
        with self.locks[side]:
            return int(self.arms[side].rm_movej(target_deg, speed_percent, 0, 0, 0))

    def slow_stop(self, side: str) -> int:
        with self.locks[side]:
            status = int(self.arms[side].rm_set_arm_slow_stop())
        self.events.emit("safety", "slow_stop", {"side": side, "sdk_status": status})
        return status

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for side, arm in self.arms.items():
            if hasattr(arm, "handle"):
                try:
                    arm.rm_delete_robot_arm()
                except Exception as exc:
                    self.events.emit("safety", "robot_handle_close_failure", {"side": side, "error": repr(exc)})
        if self.module is not None:
            try:
                self.module.RoboticArm.rm_destroy()
            except Exception as exc:
                self.events.emit("safety", "robot_sdk_destroy_failure", {"error": repr(exc)})


class RobotStatePoller:
    def __init__(self, backend: RealManBackend, events: EventBundle, *, hz: float = 20.0) -> None:
        self.backend = backend
        self.events = events
        self.hz = float(hz)
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest: dict[str, StateSample] = {}
        self._history = {side: deque(maxlen=400) for side in ("left", "right")}
        self._errors: dict[str, str] = {}
        self._threads: list[threading.Thread] = []
        self._controller_errors = {side: 0 for side in ("left", "right")}

    def start(self) -> None:
        for side in ("left", "right"):
            thread = threading.Thread(
                target=self._loop,
                args=(side,),
                name=f"rr-dep-003-state-{side}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _loop(self, side: str) -> None:
        period = 1.0 / self.hz
        count = 0
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                if count % 4 == 0:
                    self._controller_errors[side] = self.backend.controller_error(side)
                sample = self.backend.read_state(side, self._controller_errors[side])
                self.events.emit("state", "state_sample", asdict(sample))
                with self._condition:
                    self._latest[side] = sample
                    self._history[side].append(sample)
                    self._condition.notify_all()
                count += 1
            except Exception as exc:
                with self._condition:
                    self._errors[side] = repr(exc)
                    self._condition.notify_all()
                self.events.emit("safety", "robot_state_poll_failure", {"side": side, "error": repr(exc)})
                return
            deadline += period
            self._stop.wait(max(0.0, deadline - time.monotonic()))

    def latest(self, side: str) -> StateSample | None:
        with self._condition:
            return self._latest.get(side)

    def wait_stable(self, seconds: float = 2.0, timeout_s: float = 8.0) -> dict[str, Any]:
        start_ns = time.monotonic_ns()
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._errors:
                    raise MotionRejected(f"robot_state_poll_errors:{self._errors}")
                now_ns = time.monotonic_ns()
                ready = True
                summaries = {}
                for side in ("left", "right"):
                    samples = [sample for sample in self._history[side] if sample.receive_monotonic_ns >= start_ns]
                    if not samples or samples[-1].receive_monotonic_ns - samples[0].receive_monotonic_ns < seconds * 1e9:
                        ready = False
                        continue
                    if len(samples) < int(seconds * 10) + 1:
                        raise MotionRejected(f"watchdog_below_10hz:{side}:samples={len(samples)}")
                    span_s = (samples[-1].receive_monotonic_ns - samples[0].receive_monotonic_ns) / 1e9
                    measured_hz = (len(samples) - 1) / span_s
                    max_span = max(
                        max(sample.joints_rad[index] for sample in samples)
                        - min(sample.joints_rad[index] for sample in samples)
                        for index in range(7)
                    )
                    if measured_hz < 10.0:
                        raise MotionRejected(f"watchdog_below_10hz:{side}:{measured_hz:.3f}")
                    if max_span > 0.001:
                        raise MotionRejected(f"robot_state_not_stable:{side}:{max_span:.6f}")
                    if any(sample.sdk_status or sample.arm_errors or sample.controller_error for sample in samples):
                        raise MotionRejected(f"robot_alarm_during_stability_window:{side}")
                    summaries[side] = {
                        "samples": len(samples),
                        "span_sec": span_s,
                        "measured_hz": measured_hz,
                        "max_joint_span_rad": max_span,
                        "initial_rad": list(samples[-1].joints_rad),
                    }
                if ready:
                    return summaries
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MotionRejected("robot_state_stability_timeout")
                self._condition.wait(min(0.1, remaining))

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)


def _tuple_result(binding: dict[str, Any], key: str, expected_length: int = 7) -> tuple[int, list[float]]:
    value = binding.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise MotionRejected(f"invalid_binding_result:{key}:{value}")
    code, values = value
    result = [float(item) for item in values]
    if int(code) != 0 or len(result) != expected_length or not all(math.isfinite(item) for item in result):
        raise MotionRejected(f"binding_call_failed:{key}:{value}")
    return int(code), result


def evaluate_robot_gate(bindings: dict[str, dict[str, Any]], stability: dict[str, Any]) -> dict[str, Any]:
    result = {"status": "pass", "arms": {}}
    for side in ("left", "right"):
        binding = bindings[side]
        robot_info = binding.get("rm_get_robot_info")
        if not isinstance(robot_info, list) or len(robot_info) != 2 or int(robot_info[0]) != 0:
            raise MotionRejected(f"robot_info_failed:{side}:{robot_info}")
        info = robot_info[1]
        if int(info.get("arm_dof", 0)) != 7:
            raise MotionRejected(f"robot_not_7dof:{side}:{info}")
        if info.get("arm_model") != "RM_75":
            raise MotionRejected(f"unexpected_robot_model:{side}:{info.get('arm_model')}")
        _, lower_deg = _tuple_result(binding, "rm_get_joint_min_pos")
        _, upper_deg = _tuple_result(binding, "rm_get_joint_max_pos")
        _, drive_lower_deg = _tuple_result(binding, "rm_get_joint_drive_min_pos")
        _, drive_upper_deg = _tuple_result(binding, "rm_get_joint_drive_max_pos")
        _tuple_result(binding, "rm_get_joint_max_speed")
        _tuple_result(binding, "rm_get_joint_max_acc")
        _tuple_result(binding, "rm_get_joint_drive_max_speed")
        _tuple_result(binding, "rm_get_joint_drive_max_acc")
        lower_rad = [math.radians(max(a, b)) for a, b in zip(lower_deg, drive_lower_deg)]
        upper_rad = [math.radians(min(a, b)) for a, b in zip(upper_deg, drive_upper_deg)]
        initial_rad = stability[side]["initial_rad"]
        direction = select_q7_direction(initial_rad, lower_rad, upper_rad)
        en_state = binding.get("rm_get_joint_en_state")
        if not isinstance(en_state, list) or len(en_state) != 2 or int(en_state[0]) != 0 or list(en_state[1]) != [1] * 7:
            raise MotionRejected(f"joint_not_fully_enabled:{side}:{en_state}")
        error_state = binding.get("rm_get_joint_err_flag")
        if not isinstance(error_state, dict) or int(error_state.get("return_code", -1)) != 0:
            raise MotionRejected(f"joint_error_query_failed:{side}:{error_state}")
        if any(int(value) != 0 for value in error_state.get("err_flag", [])):
            raise MotionRejected(f"joint_alarm:{side}:{error_state}")
        if list(error_state.get("brake_state", [])) != [1] * 7:
            raise MotionRejected(f"joint_brake_not_released:{side}:{error_state}")
        controller = binding.get("rm_get_controller_state")
        if not isinstance(controller, dict) or int(controller.get("return_code", -1)) != 0 or int(controller.get("system_error", -1)) != 0:
            raise MotionRejected(f"controller_not_ready:{side}:{controller}")
        power = binding.get("rm_get_arm_power_state")
        if not isinstance(power, list) or power != [0, 1]:
            raise MotionRejected(f"arm_power_not_ready:{side}:{power}")
        collision = binding.get("rm_get_collision_stage")
        if not isinstance(collision, list) or len(collision) != 2 or int(collision[0]) != 0 or not 1 <= int(collision[1]) <= 8:
            raise MotionRejected(f"collision_protection_not_ready:{side}:{collision}")
        result["arms"][side] = {
            "model": info.get("arm_model"),
            "force_type": info.get("force_type"),
            "initial_rad": initial_rad,
            "lower_rad": lower_rad,
            "upper_rad": upper_rad,
            "q7_direction": direction,
            "q7_plus_margin_rad": upper_rad[6] - initial_rad[6],
            "q7_minus_margin_rad": initial_rad[6] - lower_rad[6],
            "watchdog": stability[side],
            "homing_zero_calibration": "no_direct_SDK_ready_bit_operator_attestation_required_no_alarm_and_enabled_absolute_state_verified",
        }
    return result


def other_robot_processes() -> list[str]:
    output = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, text=True, stdout=subprocess.PIPE
    ).stdout
    current = os.getpid()
    matches = []
    patterns = (
        "realman_deployment.client",
        "run_realman_fastwam_client",
        "spacemouse_control_arm",
        "robot_client_runtime.rr_dep_003",
        "run_rr_dep_003.py",
    )
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        pid, args = fields
        if int(pid) == current:
            continue
        if any(pattern in args for pattern in patterns):
            matches.append(line.strip())
    return matches


def _prompt(events: EventBundle, prompt: str, *, expected: str | None = None, field: str) -> str:
    value = input(prompt).strip()
    events.emit("operator", "operator_input", {"field": field, "value": value})
    if expected is not None and value != expected:
        raise MotionRejected(f"operator_input_mismatch:{field}")
    return value


def run_camera_stage(
    config: dict[str, Any],
    artifact_root: Path,
    events: EventBundle,
    *,
    correction_id: str | None = None,
) -> dict[str, Any]:
    suffix = f"-correction-{correction_id}" if correction_id else ""
    binding_path = artifact_root / f"camera-binding{suffix}.yaml"
    summary_path = artifact_root / f"camera-no-motion-summary{suffix}.yaml"
    local_summary_path = artifact_root / f"local-openpi-view-summary{suffix}.yaml"
    rig = RealSenseRig(config, events)
    try:
        rig.enumerate_and_start()
        rig.wait_ready()
        write_yaml(
            binding_path,
            {"status": "pass", "binding": rig.binding, "active_profiles": rig.profiles},
        )
        record_root = artifact_root / "camera-no-motion-video"
        rig.start_recording(record_root, f"stage-a-5s{suffix}")
        started = time.monotonic()
        while time.monotonic() - started < 5.05:
            if not rig.healthy():
                raise MotionRejected("camera_unhealthy_during_stage_a")
            time.sleep(0.05)
        summary = rig.stop_recording()
        if summary["wall_elapsed_sec"] < 5.0:
            raise MotionRejected(f"stage_a_recording_short:{summary['wall_elapsed_sec']}")
        sample_dir = artifact_root / f"local-openpi-view-samples{suffix}"
        sample_dir.mkdir(parents=True, exist_ok=False)
        samples = {}
        from PIL import Image
        import numpy as np

        for role, image in rig.latest_images().items():
            converted = resize_with_pad_rgb(image)
            path = sample_dir / f"{SOURCE_KEYS[role]}-224.png"
            Image.fromarray(converted, mode="RGB").save(path)
            samples[role] = {
                "path": str(path),
                "shape": list(converted.shape),
                "dtype": str(converted.dtype),
                "min": int(np.min(converted)),
                "max": int(np.max(converted)),
                "sha256": sha256_file(path),
            }
        local_summary = {
            "status": "pass",
            "conversion": "BGR_uint8_640x480_to_RGB_uint8_resize_with_pad_224x224",
            "policy_connections": 0,
            "requests": 0,
            "responses": 0,
            "samples": samples,
        }
        write_yaml(local_summary_path, local_summary)
        stage_summary = {
            "status": "pass",
            "robot_commands": 0,
            "binding": rig.binding,
            "active_profiles": rig.profiles,
            "recording": summary,
        }
        write_yaml(summary_path, stage_summary)
        return stage_summary
    except Exception as exc:
        failure = {
            "status": "failed",
            "error": repr(exc),
            "robot_commands": 0,
            "binding": rig.binding,
            "active_profiles": rig.profiles,
        }
        if not binding_path.exists():
            write_yaml(binding_path, failure)
        if not summary_path.exists():
            write_yaml(summary_path, failure)
        events.emit("safety", "stage_a_failed", failure)
        raise
    finally:
        rig.close()


def run_motion_session(config: dict[str, Any], artifact_root: Path, events: EventBundle) -> dict[str, Any]:
    source_check = verify_frozen_sources(artifact_root / "source-manifest.tsv")
    process_conflicts = other_robot_processes()
    if process_conflicts:
        raise MotionRejected(f"other_robot_command_processes:{process_conflicts}")
    correction_summaries = sorted(artifact_root.glob("camera-no-motion-summary-correction-*.yaml"))
    stage_a_path = (
        correction_summaries[-1]
        if correction_summaries
        else artifact_root / "camera-no-motion-summary.yaml"
    )
    import yaml

    stage_a = yaml.safe_load(stage_a_path.read_text(encoding="utf-8"))
    if stage_a.get("status") != "pass":
        raise MotionRejected("stage_a_not_passed")
    backend = RealManBackend(config, events)
    poller: RobotStatePoller | None = None
    rig: RealSenseRig | None = None
    active_side: str | None = None
    physical = {"left": "not_requested", "right": "not_requested"}
    try:
        backend.open()
        bindings = {side: backend.gather_binding(side) for side in ("left", "right")}
        write_yaml(artifact_root / "robot-sdk-binding.yaml", {"status": "captured", "arms": bindings})
        poller = RobotStatePoller(backend, events, hz=20.0)
        poller.start()
        stability = poller.wait_stable(seconds=2.0)
        robot_gate = evaluate_robot_gate(bindings, stability)
        gate = {
            "status": "awaiting_operator",
            "source_check": source_check,
            "stage_a": "pass",
            "robot": robot_gate,
            "other_robot_processes": [],
            "watchdog_requested_hz": 20.0,
            "watchdog_min_hz": 10.0,
            "stale_threshold_sec": 0.25,
            "command_timeout_sec": 5.0,
            "speed_percent": 5,
            "operator_items": {
                "identity_role": "pending",
                "physically_present": "pending",
                "stop_estop_reachable": "pending",
                "swept_volumes_clear": "pending",
                "wrist_camera_tool_cables_slack": "pending",
                "zero_calibration_no_loss_or_recent_joint_replacement": "pending",
                "workspace_phrase": "pending",
            },
        }
        print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
        identity = _prompt(events, "OPERATOR IDENTITY: ", field="operator_identity")
        role = _prompt(events, "OPERATOR ROLE: ", field="operator_role")
        if not identity or not role:
            raise MotionRejected("operator_identity_or_role_missing")
        for field, prompt in (
            ("physically_present", "CONFIRM PHYSICALLY PRESENT (YES): "),
            ("stop_estop_reachable", "CONFIRM STOP/E-STOP REACHABLE (YES): "),
            ("swept_volumes_clear", "CONFIRM BOTH SWEPT VOLUMES CLEAR (YES): "),
            ("wrist_camera_tool_cables_slack", "CONFIRM WRIST CAMERA/TOOL CABLES SLACK (YES): "),
            (
                "zero_calibration_no_loss_or_recent_joint_replacement",
                "CONFIRM NO ZERO/CALIBRATION LOSS OR UNCALIBRATED JOINT REPLACEMENT (YES): ",
            ),
        ):
            _prompt(events, prompt, expected="YES", field=field)
            gate["operator_items"][field] = "confirmed"
        _prompt(events, "WORKSPACE AUTHORIZATION: ", expected=WORKSPACE_PHRASE, field="workspace_phrase")
        gate["operator_items"].update(
            identity_role={"identity": identity, "role": role},
            workspace_phrase="accepted_exact",
        )
        gate["status"] = "pass_task_local_motion_unlock"
        write_yaml(artifact_root / "motion-unlock-gate.yaml", gate)
        limits_artifact = {
            "status": "pass",
            "robot_gate": robot_gate,
            "units": {"canonical": "radian", "SDK_state_and_command": "degree"},
            "speed_scale": "rm_movej_v_integer_percent_1_to_100",
            "selected_speed_percent": 5,
            "command_timeout_sec": 5.0,
        }
        write_yaml(artifact_root / "robot-limits-and-ready-state.yaml", limits_artifact)

        verify_frozen_sources(artifact_root / "source-manifest.tsv")
        rig = RealSenseRig(config, events)
        rig.enumerate_and_start()
        rig.wait_ready()
        if not rig.healthy():
            raise MotionRejected("motion_camera_gate_failed")
        motion_session_id = str(uuid.uuid4())
        thresholds = MotionThresholds()

        def sink(kind: str, payload: dict[str, Any]) -> None:
            target = "ack" if kind in ("sdk_acceptance", "observed_target") else (
                "safety" if kind == "safety_stop" else "command"
            )
            events.emit(target, kind, {"motion_session_id": motion_session_id, **payload})

        executor = BoundedCommandExecutor(
            backend,
            poller.latest,
            rig.healthy,
            thresholds=thresholds,
            event_sink=sink,
        )
        recording_summaries = {}
        for side, phrase in (("left", LEFT_PHRASE), ("right", RIGHT_PHRASE)):
            if side == "right" and physical["left"] != "confirmed":
                raise MotionRejected("right_unlock_without_left_physical_ack")
            stable = poller.wait_stable(seconds=2.0)
            arm_gate = robot_gate["arms"][side]
            initial = tuple(stable[side]["initial_rad"])
            reference_initial = tuple(arm_gate["initial_rad"])
            if max(abs(a - b) for a, b in zip(initial, reference_initial)) > 0.005:
                raise MotionRejected(f"initial_state_drift_before_{side}")
            direction = int(arm_gate["q7_direction"])
            targets = build_arm_targets(initial, direction)
            _prompt(events, f"{side.upper()} AUTHORIZATION: ", expected=phrase, field=f"{side}_phrase")
            label = f"{side}-{motion_session_id}"
            rig.start_recording(artifact_root / "camera-motion-video", label)
            preroll_started = time.monotonic()
            while time.monotonic() - preroll_started < 2.05:
                if not rig.healthy():
                    raise MotionRejected(f"camera_loss_during_{side}_preroll")
                sample = poller.latest(side)
                if sample is None or time.monotonic_ns() - sample.receive_monotonic_ns > 250_000_000:
                    raise MotionRejected(f"stale_state_during_{side}_preroll")
                time.sleep(0.02)
            active_side = side
            for suffix, kind in (("hold", "hold"), ("q7_micro", "micro"), ("return", "return")):
                command_id = f"{side}_{suffix}"
                target_key = "micro" if kind == "micro" else kind
                executor.execute(
                    command_id=command_id,
                    side=side,
                    kind=kind,
                    initial_rad=initial,
                    target_rad=targets[target_key],
                    lower_rad=arm_gate["lower_rad"],
                    upper_rad=arm_gate["upper_rad"],
                    direction=direction,
                )
            active_side = None
            postroll_started = time.monotonic()
            while time.monotonic() - postroll_started < 2.05:
                if not rig.healthy():
                    raise MotionRejected(f"camera_loss_during_{side}_postroll")
                time.sleep(0.02)
            recording = rig.stop_recording()
            if recording["wall_elapsed_sec"] > 30.0:
                raise MotionRejected(f"motion_recording_exceeded_30s:{side}")
            recording_summaries[side] = recording
            final = poller.latest(side)
            if final is None or max(abs(a - b) for a, b in zip(final.joints_rad, initial)) > 0.005:
                raise MotionRejected(f"unsafe_return_state:{side}")
            if final.arm_errors or final.controller_error:
                raise MotionRejected(f"alarm_after_return:{side}")
            _prompt(events, f"CONFIRM {side.upper()} PHYSICAL RETURN OBSERVED (YES): ", expected="YES", field=f"{side}_physical_ack")
            physical[side] = "confirmed"
            events.emit(
                "ack",
                "physical_return_ack",
                {"side": side, "operator_identity": identity, "operator_role": role, "observed_rad": list(final.joints_rad)},
            )
        acknowledgement = {
            "status": "pass",
            "motion_session_id": motion_session_id,
            "operator_identity": identity,
            "operator_role": role,
            "per_arm_physical_return_ack": physical,
            "sdk_acceptance": "see_acknowledgements_jsonl",
            "observed_movement": "see_robot_state_trajectory_jsonl",
            "return_to_start": "within_0_005rad_both_arms",
        }
        write_yaml(artifact_root / "physical-acknowledgement.yaml", acknowledgement)
        return {
            "status": "completed",
            "motion_session_id": motion_session_id,
            "command_ids": list(executor.sequence.submitted),
            "command_count": len(executor.sequence.submitted),
            "model_requests": 0,
            "gripper_commands": 0,
            "extra_joint_targets": 0,
            "recordings": recording_summaries,
            "physical_acknowledgement": acknowledgement,
        }
    except Exception as exc:
        if active_side is not None:
            try:
                backend.slow_stop(active_side)
            except Exception as stop_exc:
                events.emit("safety", "slow_stop_failure", {"side": active_side, "error": repr(stop_exc)})
        events.emit(
            "safety",
            "motion_session_failed",
            {"error": repr(exc), "active_side": active_side, "physical_acknowledgement": physical},
        )
        if not (artifact_root / "physical-acknowledgement.yaml").exists():
            write_yaml(
                artifact_root / "physical-acknowledgement.yaml",
                {"status": "not_complete", "error": repr(exc), "per_arm_physical_return_ack": physical},
            )
        raise
    finally:
        if rig is not None:
            rig.close()
        if poller is not None:
            poller.close()
        backend.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RR-DEP-003 bounded camera and Q7 runner")
    parser.add_argument("stage", choices=("camera", "motion"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--correction-id",
        choices=("001",),
        help="append-only mechanical Stage A correction identifier",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_root = args.artifact_root.resolve()
    if artifact_root.name != RUN_ID or artifact_root.parent.name != TASK_ID:
        print(json.dumps({"status": "rejected", "error": "unexpected_artifact_root"}), flush=True)
        return 2
    config = load_config(args.config.resolve())
    events = EventBundle(artifact_root)
    try:
        source_check = verify_frozen_sources(artifact_root / "source-manifest.tsv")
        if args.stage == "camera":
            result = run_camera_stage(
                config,
                artifact_root,
                events,
                correction_id=args.correction_id,
            )
        else:
            if args.correction_id is not None:
                raise MotionRejected("correction_id_is_camera_stage_only")
            result = run_motion_session(config, artifact_root, events)
        print(json.dumps({"stage": args.stage, "source_check": source_check, "result": result}, indent=2), flush=True)
        return 0
    except (MotionRejected, KeyboardInterrupt, EOFError) as exc:
        events.emit("safety", "terminal_rejection", {"stage": args.stage, "error": repr(exc)})
        print(json.dumps({"stage": args.stage, "status": "blocked", "error": repr(exc)}, indent=2), flush=True)
        return 2
    except Exception as exc:
        events.emit("safety", "terminal_failure", {"stage": args.stage, "error": repr(exc)})
        print(json.dumps({"stage": args.stage, "status": "failed", "error": repr(exc)}, indent=2), flush=True)
        return 1
    finally:
        events.close()


if __name__ == "__main__":
    raise SystemExit(main())
