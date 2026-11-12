"""RealMan client: observe, request FastWAM actions, and execute action chunks."""

from __future__ import annotations

import argparse
import json
import logging
import queue
import signal
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from .hardware import MockRealManHardware, RealManHardware
from .protocol import PROTOCOL_VERSION, ProtocolError, json_bytes, make_infer_request, validate_actions

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "The right arm first picks up the object on the right and places it into the basket, "
    "then the left arm picks up the two objects on the left and places them into the basket."
)


class RemoteInferenceClient:
    def __init__(self, server_url: str, *, timeout: float = 120.0, api_token: str = ""):
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)
        self.api_token = api_token

    def _request(self, path: str, payload: dict[str, Any] | None = None, method: str = "POST"):
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            data = json_bytes(payload)
            headers["Content-Type"] = "application/json"
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        request = urllib.request.Request(self.server_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"server returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"inference server unavailable: {exc}") from exc
        if not result.get("ok"):
            raise RuntimeError(f"server rejected request: {result.get('error', result)}")
        return result

    def health(self) -> dict[str, Any]:
        return self._request("/health", method="GET")

    def infer(self, images, state, instruction: str, jpeg_quality: int = 90):
        request = make_infer_request(images, state, instruction, jpeg_quality=jpeg_quality)
        response = self._request("/v1/infer", request)
        if response.get("version") != PROTOCOL_VERSION or response.get("request_id") != request["request_id"]:
            raise ProtocolError("server response version/request_id mismatch")
        return validate_actions(response.get("actions")), response.get("metadata", {})

    def reset(self) -> None:
        self._request("/v1/reset", {"version": PROTOCOL_VERSION})


def align_action_chunk(
    actions: list[list[float]],
    current_left,
    current_right,
    *,
    max_skip: int,
    min_improvement_rad: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Align a new absolute-action chunk to the robot state measured after inference."""
    import numpy as np

    if max_skip < 0:
        raise ValueError("action_align_max_skip must be non-negative")
    if min_improvement_rad < 0:
        raise ValueError("action_align_min_improvement_rad must be non-negative")
    prediction = np.asarray(actions, dtype=np.float64)
    left = np.asarray(current_left, dtype=np.float64)
    right = np.asarray(current_right, dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape[1] != 16 or not len(prediction):
        raise ValueError(f"actions must have shape [T,16], got {prediction.shape}")
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError("current left/right joints must each contain 7 values")
    joint_indices = np.asarray([*range(7), *range(8, 15)])
    current = np.concatenate([right, left])
    candidate_count = min(len(prediction), int(max_skip) + 1)
    distances = np.max(
        np.abs(prediction[:candidate_count, joint_indices] - current[None, :]),
        axis=1,
    )
    best_index = int(np.argmin(distances))
    improvement = float(distances[0] - distances[best_index])
    applied_index = best_index if best_index > 0 and improvement >= min_improvement_rad else 0
    diagnostics = {
        "selected_index": applied_index,
        "best_index": best_index,
        "first_error_rad": float(distances[0]),
        "selected_error_rad": float(distances[applied_index]),
        "improvement_rad": improvement,
        "candidate_count": candidate_count,
    }
    return actions[applied_index:], diagnostics

class BackgroundCameraVideoRecorder:
    """Record every native camera frame on dedicated writer threads."""

    CAMERA_NAMES = ("high_rgb", "left_wrist_rgb", "right_wrist_rgb")

    def __init__(
        self,
        root_dir: str,
        *,
        fps: float = 30.0,
        codec: str = "h264_vaapi",
        queue_size: int = 256,
    ) -> None:
        if fps <= 0:
            raise ValueError("record_fps must be positive")
        if codec not in ("h264_vaapi", "MJPG"):
            raise ValueError("record_codec must be h264_vaapi or MJPG")
        if queue_size <= 0:
            raise ValueError("record_queue_size must be positive")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(root_dir).expanduser() / f"client_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.fps = float(fps)
        self.codec = codec
        self._queues = {
            name: queue.Queue(maxsize=int(queue_size))
            for name in self.CAMERA_NAMES
        }
        self._errors: dict[str, BaseException] = {}
        self._error_lock = threading.Lock()
        self._metadata_lock = threading.Lock()
        self._metadata_fp = (self.run_dir / "camera_frames.jsonl").open("w", encoding="utf-8")
        self._closed = False
        self._threads = []
        for name in self.CAMERA_NAMES:
            thread = threading.Thread(
                target=self._writer_loop,
                args=(name,),
                name=f"video-writer-{name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        logger.info("Recording native camera streams to %s", self.run_dir)

    def submit(self, name: str, image, *, sequence: int, timestamp_ns: int) -> None:
        if self._closed:
            return
        if name not in self._queues:
            raise ValueError(f"unknown recording camera: {name}")
        with self._error_lock:
            if self._errors:
                failed_name, error = next(iter(self._errors.items()))
                raise RuntimeError(f"video writer failed for {failed_name}: {error}") from error
        try:
            self._queues[name].put((int(sequence), int(timestamp_ns), image), timeout=2.0)
        except queue.Full as exc:
            raise RuntimeError(f"video writer queue is full for {name}; refusing to drop frames") from exc

    def _writer_loop(self, name: str) -> None:
        import cv2
        import numpy as np
        import subprocess

        writer = None
        process = None
        size = None
        use_vaapi = self.codec == "h264_vaapi"
        output_path = self.run_dir / f"{name}.{'mp4' if use_vaapi else 'avi'}"
        try:
            while True:
                item = self._queues[name].get()
                if item is None:
                    break
                sequence, timestamp_ns, frame = item
                frame = np.asarray(frame)
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                    raise ValueError(f"invalid recording frame {name}: {frame.shape}/{frame.dtype}")
                if name != "high_rgb":
                    frame = frame[::-1, ::-1]
                frame = np.ascontiguousarray(frame)
                height, width = frame.shape[:2]
                if size is None:
                    size = (width, height)
                    if use_vaapi:
                        command = [
                            "ffmpeg", "-hide_banner", "-loglevel", "error",
                            "-vaapi_device", "/dev/dri/renderD128",
                            "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{width}x{height}", "-r", str(self.fps),
                            "-i", "pipe:0", "-vf", "format=nv12,hwupload",
                            "-c:v", "h264_vaapi", "-qp", "18", "-bf", "0",
                            "-y", str(output_path),
                        ]
                        # Keep ffmpeg outside the terminal process group.  The
                        # parent handles Ctrl+C and closes stdin so ffmpeg can
                        # flush a complete MP4 trailer and exit successfully.
                        process = subprocess.Popen(
                            command,
                            stdin=subprocess.PIPE,
                            start_new_session=True,
                        )
                    else:
                        writer = cv2.VideoWriter(
                            str(output_path), cv2.VideoWriter_fourcc(*self.codec), self.fps, size
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"failed to open video writer: {output_path}")
                elif size != (width, height):
                    raise ValueError(f"recording frame size changed for {name}: {(width, height)}")
                if process is not None:
                    process.stdin.write(frame.tobytes())
                else:
                    writer.write(frame)
                payload = {
                    "camera": name,
                    "sequence": sequence,
                    "timestamp_ns": timestamp_ns,
                }
                with self._metadata_lock:
                    self._metadata_fp.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except BaseException as exc:
            with self._error_lock:
                self._errors[name] = exc
            logger.exception("Camera video writer failed: %s", name)
        finally:
            if writer is not None:
                writer.release()
            if process is not None:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
                return_code = process.wait()
                if return_code != 0:
                    with self._error_lock:
                        self._errors[name] = RuntimeError(
                            f"ffmpeg exited with status {return_code} for {name}"
                        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name, frame_queue in self._queues.items():
            try:
                frame_queue.put(None, timeout=5.0)
            except queue.Full:
                logger.error("Timed out draining video queue for %s", name)
        for thread in self._threads:
            thread.join(timeout=30.0)
            if thread.is_alive():
                logger.error("Video writer thread did not stop: %s", thread.name)
        with self._metadata_lock:
            self._metadata_fp.flush()
            self._metadata_fp.close()
        with self._error_lock:
            if self._errors:
                logger.error("Recording completed with writer errors: %s", self._errors)
        logger.info("Closed native camera recording: %s", self.run_dir)

def run_control_loop(

    hardware,
    remote: RemoteInferenceClient,
    *,
    instruction: str,
    action_hz: float = 30.0,
    jpeg_quality: int = 90,
    max_replans: int = 0,
    dry_run: bool = True,
    stop_event=None,
    action_alignment: bool = False,
    action_align_max_skip: int = 8,
    action_align_min_improvement_rad: float = 0.01,
    tracking_gate: bool = False,
    tracking_error_rad: float = 0.05,
    tracking_timeout_s: float = 2.0,
) -> int:
    import threading

    stop_event = stop_event or threading.Event()
    replans = 0
    if action_hz <= 0:
        raise ValueError("action_hz must be positive")
    period = 1.0 / float(action_hz)
    while not stop_event.is_set() and (max_replans <= 0 or replans < max_replans):
        images, state = hardware.observe()
        started = time.perf_counter()
        actions, metadata = remote.infer(images, state, instruction, jpeg_quality)
        received_count = len(actions)
        round_trip_ms = (time.perf_counter() - started) * 1000
        if action_alignment:
            current_left, current_right = hardware.current_joints()
            actions, alignment = align_action_chunk(
                actions, current_left, current_right,
                max_skip=action_align_max_skip,
                min_improvement_rad=action_align_min_improvement_rad,
            )
            logger.info("action chunk alignment: %s", alignment)
        logger.info(
            "received %d actions, executing %d (server inference=%sms, round-trip=%.1fms)",
            received_count, len(actions), metadata.get("inference_ms", "?"), round_trip_ms,
        )
        replans += 1
        deadline = time.monotonic()
        for action in actions:
            if stop_event.is_set():
                break
            step_started = time.monotonic()
            hardware.execute_action(action, dry_run=dry_run)
            if tracking_gate and not dry_run:
                tracking = hardware.wait_for_joint_target(
                    action,
                    tolerance_rad=tracking_error_rad,
                    timeout_s=tracking_timeout_s,
                    resend_hz=action_hz,
                    stop_event=stop_event,
                )
                logger.info("tracking gate reached target: %s", tracking)
            if tracking_gate:
                remaining = period - (time.monotonic() - step_started)
            else:
                deadline += period
                remaining = deadline - time.monotonic()
            if remaining > 0:
                stop_event.wait(remaining)
    return replans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FastWAM on a RealMan dual-arm robot")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-token", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--left-ip", default="192.168.1.18")
    parser.add_argument("--right-ip", default="192.168.1.19")
    parser.add_argument("--arm-port", type=int, default=8080)
    parser.add_argument("--high-camera-id", default="")
    parser.add_argument("--left-camera-id", default="")
    parser.add_argument("--right-camera-id", default="")
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.5)
    parser.add_argument("--gripper-mode", choices=("position", "binary-hysteresis"), default="binary-hysteresis")
    parser.add_argument("--gripper-close-threshold", type=float, default=300.0)
    parser.add_argument("--gripper-open-threshold", type=float, default=700.0)
    parser.add_argument("--gripper-confirm-actions", type=int, default=3,
                        help="binary mode: consecutive samples; 0 disables")
    parser.add_argument("--align-action-chunk", action="store_true",
                        help="skip stale leading actions by matching the post-inference robot state")
    parser.add_argument("--left-gripper-close-threshold", type=float, default=None,
                        help="left-side override for --gripper-close-threshold")
    parser.add_argument("--right-gripper-close-threshold", type=float, default=None,
                        help="right-side override for --gripper-close-threshold")
    parser.add_argument("--left-gripper-confirm-actions", type=int, default=None,
                        help="left-side override for --gripper-confirm-actions")
    parser.add_argument("--right-gripper-confirm-actions", type=int, default=None,
                        help="right-side override for --gripper-confirm-actions")
    parser.add_argument("--action-align-max-skip", type=int, default=8,
                        help="maximum number of leading actions that alignment may skip")
    parser.add_argument("--action-align-min-improvement-rad", type=float, default=0.01,
                        help="minimum max-joint-error improvement required before skipping")
    parser.add_argument("--gripper-confirm-seconds", type=float, default=0.0,
                        help="binary mode: duration; 0 disables; actions/time use OR")
    parser.add_argument("--max-sync-skew-ms", type=float, default=50.0,
                        help="reject observations whose three frames/dual-arm state span exceeds this")
    parser.add_argument("--record-dir", default="log/realman_demo",
                        help="record all native camera frames to three background videos")
    parser.add_argument("--gripper-before-arm", action="store_true",
                        help="wait for changed grippers to finish before sending arm targets")
    parser.add_argument("--gripper-blocking-timeout", type=float, default=5.0,
                        help="seconds to wait for each blocking gripper command")
    parser.add_argument("--record-fps", type=float, default=30.0, help="camera video FPS")
    parser.add_argument("--tracking-gate", action="store_true",
                        help="wait and resend each joint target until physical tracking error is small")
    parser.add_argument("--tracking-error-rad", type=float, default=0.05,
                        help="maximum physical joint error before advancing to the next action")
    parser.add_argument("--tracking-timeout", type=float, default=2.0,
                        help="maximum seconds to wait for each action target")
    parser.add_argument("--record-codec", default="h264_vaapi",
                        help="h264_vaapi (default, hardware MP4) or MJPG (software AVI)")
    parser.add_argument("--record-queue-size", type=int, default=256,
                        help="per-camera writer queue; recording fails instead of silently dropping frames")
    parser.add_argument("--sync-timeout", type=float, default=8.0,
                        help="seconds to wait for a synchronized camera tuple")
    parser.add_argument("--max-replans", type=int, default=0, help="0 means run until interrupted")
    parser.add_argument("--execute", action="store_true", help="actually command hardware; default is dry-run")
    parser.add_argument("--mock-hardware", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    import os
    import threading

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = args.api_token or os.environ.get("FASTWAM_API_TOKEN", "")
    remote = RemoteInferenceClient(args.server_url, timeout=args.timeout, api_token=token)
    health = remote.health()
    logger.info("Connected to inference server: %s", health)
    recorder = BackgroundCameraVideoRecorder(
        args.record_dir, fps=args.record_fps, codec=args.record_codec,
        queue_size=args.record_queue_size,
    ) if args.record_dir else None
    try:
        hardware = MockRealManHardware() if args.mock_hardware else RealManHardware(
            left_ip=args.left_ip, right_ip=args.right_ip, arm_port=args.arm_port,
            high_camera_id=args.high_camera_id, left_camera_id=args.left_camera_id,
            right_camera_id=args.right_camera_id, max_joint_step_rad=args.max_joint_step_rad,
            gripper_mode=args.gripper_mode,
            gripper_close_threshold=args.gripper_close_threshold,
            gripper_open_threshold=args.gripper_open_threshold,
            left_gripper_close_threshold=args.left_gripper_close_threshold,
            right_gripper_close_threshold=args.right_gripper_close_threshold,
            left_gripper_confirm_actions=args.left_gripper_confirm_actions,
            right_gripper_confirm_actions=args.right_gripper_confirm_actions,
            gripper_confirm_actions=args.gripper_confirm_actions,
            gripper_confirm_seconds=args.gripper_confirm_seconds,
            gripper_before_arm=args.gripper_before_arm,
            gripper_blocking_timeout_s=args.gripper_blocking_timeout,
            max_sync_skew_ms=args.max_sync_skew_ms, sync_timeout_s=args.sync_timeout,
            frame_recorder=recorder,
        )
    except Exception:
        if recorder is not None:
            recorder.close()
        raise
    stop_event = threading.Event()

    def request_stop(signum, _frame):
        logger.warning("Received signal %s; stopping arms", signum)
        stop_event.set()
        hardware.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        run_control_loop(
            hardware, remote, instruction=args.instruction, action_hz=args.action_hz,
            jpeg_quality=args.jpeg_quality, max_replans=args.max_replans,
            dry_run=not args.execute, stop_event=stop_event,
            action_alignment=args.align_action_chunk,
            action_align_max_skip=args.action_align_max_skip,
            action_align_min_improvement_rad=args.action_align_min_improvement_rad,
            tracking_gate=args.tracking_gate,
            tracking_error_rad=args.tracking_error_rad,
            tracking_timeout_s=args.tracking_timeout,
        )
    except Exception:
        hardware.stop()
        raise
    finally:
        hardware.close()
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
