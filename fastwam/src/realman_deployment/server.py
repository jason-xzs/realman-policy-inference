"""Threaded HTTP inference server for FastWAM."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .policy import DEFAULT_CHECKPOINT, FastWAMPolicy, MockPolicy
from .protocol import MAX_REQUEST_BYTES, PROTOCOL_VERSION, ProtocolError, json_bytes, validate_actions

logger = logging.getLogger(__name__)


def build_action_diagnostics(
    raw_state: list[float],
    actions: list[list[float]],
    returned_count: int,
    *,
    clipped_values: int = 0,
) -> dict[str, Any]:
    """Summarize model-space actions relative to the request observation."""
    import numpy as np

    # Request state: left joints/gripper/EEF, then right joints/gripper/EEF.
    # Model action: right joints/gripper, then left joints/gripper.
    current = np.asarray(
        raw_state[14:21] + [raw_state[21] * 1000.0]
        + raw_state[0:7] + [raw_state[7] * 1000.0],
        dtype=np.float64,
    )
    prediction = np.asarray(actions, dtype=np.float64)
    returned = prediction[:returned_count]
    if current.shape != (16,) or prediction.ndim != 2 or prediction.shape[1] != 16 or not len(returned):
        raise ValueError("cannot diagnose invalid/empty RealMan action data")

    joint_indices = np.asarray([*range(7), *range(8, 15)])
    current_joints = current[joint_indices]
    returned_joints = returned[:, joint_indices]
    full_joints = prediction[:, joint_indices]
    transitions = np.vstack([current_joints, returned_joints])
    step_max = np.max(np.abs(np.diff(transitions, axis=0)), axis=1)

    def rounded(value):
        return np.round(value, 6).tolist()

    return {
        "raw_state_28d": rounded(np.asarray(raw_state, dtype=np.float64)),
        "current_control_16d": rounded(current),
        "predicted_horizon": int(len(prediction)),
        "returned_count": int(len(returned)),
        "returned_actions_16d": rounded(returned),
        "returned_step_joint_max_abs_rad": rounded(step_max),
        "returned_endpoint_joint_max_abs_rad": round(
            float(np.max(np.abs(returned_joints[-1] - current_joints))), 6
        ),
        "full_endpoint_joint_max_abs_rad": round(
            float(np.max(np.abs(full_joints[-1] - current_joints))), 6
        ),
        "full_joint_min_rad": rounded(np.min(full_joints, axis=0)),
        "full_joint_max_rad": rounded(np.max(full_joints, axis=0)),
        "returned_grippers": {
            "right": rounded(returned[:, 7]),
            "left": rounded(returned[:, 15]),
        },
        "clipped_values": int(clipped_values),
    }


class PolicyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, policy, *, api_token: str = "", max_actions: int = 8):
        super().__init__(address, InferenceHandler)
        self.policy = policy
        self.api_token = api_token
        self.max_actions = int(max_actions)
        self.inference_lock = threading.Lock()
        self.started_at = time.time()
        self.requests = 0


class InferenceHandler(BaseHTTPRequestHandler):
    server: PolicyHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = self.server.api_token
        if not expected:
            return True
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {expected}")

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError as exc:
            raise ProtocolError("invalid Content-Length") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ProtocolError(f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
        try:
            return json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"invalid JSON: {exc}") from exc

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        self._send(HTTPStatus.OK, {
            "ok": True,
            "version": PROTOCOL_VERSION,
            "ready": True,
            "requests": self.server.requests,
            "uptime_s": round(time.time() - self.server.started_at, 3),
            "mock": isinstance(self.server.policy, MockPolicy),
        })

    def do_POST(self) -> None:
        request_id = None
        if not self._authorized():
            self._send(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
            return
        try:
            payload = self._read_json()
            request_id = payload.get("request_id") if isinstance(payload, dict) else None
            if self.path == "/v1/reset":
                with self.server.inference_lock:
                    self.server.policy.reset()
                self._send(HTTPStatus.OK, {"ok": True})
                return
            if self.path != "/v1/infer":
                self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            with self.server.inference_lock:
                actions, metadata = self.server.policy.infer(payload)
            all_actions = validate_actions(actions)
            actions = all_actions[: self.server.max_actions]
            diagnostics = build_action_diagnostics(
                payload["state"],
                all_actions,
                self.server.max_actions,
                clipped_values=metadata.get("clipped_values", 0),
            )
            logger.info(
                "action_diagnostics request_id=%s %s",
                request_id,
                json.dumps(diagnostics, separators=(",", ":"), ensure_ascii=False),
            )
            self.server.requests += 1
            self._send(HTTPStatus.OK, {
                "ok": True,
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "actions": actions,
                "metadata": metadata,
            })
        except ProtocolError as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "request_id": request_id, "error": str(exc)})
        except Exception as exc:
            logger.exception("Inference request failed")
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {
                "ok": False, "request_id": request_id, "error": f"inference failed: {type(exc).__name__}: {exc}"
            })


def create_server(host: str, port: int, policy, *, api_token: str = "", max_actions: int = 8):
    if int(max_actions) <= 0:
        raise ValueError("max_actions/replan_steps must be positive")
    return PolicyHTTPServer((host, int(port)), policy, api_token=api_token, max_actions=max_actions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve FastWAM inference for the RealMan client")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=24, help="maximum actions returned per request")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--no-action-clip", action="store_true")
    parser.add_argument("--api-token", default="", help="optional bearer token; prefer FASTWAM_API_TOKEN")
    parser.add_argument("--mock", action="store_true", help="serve a deterministic policy without loading the model")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    import os

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action_horizon <= 0 or args.replan_steps <= 0 or args.num_inference_steps <= 0:
        parser.error("action-horizon, replan-steps, and num-inference-steps must be positive")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.mock:
        policy = MockPolicy(action_horizon=args.action_horizon)
    else:
        policy = FastWAMPolicy(
            args.checkpoint, config=args.config, stats=args.stats, device=args.device, dtype=args.dtype,
            action_horizon=args.action_horizon, num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift, seed=args.seed, text_cfg_scale=args.text_cfg_scale,
            negative_prompt=args.negative_prompt, rand_device=args.rand_device, tiled=args.tiled,
            clip_to_training_range=not args.no_action_clip,
        )
    token = args.api_token or os.environ.get("FASTWAM_API_TOKEN", "")
    server = create_server(args.host, args.port, policy, api_token=token, max_actions=args.replan_steps)
    logger.info("FastWAM server ready at http://%s:%d (mock=%s)", args.host, args.port, args.mock)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        logger.info("Stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
