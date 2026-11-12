"""RR-DEP-007 one-command, no-recording RealMan client preset.

The module reuses the established camera, capture-state, 28D-to-16D mapping,
robot gate, projection, and SDK paths.  It adds only RR-DEP-007 orchestration:
server-context discovery, two internally identified requests, simple onsite
operator confirmations, pre-send-only projection enforcement, compact terminal
artifacts, and Compute finalization/reconciliation.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any, Mapping, Sequence
import uuid

from .rr_dep_003 import RealSenseRig, RobotStatePoller, json_safe, sha256_file, write_yaml
from .rr_dep_003_motion import MotionRejected
from .rr_dep_004 import evaluate_robot_gate_004, map_action28_to_action16, project_model_row, require_uuid4
from .rr_dep_005 import latest_camera_observation, settle_after_return
from .rr_dep_006 import RR006Backend


PRESET = "pi05-realman-30000"
TASK_ID = "RR-DEP-007"
EXPERIMENT_ID = TASK_ID
RUN_ID = "RR-DEP-007-RUN-001"
SCHEMA_ID = "rr_dep_openpi_robot_live_v3"
PROMPT = "Place the three objects into the blue container."
CLIENT_ENDPOINT = "127.0.0.1:18200"
COMPUTE_ENDPOINT = "127.0.0.1:8020"
OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
OPENPI_CONFIG = "pi05_realman_joint_201_step30000_ema"
CHECKPOINT_PATH = (
    "/share/jhk/phywam-mope-real/checkpoints/finetuned/pi05/"
    "RR-BL-003/RR-BL-003-RUN-001/attempt-001/30000"
)
MAPPING_INDICES = list(range(8)) + list(range(14, 22))
FINAL_ARTIFACTS = ("client-receipt.yaml", "run-manifest.yaml", "SHA256SUMS")
COMPUTE_ARTIFACT_ROOT = Path(
    "/share/jhk/phywam-mope-real/runs/RR-DEP-007/RR-DEP-007-RUN-001/compute"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2].parent


def runtime_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifact_root() -> Path:
    return project_root() / "runs" / TASK_ID / RUN_ID / "client"


def cache_root() -> Path:
    return project_root() / "cache" / "rr-dep-007-run-001"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="RR-DEP-007 quick pi0.5 RealMan inference")
    parser.add_argument("--preset", choices=(PRESET,), required=True)
    return parser


def preset_args() -> Namespace:
    """Fixed values consumed by the already established runtime components."""
    return Namespace(
        head_serial="420122071413",
        left_wrist_serial="323622272614",
        right_wrist_serial="230422272673",
        left_robot_ip="192.168.1.18",
        right_robot_ip="192.168.1.19",
        robot_port=8080,
        image_width=640,
        image_height=480,
        camera_fps=30,
        model_image_size=224,
        state_dim=28,
        action_dim=28,
        action_row=0,
        joint_delta_limit_rad=0.01,
        speed_percent=5,
        target_tolerance_rad=0.005,
        state_stale_sec=0.25,
        response_to_command_sec=2.0,
        command_timeout_sec=5.0,
        watchdog_hz=20.0,
        controller_query_retries=10,
        controller_query_backoff_sec=0.05,
        settle_grace_sec=2.0,
        stable_window_sec=2.0,
        stable_span_threshold_rad=0.001,
        settle_timeout_sec=10.0,
    )


class CompactEvents:
    """Keep counters and compact terminal facts only, never frames or trajectories."""

    def __init__(self, rollout_id: str) -> None:
        self.rollout_id = require_uuid4(rollout_id, "rollout_id")
        self.counters: Counter[str] = Counter()
        self.safety_events: list[dict[str, str]] = []
        self.operator_confirmations: list[dict[str, Any]] = []
        self.request_facts: dict[str, dict[str, Any]] = {}
        self.command_facts: dict[str, dict[str, Any]] = {}
        self.service: dict[str, Any] = {}
        self._lock = threading.Lock()

    def emit(self, name: str, event: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self.counters[f"{name}.{event}"] += 1
            if name == "safety":
                self.safety_events.append({
                    "event": event,
                    "error": str(payload.get("error", ""))[:240],
                })
            elif name == "operator" and event == "confirmation":
                value = str(payload.get("value", ""))
                self.operator_confirmations.append({
                    "field": str(payload.get("field", "")),
                    "accepted": value == "yes",
                    "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                    "at": utc_now(),
                })
            elif name == "policy" and event == "connection_verified":
                self.service = json_safe(dict(payload))
            elif name == "policy" and event in ("request_sending", "response_validated"):
                request_id = str(payload.get("request_id", ""))
                fact = self.request_facts.setdefault(request_id, {"request_id": request_id})
                fact.update({
                    key: json_safe(payload[key])
                    for key in (
                        "request_ordinal", "execution_intent", "source_shape", "source_dtype",
                        "source_finite", "source_sha256", "canonical_shape", "canonical_dtype",
                        "canonical_finite", "canonical_sha256", "mapping_indices", "client_round_trip_ms",
                    )
                    if key in payload
                })
                fact["outcome"] = "response_validated" if event == "response_validated" else "sending_outcome_unknown"
            elif name == "command" and event in ("submitting", "sdk_acceptance", "observed_target"):
                command_id = str(payload.get("command_id", ""))
                fact = self.command_facts.setdefault(command_id, {"command_id": command_id})
                fact.update({
                    key: json_safe(payload[key])
                    for key in ("side", "kind", "sdk_status", "target_error_rad", "observed_direction")
                    if key in payload
                })
                fact["outcome"] = event

    def compact(self) -> dict[str, Any]:
        with self._lock:
            return {
                "event_counters": dict(sorted(self.counters.items())),
                "safety_events": list(self.safety_events),
                "operator_confirmations": list(self.operator_confirmations),
                "request_facts": list(self.request_facts.values()),
                "command_facts": list(self.command_facts.values()),
                "service": dict(self.service),
            }


def exact_runtime_provenance() -> dict[str, Any]:
    root = runtime_root().resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    return {"root": str(root), "commit": commit, "clean": not dirty}


def validate_client_identity() -> dict[str, Any]:
    user = subprocess.check_output(["id", "-un"], text=True).strip()
    if user != "robot2":
        raise MotionRejected(f"client_login_user_mismatch:{user}")
    home = Path.home().resolve()
    expected_project = (home / "jhk" / "phywam-mope-real").resolve()
    if expected_project != project_root().resolve():
        raise MotionRejected("client_project_realpath_mismatch")
    control = (expected_project / "control").resolve()
    environment = (expected_project / "envs" / "robot-client-py311").resolve()
    expected_python = (environment / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise MotionRejected("client_python_environment_mismatch")
    runtime = exact_runtime_provenance()
    if not runtime["clean"]:
        raise MotionRejected("client_runtime_must_be_clean_before_live_execution")
    sdk_source = (home / "deltawam" / "src" / "realman_deployment" / "Robotic_Arm" / "rm_robot_interface.py").resolve()
    if not sdk_source.is_file():
        raise MotionRejected("realman_sdk_source_missing")
    return {
        "login_user": user,
        "home_realpath": str(home),
        "project_realpath": str(expected_project),
        "control_realpath": str(control),
        "runtime_realpath": runtime["root"],
        "environment_realpath": str(environment),
        "python_realpath": str(expected_python),
        "runtime_commit": runtime["commit"],
        "runtime_clean": True,
        "python_version": sys.version.split()[0],
        "realman_sdk_source": str(sdk_source),
        "realman_sdk_source_sha256": sha256_file(sdk_source),
    }


def _ancestor_pids() -> set[int]:
    ancestors = {os.getpid()}
    current = os.getppid()
    while current > 1 and current not in ancestors:
        ancestors.add(current)
        try:
            stat = Path(f"/proc/{current}/stat").read_text(encoding="utf-8")
            current = int(stat.rsplit(")", 1)[1].split()[1])
        except (FileNotFoundError, IndexError, ValueError):
            break
    return ancestors


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
    matches = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid not in ignored and any(pattern in fields[1] for pattern in patterns):
            matches.append(line.strip())
    return matches


def verify_existing_tunnel() -> dict[str, Any]:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", 18200), timeout=2.0):
            pass
    except OSError as exc:
        raise MotionRejected("existing_manual_tunnel_not_listening") from exc
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    matches = [
        line.strip() for line in output.splitlines()
        if "ssh" in line and "18200:127.0.0.1:8020" in line and "xzs_server" in line
    ]
    if len(matches) != 1:
        raise MotionRejected(f"manual_tunnel_process_identity_mismatch:{len(matches)}")
    for key in ("NO_PROXY", "no_proxy"):
        values = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        for value in ("127.0.0.1", "localhost"):
            if value not in values:
                values.append(value)
        os.environ[key] = ",".join(values)
    return {
        "status": "pass_existing_manual_tunnel_preserved",
        "local": CLIENT_ENDPOINT,
        "target": f"xzs_server:{COMPUTE_ENDPOINT}",
        "matching_processes": 1,
        "started_stopped_or_reconfigured_by_client": 0,
    }


def validate_service_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "schema_id": SCHEMA_ID,
        "bind": COMPUTE_ENDPOINT,
        "port": 8020,
        "checkpoint_role": "integration_only",
        "checkpoint_selection_status": "not_selected",
        "maximum_unique_matching_Client_requests": 2,
        "OpenPI_commit": OPENPI_COMMIT,
        "config": OPENPI_CONFIG,
        "checkpoint": CHECKPOINT_PATH,
        "artifact_root": str(COMPUTE_ARTIFACT_ROOT),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise MotionRejected(f"server_metadata_mismatch:{field}:{metadata.get(field)!r}")
    service_session_id = require_uuid4(str(metadata.get("service_session_id", "")), "service_session_id")
    intents = metadata.get("execution_intents")
    if intents != ["no_motion_validation", "one_projected_physical_batch"]:
        raise MotionRejected("server_execution_intents_mismatch")
    runtime_commit = str(metadata.get("stable_runtime_commit", ""))
    if len(runtime_commit) != 40 or any(value not in "0123456789abcdef" for value in runtime_commit):
        raise MotionRejected("server_runtime_commit_invalid")
    if metadata.get("stable_runtime_clean") is not True:
        raise MotionRejected("server_runtime_not_clean")
    return {
        **expected,
        "service_session_id": service_session_id,
        "stable_runtime_commit": runtime_commit,
        "stable_runtime_clean": True,
    }


def build_backend(args: Namespace, events: CompactEvents) -> RR006Backend:
    config = {
        "cameras": {
            "head": args.head_serial,
            "left_wrist": args.left_wrist_serial,
            "right_wrist": args.right_wrist_serial,
            "width": args.image_width,
            "height": args.image_height,
            "fps": args.camera_fps,
            "format": "bgr8",
        },
        "robots": {
            "left": {
                "ip": args.left_robot_ip,
                "port": args.robot_port,
                "local_ip": "192.168.1.100",
                "udp_port": 8090,
            },
            "right": {
                "ip": args.right_robot_ip,
                "port": args.robot_port,
                "local_ip": "192.168.1.101",
                "udp_port": 8089,
            },
        },
        "sources": {"deltawam_src": str(Path.home() / "deltawam" / "src")},
    }
    return RR006Backend(
        config,
        events,
        retries=args.controller_query_retries,
        backoff_sec=args.controller_query_backoff_sec,
    )


def require_fresh_safe(
    poller: RobotStatePoller,
    rig: RealSenseRig,
    args: Namespace,
    stage: str,
) -> dict[str, Any]:
    if not rig.healthy(args.state_stale_sec):
        raise MotionRejected(f"camera_unhealthy:{stage}")
    with poller._condition:
        if poller._errors:
            raise MotionRejected(f"robot_state_poll_errors:{stage}:{poller._errors}")
    now_ns = time.monotonic_ns()
    compact = {}
    for side in ("left", "right"):
        sample = poller.latest(side)
        if sample is None:
            raise MotionRejected(f"feedback_missing:{stage}:{side}")
        age_sec = (now_ns - sample.receive_monotonic_ns) / 1e9
        if age_sec < 0 or age_sec > args.state_stale_sec:
            raise MotionRejected(f"feedback_stale:{stage}:{side}:{age_sec:.6f}")
        if sample.sdk_status or sample.arm_errors or sample.controller_error:
            raise MotionRejected(f"robot_alarm:{stage}:{side}")
        compact[side] = {
            "age_sec": age_sec,
            "joint_state_sha256": hashlib.sha256(
                json.dumps(list(sample.joints_rad), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    return compact


def open_live_stack(args: Namespace, events: CompactEvents):
    conflicts = other_robot_command_processes()
    if conflicts:
        raise MotionRejected(f"other_robot_command_source:{conflicts}")
    backend = build_backend(args, events)
    rig: RealSenseRig | None = None
    poller: RobotStatePoller | None = None
    try:
        rig = RealSenseRig(backend.config, events)
        rig.enumerate_and_start()
        rig.wait_ready()
        backend.open()
        bindings = {side: backend.gather_binding(side) for side in ("left", "right")}
        poller = RobotStatePoller(backend, events, hz=args.watchdog_hz)
        poller.start()
        stability = poller.wait_stable(seconds=2.0)
        gate = evaluate_robot_gate_004(bindings, stability)
        udp = backend.start_capture_state()
        state, state_ns, _ = backend.capture_observation_state()
        if time.monotonic_ns() - state_ns > int(args.state_stale_sec * 1e9):
            raise MotionRejected("capture_state_stale_at_gate")
        if list(state.shape) != [28] or str(state.dtype) != "float32":
            raise MotionRejected("capture_state_contract_mismatch")
        images, _, _ = latest_camera_observation(rig, args)
        if any(list(value.shape) != [224, 224, 3] or str(value.dtype) != "uint8" for value in images.values()):
            raise MotionRejected("model_image_contract_mismatch")
        require_fresh_safe(poller, rig, args, "post_gate")
        sensing = {
            "camera_profiles": {
                role: {
                    "serial": profile["serial"],
                    "width": profile["width"],
                    "height": profile["height"],
                    "fps": profile["fps"],
                    "format": profile["format"],
                }
                for role, profile in rig.profiles.items()
            },
            "model_images": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in images.items()
            },
            "state": {"shape": [28], "dtype": "float32", "finite": True, "order": "left_first"},
            "capture_state_status": udp.get("status"),
        }
        return backend, rig, poller, gate, sensing
    except BaseException:
        if poller is not None:
            poller.close()
        if rig is not None:
            rig.close()
        backend.close()
        raise


def close_live_stack(backend: RR006Backend, rig: RealSenseRig, poller: RobotStatePoller) -> None:
    failure: BaseException | None = None
    for close in (rig.close, poller.close, backend.close):
        try:
            close()
        except BaseException as exc:
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure


def open_policy() -> tuple[Any, dict[str, Any]]:
    module = importlib.import_module("openpi_client.websocket_client_policy")
    policy = module.WebsocketClientPolicy(host="127.0.0.1", port=18200)
    try:
        metadata = validate_service_metadata(policy.get_server_metadata())
    except BaseException:
        close = getattr(policy, "close", None)
        if close:
            close()
        raise
    return policy, metadata


def send_request(
    *,
    policy: Any,
    metadata: Mapping[str, Any],
    rollout_id: str,
    request_ordinal: int,
    intent: str,
    rig: RealSenseRig,
    backend: RR006Backend,
    args: Namespace,
    events: CompactEvents,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    if request_ordinal not in (1, 2):
        raise MotionRejected("request_ordinal_outside_exact_budget")
    expected_intent = "no_motion_validation" if request_ordinal == 1 else "one_projected_physical_batch"
    if intent != expected_intent:
        raise MotionRejected("request_intent_or_order_mismatch")
    images, camera_ns, camera_metadata = latest_camera_observation(rig, args)
    state, state_ns, state_metadata = backend.capture_observation_state()
    if state.shape != (28,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise MotionRejected("invalid_live_left_first_state")
    capture_ns = min(camera_ns, state_ns)
    send_ns = time.monotonic_ns()
    age_sec = (send_ns - capture_ns) / 1e9
    if age_sec < 0 or age_sec > args.state_stale_sec:
        raise MotionRejected(f"combined_observation_stale_before_send:{age_sec:.6f}")
    request_id = str(uuid.uuid4())
    context = {
        "schema_id": SCHEMA_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "rollout_id": require_uuid4(rollout_id, "rollout_id"),
        "service_session_id": metadata["service_session_id"],
        "request_id": require_uuid4(request_id, "request_id"),
        "request_ordinal": request_ordinal,
        "execution_intent": intent,
        "client_send_utc": utc_now(),
        "client_send_monotonic_ns": send_ns,
        "observation_capture_monotonic_ns": capture_ns,
    }
    observation = {**images, "state": state, "prompt": PROMPT, "__rr_context__": context}
    events.emit("policy", "request_sending", {
        "request_id": request_id,
        "request_ordinal": request_ordinal,
        "execution_intent": intent,
    })
    started_ns = time.monotonic_ns()
    try:
        response = policy.infer(observation)
    except BaseException as exc:
        events.emit("safety", "policy_request_outcome_unknown", {"error": f"{type(exc).__name__}:{request_id}"})
        raise MotionRejected(f"policy_request_outcome_unknown:{request_id}") from exc
    received_ns = time.monotonic_ns()
    if not isinstance(response, Mapping) or response.get("__rr_context__") != context:
        raise MotionRejected("policy_response_context_mismatch")
    actions = np.asarray(response.get("actions"))
    canonical = map_action28_to_action16(actions)
    summary = {
        "request_id": request_id,
        "request_ordinal": request_ordinal,
        "execution_intent": intent,
        "source_shape": [50, 28],
        "source_dtype": "float32",
        "source_finite": True,
        "source_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
        "canonical_shape": [50, 16],
        "canonical_dtype": "float32",
        "canonical_finite": True,
        "canonical_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "mapping_indices": MAPPING_INDICES,
        "camera_capture_digest": hashlib.sha256(json.dumps(json_safe(camera_metadata), sort_keys=True).encode()).hexdigest(),
        "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "capture_state_digest": hashlib.sha256(json.dumps(json_safe(state_metadata), sort_keys=True).encode()).hexdigest(),
        "observation_age_sec": age_sec,
        "client_round_trip_ms": (received_ns - started_ns) / 1_000_000.0,
        "client_receive_monotonic_ns": received_ns,
        "raw_action_or_trajectory_written": False,
    }
    events.emit("policy", "response_validated", summary)
    return canonical, summary


def prompt_simple_yes(
    events: CompactEvents,
    text: str,
    *,
    field: str,
    safety_check: Any,
) -> None:
    import select

    print(text, end="", flush=True)
    while True:
        safety_check()
        readable, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not readable:
            continue
        line = sys.stdin.readline()
        if line == "":
            raise MotionRejected(f"operator_input_closed:{field}")
        value = line.strip()
        safety_check()
        break
    events.emit("operator", "confirmation", {"field": field, "value": value})
    if value != "yes":
        raise MotionRejected(f"operator_input_mismatch:{field}")


class ProjectedCommandExecutor:
    """Submit each bounded target once and validate controller feedback.

    The 0.01 rad limit applies only to the target projection before send.
    Feedback is judged by SDK/controller/watchdog health and target error; a
    raw sample just beyond the projection boundary is not itself a rejection.
    """

    def __init__(
        self,
        backend: RR006Backend,
        poller: RobotStatePoller,
        rig: RealSenseRig,
        events: CompactEvents,
        args: Namespace,
        pre_model: Mapping[str, Sequence[float]],
    ) -> None:
        self.backend = backend
        self.poller = poller
        self.rig = rig
        self.events = events
        self.args = args
        self.pre_model = {side: tuple(float(value) for value in pre_model[side]) for side in ("left", "right")}
        self.submitted: set[str] = set()
        self.active_side: str | None = None

    def _fresh_samples(self, stage: str) -> dict[str, Any]:
        try:
            require_fresh_safe(self.poller, self.rig, self.args, stage)
        except BaseException:
            if self.active_side is not None:
                self.backend.slow_stop(self.active_side)
            raise
        return {side: self.poller.latest(side) for side in ("left", "right")}

    def execute(
        self,
        *,
        side: str,
        kind: str,
        target: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> dict[str, Any]:
        command_id = require_uuid4(str(uuid.uuid4()), "command_id")
        values = tuple(float(value) for value in target)
        if command_id in self.submitted or len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise MotionRejected(f"invalid_or_duplicate_command:{side}")
        for index, (value, lo, hi, initial) in enumerate(
            zip(values, lower, upper, self.pre_model[side]), start=1
        ):
            if value < float(lo) or value > float(hi):
                raise MotionRejected(f"command_out_of_limit:{side}:q{index}")
            if abs(value - initial) > self.args.joint_delta_limit_rad + 1e-12:
                raise MotionRejected(f"pre_send_projection_exceeded:{side}:q{index}")
        before = self._fresh_samples(f"before_command:{kind}")[side]
        if before is None:
            raise MotionRejected(f"feedback_missing_before_command:{side}")
        self.submitted.add(command_id)
        submitted_ns = time.monotonic_ns()
        self.events.emit("command", "submitting", {"command_id": command_id, "side": side, "kind": kind})
        self.active_side = side
        try:
            status = self.backend.command_absolute(side, values, self.args.speed_percent)
        except BaseException as exc:
            self.backend.slow_stop(side)
            raise MotionRejected(f"command_acceptance_unknown:{command_id}") from exc
        accepted_ns = time.monotonic_ns()
        self.events.emit("command", "sdk_acceptance", {
            "command_id": command_id,
            "side": side,
            "kind": kind,
            "sdk_status": status,
        })
        if status != 0:
            self.backend.slow_stop(side)
            raise MotionRejected(f"command_sdk_rejected:{command_id}:{status}")
        deadline = time.monotonic() + self.args.command_timeout_sec
        while True:
            commanded = self._fresh_samples(f"during_command:{command_id}")[side]
            if commanded is None:
                self.backend.slow_stop(side)
                raise MotionRejected(f"feedback_missing_during_command:{side}")
            error = max(abs(actual - expected) for actual, expected in zip(commanded.joints_rad, values))
            direction_ok = all(
                (target_value - prior) * (actual - prior) >= -1e-12
                for target_value, prior, actual in zip(values, before.joints_rad, commanded.joints_rad)
            )
            if error <= self.args.target_tolerance_rad and time.monotonic_ns() - submitted_ns >= 250_000_000:
                if not direction_ok:
                    self.backend.slow_stop(side)
                    raise MotionRejected(f"observed_direction_mismatch:{command_id}")
                self.active_side = None
                result = {
                    "command_id": command_id,
                    "side": side,
                    "kind": kind,
                    "sdk_status": status,
                    "target_error_rad": error,
                    "accepted_monotonic_ns": accepted_ns,
                    "observed_direction": "pass",
                }
                self.events.emit("command", "observed_target", result)
                return result
            if time.monotonic() >= deadline:
                self.backend.slow_stop(side)
                raise MotionRejected(f"command_timeout:{command_id}")
            time.sleep(0.01)


def compact_command(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "command_id": result["command_id"],
        "side": result["side"],
        "kind": result["kind"],
        "sdk_status": result["sdk_status"],
        "target_error_rad": result["target_error_rad"],
        "observed_direction": result["observed_direction"],
    }


def run_live(args: Namespace, events: CompactEvents) -> dict[str, Any]:
    tunnel = verify_existing_tunnel()
    backend, rig, poller, gate, sensing = open_live_stack(args, events)
    policy = None
    executor: ProjectedCommandExecutor | None = None
    try:
        policy, service = open_policy()
        events.emit("policy", "connection_verified", service)
        canonical_a, no_motion = send_request(
            policy=policy,
            metadata=service,
            rollout_id=events.rollout_id,
            request_ordinal=1,
            intent="no_motion_validation",
            rig=rig,
            backend=backend,
            args=args,
            events=events,
        )
        require_fresh_safe(poller, rig, args, "after_no_motion_response")
        if canonical_a.shape != (50, 16) or backend.motion_started:
            raise MotionRejected("no_motion_zero_command_proof_failed")
        if int(events.counters.get("command.submitting", 0)) != 0:
            raise MotionRejected("no_motion_command_counter_nonzero")
        no_motion["zero_robot_gripper_commands_proven"] = True
        events.emit("stage", "no_motion_pass", {
            "request_id": no_motion["request_id"],
            "zero_robot_gripper_commands_proven": True,
        })
        print(
            "No-motion validation passed: finite float32 [50,28] -> left-first [50,16]; "
            "robot/gripper commands=0.",
            flush=True,
        )
        prompt_simple_yes(
            events,
            "现场确认工作区安全并授权一次受限 physical batch，请输入 yes: ",
            field="physical_batch_authorization",
            safety_check=lambda: require_fresh_safe(poller, rig, args, "awaiting_physical_authorization"),
        )
        require_fresh_safe(poller, rig, args, "before_physical_request")
        canonical_b, physical_response = send_request(
            policy=policy,
            metadata=service,
            rollout_id=events.rollout_id,
            request_ordinal=2,
            intent="one_projected_physical_batch",
            rig=rig,
            backend=backend,
            args=args,
            events=events,
        )
        require_fresh_safe(poller, rig, args, "after_physical_response")
        pre_samples = {side: poller.latest(side) for side in ("left", "right")}
        if any(sample is None for sample in pre_samples.values()):
            raise MotionRejected("fresh_pre_command_state_missing")
        pre_model = {side: list(pre_samples[side].joints_rad) for side in ("left", "right")}
        limits = {
            side: {
                "lower_rad": gate["arms"][side]["lower_rad"],
                "upper_rad": gate["arms"][side]["upper_rad"],
            }
            for side in ("left", "right")
        }
        projected = project_model_row(
            canonical_b[0],
            pre_model["left"],
            pre_model["right"],
            {side: limits[side]["lower_rad"] for side in ("left", "right")},
            {side: limits[side]["upper_rad"] for side in ("left", "right")},
            delta_max_rad=args.joint_delta_limit_rad,
        )
        projection_max = max(
            abs(projected["projected_joint_targets"][side][index] - pre_model[side][index])
            for side in ("left", "right") for index in range(7)
        )
        if projection_max > args.joint_delta_limit_rad + 1e-12:
            raise MotionRejected("projection_limit_failed_before_send")
        if (time.monotonic_ns() - int(physical_response["client_receive_monotonic_ns"])) / 1e9 > args.response_to_command_sec:
            raise MotionRejected("response_to_first_command_timeout")
        executor = ProjectedCommandExecutor(backend, poller, rig, events, args, pre_model)
        model_commands = []
        for side in ("right", "left"):
            model_commands.append(executor.execute(
                side=side,
                kind="projected_model_row_0",
                target=projected["projected_joint_targets"][side],
                lower=limits[side]["lower_rad"],
                upper=limits[side]["upper_rad"],
            ))
        if len(model_commands) != 2:
            raise MotionRejected("model_command_count_mismatch")
        return_commands = []
        for side in ("right", "left"):
            return_commands.append(executor.execute(
                side=side,
                kind="return_to_pre_model",
                target=pre_model[side],
                lower=limits[side]["lower_rad"],
                upper=limits[side]["upper_rad"],
            ))
        settling = settle_after_return(
            args=args,
            rig=rig,
            poller=poller,
            pre_model=pre_model,
            return_accepted_ns=int(return_commands[-1]["accepted_monotonic_ns"]),
            events=events,
        )

        def monitor_return_confirmation() -> None:
            require_fresh_safe(poller, rig, args, "awaiting_return_confirmation")
            for side in ("left", "right"):
                sample = poller.latest(side)
                if sample is None:
                    raise MotionRejected(f"return_feedback_missing:{side}")
                error = max(abs(actual - target) for actual, target in zip(sample.joints_rad, pre_model[side]))
                if error > args.target_tolerance_rad:
                    raise MotionRejected(f"return_error_while_awaiting_confirmation:{side}:{error:.9f}")

        print(
            "Bounded return and stable feedback passed; visually verify both arms returned.",
            flush=True,
        )
        prompt_simple_yes(
            events,
            "现场确认双臂已安全回位，请输入 yes: ",
            field="return_confirmation",
            safety_check=monitor_return_confirmation,
        )
        return {
            "status": "completed_approved_scope",
            "tunnel": tunnel,
            "service": service,
            "rollout_id": events.rollout_id,
            "sensing": sensing,
            "no_motion_validation": no_motion,
            "physical_response": physical_response,
            "projection": {
                "row": 0,
                "mapping_indices": MAPPING_INDICES,
                "max_abs_delta_rad": projection_max,
                "policy": "pre_send_target_projection_only",
                "gripper": "held_current_zero_commands",
            },
            "model_commands": [compact_command(item) for item in model_commands],
            "return_commands": [compact_command(item) for item in return_commands],
            "settling": settling,
            "return_status": "completed_stable_and_operator_confirmed",
        }
    except BaseException:
        if executor is not None and executor.active_side is not None:
            try:
                backend.slow_stop(executor.active_side)
            except BaseException as stop_exc:
                events.emit("safety", "slow_stop_failure", {"error": repr(stop_exc)})
        raise
    finally:
        if policy is not None:
            close = getattr(policy, "close", None)
            if close:
                close()
        close_live_stack(backend, rig, poller)


def request_compute_finalize() -> dict[str, Any]:
    command = (
        "cd /share/jhk/phywam-mope-real/pi05-service-runtime && "
        "./scripts/run_pi05_service.py finalize --preset pi05-realman-30000"
    )
    run = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "xzs_server", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=330,
    )
    if run.returncode != 0:
        raise MotionRejected(f"compute_finalize_failed:{run.returncode}:{run.stderr.strip()[-240:]}")
    return {"status": "requested_and_completed", "preset": PRESET, "exit_code": 0}


def fetch_compute_bundle() -> dict[str, Any]:
    names = ("compute-receipt.yaml", "run-manifest.yaml", "SHA256SUMS")
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "xzs_server",
        "tar", "-C", str(COMPUTE_ARTIFACT_ROOT), "-cf", "-", *names,
    ]
    run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    if run.returncode != 0:
        raise MotionRejected(f"compute_receipt_pull_failed:{run.returncode}:{run.stderr.decode(errors='replace')[-240:]}")
    with tarfile.open(fileobj=io.BytesIO(run.stdout), mode="r:") as archive:
        members = archive.getmembers()
        if sorted(member.name for member in members) != sorted(names) or any(not member.isfile() for member in members):
            raise MotionRejected("compute_bundle_allowlist_mismatch")
        payloads = {}
        for member in members:
            stream = archive.extractfile(member)
            if stream is None:
                raise MotionRejected(f"compute_bundle_member_unreadable:{member.name}")
            payloads[member.name] = stream.read()
    ledger_lines = payloads["SHA256SUMS"].decode("utf-8").splitlines()
    ledger = {}
    for line in ledger_lines:
        digest, name = line.split(maxsplit=1)
        ledger[name.lstrip(" *")] = digest
    for name in ("compute-receipt.yaml", "run-manifest.yaml"):
        digest = hashlib.sha256(payloads[name]).hexdigest()
        if ledger.get(name) != digest:
            raise MotionRejected(f"compute_bundle_digest_mismatch:{name}")
    import yaml

    receipt = yaml.safe_load(payloads["compute-receipt.yaml"])
    manifest = yaml.safe_load(payloads["run-manifest.yaml"])
    if not isinstance(receipt, Mapping) or not isinstance(manifest, Mapping):
        raise MotionRejected("compute_bundle_document_type_mismatch")
    if receipt.get("task_id") != TASK_ID or receipt.get("run_id") != RUN_ID:
        raise MotionRejected("compute_receipt_identity_mismatch")
    if manifest.get("task_id") != TASK_ID or manifest.get("run_id") != RUN_ID:
        raise MotionRejected("compute_manifest_identity_mismatch")
    require_uuid4(str(receipt.get("service_session_id", "")), "compute_receipt_service_session_id")
    return {
        "status": "pulled_and_verified_read_only",
        "artifact_root": str(COMPUTE_ARTIFACT_ROOT),
        "receipt_sha256": hashlib.sha256(payloads["compute-receipt.yaml"]).hexdigest(),
        "manifest_sha256": hashlib.sha256(payloads["run-manifest.yaml"]).hexdigest(),
        "ledger_sha256": hashlib.sha256(payloads["SHA256SUMS"]).hexdigest(),
        "receipt": json_safe(receipt),
    }


def reconcile_compute_receipt(compute: Mapping[str, Any], events: CompactEvents) -> dict[str, Any]:
    receipt = compute.get("receipt", {})
    if not isinstance(receipt, Mapping):
        raise MotionRejected("compute_receipt_missing_for_reconciliation")
    counts = receipt.get("request_counts", {})
    if not isinstance(counts, Mapping):
        raise MotionRejected("compute_request_counts_missing")
    expected_responses = int(events.counters.get("policy.response_validated", 0))
    mismatches = []
    expected = {
        "client_requests_accepted": expected_responses,
        "client_responses_sent": expected_responses,
        "inference_failures": 0,
        "response_send_failures": 0,
    }
    for field, value in expected.items():
        if counts.get(field) != value:
            mismatches.append(f"{field}:{counts.get(field)!r}!={value!r}")
    if events.service:
        if receipt.get("service_session_id") != events.service.get("service_session_id"):
            mismatches.append("service_session_id")
        if expected_responses and receipt.get("client_rollout_id") != events.rollout_id:
            mismatches.append("client_rollout_id")
    if receipt.get("client_commands") != 0:
        mismatches.append("compute_client_commands_nonzero")
    if "finalized" not in str(receipt.get("status", "")):
        mismatches.append(f"compute_status_not_finalized:{receipt.get('status')!r}")
    return {
        "status": "pass" if not mismatches else "mismatch",
        "expected_client_responses": expected_responses,
        "compute_request_counts": json_safe(counts),
        "client_rollout_id": receipt.get("client_rollout_id"),
        "mismatches": mismatches,
    }


def terminal_counts(events: CompactEvents) -> dict[str, int]:
    counters = events.counters
    commands = int(counters.get("command.submitting", 0))
    accepted = int(counters.get("command.sdk_acceptance", 0))
    observed = int(counters.get("command.observed_target", 0))
    return {
        "policy_connections": 1 if int(counters.get("policy.request_sending", 0)) else 0,
        "policy_requests": int(counters.get("policy.request_sending", 0)),
        "policy_responses": int(counters.get("policy.response_validated", 0)),
        "joint_commands_submitted": commands,
        "joint_commands_sdk_accepted": accepted,
        "joint_commands_observed_target": observed,
        "model_joint_commands": min(commands, 2),
        "return_joint_commands": max(0, commands - 2),
        "gripper_commands": 0,
        "physical_batches": 1 if commands >= 2 else 0,
        "video_writers_encoders_recorders": 0,
        "raw_action_or_full_trajectory_files": 0,
    }


def write_terminal_artifacts(
    *,
    identity: Mapping[str, Any],
    events: CompactEvents,
    live: Mapping[str, Any] | None,
    stop_reason: str | None,
    finalize: Mapping[str, Any] | None,
    compute: Mapping[str, Any] | None,
) -> dict[str, Any]:
    root = artifact_root()
    root.mkdir(parents=True, exist_ok=False)
    counts = terminal_counts(events)
    completed = stop_reason is None and live is not None and finalize is not None and compute is not None
    service = dict(live.get("service", {})) if live else dict(events.service)
    request_summaries = sorted(events.request_facts.values(), key=lambda item: int(item.get("request_ordinal", 99)))
    commands = list(events.command_facts.values())
    no_motion_pass = int(events.counters.get("stage.no_motion_pass", 0)) == 1
    model_observed = sum(
        item.get("kind") == "projected_model_row_0" and item.get("outcome") == "observed_target"
        for item in commands
    )
    return_observed = sum(
        item.get("kind") == "return_to_pre_model" and item.get("outcome") == "observed_target"
        for item in commands
    )
    return_confirmed = any(
        item.get("field") == "return_confirmation" and item.get("accepted") is True
        for item in events.operator_confirmations
    )
    receipt = {
        "schema_version": 1,
        "status": "terminal",
        "identity": {
            "task_id": TASK_ID,
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "endpoint_role": "robot_client_executor_and_integrator",
            "preset": PRESET,
            "service_session_id": service.get("service_session_id"),
            "rollout_id": live.get("rollout_id") if live else events.rollout_id,
        },
        "source_chain": {
            "client_runtime_commit": identity["runtime_commit"],
            "client_runtime_clean": identity["runtime_clean"],
            "compute_runtime_commit": service.get("stable_runtime_commit"),
            "OpenPI_commit": service.get("OpenPI_commit"),
            "compute_receipt_sha256": compute.get("receipt_sha256") if compute else None,
        },
        "execution": {
            "execution_status": "completed" if completed else "blocked",
            "no_motion_validation": "pass_zero_commands" if no_motion_pass else "not_completed",
            "physical_command_acceptance": "pass_sdk_and_target_error" if model_observed == 2 else "not_completed",
            "return_status": "completed_stable_and_operator_confirmed" if return_observed == 2 and return_confirmed else "not_completed",
            "stop_reason": stop_reason,
            "compute_finalize": json_safe(finalize) if finalize else "failed_or_not_completed",
            "continuation_decision": "return_to_mac",
        },
        "requests": request_summaries,
        "commands": commands,
        "counters": counts,
        "sdk_acceptance": [command.get("sdk_status") for command in commands if "sdk_status" in command],
        "projection": live.get("projection") if live else None,
        "settling": live.get("settling") if live else None,
        "operator_confirmations": events.operator_confirmations,
        "compute_reconciliation": {
            key: compute[key]
            for key in (
                "status", "artifact_root", "receipt_sha256", "manifest_sha256", "ledger_sha256",
                "reconciliation",
            )
            if compute and key in compute
        },
        "status_dimensions": {
            "execution_status": "completed" if completed else "blocked",
            "provenance_status": "complete" if compute is not None else "partial",
            "gate_result": "not_run",
            "review_status": "not_requested",
            "deployment_readiness": "not_ready",
            "safety_readiness": "not_ready",
            "claim_status": "unsupported",
            "continuation_decision": "return_to_mac",
        },
        "artifact_boundary": {
            "record_mode": "off",
            "video_or_encoder_output": 0,
            "raw_action_dump": 0,
            "full_trajectory_dump": 0,
        },
        "created_at": utc_now(),
    }
    write_yaml(root / "client-receipt.yaml", receipt)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "artifact_root": str(root),
        "required_files": list(FINAL_ARTIFACTS),
        "file_count": 3,
        "forbidden_artifacts_present": False,
        "client_runtime_commit": identity["runtime_commit"],
        "compute_receipt_sha256": compute.get("receipt_sha256") if compute else None,
    }
    write_yaml(root / "run-manifest.yaml", manifest)
    ledger = "\n".join(
        f"{sha256_file(root / name)}  {name}"
        for name in ("client-receipt.yaml", "run-manifest.yaml")
    ) + "\n"
    with (root / "SHA256SUMS").open("x", encoding="utf-8") as stream:
        stream.write(ledger)
        stream.flush()
        os.fsync(stream.fileno())
    existing = sorted(path.name for path in root.iterdir() if path.is_file())
    if existing != sorted(FINAL_ARTIFACTS):
        raise MotionRejected(f"client_artifact_allowlist_mismatch:{existing}")
    return {"artifact_root": str(root), "files": list(FINAL_ARTIFACTS), "completed": completed}


def append_attempt(payload: Mapping[str, Any]) -> None:
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "attempt-ledger.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(dict(payload)), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    if parsed.preset != PRESET:
        raise SystemExit(2)
    identity: dict[str, Any] | None = None
    live: dict[str, Any] | None = None
    finalize: dict[str, Any] | None = None
    compute: dict[str, Any] | None = None
    stop_reason: str | None = None
    live_attempted = False
    events = CompactEvents(str(uuid.uuid4()))
    append_attempt({"event": "started", "attempt": 1, "rollout_id": events.rollout_id, "at": utc_now()})
    try:
        if artifact_root().exists():
            raise MotionRejected("client_artifact_root_already_exists_preserve_prior_run")
        identity = validate_client_identity()
        live_attempted = True
        live = run_live(preset_args(), events)
    except BaseException as exc:
        stop_reason = f"{type(exc).__name__}:{exc}"
    if live_attempted:
        try:
            finalize = request_compute_finalize()
            compute = fetch_compute_bundle()
            compute["reconciliation"] = reconcile_compute_receipt(compute, events)
            if stop_reason is None and compute["reconciliation"]["status"] != "pass":
                stop_reason = "delivery_failure:compute_receipt_reconciliation_mismatch"
        except BaseException as exc:
            if stop_reason is None:
                stop_reason = f"delivery_failure:{type(exc).__name__}:{exc}"
            events.emit("safety", "compute_delivery_failure", {"error": repr(exc)})
    if identity is None:
        try:
            provenance = exact_runtime_provenance()
            identity = {
                "runtime_commit": provenance["commit"],
                "runtime_clean": provenance["clean"],
                "login_user": subprocess.check_output(["id", "-un"], text=True).strip(),
            }
        except BaseException as exc:
            print(json.dumps({
                "status": "failed",
                "blocker": stop_reason or repr(exc),
                "artifact_root": str(artifact_root()),
                "attempts": "1/10",
                "continuation_decision": "return_to_mac",
            }, ensure_ascii=False), flush=True)
            return 1
    try:
        artifact = write_terminal_artifacts(
            identity=identity,
            events=events,
            live=live,
            stop_reason=stop_reason,
            finalize=finalize,
            compute=compute,
        )
    except BaseException as exc:
        print(json.dumps({
            "status": "failed",
            "blocker": stop_reason or repr(exc),
            "artifact_root": str(artifact_root()),
            "attempts": "1/10",
            "service_tunnel": "preserved_unknown_after_artifact_failure",
            "continuation_decision": "return_to_mac",
        }, ensure_ascii=False), flush=True)
        return 1
    append_attempt({
        "event": "terminal",
        "attempt": 1,
        "status": "completed" if stop_reason is None else "blocked",
        "rollout_id": events.rollout_id,
        "stop_reason": stop_reason,
        "event_counters": dict(events.counters),
        "at": utc_now(),
    })
    output = {
        "status": "completed_approved_scope" if stop_reason is None else "blocked",
        "blocker": stop_reason,
        "artifact_root": artifact["artifact_root"],
        "attempts": "1/10",
        "service_finalize": finalize,
        "tunnel_status": "preserved",
        "continuation_decision": "return_to_mac",
    }
    print(json.dumps(json_safe(output), indent=2, ensure_ascii=False), flush=True)
    return 0 if stop_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
