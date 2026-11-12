"""RR-DEP-006 low-latency, no-recording pi0.5 Client runtime.

The runtime intentionally splits the task into four explicit invocations:

* ``dry-run`` freezes help and zero-side-effect validation;
* ``no-motion`` performs the one Stage A request and stores only a compact
  task-cache checkpoint;
* ``staged-one-batch`` requires the Stage A checkpoint, asks the onsite
  operator for the exact motion phrase, and performs only the new physical
  request plus the bounded command/return sequence;
* ``finalize`` reconciles the Compute receipt and freezes the six permitted
  Client artifacts.

No mode creates a video writer, encoder, raw action file, canonical action
file, camera recording, state trajectory, or full telemetry ledger.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import http.client
import importlib
import json
import math
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
import uuid

from .rr_dep_003 import RealSenseRig, RobotStatePoller, json_safe, sha256_file, write_yaml
from .rr_dep_003_motion import MotionRejected
from .rr_dep_004 import (
    CaptureUDPReceiver,
    evaluate_robot_gate_004,
    map_action28_to_action16,
    project_model_row,
    require_uuid4,
)
from .rr_dep_005 import (
    JointCommandExecutor005,
    RR005Backend,
    latest_camera_observation,
    settle_after_return,
)


TASK_ID = "RR-DEP-006"
EXPERIMENT_ID = TASK_ID
RUN_ID = "RR-DEP-006-RUN-001"
SERVICE_SCHEMA_ID = "rr_dep_openpi_robot_live_v2"
ACTION_SCHEMA_ID = "realman.dual_arm.absolute.v1"
SERVICE_SESSION_ID = "39efea6c-db55-4390-8fa3-e44279d526f4"
SERVICE_RUNTIME_BASE_COMMIT = "d438df89fe8f46787c342e6e48963643690cf484"
SERVICE_RUNTIME_COMMIT = "bc9de46eb8bc92c8ae42e714f12ed1f89c0dbcc2"
PROMPT = "Place the three objects into the blue container."
OPERATOR_PHRASE = "AUTHORIZE RR-DEP-006 ONE PROJECTED MODEL STEP"
RETURN_ACK_PHRASE = "ACK RR-DEP-006 RETURN SAFE"
CLIENT_PORT = 18200
COMPUTE_PORT = 8020
MAPPING_INDICES = list(range(8)) + list(range(14, 22))
FINAL_ARTIFACTS = (
    "client-help.txt",
    "client-dry-run.json",
    "execution-summary.yaml",
    "client-receipt.yaml",
    "run-manifest.yaml",
    "SHA256SUMS",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MotionRejected(f"expected_mapping:{path}")
    return value


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(dict(payload)), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _digest_payload(value: Any) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def port_number(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError("must be in 1..65535")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def canonical_uuid4_or_auto(text: str) -> str:
    if text == "auto":
        return text
    try:
        parsed = uuid.UUID(text)
    except (TypeError, ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("must be canonical UUIDv4 or auto") from exc
    if parsed.version != 4 or str(parsed) != text:
        raise argparse.ArgumentTypeError("must be canonical UUIDv4 or auto")
    return text


def project_root() -> Path:
    return Path(__file__).resolve().parents[2].parent


def artifact_root() -> Path:
    return project_root() / "runs" / TASK_ID / RUN_ID / "client"


def cache_root() -> Path:
    return project_root() / "cache" / "rr-dep-006-run-001"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RR-DEP-006 low-latency no-recording Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("dry-run", "no-motion", "staged-one-batch", "finalize"), required=True,
                        help="Explicit side-effect stage; motion exists only in staged-one-batch after interactive authorization")
    parser.add_argument("--record-mode", choices=("off",), default="off",
                        help="Live modes permit only off; no recorder, writer, encoder, video, or trajectory dump is created")
    parser.add_argument("--server-url", default="ws://127.0.0.1:18200")
    parser.add_argument("--ssh-alias", default="xzs_server")
    parser.add_argument("--ssh-local-port", type=port_number, default=CLIENT_PORT)
    parser.add_argument("--compute-port", type=port_number, default=COMPUTE_PORT)
    parser.add_argument("--service-session-id", type=canonical_uuid4_or_auto, default=SERVICE_SESSION_ID)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--rollout-id", type=canonical_uuid4_or_auto, default="auto")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--head-serial", default="420122071413")
    parser.add_argument("--left-wrist-serial", default="323622272614")
    parser.add_argument("--right-wrist-serial", default="230422272673")
    parser.add_argument("--left-robot-ip", default="192.168.1.18")
    parser.add_argument("--right-robot-ip", default="192.168.1.19")
    parser.add_argument("--robot-port", type=port_number, default=8080)
    parser.add_argument("--image-width", type=positive_int, default=640)
    parser.add_argument("--image-height", type=positive_int, default=480)
    parser.add_argument("--camera-fps", type=positive_int, default=30)
    parser.add_argument("--model-image-size", type=positive_int, default=224)
    parser.add_argument("--state-dim", type=positive_int, default=28)
    parser.add_argument("--action-dim", type=positive_int, default=28)
    parser.add_argument("--action-row", type=int, default=0)
    parser.add_argument("--joint-delta-limit-rad", type=positive_float, default=0.01)
    parser.add_argument("--speed-percent", type=positive_int, default=5)
    parser.add_argument("--target-tolerance-rad", type=positive_float, default=0.005)
    parser.add_argument("--state-stale-sec", type=positive_float, default=0.25)
    parser.add_argument("--response-to-command-sec", type=positive_float, default=2.0)
    parser.add_argument("--command-timeout-sec", type=positive_float, default=5.0)
    parser.add_argument("--watchdog-hz", type=positive_float, default=20.0)
    parser.add_argument("--controller-query-retries", type=positive_int, default=10)
    parser.add_argument("--controller-query-backoff-sec", type=positive_float, default=0.05)
    parser.add_argument("--settle-grace-sec", type=nonnegative_float, default=2.0)
    parser.add_argument("--stable-window-sec", type=positive_float, default=2.0)
    parser.add_argument("--stable-span-threshold-rad", type=positive_float, default=0.001)
    parser.add_argument("--settle-timeout-sec", type=positive_float, default=10.0)
    parser.add_argument("--max-attempts", type=positive_int, default=10)
    parser.add_argument("--max-policy-requests", type=positive_int, default=10)
    parser.add_argument("--gripper-mode", choices=("hold-unproven-conversion",), default="hold-unproven-conversion")
    parser.add_argument("--return-to-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root())
    parser.add_argument("--operator-phrase", default=OPERATOR_PHRASE,
                        help="Expected interactive text only; supplying this flag never authorizes motion")
    parser.add_argument("--return-ack-phrase", default=RETURN_ACK_PHRASE,
                        help="Expected interactive text only; supplying this flag never acknowledges return")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    parsed = urlparse(args.server_url)
    if parsed.scheme != "ws" or parsed.hostname != "127.0.0.1" or parsed.port != CLIENT_PORT:
        raise MotionRejected("server_url_must_be_exact_existing_loopback_tunnel")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise MotionRejected("server_url_must_not_have_path_query_or_fragment")
    expected = {
        "record_mode": "off", "ssh_alias": "xzs_server", "ssh_local_port": CLIENT_PORT,
        "compute_port": COMPUTE_PORT, "service_session_id": SERVICE_SESSION_ID,
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "prompt": PROMPT,
        "head_serial": "420122071413", "left_wrist_serial": "323622272614",
        "right_wrist_serial": "230422272673", "left_robot_ip": "192.168.1.18",
        "right_robot_ip": "192.168.1.19", "robot_port": 8080, "image_width": 640,
        "image_height": 480, "camera_fps": 30, "model_image_size": 224,
        "state_dim": 28, "action_dim": 28, "action_row": 0,
        "settle_grace_sec": 2.0, "stable_window_sec": 2.0,
        "controller_query_retries": 10, "controller_query_backoff_sec": 0.05,
        "operator_phrase": OPERATOR_PHRASE, "return_ack_phrase": RETURN_ACK_PHRASE,
    }
    for field, value in expected.items():
        if getattr(args, field) != value:
            raise MotionRejected(f"contract_value_mismatch:{field}")
    if args.artifact_root.expanduser().resolve() != artifact_root().resolve():
        raise MotionRejected("artifact_root_must_match_contract")
    if args.joint_delta_limit_rad > 0.01 or args.speed_percent > 5:
        raise MotionRejected("projection_or_speed_limit_exceeds_contract")
    if args.target_tolerance_rad > 0.005 or args.state_stale_sec > 0.25:
        raise MotionRejected("target_tolerance_or_freshness_exceeds_contract")
    if args.response_to_command_sec > 2.0 or args.command_timeout_sec > 5.0:
        raise MotionRejected("response_or_command_timeout_exceeds_contract")
    if args.watchdog_hz < 10.0:
        raise MotionRejected("watchdog_below_contract_minimum")
    if args.stable_span_threshold_rad > 0.001 or args.settle_timeout_sec > 10.0:
        raise MotionRejected("settling_threshold_or_timeout_exceeds_contract")
    if args.max_attempts > 10 or args.max_policy_requests > 10:
        raise MotionRejected("attempt_or_policy_request_budget_exceeds_contract")
    if args.mode == "staged-one-batch" and not args.return_to_start:
        raise MotionRejected("physical_stage_requires_return_to_start")
    if args.gripper_mode != "hold-unproven-conversion":
        raise MotionRejected("gripper_conversion_unproven_hold_required")


def command_array(args: argparse.Namespace) -> list[str]:
    result = [str(Path(__file__).resolve().parents[2] / "scripts" / "run_rr_dep_006.py")]
    values = vars(args)
    for action in build_parser()._actions:
        if not action.option_strings or action.dest == "help":
            continue
        value = values[action.dest]
        if isinstance(action, argparse.BooleanOptionalAction):
            result.append(action.option_strings[0] if value else action.option_strings[1])
        else:
            result.extend((action.option_strings[0], str(value)))
    return result


class CompactEventSink:
    """In-memory counters only; high-rate payloads and operator text are discarded."""

    def __init__(self, rollout_id: str) -> None:
        self.rollout_id = rollout_id
        self.counters: Counter[str] = Counter()
        self.safety_events: list[dict[str, str]] = []
        self.operator_facts: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def emit(self, name: str, event: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self.counters[f"{name}.{event}"] += 1
            if name == "safety":
                self.safety_events.append({"event": event, "error": str(payload.get("error", ""))[:240]})
            elif name == "operator" and event == "operator_input":
                value = str(payload.get("value", ""))
                self.operator_facts.append({
                    "field": str(payload.get("field", "")),
                    "present": bool(value),
                    "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                })

    def compact(self) -> dict[str, Any]:
        with self._lock:
            return {
                "event_counters": dict(sorted(self.counters.items())),
                "safety_events": list(self.safety_events),
                "operator_facts": list(self.operator_facts),
            }


class RR006Backend(RR005Backend):
    """RR-DEP-005 backend with bounded read-only controller-query recovery."""

    def __init__(self, config: dict[str, Any], events: CompactEventSink, *, retries: int, backoff_sec: float) -> None:
        super().__init__(config, events)
        self.controller_query_retries = int(retries)
        self.controller_query_backoff_sec = float(backoff_sec)
        self.controller_retry_counts = {"left": 0, "right": 0}
        self.controller_minus_2_events = {"left": 0, "right": 0}
        self.motion_started = False

    def _call(self, side: str, method: str) -> Any:
        result = super()._call(side, method)
        if method != "rm_get_controller_state" or not isinstance(result, Mapping):
            return result
        if int(result.get("return_code", -1)) != -2:
            return result
        self.controller_minus_2_events[side] += 1
        self.events.emit("ack", "controller_query_minus_2", {"side": side})
        if self.motion_started:
            return result
        for ordinal in range(1, self.controller_query_retries + 1):
            time.sleep(self.controller_query_backoff_sec)
            self.controller_retry_counts[side] += 1
            result = super()._call(side, method)
            code = int(result.get("return_code", -1)) if isinstance(result, Mapping) else -1
            self.events.emit("ack", "controller_query_retry", {"side": side, "ordinal": ordinal, "return_code": code})
            if code != -2:
                return result
        return result

    def command_absolute(self, side: str, target_rad: Sequence[float], speed_percent: int) -> int:
        self.motion_started = True
        return super().command_absolute(side, target_rad, speed_percent)

    def start_capture_state(self) -> dict[str, Any]:
        """Configure capture once and resolve ``-2`` only by exact readback.

        The setter is never retried. A nonzero status is accepted only when a
        fresh getter proves that the complete selected configuration is
        already active.
        """
        for side in ("left", "right"):
            item = self.config["robots"][side]
            receiver = CaptureUDPReceiver(str(item["local_ip"]), int(item["udp_port"]), side, self.events)
            receiver.start()
            self.udp_receivers[side] = receiver
        evidence: dict[str, Any] = {}
        for side in ("left", "right"):
            mode_code, prior_mode = self._call(side, "rm_get_rm_plus_mode")
            if int(mode_code) != 0 or int(prior_mode) not in (0, 9600, 115200, 256000, 460800):
                raise MotionRejected(f"get_rm_plus_mode_failed:{side}:{mode_code}:{prior_mode}")
            self.prior_rm_plus_mode[side] = int(prior_mode)
            mode_response = None
            if int(prior_mode) != 115200:
                mode_response = self._set_rm_plus_json(side, 115200)
                self.rm_plus_changed.add(side)
            code, prior = self._call(side, "rm_get_realtime_push")
            if int(code) != 0 or not isinstance(prior, dict):
                raise MotionRejected(f"get_realtime_push_failed:{side}:{code}:{prior}")
            self.prior_udp_config[side] = json_safe(prior)
            item = self.config["robots"][side]
            custom = self.module.rm_udp_custom_config_t(
                joint_speed=1, arm_current_status=1, plus_base=1, plus_state=1,
            )
            selected = self.module.rm_realtime_push_config_t(
                cycle=1,
                enable=True,
                port=int(item["udp_port"]),
                force_coordinate=0,
                ip=str(item["local_ip"]),
                custom_config=custom,
            )
            selected_dict = json_safe(selected.to_dict())
            effective_selected = json.loads(json.dumps(self.prior_udp_config[side]))
            effective_selected.update({
                "cycle": 1,
                "enable": True,
                "port": int(item["udp_port"]),
                "force_coordinate": 0,
                "ip": str(item["local_ip"]),
            })
            effective_custom = effective_selected.setdefault("custom_config", {})
            effective_custom.update({
                "joint_speed": 1,
                "arm_current_status": 1,
                "plus_base": 1,
                "plus_state": 1,
            })
            status = int(self.arms[side].rm_set_realtime_push(selected))
            read_code, readback = self._call(side, "rm_get_realtime_push")
            exact = int(read_code) == 0 and json_safe(readback) == effective_selected
            if status != 0 and not (status == -2 and exact):
                raise MotionRejected(f"set_realtime_push_unresolved:{side}:{status}:{read_code}:{exact}")
            if not exact:
                raise MotionRejected(f"set_realtime_push_readback_mismatch:{side}:{status}:{read_code}")
            self.udp_configured.add(side)
            evidence[side] = {
                "prior_digest": _digest_payload(self.prior_udp_config[side]),
                "setter_argument_digest": _digest_payload(selected_dict),
                "effective_selected_digest": _digest_payload(effective_selected),
                "sdk_status": status,
                "readback_code": int(read_code),
                "readback_exact_selected": exact,
                "status_minus_2_resolved_without_resend": status == -2 and exact,
                "prior_rm_plus_mode": int(prior_mode),
                "selected_rm_plus_mode": 115200,
                "rm_plus_json_response_present": mode_response is not None,
            }
        for receiver in self.udp_receivers.values():
            receiver.wait_ready()
        evidence["status"] = "pass_exact_capture_generator_fields_units_and_RMPlus_gripper"
        return evidence

    def retry_summary(self) -> dict[str, Any]:
        return {
            "minus_2_events": dict(self.controller_minus_2_events),
            "read_only_retries": dict(self.controller_retry_counts),
            "maximum_retries_per_event": self.controller_query_retries,
            "backoff_sec": self.controller_query_backoff_sec,
            "retry_after_motion_started": False,
        }


def ensure_loopback_no_proxy() -> dict[str, Any]:
    added = []
    for key in ("NO_PROXY", "no_proxy"):
        current = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        for value in ("127.0.0.1", "localhost"):
            if value not in current:
                current.append(value)
                added.append(f"{key}:{value}")
        os.environ[key] = ",".join(current)
    return {"loopback_bypass_present": True, "process_local_entries_added": added}


def verify_existing_tunnel(args: argparse.Namespace) -> dict[str, Any]:
    try:
        with socket.create_connection(("127.0.0.1", CLIENT_PORT), timeout=2.0):
            pass
    except OSError as exc:
        raise MotionRejected("existing_manual_tunnel_not_listening") from exc
    output = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, text=True, stdout=subprocess.PIPE,
    ).stdout
    matches = []
    for line in output.splitlines():
        if "ssh" in line and "18200:127.0.0.1:8020" in line and args.ssh_alias in line:
            matches.append(line.strip())
    if len(matches) != 1:
        raise MotionRejected(f"manual_tunnel_process_identity_mismatch:{len(matches)}")
    return {
        "status": "pass_existing_manual_tunnel_preserved",
        "local": "127.0.0.1:18200",
        "target": "xzs_server:127.0.0.1:8020",
        "matching_processes": 1,
        "started_stopped_or_reconfigured_by_Client": 0,
    }


def validate_service_metadata(metadata: Mapping[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "schema_id": SERVICE_SCHEMA_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "service_session_id": SERVICE_SESSION_ID,
        "bind": "127.0.0.1:8020",
        "port": COMPUTE_PORT,
        "checkpoint_role": "integration_only",
        "checkpoint_selection_status": "not_selected",
        "stable_runtime_commit": SERVICE_RUNTIME_COMMIT,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise MotionRejected(f"server_metadata_mismatch:{field}:{metadata.get(field)!r}")
    if metadata.get("task_id") != TASK_ID:
        raise MotionRejected("server_task_id_mismatch")


def open_verified_policy(args: argparse.Namespace) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    tunnel = verify_existing_tunnel(args)
    no_proxy = ensure_loopback_no_proxy()
    module = importlib.import_module("openpi_client.websocket_client_policy")
    policy = module.WebsocketClientPolicy(host="127.0.0.1", port=CLIENT_PORT)
    metadata = policy.get_server_metadata()
    try:
        validate_service_metadata(metadata, args)
    except Exception:
        if hasattr(policy, "close"):
            policy.close()
        raise
    compact = {
        "schema_id": metadata["schema_id"],
        "experiment_id": metadata["experiment_id"],
        "run_id": metadata["run_id"],
        "service_session_id": metadata["service_session_id"],
        "bind": metadata["bind"],
        "port": metadata["port"],
        "stable_runtime_base_commit": SERVICE_RUNTIME_BASE_COMMIT,
        "stable_runtime_commit": metadata["stable_runtime_commit"],
        "checkpoint_role": metadata["checkpoint_role"],
        "checkpoint_selection_status": metadata["checkpoint_selection_status"],
    }
    return policy, compact, {"tunnel": tunnel, "no_proxy": no_proxy}


def other_robot_command_processes() -> list[str]:
    output = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True, stdout=subprocess.PIPE).stdout
    patterns = (
        "realman_deployment.client", "run_realman_fastwam_client", "spacemouse_control_arm",
        "run_rr_dep_003.py", "run_rr_dep_004.py", "run_pi05_realman.py", "run_rr_dep_006.py",
    )
    current = os.getpid()
    matches = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or int(fields[0]) == current:
            continue
        if any(pattern in fields[1] for pattern in patterns):
            matches.append(line.strip())
    return matches


def require_fresh_safe(
    poller: RobotStatePoller, rig: RealSenseRig, args: argparse.Namespace, stage: str,
) -> dict[str, Any]:
    if not rig.healthy(args.state_stale_sec):
        raise MotionRejected(f"camera_unhealthy:{stage}")
    with poller._condition:
        if poller._errors:
            raise MotionRejected(f"robot_state_poll_errors:{stage}:{poller._errors}")
    now_ns = time.monotonic_ns()
    samples = {side: poller.latest(side) for side in ("left", "right")}
    compact = {}
    for side, sample in samples.items():
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


def request_ordinal(args: argparse.Namespace) -> int:
    stage_a = cache_root() / "stage-a-summary.json"
    stage_b = cache_root() / "stage-b-summary.json"
    if args.mode == "no-motion":
        if stage_a.exists() or stage_b.exists():
            raise MotionRejected("stage_a_or_b_cache_already_exists_no_request_reuse")
        return 1
    if args.mode == "staged-one-batch":
        if not stage_a.is_file() or stage_b.exists():
            raise MotionRejected("physical_stage_requires_unique_completed_stage_a_and_absent_stage_b")
        return 2
    raise MotionRejected("request_ordinal_not_available_for_mode")


def prior_attempts(mode: str) -> list[dict[str, Any]]:
    path = cache_root() / "attempt-ledger.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if isinstance(value, dict) and value.get("mode") == mode and value.get("event") == "attempt_terminal":
            records.append(value)
    return records


def resolved_rejected_rollouts() -> set[str]:
    path = cache_root() / "request-outcome-resolutions.jsonl"
    if not path.exists():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if isinstance(value, dict) and value.get("outcome") == "rejected_no_inference":
            result.add(str(value.get("rollout_id", "")))
    return result


def query_request_outcome(request_id: str) -> dict[str, Any]:
    require_uuid4(request_id, "request_outcome_id")
    connection = http.client.HTTPConnection("127.0.0.1", CLIENT_PORT, timeout=5.0)
    try:
        connection.request("GET", f"/request-outcome/{request_id}")
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise MotionRejected(f"request_outcome_query_failed:{request_id}:{response.status}")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("request_id") != request_id:
        raise MotionRejected("request_outcome_query_identity_mismatch")
    return payload


def record_request_outcome_resolution(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("known") is not True:
        raise MotionRejected("request_outcome_remains_unknown")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != 1 or not isinstance(events[0], Mapping):
        raise MotionRejected("request_outcome_resolution_requires_one_event")
    event = events[0]
    if event.get("outcome") != "rejected_no_inference" or event.get("event_type") != "request_rejection":
        raise MotionRejected("request_outcome_not_rejected_no_inference")
    context = event.get("context")
    if not isinstance(context, Mapping) or context.get("request_id") != payload.get("request_id"):
        raise MotionRejected("request_outcome_context_mismatch")
    record = {
        "request_id": payload["request_id"],
        "rollout_id": context.get("rollout_id"),
        "request_ordinal": context.get("request_ordinal"),
        "execution_intent": context.get("execution_intent"),
        "outcome": "rejected_no_inference",
        "reason": str(event.get("detail", {}).get("reason", "")),
        "event_digest": _digest_payload(event),
        "recorded_at": utc_now(),
    }
    existing = resolved_rejected_rollouts()
    if str(record["rollout_id"]) not in existing:
        _append_jsonl(cache_root() / "request-outcome-resolutions.jsonl", record)
    return record


def validate_attempt_boundary(args: argparse.Namespace) -> int:
    attempts = prior_attempts(args.mode)
    resolved = resolved_rejected_rollouts()
    if len(attempts) >= args.max_attempts:
        raise MotionRejected(f"attempt_budget_exhausted:{args.mode}:{len(attempts)}")
    for record in attempts:
        counters = record.get("events", {}).get("event_counters", {})
        if int(counters.get("policy.request_sending", 0)) > 0 and str(record.get("rollout_id", "")) not in resolved:
            raise MotionRejected("prior_policy_request_may_have_been_accepted_no_retry")
        if int(counters.get("model.command_submitting", 0)) > 0:
            raise MotionRejected("prior_robot_command_may_have_been_accepted_no_retry")
    return len(attempts) + 1


def select_rollout_id(args: argparse.Namespace) -> str:
    if args.mode == "staged-one-batch":
        expected = str(_read_json(cache_root() / "stage-a-summary.json").get("rollout_id", ""))
        require_uuid4(expected, "stage_a_rollout_id")
        if args.rollout_id not in ("auto", expected):
            raise MotionRejected("physical_stage_must_reuse_stage_a_rollout_id")
        return expected
    value = str(uuid.uuid4()) if args.rollout_id == "auto" else args.rollout_id
    return require_uuid4(value, "rollout_id")


def send_compact_request(
    *, args: argparse.Namespace, intent: str, ordinal: int, rig: RealSenseRig,
    backend: RR006Backend, events: CompactEventSink,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    policy, metadata, connection = open_verified_policy(args)
    events.emit("policy", "connection_verified", metadata)
    try:
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
        require_uuid4(request_id, "request_id")
        context = {
            "schema_id": SERVICE_SCHEMA_ID,
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "rollout_id": events.rollout_id,
            "service_session_id": SERVICE_SESSION_ID,
            "request_id": request_id,
            "request_ordinal": ordinal,
            "execution_intent": intent,
            "client_send_utc": utc_now(),
            "client_send_monotonic_ns": send_ns,
            "observation_capture_monotonic_ns": capture_ns,
        }
        observation = {**images, "state": state, "prompt": PROMPT, "__rr_context__": context}
        events.emit("policy", "request_sending", {"request_id": request_id, "intent": intent})
        started_ns = time.monotonic_ns()
        try:
            response = policy.infer(observation)
        except Exception as exc:
            try:
                resolution = record_request_outcome_resolution(query_request_outcome(request_id))
            except Exception as outcome_exc:
                events.emit("safety", "policy_request_outcome_unknown", {"error": repr(exc)})
                raise MotionRejected(f"policy_request_outcome_unknown:{request_id}") from outcome_exc
            events.emit("policy", "request_rejected_no_inference_resolved", resolution)
            raise MotionRejected(f"policy_request_rejected_no_inference:{request_id}") from exc
        received_ns = time.monotonic_ns()
    finally:
        if hasattr(policy, "close"):
            policy.close()
    if not isinstance(response, Mapping) or response.get("__rr_context__") != context:
        raise MotionRejected("policy_response_context_mismatch")
    actions = np.asarray(response.get("actions"))
    canonical = map_action28_to_action16(actions)
    summary = {
        "status": "pass",
        "context": context,
        "connection": connection,
        "service": metadata,
        "source_shape": [50, 28],
        "source_dtype": "float32",
        "source_finite": True,
        "source_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
        "canonical_shape": [50, 16],
        "canonical_dtype": "float32",
        "canonical_finite": True,
        "canonical_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "mapping_indices": MAPPING_INDICES,
        "camera_capture_digest": _digest_payload({
            role: {"frame_number": item.get("frame_number"), "receive_monotonic_ns": item.get("receive_monotonic_ns")}
            for role, item in camera_metadata.items()
        }),
        "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "capture_state_digest": _digest_payload(state_metadata),
        "observation_age_sec": age_sec,
        "client_round_trip_ms": (received_ns - started_ns) / 1_000_000.0,
        "client_receive_monotonic_ns": received_ns,
        "server_timing": json_safe(response.get("server_timing")),
        "raw_or_canonical_payload_written": False,
        "cross_host_timestamp_subtraction_used": False,
    }
    events.emit("policy", "response_validated", {"request_id": request_id, "intent": intent})
    return canonical, summary


def prompt_exact(
    events: CompactEventSink,
    text: str,
    *,
    expected: str,
    field: str,
    safety_check: Any | None = None,
) -> None:
    if safety_check is None:
        value = input(text).strip()
    else:
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
    events.emit("operator", "operator_input", {"field": field, "value": value})
    if value != expected:
        raise MotionRejected(f"operator_input_mismatch:{field}")


def create_backend(args: argparse.Namespace, events: CompactEventSink) -> RR006Backend:
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
                "ip": args.left_robot_ip, "port": args.robot_port,
                "local_ip": "192.168.1.100", "udp_port": 8090,
            },
            "right": {
                "ip": args.right_robot_ip, "port": args.robot_port,
                "local_ip": "192.168.1.101", "udp_port": 8089,
            },
        },
        "sources": {"deltawam_src": str(Path.home() / "deltawam" / "src")},
    }
    return RR006Backend(
        config, events,
        retries=args.controller_query_retries,
        backoff_sec=args.controller_query_backoff_sec,
    )


def open_live_stack(args: argparse.Namespace, events: CompactEventSink):
    conflicts = other_robot_command_processes()
    if conflicts:
        raise MotionRejected(f"other_robot_command_source:{conflicts}")
    backend = create_backend(args, events)
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
        return backend, rig, poller, gate, {
            "camera_profiles": {
                role: {
                    "serial": profile["serial"], "width": profile["width"],
                    "height": profile["height"], "fps": profile["fps"], "format": profile["format"],
                }
                for role, profile in rig.profiles.items()
            },
            "model_images": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in images.items()},
            "state": {"shape": [28], "dtype": "float32", "finite": True, "order": "left_first"},
            "capture_state_status": udp.get("status"),
        }
    except Exception:
        if poller is not None:
            poller.close()
        if rig is not None:
            rig.close()
        backend.close()
        raise


def close_live_stack(backend: RR006Backend, rig: RealSenseRig, poller: RobotStatePoller) -> None:
    failure: Exception | None = None
    for close in (rig.close, poller.close, backend.close):
        try:
            close()
        except Exception as exc:
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure


def run_stage_a(args: argparse.Namespace, events: CompactEventSink) -> dict[str, Any]:
    ordinal = request_ordinal(args)
    verify_existing_tunnel(args)
    backend, rig, poller, gate, sensing = open_live_stack(args, events)
    result: dict[str, Any]
    try:
        canonical, response = send_compact_request(
            args=args, intent="no_motion_validation", ordinal=ordinal,
            rig=rig, backend=backend, events=events,
        )
        post = require_fresh_safe(poller, rig, args, "after_no_motion_response")
        result = {
            "schema_version": 1,
            "status": "pass",
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "rollout_id": events.rollout_id,
            "attempt": 1,
            "record_mode": "off",
            "recorders_writers_encoders": 0,
            "policy_connections_requests_responses": [1, 1, 1],
            "joint_gripper_commands": [0, 0],
            "physical_batches": 0,
            "response": response,
            "canonical_shape": list(canonical.shape),
            "sensing": sensing,
            "post_response_fresh_feedback": post,
            "robot_limit_digest": _digest_payload({
                side: {"lower": gate["arms"][side]["lower_rad"], "upper": gate["arms"][side]["upper_rad"]}
                for side in ("left", "right")
            }),
            "controller_query_retry": backend.retry_summary(),
            "events": events.compact(),
            "created_at": utc_now(),
        }
    finally:
        close_live_stack(backend, rig, poller)
    _write_new_json(cache_root() / "stage-a-summary.json", result)
    return result


def compact_command(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "command_id": result["command_id"],
        "side": result["side"],
        "kind": result["kind"],
        "sdk_status": result["sdk_status"],
        "target_error_rad": result["target_error_rad"],
        "observed_direction": result.get("observed_direction"),
    }


def run_stage_b(args: argparse.Namespace, events: CompactEventSink) -> dict[str, Any]:
    ordinal = request_ordinal(args)
    stage_a = _read_json(cache_root() / "stage-a-summary.json")
    if stage_a.get("status") != "pass" or stage_a.get("policy_connections_requests_responses") != [1, 1, 1]:
        raise MotionRejected("stage_a_pass_and_exact_counts_required")
    if stage_a.get("joint_gripper_commands") != [0, 0] or stage_a.get("physical_batches") != 0:
        raise MotionRejected("stage_a_zero_command_boundary_violated")
    verify_existing_tunnel(args)
    backend, rig, poller, gate, sensing = open_live_stack(args, events)
    executor: JointCommandExecutor005 | None = None
    result: dict[str, Any]
    try:
        limits = {
            side: {"lower_rad": gate["arms"][side]["lower_rad"], "upper_rad": gate["arms"][side]["upper_rad"]}
            for side in ("left", "right")
        }
        display = {
            "task_id": TASK_ID,
            "projected_joint_delta_limit_rad": args.joint_delta_limit_rad,
            "speed_percent_max": args.speed_percent,
            "target_and_return_tolerance_rad": args.target_tolerance_rad,
            "source_verified_absolute_limits": limits,
            "return_plan": "right_then_left_projected_row_0_then_right_then_left_exact_pre_command_return",
            "settling": {
                "grace_sec": args.settle_grace_sec,
                "stable_window_sec": args.stable_window_sec,
                "stable_span_threshold_rad": args.stable_span_threshold_rad,
                "timeout_sec": args.settle_timeout_sec,
            },
            "gripper": "hold_current_zero_commands",
            "required_phrase": OPERATOR_PHRASE,
        }
        print("RR-DEP-006 STAGE A PASS; PHYSICAL LIMITS AND RETURN PLAN", flush=True)
        print(json.dumps(display, indent=2, ensure_ascii=False), flush=True)
        prompt_exact(
            events,
            "MOTION AUTHORIZATION: ",
            expected=OPERATOR_PHRASE,
            field="exact_motion_phrase",
            safety_check=lambda: require_fresh_safe(poller, rig, args, "awaiting_motion_authorization"),
        )
        require_fresh_safe(poller, rig, args, "before_physical_request")
        canonical, response = send_compact_request(
            args=args, intent="one_projected_physical_batch", ordinal=ordinal,
            rig=rig, backend=backend, events=events,
        )
        require_fresh_safe(poller, rig, args, "after_physical_response")
        pre_samples = {side: poller.latest(side) for side in ("left", "right")}
        if any(sample is None for sample in pre_samples.values()):
            raise MotionRejected("fresh_pre_command_state_missing")
        pre_model = {side: list(pre_samples[side].joints_rad) for side in ("left", "right")}
        projected = project_model_row(
            canonical[0], pre_model["left"], pre_model["right"],
            {side: limits[side]["lower_rad"] for side in ("left", "right")},
            {side: limits[side]["upper_rad"] for side in ("left", "right")},
            delta_max_rad=args.joint_delta_limit_rad,
        )
        if (time.monotonic_ns() - int(response["client_receive_monotonic_ns"])) / 1e9 > args.response_to_command_sec:
            raise MotionRejected("response_to_first_command_timeout")
        executor = JointCommandExecutor005(backend, poller, rig, events, args, pre_model)
        command_results = []
        for side in ("right", "left"):
            command_results.append(executor.execute(
                side=side, kind="projected_model_row_0",
                target=projected["projected_joint_targets"][side],
                lower=limits[side]["lower_rad"], upper=limits[side]["upper_rad"],
            ))
        if executor.count != 2:
            raise MotionRejected("model_joint_command_count_mismatch")
        for side in ("right", "left"):
            command_results.append(executor.execute(
                side=side, kind="return_to_pre_command",
                target=pre_model[side], lower=limits[side]["lower_rad"], upper=limits[side]["upper_rad"],
            ))
        if executor.count != 4:
            raise MotionRejected("total_joint_command_count_mismatch")
        settling = settle_after_return(
            args=args, rig=rig, poller=poller, pre_model=pre_model,
            return_accepted_ns=int(command_results[-1]["accepted_monotonic_ns"]), events=events,
        )
        print("RR-DEP-006 RETURN AND STABLE WINDOW PASS", flush=True)
        print(json.dumps({"settling": settling, "required_ack": RETURN_ACK_PHRASE}, indent=2), flush=True)

        def monitor_return_wait() -> None:
            require_fresh_safe(poller, rig, args, "awaiting_return_acknowledgement")
            samples = {side: poller.latest(side) for side in ("left", "right")}
            for side, sample in samples.items():
                if sample is None:
                    raise MotionRejected(f"return_feedback_missing_while_awaiting_ack:{side}")
                error = max(abs(actual - target) for actual, target in zip(sample.joints_rad, pre_model[side]))
                if error > args.target_tolerance_rad:
                    raise MotionRejected(f"return_error_while_awaiting_ack:{side}:{error:.9f}")

        prompt_exact(
            events,
            "RETURN ACKNOWLEDGEMENT: ",
            expected=RETURN_ACK_PHRASE,
            field="exact_return_ack_phrase",
            safety_check=monitor_return_wait,
        )
        compact_commands = [compact_command(item) for item in command_results]
        result = {
            "schema_version": 1,
            "status": "pass",
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "rollout_id": events.rollout_id,
            "record_mode": "off",
            "recorders_writers_encoders": 0,
            "policy_connections_requests_responses": [1, 1, 1],
            "joint_commands": 4,
            "model_joint_commands": 2,
            "return_joint_commands": 2,
            "gripper_commands": 0,
            "physical_batches": 1,
            "response": response,
            "projected_row_digest": _digest_payload(projected),
            "projection_max_abs_delta_rad": max(
                abs(projected["projected_joint_targets"][side][index] - pre_model[side][index])
                for side in ("left", "right") for index in range(7)
            ),
            "commands": compact_commands,
            "settling": settling,
            "operator_authorization": events.operator_facts[0] if events.operator_facts else None,
            "return_acknowledgement": events.operator_facts[-1] if len(events.operator_facts) >= 2 else None,
            "sensing": sensing,
            "controller_query_retry": backend.retry_summary(),
            "events": events.compact(),
            "created_at": utc_now(),
        }
    except Exception:
        if executor is not None and executor.active_side is not None:
            try:
                backend.slow_stop(executor.active_side)
            except Exception as stop_exc:
                events.emit("safety", "slow_stop_failure", {"error": repr(stop_exc)})
        raise
    finally:
        close_live_stack(backend, rig, poller)
    _write_new_json(cache_root() / "stage-b-summary.json", result)
    return result


def run_dry(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    root = args.artifact_root
    root.mkdir(parents=True, exist_ok=False)
    help_path = root / "client-help.txt"
    help_path.write_text(parser.format_help(), encoding="utf-8")
    result = {
        "schema_version": 1,
        "status": "pass",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "mode": "dry-run",
        "record_mode": "off",
        "network_operations": 0,
        "device_operations": 0,
        "sdk_calls": 0,
        "joint_commands": 0,
        "gripper_commands": 0,
        "physical_batches": 0,
        "video_writers_encoders_recorders": 0,
        "raw_action_or_full_trajectory_writes": 0,
        "validated_command": command_array(args),
        "operator_phrase_flag_semantics": "expected_interactive_match_only_not_authorization",
        "return_ack_phrase_flag_semantics": "expected_interactive_match_only_not_acknowledgement",
        "created_at": utc_now(),
    }
    _write_new_json(root / "client-dry-run.json", result)
    return result


def fetch_compute_receipt(args: argparse.Namespace) -> dict[str, Any]:
    remote_root = "/share/jhk/phywam-mope-real/runs/RR-DEP-006/RR-DEP-006-RUN-001/compute"
    receipt_path = f"{remote_root}/compute-receipt.yaml"
    digest_path = f"{remote_root}/compute-receipt.sha256"
    common = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.ssh_alias]
    receipt_run = subprocess.run(common + ["cat", receipt_path], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest_run = subprocess.run(common + ["cat", digest_path], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if receipt_run.returncode != 0 or digest_run.returncode != 0:
        raise MotionRejected("compute_receipt_not_available_do_not_finalize_service_from_Client")
    raw = receipt_run.stdout.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    expected = digest_run.stdout.strip().split()[0] if digest_run.stdout.strip() else ""
    if digest != expected:
        raise MotionRejected("compute_receipt_digest_mismatch")
    receipt = json.loads(receipt_run.stdout)
    identity = receipt.get("identity", {})
    if identity.get("task_id") != TASK_ID or identity.get("run_id") != RUN_ID:
        raise MotionRejected("compute_receipt_task_or_run_mismatch")
    if identity.get("service_session_id") != SERVICE_SESSION_ID:
        raise MotionRejected("compute_receipt_session_mismatch")
    return {"sha256": digest, "receipt": receipt}


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = args.artifact_root
    if sorted(path.name for path in root.iterdir() if path.is_file()) != ["client-dry-run.json", "client-help.txt"]:
        raise MotionRejected("final_artifact_root_must_contain_only_help_and_dry_run_before_finalize")
    stage_a = _read_json(cache_root() / "stage-a-summary.json")
    stage_b = _read_json(cache_root() / "stage-b-summary.json")
    if stage_a.get("status") != "pass" or stage_b.get("status") != "pass":
        raise MotionRejected("both_live_stages_must_pass_before_finalize")
    compute = fetch_compute_receipt(args)
    compute_counts = compute["receipt"].get("request_inference_response_accounting", {})
    if compute_counts.get("requests_by_intent", {}).get("no_motion_validation") != 1:
        raise MotionRejected("compute_no_motion_request_count_mismatch")
    if compute_counts.get("requests_by_intent", {}).get("one_projected_physical_batch") != 1:
        raise MotionRejected("compute_physical_request_count_mismatch")
    runtime = Path(__file__).resolve().parents[2]
    runtime_commit = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    runtime_dirty = bool(subprocess.check_output(["git", "-C", str(runtime), "status", "--porcelain"], text=True).strip())
    if runtime_dirty:
        raise MotionRejected("runtime_must_be_clean_before_terminal_freeze")
    execution = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "execution_status": "completed",
        "record_mode": "off",
        "service_session_id": SERVICE_SESSION_ID,
        "service_runtime_base_commit": SERVICE_RUNTIME_BASE_COMMIT,
        "service_runtime_commit": SERVICE_RUNTIME_COMMIT,
        "client_runtime_commit": runtime_commit,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "total_policy_connections_requests_responses": [2, 2, 2],
        "joint_commands": 4,
        "gripper_commands": 0,
        "physical_model_batches": 1,
        "recorders_writers_encoders": 0,
        "raw_action_and_full_trajectory_artifacts": 0,
        "compute_receipt_sha256": compute["sha256"],
        "continuation_decision": "return_to_mac",
        "completed_at": utc_now(),
    }
    write_yaml(root / "execution-summary.yaml", execution)
    receipt = {
        "schema_version": 3,
        "status": "terminal",
        "identity": {
            "task_id": TASK_ID, "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
            "endpoint_role": "robot_client_executor_and_integrator", "risk_profile": "robot_motion",
        },
        "source_chain": {
            "runtime_start_commit": "5f080583d67430a426a4828badc079e9af4d0062",
            "runtime_final_commit": runtime_commit,
            "runtime_final_clean": True,
            "compute_runtime_base_commit": SERVICE_RUNTIME_BASE_COMMIT,
            "compute_runtime_final_commit": SERVICE_RUNTIME_COMMIT,
            "compute_receipt_sha256": compute["sha256"],
        },
        "execution": {
            "execution_status": "completed", "provenance_status": "complete",
            "record_mode": "off", "stage_a": "pass", "physical_batch": "pass",
            "return_and_settling": "pass", "continuation_decision": "return_to_mac",
        },
        "counters": {
            "Client_policy_connections_requests_responses": [2, 2, 2],
            "joint_commands": 4, "model_joint_commands": 2, "return_joint_commands": 2,
            "gripper_commands": 0, "physical_model_batches": 1,
            "video_writers_encoders_recorders": 0, "raw_action_or_full_trajectory_files": 0,
        },
        "status_dimensions": {
            "execution_status": "completed", "provenance_status": "complete", "gate_result": "not_run",
            "review_status": "not_requested", "deployment_readiness": "not_ready",
            "safety_readiness": "not_ready", "claim_status": "unsupported",
            "continuation_decision": "return_to_mac",
        },
        "claim_boundary": "one bounded no-recording live inference and projected row-0 batch only",
    }
    write_yaml(root / "client-receipt.yaml", receipt)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "artifact_root": str(root),
        "required_files": list(FINAL_ARTIFACTS),
        "file_count": 6,
        "record_mode": "off",
        "forbidden_artifacts_present": False,
        "runtime_commit": runtime_commit,
        "compute_receipt_sha256": compute["sha256"],
    }
    write_yaml(root / "run-manifest.yaml", manifest)
    existing = sorted(path.name for path in root.iterdir() if path.is_file())
    expected_before_ledger = sorted(name for name in FINAL_ARTIFACTS if name != "SHA256SUMS")
    if existing != expected_before_ledger:
        raise MotionRejected(f"final_artifact_allowlist_mismatch_before_ledger:{existing}")
    ledger = "\n".join(f"{sha256_file(root / name)}  {name}" for name in expected_before_ledger) + "\n"
    with (root / "SHA256SUMS").open("x", encoding="utf-8") as stream:
        stream.write(ledger)
        stream.flush()
        os.fsync(stream.fileno())
    if sorted(path.name for path in root.iterdir() if path.is_file()) != sorted(FINAL_ARTIFACTS):
        raise MotionRejected("terminal_artifact_allowlist_mismatch")
    return {"status": "pass", "artifact_root": str(root), "files": list(FINAL_ARTIFACTS)}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.artifact_root = args.artifact_root.expanduser().resolve()
    events: CompactEventSink | None = None
    attempt_number: int | None = None
    try:
        validate_args(args)
        cache_root().mkdir(parents=True, exist_ok=True)
        if args.mode == "dry-run":
            result = run_dry(args, parser)
        elif args.mode == "finalize":
            result = run_finalize(args)
        else:
            attempt_number = validate_attempt_boundary(args)
            rollout_id = select_rollout_id(args)
            events = CompactEventSink(rollout_id)
            _append_jsonl(cache_root() / "attempt-ledger.jsonl", {
                "event": "attempt_started", "mode": args.mode, "attempt": attempt_number,
                "rollout_id": rollout_id, "started_at": utc_now(),
            })
            result = run_stage_a(args, events) if args.mode == "no-motion" else run_stage_b(args, events)
            _append_jsonl(cache_root() / "attempt-ledger.jsonl", {
                "event": "attempt_terminal", "status": "pass", "mode": args.mode,
                "attempt": attempt_number, "rollout_id": rollout_id,
                "events": events.compact(), "ended_at": utc_now(),
            })
        print(json.dumps(json_safe(result), indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = {"status": "failed", "mode": args.mode, "error": repr(exc), "captured_at": utc_now()}
        if events is not None and attempt_number is not None:
            _append_jsonl(cache_root() / "attempt-ledger.jsonl", {
                "event": "attempt_terminal", "status": "failed", "mode": args.mode,
                "attempt": attempt_number, "rollout_id": events.rollout_id,
                "error": repr(exc), "events": events.compact(), "ended_at": utc_now(),
            })
        print(json.dumps(failure, indent=2, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
