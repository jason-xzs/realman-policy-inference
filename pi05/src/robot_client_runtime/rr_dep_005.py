"""Stable RR-DEP-005 live pi0.5 Client CLI.

The CLI has three explicit modes.  ``dry-run`` never imports the camera,
policy, or robot SDK paths.  ``no-motion`` opens only live sensing/state and
performs one policy response validation.  ``staged-one-batch`` repeats that
zero-command validation, asks the onsite operator for interactive approval,
uses a fresh physical-intent request, executes one projected row, returns, and
requires a settling-aware visual acknowledgement.
"""

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
import socket
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
import uuid

from .rr_dep_003 import (
    CameraRecorder,
    RealManBackend,
    RealSenseRig,
    RobotStatePoller,
    json_safe,
    resize_with_pad_rgb,
    sha256_file,
    verify_frozen_sources,
    write_yaml,
)
from .rr_dep_003_motion import MotionRejected
from .rr_dep_004 import (
    RR004Backend,
    evaluate_robot_gate_004,
    map_action28_to_action16,
    project_model_row,
    require_uuid4,
)


TASK_ID = "RR-DEP-005"
EXPERIMENT_ID = TASK_ID
RUN_ID = "RR-DEP-005-RUN-001"
SCHEMA_ID = "rr_dep_openpi_robot_live_v2"
SERVICE_SESSION_ID = "ff87572f-c35e-41d6-93bf-5d4741949233"
COMPUTE_RUNTIME_COMMIT = "d438df89fe8f46787c342e6e48963643690cf484"
PROMPT = "Place the three objects into the blue container."
OPERATOR_PHRASE = "AUTHORIZE RR-DEP-005 ONE PROJECTED MODEL STEP"
RETURN_ACK_PHRASE = "ACK RR-DEP-005 RETURN SAFE"
CAMERA_TO_OBSERVATION = {
    "head": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}
TUNNEL_SESSION = "rr-dep-005-client-tunnel"
CLIENT_PORT_FIRST = 18200
CLIENT_PORT_LAST = 18209


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(dict(payload)), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def versioned_path(root: Path, name: str, label: str) -> Path:
    path = root / name
    if not path.exists():
        return path
    source = Path(name)
    for ordinal in range(1, 11):
        candidate = root / f"{source.stem}-{label}-attempt-{ordinal:02d}{source.suffix}"
        if not candidate.exists():
            return candidate
    raise MotionRejected(f"artifact_version_budget_exhausted:{name}:{label}")


class EventBundle005:
    FILES = {
        "attempt": "attempt-ledger.jsonl",
        "operator": "operator-authorizations.jsonl",
        "policy": "policy-request-ledger.jsonl",
        "response": "response-summaries.jsonl",
        "canonical": "canonical-actions.jsonl",
        "rejection": "action-rejections.jsonl",
        "model": "model-command-ledger.jsonl",
        "state": "robot-state-trajectory.jsonl",
        "settling": "settling-window.jsonl",
        "safety": "safety-events.jsonl",
        "camera": "camera-frames.jsonl",
        "ack": "acknowledgements.jsonl",
    }

    def __init__(self, root: Path, rollout_id: str) -> None:
        self.root = root
        self.rollout_id = rollout_id
        self._locks = {name: threading.Lock() for name in self.FILES}
        self._streams = {
            name: (root / relative).open("a", encoding="utf-8", buffering=1)
            for name, relative in self.FILES.items()
        }

    def emit(self, name: str, event: str, payload: Mapping[str, Any]) -> None:
        record = {
            "task_id": TASK_ID,
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "rollout_id": self.rollout_id,
            "event": event,
            "utc": utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            **json_safe(dict(payload)),
        }
        with self._locks[name]:
            self._streams[name].write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def close(self) -> None:
        for stream in self._streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()


class RR005Backend(RR004Backend):
    """RR-DEP-004 backend with readback-verified task-owned state restore."""

    def close_capture_state(self) -> None:
        failures = []
        for side in reversed(("left", "right")):
            if side in self.udp_configured:
                try:
                    prior = self._udp_config_from_dict(self.prior_udp_config[side])
                    status = int(self.arms[side].rm_set_realtime_push(prior))
                    read_code, readback = self._call(side, "rm_get_realtime_push")
                    readback_matches = int(read_code) == 0 and json_safe(readback) == self.prior_udp_config[side]
                    self.events.emit("ack", "capture_udp_config_restore_verified", {
                        "side": side, "sdk_status": status, "readback_code": read_code,
                        "readback_matches_exact_prior": readback_matches,
                        "known_status_minus_2_resolved_by_readback": status == -2 and readback_matches,
                    })
                    if not readback_matches:
                        failures.append(f"capture_udp_restore_mismatch:{side}:{status}:{read_code}")
                except Exception as exc:
                    failures.append(f"capture_udp_restore_unknown:{side}:{exc!r}")
            if side in self.rm_plus_changed:
                try:
                    response = self._set_rm_plus_json(side, self.prior_rm_plus_mode[side])
                    mode_code, mode = self._call(side, "rm_get_rm_plus_mode")
                    restored = int(mode_code) == 0 and int(mode) == self.prior_rm_plus_mode[side]
                    self.events.emit("ack", "rm_plus_mode_restore_verified", {
                        "side": side, "mode": self.prior_rm_plus_mode[side], "response": response,
                        "readback_code": mode_code, "readback_mode": mode, "restored": restored,
                    })
                    if not restored:
                        failures.append(f"rm_plus_restore_mismatch:{side}:{mode_code}:{mode}")
                except Exception as exc:
                    failures.append(f"rm_plus_restore_unknown:{side}:{exc!r}")
        for receiver in self.udp_receivers.values():
            receiver.close()
        self.udp_receivers.clear()
        self.udp_configured.clear()
        self.rm_plus_changed.clear()
        if failures:
            for failure in failures:
                self.events.emit("safety", "task_owned_state_restore_failure", {"error": failure})
            raise MotionRejected(";".join(failures))

    def close(self) -> None:
        failure = None
        try:
            self.close_capture_state()
        except Exception as exc:
            failure = exc
        finally:
            RealManBackend.close(self)
        if failure is not None:
            raise failure


def start_warmed_recording(rig: RealSenseRig, root: Path, label: str, warmup_sec: float = 0.75) -> None:
    """Start all ffmpeg writers before attaching native camera frame producers."""
    recorder = CameraRecorder(
        root,
        label,
        width=int(rig.config["cameras"]["width"]),
        height=int(rig.config["cameras"]["height"]),
        fps=int(rig.config["cameras"]["fps"]),
    )
    time.sleep(warmup_sec)
    with rig._recorder_lock:
        if rig._recorder is not None:
            raise MotionRejected("camera_recorder_already_active")
        rig._recorder = recorder


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


def nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def canonical_uuid4_or_auto(text: str) -> str:
    if text == "auto":
        return text
    try:
        parsed = uuid.UUID(text)
    except (ValueError, TypeError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("must be canonical UUIDv4 or auto") from exc
    if parsed.version != 4 or str(parsed) != text:
        raise argparse.ArgumentTypeError("must be canonical UUIDv4 or auto")
    return text


def build_parser() -> argparse.ArgumentParser:
    runtime_root = Path(__file__).resolve().parents[2]
    client_root = runtime_root.parent
    parser = argparse.ArgumentParser(
        description="Stable RR-DEP-005 no-motion and one projected RealMan batch CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("dry-run", "no-motion", "staged-one-batch"), required=True,
                        help="Explicit side-effect mode; only staged-one-batch contains a motion path")
    parser.add_argument("--server-url", default="auto", help="Loopback WebSocket URL or auto for the selected SSH local port")
    parser.add_argument("--ssh-alias", default="xzs_server", help="Existing OpenSSH alias; credentials are never read")
    parser.add_argument("--ssh-local-port", type=int, default=0, help="0 selects the first free port in 18200..18209")
    parser.add_argument("--compute-port", type=port_number, default=8020)
    parser.add_argument("--service-session-id", type=canonical_uuid4_or_auto, default=SERVICE_SESSION_ID)
    parser.add_argument("--compute-runtime-commit", default=COMPUTE_RUNTIME_COMMIT)
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
    parser.add_argument("--left-local-ip", default="192.168.1.100")
    parser.add_argument("--right-local-ip", default="192.168.1.101")
    parser.add_argument("--left-udp-port", type=port_number, default=8090)
    parser.add_argument("--right-udp-port", type=port_number, default=8089)
    parser.add_argument("--deltawam-src", type=Path, default=Path.home() / "deltawam" / "src")
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
    parser.add_argument("--settle-grace-sec", type=nonnegative_float, default=2.0)
    parser.add_argument("--stable-window-sec", type=positive_float, default=2.0)
    parser.add_argument("--stable-span-threshold-rad", type=positive_float, default=0.001)
    parser.add_argument("--settle-timeout-sec", type=positive_float, default=10.0)
    parser.add_argument("--max-attempts", type=positive_int, default=10)
    parser.add_argument("--max-policy-requests", type=positive_int, default=10)
    parser.add_argument("--gripper-mode", choices=("hold-unproven-conversion", "execute-source-verified"),
                        default="hold-unproven-conversion")
    parser.add_argument("--return-to-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--artifact-root", type=Path,
                        default=client_root / "runs" / TASK_ID / RUN_ID / "client")
    parser.add_argument("--record-dir", type=Path,
                        default=client_root / "runs" / TASK_ID / RUN_ID / "client" / "camera-video")
    parser.add_argument("--source-manifest", type=Path,
                        default=client_root / "runs" / TASK_ID / RUN_ID / "client" / "source-manifest.tsv")
    parser.add_argument("--operator-phrase", default=OPERATOR_PHRASE,
                        help="Expected interactive text only; this flag is not authorization")
    parser.add_argument("--return-ack-phrase", default=RETURN_ACK_PHRASE,
                        help="Expected interactive text only; this flag is not acknowledgement")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    expected_root = Path(__file__).resolve().parents[2].parent / "runs" / TASK_ID / RUN_ID / "client"
    if args.artifact_root.resolve() != expected_root.resolve():
        raise MotionRejected("artifact_root_must_match_contract_run_root")
    if args.experiment_id != EXPERIMENT_ID or args.run_id != RUN_ID:
        raise MotionRejected("experiment_or_run_id_mismatch")
    if args.service_session_id == "auto":
        raise MotionRejected("service_session_id_cannot_be_auto_for_live_service")
    if args.compute_runtime_commit != COMPUTE_RUNTIME_COMMIT or len(args.compute_runtime_commit) != 40:
        raise MotionRejected("compute_runtime_commit_mismatch")
    if args.prompt != PROMPT:
        raise MotionRejected("prompt_mismatch")
    if len({args.head_serial, args.left_wrist_serial, args.right_wrist_serial}) != 3:
        raise MotionRejected("camera_serials_must_be_distinct")
    if args.ssh_local_port not in (0, *range(CLIENT_PORT_FIRST, CLIENT_PORT_LAST + 1)):
        raise MotionRejected("ssh_local_port_outside_0_or_18200_18209")
    if args.action_row != 0 or args.state_dim != 28 or args.action_dim != 28:
        raise MotionRejected("live_contract_requires_action_row_0_state_28_action_28")
    if args.image_width != 640 or args.image_height != 480 or args.camera_fps != 30 or args.model_image_size != 224:
        raise MotionRejected("live_contract_requires_640x480_30fps_and_model_size_224")
    if args.joint_delta_limit_rad > 0.01 or args.speed_percent > 5:
        raise MotionRejected("motion_limit_or_speed_exceeds_contract")
    if args.target_tolerance_rad > 0.005 or args.state_stale_sec > 0.25:
        raise MotionRejected("tolerance_or_stale_limit_exceeds_contract")
    if args.response_to_command_sec > 2.0 or args.command_timeout_sec > 5.0:
        raise MotionRejected("response_or_command_timeout_exceeds_contract")
    if args.watchdog_hz < 10.0:
        raise MotionRejected("watchdog_must_be_at_least_10hz")
    if args.settle_grace_sec != 2.0 or args.stable_window_sec != 2.0:
        raise MotionRejected("settling_grace_and_window_must_match_contract")
    if args.stable_span_threshold_rad > 0.001 or args.settle_timeout_sec > 10.0:
        raise MotionRejected("settling_threshold_or_timeout_exceeds_contract")
    if args.max_attempts > 10 or args.max_policy_requests > 10:
        raise MotionRejected("attempt_or_request_budget_exceeds_contract")
    if args.mode == "staged-one-batch" and not args.return_to_start:
        raise MotionRejected("staged_mode_requires_return_to_start")
    if args.mode == "staged-one-batch" and args.gripper_mode == "execute-source-verified":
        raise MotionRejected("gripper_execute_equivalence_not_proven_use_hold")
    for path_name in ("artifact_root", "record_dir", "source_manifest", "deltawam_src"):
        if not getattr(args, path_name).expanduser().is_absolute():
            raise MotionRejected(f"{path_name}_must_be_absolute")


def command_array(args: argparse.Namespace) -> list[str]:
    values = vars(args)
    result = [str(Path(__file__).resolve().parents[2] / "scripts" / "run_pi05_realman.py")]
    for action in build_parser()._actions:
        if not action.option_strings or action.dest == "help":
            continue
        flag = action.option_strings[0]
        value = values[action.dest]
        if isinstance(action, argparse.BooleanOptionalAction):
            result.append(flag if value else action.option_strings[1])
        else:
            result.extend((flag, str(value)))
    return result


def cli_contract(parser: argparse.ArgumentParser) -> dict[str, Any]:
    flags = []
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        type_name = "boolean" if isinstance(action, argparse.BooleanOptionalAction) else (
            getattr(action.type, "__name__", "string") if action.type else "string"
        )
        flags.append({
            "flags": action.option_strings,
            "dest": action.dest,
            "type": type_name,
            "default": json_safe(action.default),
            "allowed": list(action.choices) if action.choices else None,
            "required": bool(action.required),
            "meaning": action.help or "",
        })
    return {
        "schema_version": 1,
        "entrypoint": str(Path(__file__).resolve().parents[2] / "scripts" / "run_pi05_realman.py"),
        "operator_phrase_flag_semantics": "expected_interactive_match_only_not_authorization",
        "return_ack_flag_semantics": "expected_interactive_match_only_not_acknowledgement",
        "flags": flags,
    }


def freeze_source_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    snapshot_root = root / "source-snapshot"
    if manifest_path.exists():
        return verify_frozen_sources(manifest_path)
    rows = []
    for snapshot in sorted(path for path in snapshot_root.rglob("*") if path.is_file()):
        relative = snapshot.relative_to(snapshot_root)
        if relative.parts[:2] == ("home", "robot2"):
            source = Path("/") / relative
        elif relative.parts[0] == "robot-client-runtime":
            source = Path(__file__).resolve().parents[2].parent / relative
        else:
            raise MotionRejected(f"unknown_snapshot_mapping:{relative}")
        if not source.is_file():
            raise MotionRejected(f"snapshot_source_missing:{source}")
        digest = sha256_file(snapshot)
        if sha256_file(source) != digest:
            raise MotionRejected(f"snapshot_source_drift_before_manifest:{source}")
        rows.append((source, snapshot, snapshot.stat().st_size,
                     datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(), digest))
    if not rows:
        raise MotionRejected("source_snapshot_empty")
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write("source_path\tsnapshot_path\tsize_bytes\tmtime_utc\tsha256\n")
        for row in rows:
            stream.write("\t".join(str(value) for value in row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return verify_frozen_sources(manifest_path)


def run_dry(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    root = args.artifact_root.resolve()
    source = freeze_source_manifest(root, args.source_manifest.resolve())
    help_path = versioned_path(root, "client-help.txt", "dry-run")
    help_path.write_text(parser.format_help(), encoding="utf-8")
    contract_path = versioned_path(root, "manual-cli-contract.yaml", "dry-run")
    write_yaml(contract_path, cli_contract(parser))
    result = {
        "schema_version": 1,
        "status": "pass",
        "mode": "dry-run",
        "network_operations": 0,
        "device_operations": 0,
        "sdk_commands": 0,
        "joint_commands": 0,
        "gripper_commands": 0,
        "physical_batches": 0,
        "validated_command": command_array(args),
        "source_snapshot": source,
        "client_help": str(help_path),
        "manual_cli_contract": str(contract_path),
    }
    output_path = versioned_path(root, "client-dry-run.json", "dry-run")
    write_new_json(output_path, result)
    result["client_dry_run"] = str(output_path)
    return result


def build_live_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
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
            "left": {"ip": args.left_robot_ip, "port": args.robot_port,
                     "local_ip": args.left_local_ip, "udp_port": args.left_udp_port},
            "right": {"ip": args.right_robot_ip, "port": args.robot_port,
                      "local_ip": args.right_local_ip, "udp_port": args.right_udp_port},
        },
        "sources": {"deltawam_src": str(args.deltawam_src.resolve())},
    }


def latest_camera_observation(rig: RealSenseRig, args: argparse.Namespace) -> tuple[dict[str, Any], int, dict[str, Any]]:
    import numpy as np

    with rig._condition:
        if not all(role in rig._latest for role in CAMERA_TO_OBSERVATION):
            raise MotionRejected("camera_observation_not_ready")
        samples = {
            role: (rig._latest[role][0].copy(), dict(rig._latest[role][1]))
            for role in CAMERA_TO_OBSERVATION
        }
    result: dict[str, Any] = {}
    metadata = {}
    for role, (image, item) in samples.items():
        if role != "head":
            image = image[::-1, ::-1].copy()
        converted = resize_with_pad_rgb(image, args.model_image_size)
        result[CAMERA_TO_OBSERVATION[role]] = np.ascontiguousarray(converted, dtype=np.uint8)
        metadata[role] = item
    capture_ns = min(int(item["receive_monotonic_ns"]) for item in metadata.values())
    age_sec = (time.monotonic_ns() - capture_ns) / 1e9
    if age_sec < 0 or age_sec > args.state_stale_sec:
        raise MotionRejected(f"camera_observation_stale:{age_sec:.6f}")
    return result, capture_ns, metadata


def other_robot_command_processes() -> list[str]:
    output = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True, stdout=subprocess.PIPE).stdout
    patterns = (
        "realman_deployment.client", "run_realman_fastwam_client", "spacemouse_control_arm",
        "run_rr_dep_003.py", "run_rr_dep_004.py",
    )
    current = os.getpid()
    matches = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and int(fields[0]) != current and any(pattern in fields[1] for pattern in patterns):
            matches.append(line.strip())
    return matches


def select_free_loopback_port(requested: int) -> int:
    candidates = [requested] if requested else list(range(CLIENT_PORT_FIRST, CLIENT_PORT_LAST + 1))
    for port in candidates:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            probe.close()
    raise MotionRejected("no_free_authorized_client_tunnel_port")


def existing_tunnel_port(args: argparse.Namespace) -> int | None:
    present = subprocess.run(
        ["tmux", "has-session", "-t", TUNNEL_SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0
    if not present:
        return None
    command = subprocess.run(
        ["tmux", "list-panes", "-t", TUNNEL_SESSION, "-F", "#{pane_start_command}"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    for port in range(CLIENT_PORT_FIRST, CLIENT_PORT_LAST + 1):
        token = f"127.0.0.1:{port}:127.0.0.1:{args.compute_port}"
        if token in command and args.ssh_alias in command:
            if args.ssh_local_port not in (0, port):
                raise MotionRejected("existing_task_tunnel_port_conflicts_with_cli")
            return port
    raise MotionRejected("existing_rr_dep_005_tunnel_identity_unknown")


def ensure_tunnel(args: argparse.Namespace, events: EventBundle005) -> tuple[int, dict[str, Any]]:
    existing = existing_tunnel_port(args)
    if existing is not None:
        return existing, {"status": "reused_same_task_tunnel", "local_port": existing,
                          "compute_port": args.compute_port, "tmux_session": TUNNEL_SESSION}
    local_port = select_free_loopback_port(args.ssh_local_port)
    command = [
        "tmux", "new-session", "-d", "-s", TUNNEL_SESSION, "exec", "ssh", "-N",
        "-L", f"127.0.0.1:{local_port}:127.0.0.1:{args.compute_port}",
        "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
        args.ssh_alias,
    ]
    subprocess.run(command, check=True)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=0.5):
                result = {"status": "started_task_owned_loopback_only", "local_port": local_port,
                          "compute_port": args.compute_port, "tmux_session": TUNNEL_SESSION,
                          "command": command, "credentials_included": False}
                events.emit("policy", "tunnel_started", result)
                return local_port, result
        except OSError:
            time.sleep(0.1)
    raise MotionRejected("task_tunnel_start_timeout")


def validate_server_url(args: argparse.Namespace, local_port: int) -> None:
    if args.server_url == "auto":
        return
    parsed = urlparse(args.server_url)
    if parsed.scheme not in ("ws", "http") or parsed.hostname != "127.0.0.1" or parsed.port != local_port:
        raise MotionRejected("server_url_must_match_selected_loopback_tunnel")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise MotionRejected("server_url_must_not_contain_path_query_or_fragment")


def validate_service_metadata(metadata: Mapping[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "schema_id": SCHEMA_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "service_session_id": args.service_session_id,
        "stable_runtime_commit": args.compute_runtime_commit,
        "checkpoint_role": "integration_only",
        "checkpoint_selection_status": "not_selected",
        "bind": f"127.0.0.1:{args.compute_port}",
        "port": args.compute_port,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise MotionRejected(f"server_metadata_mismatch:{key}:{metadata.get(key)!r}")
    if metadata.get("schema_id") != SCHEMA_ID:
        raise MotionRejected("service_schema_mismatch")


def next_request_ordinal(root: Path, maximum: int) -> int:
    path = root / "policy-request-ledger.jsonl"
    sent = 0
    ids: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("event") == "request_sending":
                sent += 1
                request_id = record.get("context", {}).get("request_id")
                if not request_id or request_id in ids:
                    raise MotionRejected("duplicate_or_missing_prior_request_id")
                ids.add(request_id)
    ordinal = sent + 1
    if ordinal > maximum:
        raise MotionRejected("policy_request_budget_exhausted")
    return ordinal


def send_live_request(
    *, args: argparse.Namespace, local_port: int, intent: str, rig: RealSenseRig,
    backend: RR005Backend, events: EventBundle005,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    ordinal = next_request_ordinal(args.artifact_root, args.max_policy_requests)
    images, camera_ns, camera_metadata = latest_camera_observation(rig, args)
    state, state_ns, state_metadata = backend.capture_observation_state()
    if state.shape != (args.state_dim,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise MotionRejected("invalid_live_observation_state")
    capture_ns = min(camera_ns, state_ns)
    send_ns = time.monotonic_ns()
    age_sec = (send_ns - capture_ns) / 1e9
    if age_sec < 0 or age_sec > args.state_stale_sec:
        raise MotionRejected(f"combined_observation_stale_before_send:{age_sec:.6f}")
    request_id = str(uuid.uuid4())
    require_uuid4(request_id, "request_id")
    context = {
        "schema_id": SCHEMA_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "rollout_id": events.rollout_id,
        "service_session_id": args.service_session_id,
        "request_id": request_id,
        "request_ordinal": ordinal,
        "execution_intent": intent,
        "client_send_utc": utc_now(),
        "client_send_monotonic_ns": send_ns,
        "observation_capture_monotonic_ns": capture_ns,
    }
    observation = {**images, "state": state, "prompt": args.prompt, "__rr_context__": context}
    module = importlib.import_module("openpi_client.websocket_client_policy")
    policy = module.WebsocketClientPolicy(host="127.0.0.1", port=local_port)
    try:
        metadata = policy.get_server_metadata()
        validate_service_metadata(metadata, args)
        events.emit("policy", "request_sending", {
            "context": context, "camera_metadata": camera_metadata, "capture_state": state_metadata,
            "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        })
        started_ns = time.monotonic_ns()
        try:
            response = policy.infer(observation)
        except Exception as exc:
            events.emit("policy", "request_outcome_unknown_query_compute_ledger_before_retry", {
                "context": context, "error": repr(exc),
            })
            raise MotionRejected(f"policy_request_outcome_unknown:{request_id}") from exc
        received_ns = time.monotonic_ns()
    finally:
        if hasattr(policy, "close"):
            policy.close()
    if not isinstance(response, Mapping) or response.get("__rr_context__") != context:
        events.emit("rejection", "response_context_rejected", {"context": context})
        raise MotionRejected("policy_response_context_mismatch")
    actions = np.asarray(response.get("actions"))
    canonical = map_action28_to_action16(actions)
    raw_dir = args.artifact_root / "raw-actions"
    canonical_dir = args.artifact_root / "canonical-actions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{ordinal:02d}-{intent}-{request_id}.npy"
    canonical_path = canonical_dir / f"{ordinal:02d}-{intent}-{request_id}.npy"
    with raw_path.open("xb") as stream:
        np.save(stream, np.ascontiguousarray(actions), allow_pickle=False)
    with canonical_path.open("xb") as stream:
        np.save(stream, canonical, allow_pickle=False)
    summary = {
        "status": "pass",
        "context": context,
        "source_shape": [50, 28],
        "source_dtype": "float32",
        "source_finite": True,
        "source_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
        "raw_action_file": str(raw_path),
        "raw_action_file_sha256": sha256_file(raw_path),
        "canonical_shape": [50, 16],
        "canonical_file": str(canonical_path),
        "canonical_file_sha256": sha256_file(canonical_path),
        "canonical_row_0": canonical[0].tolist(),
        "server_timing": response.get("server_timing"),
        "rr_result_summary": response.get("rr_result_summary"),
        "client_round_trip_ms": (received_ns - started_ns) / 1_000_000.0,
        "client_receive_monotonic_ns": received_ns,
        "cross_host_timestamp_subtraction_used": False,
    }
    events.emit("policy", "response_validated", {"context": context, "status": "pass"})
    events.emit("response", "response_summary", summary)
    events.emit("canonical", "canonical_action_validated", {
        "context": context, "shape": [50, 16], "dtype": "float32",
        "path": str(canonical_path), "sha256": sha256_file(canonical_path),
        "mapping_indices": list(range(8)) + list(range(14, 22)),
    })
    events.emit("rejection", "validation_rejection_summary", {
        "context": context,
        "wrong_shape": "rejected", "wrong_dtype": "rejected", "nonfinite": "rejected",
        "wrong_context": "rejected", "raw_unprojected_execution": "not_attempted",
    })
    return canonical, summary


def prompt_exact(events: EventBundle005, prompt: str, *, expected: str | None, field: str) -> str:
    value = input(prompt).strip()
    events.emit("operator", "operator_input", {"field": field, "value": value})
    if not value or (expected is not None and value != expected):
        raise MotionRejected(f"operator_input_mismatch:{field}")
    return value


class JointCommandExecutor005:
    def __init__(self, backend: RR005Backend, poller: RobotStatePoller, rig: RealSenseRig,
                 events: EventBundle005, args: argparse.Namespace, pre_model: Mapping[str, Sequence[float]]) -> None:
        self.backend = backend
        self.poller = poller
        self.rig = rig
        self.events = events
        self.args = args
        self.pre_model = {side: tuple(float(v) for v in pre_model[side]) for side in ("left", "right")}
        self.submitted: set[str] = set()
        self.count = 0
        self.active_side: str | None = None

    def execute(self, *, side: str, kind: str, target: Sequence[float], lower: Sequence[float],
                upper: Sequence[float]) -> dict[str, Any]:
        command_id = str(uuid.uuid4())
        require_uuid4(command_id, "command_id")
        if command_id in self.submitted:
            raise MotionRejected("duplicate_command_id")
        values = tuple(float(value) for value in target)
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise MotionRejected(f"invalid_command_target:{side}")
        for index, (value, lo, hi) in enumerate(zip(values, lower, upper), start=1):
            if not float(lo) <= value <= float(hi):
                raise MotionRejected(f"command_out_of_limit:{side}:q{index}")
            if abs(value - self.pre_model[side][index - 1]) > self.args.joint_delta_limit_rad + 1e-12:
                raise MotionRejected(f"command_exceeds_projected_envelope:{side}:q{index}")
        samples = self._fresh_safe_samples("before_command")
        before = samples[side]
        self.submitted.add(command_id)
        self.count += 1
        if self.count > 4:
            raise MotionRejected("model_joint_command_budget_exceeded")
        submitted_ns = time.monotonic_ns()
        self.events.emit("model", "command_submitting", {
            "command_id": command_id, "side": side, "kind": kind,
            "target_rad": values, "speed_percent": self.args.speed_percent,
        })
        self.active_side = side
        try:
            status = self.backend.command_absolute(side, values, self.args.speed_percent)
        except Exception as exc:
            self.backend.slow_stop(side)
            raise MotionRejected(f"command_acceptance_unknown:{command_id}") from exc
        accepted_ns = time.monotonic_ns()
        self.events.emit("model", "sdk_acceptance", {
            "command_id": command_id, "side": side, "sdk_status": status,
            "accepted_monotonic_ns": accepted_ns,
        })
        if status != 0:
            self.backend.slow_stop(side)
            raise MotionRejected(f"command_sdk_rejected:{command_id}:{status}")
        deadline = time.monotonic() + self.args.command_timeout_sec
        while True:
            samples = self._fresh_safe_samples(f"during_command:{command_id}")
            commanded = samples[side]
            for arm, sample in samples.items():
                for index, (actual, initial) in enumerate(zip(sample.joints_rad, self.pre_model[arm]), start=1):
                    if abs(actual - initial) > self.args.joint_delta_limit_rad + 1e-12:
                        self.backend.slow_stop(side)
                        raise MotionRejected(f"unexpected_motion_envelope:{command_id}:{arm}:q{index}")
            error = max(abs(actual - expected) for actual, expected in zip(commanded.joints_rad, values))
            direction_ok = all(
                (target - prior) * (actual - prior) >= -1e-12
                for target, prior, actual in zip(values, before.joints_rad, commanded.joints_rad)
            )
            if error <= self.args.target_tolerance_rad and time.monotonic_ns() - submitted_ns >= 250_000_000:
                if not direction_ok:
                    self.backend.slow_stop(side)
                    raise MotionRejected(f"observed_direction_mismatch:{command_id}")
                self.active_side = None
                result = {
                    "command_id": command_id, "side": side, "kind": kind, "sdk_status": status,
                    "target_error_rad": error, "observed_direction": "pass",
                    "observed_rad": list(commanded.joints_rad),
                    "accepted_monotonic_ns": accepted_ns,
                }
                self.events.emit("model", "observed_target", result)
                return result
            if time.monotonic() >= deadline:
                self.backend.slow_stop(side)
                raise MotionRejected(f"command_timeout:{command_id}")
            time.sleep(0.01)

    def _fresh_safe_samples(self, stage: str) -> dict[str, Any]:
        if not self.rig.healthy(self.args.state_stale_sec):
            if self.active_side:
                self.backend.slow_stop(self.active_side)
            raise MotionRejected(f"camera_loss:{stage}")
        samples = {side: self.poller.latest(side) for side in ("left", "right")}
        now_ns = time.monotonic_ns()
        for side, sample in samples.items():
            if sample is None or now_ns - sample.receive_monotonic_ns > int(self.args.state_stale_sec * 1e9):
                if self.active_side:
                    self.backend.slow_stop(self.active_side)
                raise MotionRejected(f"feedback_loss:{stage}:{side}")
            if sample.sdk_status or sample.arm_errors or sample.controller_error:
                if self.active_side:
                    self.backend.slow_stop(self.active_side)
                raise MotionRejected(f"robot_alarm:{stage}:{side}")
        return samples  # type: ignore[return-value]


def settle_after_return(
    *, args: argparse.Namespace, rig: RealSenseRig, poller: RobotStatePoller,
    pre_model: Mapping[str, Sequence[float]], return_accepted_ns: int, events: EventBundle005,
) -> dict[str, Any]:
    start_ns = return_accepted_ns
    deadline_ns = start_ns + int(args.settle_timeout_sec * 1e9)
    windows = {side: deque() for side in ("left", "right")}
    period = 1.0 / args.watchdog_hz
    while time.monotonic_ns() <= deadline_ns:
        if not rig.healthy(args.state_stale_sec):
            raise MotionRejected("camera_loss_during_settling")
        now_ns = time.monotonic_ns()
        elapsed = (now_ns - start_ns) / 1e9
        samples = {side: poller.latest(side) for side in ("left", "right")}
        for side, sample in samples.items():
            if sample is None or now_ns - sample.receive_monotonic_ns > int(args.state_stale_sec * 1e9):
                raise MotionRejected(f"feedback_loss_during_settling:{side}")
            if sample.sdk_status or sample.arm_errors or sample.controller_error:
                raise MotionRejected(f"alarm_during_settling:{side}")
            return_error = max(abs(actual - target) for actual, target in zip(sample.joints_rad, pre_model[side]))
            if return_error > args.target_tolerance_rad:
                events.emit("settling", "return_error_observed", {
                    "side": side, "elapsed_sec": elapsed, "return_error_rad": return_error,
                    "phase": "grace" if elapsed < args.settle_grace_sec else "window_search",
                })
            if elapsed >= args.settle_grace_sec:
                windows[side].append((sample.receive_monotonic_ns, tuple(sample.joints_rad)))
                cutoff = sample.receive_monotonic_ns - int(args.stable_window_sec * 1e9)
                while len(windows[side]) >= 2 and windows[side][1][0] <= cutoff:
                    windows[side].popleft()
        record = {"elapsed_sec": elapsed, "phase": "grace" if elapsed < args.settle_grace_sec else "window_search"}
        if elapsed >= args.settle_grace_sec:
            spans = {}
            durations = {}
            for side in ("left", "right"):
                window = windows[side]
                durations[side] = (window[-1][0] - window[0][0]) / 1e9 if len(window) > 1 else 0.0
                spans[side] = max(
                    max(item[1][joint] for item in window) - min(item[1][joint] for item in window)
                    for joint in range(7)
                ) if window else math.inf
            record.update({"window_duration_sec": durations, "max_joint_span_rad": spans})
            events.emit("settling", "settling_sample", record)
            if all(durations[side] >= args.stable_window_sec for side in ("left", "right")) and all(
                spans[side] <= args.stable_span_threshold_rad for side in ("left", "right")
            ):
                final_errors = {
                    side: max(abs(actual - target) for actual, target in zip(samples[side].joints_rad, pre_model[side]))
                    for side in ("left", "right")
                }
                if any(error > args.target_tolerance_rad for error in final_errors.values()):
                    raise MotionRejected(f"settled_return_tolerance_failed:{final_errors}")
                result = {
                    "status": "pass", "elapsed_sec": elapsed, "settle_grace_sec": args.settle_grace_sec,
                    "stable_window_sec": args.stable_window_sec,
                    "stable_span_threshold_rad": args.stable_span_threshold_rad,
                    "max_joint_span_rad": spans, "return_error_rad": final_errors,
                }
                events.emit("settling", "stable_window_pass", result)
                return result
        else:
            events.emit("settling", "settling_sample", record)
        time.sleep(period)
    raise MotionRejected("settle_timeout_hard_stop")


def run_live(args: argparse.Namespace, events: EventBundle005) -> dict[str, Any]:
    source_check = verify_frozen_sources(args.source_manifest.resolve())
    conflicts = other_robot_command_processes()
    if conflicts:
        raise MotionRejected(f"other_robot_command_source:{conflicts}")
    config = build_live_config(args)
    backend = RR005Backend(config, events)
    poller: RobotStatePoller | None = None
    rig: RealSenseRig | None = None
    executor: JointCommandExecutor005 | None = None
    recording: dict[str, Any] | None = None
    try:
        rig = RealSenseRig(config, events)
        rig.enumerate_and_start()
        rig.wait_ready()
        record_label = f"{args.mode}-{events.rollout_id}"
        backend.open()
        bindings = {side: backend.gather_binding(side) for side in ("left", "right")}
        poller = RobotStatePoller(backend, events, hz=args.watchdog_hz)
        poller.start()
        stability = poller.wait_stable(seconds=2.0)
        robot_gate = evaluate_robot_gate_004(bindings, stability)
        udp_evidence = backend.start_capture_state()
        state, state_ns, capture_samples = backend.capture_observation_state()
        if time.monotonic_ns() - state_ns > int(args.state_stale_sec * 1e9):
            raise MotionRejected("capture_state_stale_at_gate")
        camera_images, _, _ = latest_camera_observation(rig, args)
        camera_binding = {
            "status": "pass", "binding": rig.binding, "active_profiles": rig.profiles,
            "openpi_outputs": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                               for key, value in camera_images.items()},
            "orientation": {"head": "unchanged", "left_wrist": "rotate_180", "right_wrist": "rotate_180"},
        }
        write_yaml(versioned_path(args.artifact_root, "camera-binding.yaml", args.mode), camera_binding)
        robot_semantics = {
            "status": "pass", "bindings": bindings, "gate": robot_gate, "capture_udp": udp_evidence,
            "capture_state": {"shape": list(state.shape), "dtype": str(state.dtype), "finite": True,
                              "samples": capture_samples},
            "capture_state_order": ["left_q1_to_q7", "left_gripper_open", "left_eef_xyz_euler",
                                    "right_q1_to_q7", "right_gripper_open", "right_eef_xyz_euler"],
            "model_dispatch_order": ["right", "left"],
            "gripper_branch": "hold_unproven_conversion",
            "slow_stop": "rm_set_arm_slow_stop",
            "command_semantics": "rm_movej_absolute_degrees_nonblocking_status_then_live_feedback",
        }
        write_yaml(versioned_path(args.artifact_root, "robot-sdk-semantics.yaml", args.mode), robot_semantics)

        start_warmed_recording(rig, args.record_dir.resolve(), record_label)
        local_port, tunnel = ensure_tunnel(args, events)
        validate_server_url(args, local_port)
        canonical_a, stage_a = send_live_request(
            args=args, local_port=local_port, intent="no_motion_validation", rig=rig, backend=backend, events=events,
        )
        stage_a_result = {
            "status": "pass", "policy_connections_min": 1, "policy_requests": 1, "policy_responses": 1,
            "joint_commands": 0, "gripper_commands": 0, "physical_batches": 0,
            "response": stage_a, "canonical_shape": list(canonical_a.shape),
        }
        events.emit("ack", "stage_a_no_motion_pass", stage_a_result)
        if args.mode == "no-motion":
            recording = rig.stop_recording()
            return {
                "status": "completed_live_no_motion", "rollout_id": events.rollout_id,
                "source_check": source_check, "tunnel": tunnel, "stage_a": stage_a_result,
                "recording": recording, "command": command_array(args),
            }

        gate = {
            "status": "awaiting_operator_exact_phrase_after_stage_a",
            "source_check": source_check, "camera": camera_binding, "robot": robot_gate,
            "stage_a": stage_a_result,
            "selected_limits": {
                side: {"lower_rad": robot_gate["arms"][side]["lower_rad"],
                       "upper_rad": robot_gate["arms"][side]["upper_rad"]}
                for side in ("left", "right")
            },
            "return_plan": "right_then_left_target_then_right_then_left_exact_pre_model_return",
            "settle": {"grace_sec": args.settle_grace_sec, "window_sec": args.stable_window_sec,
                       "span_threshold_rad": args.stable_span_threshold_rad, "timeout_sec": args.settle_timeout_sec},
            "stop_behavior": "operator_stop_or_any_alarm_stale_feedback_unexpected_motion_calls_slow_stop_and_no_new_command",
            "joint_delta_limit_rad": args.joint_delta_limit_rad,
            "speed_percent": args.speed_percent,
            "gripper_branch": "hold_unproven_conversion",
            "required_phrase": args.operator_phrase,
        }
        print("RR-DEP-005 STAGE A PASS; COMPLETE PHYSICAL GATE", flush=True)
        print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
        identity = prompt_exact(events, "OPERATOR IDENTITY: ", expected=None, field="operator_identity")
        role = prompt_exact(events, "OPERATOR ROLE: ", expected=None, field="operator_role")
        for field, text in (
            ("physically_present", "CONFIRM PHYSICALLY PRESENT (YES): "),
            ("stop_estop_reachable", "CONFIRM STOP/E-STOP REACHABLE (YES): "),
            ("workcell_and_both_swept_volumes_clear", "CONFIRM WORKCELL AND BOTH SWEPT VOLUMES CLEAR (YES): "),
            ("wrist_tool_cables_slack", "CONFIRM WRIST/TOOL CABLES SLACK (YES): "),
        ):
            prompt_exact(events, text, expected="YES", field=field)
        prompt_exact(events, "MOTION AUTHORIZATION: ", expected=args.operator_phrase, field="exact_motion_phrase")
        gate["status"] = "pass_task_local_motion_unlock"
        gate["operator"] = {"identity": identity, "role": role, "authorized_at": utc_now()}
        write_yaml(args.artifact_root / "motion-unlock-gate.yaml", gate)

        fresh = {side: poller.latest(side) for side in ("left", "right")}
        now_ns = time.monotonic_ns()
        for side, sample in fresh.items():
            if sample is None or now_ns - sample.receive_monotonic_ns > int(args.state_stale_sec * 1e9):
                raise MotionRejected(f"stale_state_before_physical_request:{side}")
            if sample.sdk_status or sample.arm_errors or sample.controller_error:
                raise MotionRejected(f"alarm_before_physical_request:{side}")
        canonical_b, stage_b = send_live_request(
            args=args, local_port=local_port, intent="one_projected_physical_batch",
            rig=rig, backend=backend, events=events,
        )
        response_received_ns = int(stage_b["client_receive_monotonic_ns"])
        pre_samples = {side: poller.latest(side) for side in ("left", "right")}
        pre_model = {side: list(pre_samples[side].joints_rad) for side in ("left", "right")}
        projected = project_model_row(
            canonical_b[args.action_row], pre_model["left"], pre_model["right"],
            {side: robot_gate["arms"][side]["lower_rad"] for side in ("left", "right")},
            {side: robot_gate["arms"][side]["upper_rad"] for side in ("left", "right")},
            delta_max_rad=args.joint_delta_limit_rad,
        )
        projected.update({
            "request_context": stage_b["context"], "response_row": args.action_row,
            "pre_model_joint_state": pre_model, "speed_percent": args.speed_percent,
            "gripper_branch": "hold_unproven_conversion", "gripper_commands": 0,
        })
        write_new_json(args.artifact_root / "projected-action.json", projected)
        if (time.monotonic_ns() - response_received_ns) / 1e9 > args.response_to_command_sec:
            raise MotionRejected("response_to_first_command_timeout")
        executor = JointCommandExecutor005(backend, poller, rig, events, args, pre_model)
        model_results = []
        for side in ("right", "left"):
            model_results.append(executor.execute(
                side=side, kind="projected_model_row_0", target=projected["projected_joint_targets"][side],
                lower=robot_gate["arms"][side]["lower_rad"], upper=robot_gate["arms"][side]["upper_rad"],
            ))
        for side in ("right", "left"):
            model_results.append(executor.execute(
                side=side, kind="return_to_pre_model", target=pre_model[side],
                lower=robot_gate["arms"][side]["lower_rad"], upper=robot_gate["arms"][side]["upper_rad"],
            ))
        final_return_accepted_ns = int(model_results[-1]["accepted_monotonic_ns"])
        settling = settle_after_return(
            args=args, rig=rig, poller=poller, pre_model=pre_model,
            return_accepted_ns=final_return_accepted_ns, events=events,
        )
        print("RR-DEP-005 RETURN AND STABLE WINDOW PASS", flush=True)
        print(json.dumps({"pre_model_joint_state": pre_model, "settling": settling}, indent=2), flush=True)
        prompt_exact(events, "RETURN ACKNOWLEDGEMENT: ", expected=args.return_ack_phrase, field="exact_return_ack_phrase")
        recording = rig.stop_recording()
        acknowledgement = {
            "status": "pass_complete_bounded_joint_pipeline",
            "rollout_id": events.rollout_id,
            "operator": {"identity": identity, "role": role},
            "no_motion_request": stage_a["context"], "physical_request": stage_b["context"],
            "model_commands": model_results, "model_joint_commands_including_return": executor.count,
            "gripper_branch": "hold_unproven_conversion", "gripper_commands": 0,
            "physical_model_batches": 1, "return": "within_0.005rad_both",
            "settling": settling, "return_ack_phrase": args.return_ack_phrase,
            "recording": recording,
        }
        write_yaml(args.artifact_root / "physical-acknowledgement.yaml", acknowledgement)
        events.emit("ack", "physical_acknowledgement_pass", acknowledgement)
        return {
            "status": "completed_bounded_joint_pipeline", "rollout_id": events.rollout_id,
            "source_check": source_check, "tunnel": tunnel, "stage_a": stage_a_result,
            "stage_b": stage_b, "projected_action": projected,
            "physical_acknowledgement": acknowledgement, "command": command_array(args),
        }
    except Exception as exc:
        if executor is not None and executor.active_side is not None:
            try:
                backend.slow_stop(executor.active_side)
            except Exception as stop_exc:
                events.emit("safety", "slow_stop_failure", {"error": repr(stop_exc)})
        events.emit("safety", "live_run_failed", {
            "mode": args.mode, "error": repr(exc),
            "last_accepted_command": None if executor is None else list(executor.submitted)[-1:] or None,
        })
        raise
    finally:
        if rig is not None:
            if recording is None:
                try:
                    recording = rig.stop_recording()
                except Exception as exc:
                    events.emit("safety", "recording_close_failure", {"error": repr(exc)})
            rig.close()
        if poller is not None:
            poller.close()
        backend.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.record_dir = args.record_dir.expanduser().resolve()
    args.source_manifest = args.source_manifest.expanduser().resolve()
    args.deltawam_src = args.deltawam_src.expanduser().resolve()
    try:
        validate_args(args)
    except Exception as exc:
        print(json.dumps({"status": "rejected", "error": repr(exc)}, sort_keys=True))
        return 2
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    rollout_id = str(uuid.uuid4()) if args.rollout_id == "auto" else args.rollout_id
    require_uuid4(rollout_id, "rollout_id")
    events = EventBundle005(args.artifact_root, rollout_id)
    events.emit("attempt", "attempt_started", {"mode": args.mode, "command": command_array(args)})
    try:
        result = run_dry(args, parser) if args.mode == "dry-run" else run_live(args, events)
        events.emit("attempt", "attempt_completed", {"mode": args.mode, "status": result["status"]})
        result_path = versioned_path(args.artifact_root, f"{args.mode}-result.json", args.mode)
        write_new_json(result_path, result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = {"status": "failed", "mode": args.mode, "error": repr(exc), "captured_at": utc_now()}
        events.emit("attempt", "attempt_failed", failure)
        failure_path = versioned_path(args.artifact_root, f"{args.mode}-result.json", args.mode)
        write_new_json(failure_path, failure)
        print(json.dumps(failure, indent=2, ensure_ascii=False), flush=True)
        return 1
    finally:
        events.close()
