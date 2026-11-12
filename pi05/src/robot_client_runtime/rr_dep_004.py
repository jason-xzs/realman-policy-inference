"""RR-DEP-004 bounded live camera, policy, and dual-RealMan runner.

The live entrypoint is deliberately interactive.  It opens only the approved
cameras and read-only robot/state paths until the complete task-local gate is
printed and the onsite operator enters the exact authorization phrase.
"""

from __future__ import annotations

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
import tomllib
from typing import Any, Mapping, Sequence
import uuid

from .pi05_action_adapter import Pi05ActionAdapter
from .rr_dep_003 import (
    RealManBackend,
    RealSenseRig,
    RobotStatePoller,
    json_safe,
    resize_with_pad_rgb,
    sha256_file,
    verify_frozen_sources,
    write_yaml,
)
from .rr_dep_003_motion import MotionRejected, StateSample, select_q7_direction


TASK_ID = "RR-DEP-004"
EXPERIMENT_ID = TASK_ID
RUN_ID = "RR-DEP-004-RUN-001"
SCHEMA_ID = "rr_dep_openpi_robot_live_v1"
PROMPT = "Place the three objects into the blue container."
OPERATOR_PHRASE = "AUTHORIZE RR-DEP-004 MICRO AND ONE MODEL STEP"
SERVICE_SESSION_ID = "495c2511-265c-4bec-b814-891d9468d1b7"
ACTION_28_TO_16 = tuple(range(8)) + tuple(range(14, 22))
CAMERA_TO_OBSERVATION = {
    "head": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class EventBundle004:
    FILES = {
        "attempt": "attempt-ledger.jsonl",
        "operator": "operator-authorizations.jsonl",
        "manual": "manual-command-ledger.jsonl",
        "policy": "policy-request-ledger.jsonl",
        "model": "model-command-ledger.jsonl",
        "state": "robot-state-trajectory.jsonl",
        "safety": "safety-events.jsonl",
        "camera": "camera-frames.jsonl",
        "ack": "acknowledgements.jsonl",
    }

    def __init__(self, root: Path) -> None:
        self.root = root
        self.rollout_id: str | None = None
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


def normalize_zero_error_entries(values: Any) -> tuple[str, ...]:
    """Return no alarms only when every raw entry is provably numeric zero."""
    if not isinstance(values, (list, tuple)):
        raise MotionRejected(f"raw_error_field_not_sequence:{values!r}")
    raw = []
    for value in values:
        if isinstance(value, bool):
            raise MotionRejected(f"raw_error_entry_unparseable:{value!r}")
        try:
            numeric = float(value.strip()) if isinstance(value, str) else float(value)
        except (TypeError, ValueError) as exc:
            raise MotionRejected(f"raw_error_entry_unparseable:{value!r}") from exc
        if not math.isfinite(numeric) or numeric != 0.0:
            raise MotionRejected(f"raw_error_entry_nonzero_or_nonfinite:{value!r}")
        raw.append(str(value))
    return ()


def require_uuid4(value: str, field: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise MotionRejected(f"{field}_must_be_canonical_uuid4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise MotionRejected(f"{field}_must_be_canonical_uuid4")
    return value


def map_action28_to_action16(actions: Any) -> Any:
    import numpy as np

    array = np.asarray(actions)
    if array.shape != (50, 28) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise MotionRejected(f"invalid_policy_actions:{array.shape}:{array.dtype}")
    mapped = np.ascontiguousarray(array[:, ACTION_28_TO_16], dtype=np.float32)
    if mapped.shape != (50, 16) or not np.isfinite(mapped).all():
        raise MotionRejected("canonical_action_mapping_failed")
    return mapped


def project_model_row(
    canonical_row: Sequence[float],
    current_left: Sequence[float],
    current_right: Sequence[float],
    lower: Mapping[str, Sequence[float]],
    upper: Mapping[str, Sequence[float]],
    *,
    delta_max_rad: float = 0.01,
) -> dict[str, Any]:
    row = [float(value) for value in canonical_row]
    if len(row) != 16 or not all(math.isfinite(value) for value in row):
        raise MotionRejected("canonical_row_must_be_finite_16D")
    current = {
        "left": tuple(float(value) for value in current_left),
        "right": tuple(float(value) for value in current_right),
    }
    raw = {"left": tuple(row[0:7]), "right": tuple(row[8:15])}
    projected: dict[str, list[float]] = {}
    for side in ("left", "right"):
        if len(current[side]) != 7 or not all(math.isfinite(value) for value in current[side]):
            raise MotionRejected(f"invalid_current_state:{side}")
        projected[side] = []
        for index, (actual, target, lo, hi) in enumerate(
            zip(current[side], raw[side], lower[side], upper[side]), start=1
        ):
            executed = actual + max(-delta_max_rad, min(delta_max_rad, target - actual))
            if not math.isfinite(executed) or executed < float(lo) or executed > float(hi):
                raise MotionRejected(f"projected_target_out_of_limit:{side}:q{index}")
            if abs(executed - actual) > delta_max_rad + 1e-12:
                raise MotionRejected(f"projection_delta_exceeded:{side}:q{index}")
            projected[side].append(executed)
    return {
        "raw_joint_targets": {side: list(raw[side]) for side in raw},
        "projected_joint_targets": projected,
        "raw_gripper_targets": {"left": row[7], "right": row[15]},
        "gripper_branch": "hold_unproven_conversion",
        "gripper_commands": 0,
    }


def resolve_gripper_branch(
    raw_targets: Mapping[str, float], *, capture_to_fastwam_conversion_proven: bool
) -> dict[str, Any]:
    """Resolve the only two contract branches without silently guessing semantics."""
    values = {side: float(raw_targets[side]) for side in ("left", "right")}
    if not all(math.isfinite(value) for value in values.values()):
        raise MotionRejected("nonfinite_gripper_target")
    if not capture_to_fastwam_conversion_proven:
        return {"gripper_branch": "hold_unproven_conversion", "commands": {}}
    if any(value < 0.0 or value > 1.0 for value in values.values()):
        raise MotionRejected("capture_gripper_target_out_of_normalized_range")
    return {
        "gripper_branch": "execute_source_verified_conversion",
        "commands": {side: max(1, min(1000, int(round(value * 1000.0)))) for side, value in values.items()},
    }


def capture_generator_gripper_semantics(rm_plus_state: Any) -> dict[str, Any] | None:
    """Mirror current capture driver: None -> zero; numeric pos -> clip(pos/1000)."""
    if rm_plus_state == "disable":
        return None
    if rm_plus_state == "offline":
        return {
            "gripper_raw_rm_plus": None,
            "gripper_open": 0.0,
            "gripper_source": "capture_generator_none_to_zero_no_last_command",
        }
    if not isinstance(rm_plus_state, dict):
        raise MotionRejected(f"unrecognized_rm_plus_state:{rm_plus_state!r}")
    positions = rm_plus_state.get("pos", [])
    if not isinstance(positions, list) or not positions:
        return {
            "gripper_raw_rm_plus": None,
            "gripper_open": 0.0,
            "gripper_source": "capture_generator_none_to_zero_no_last_command",
        }
    raw = int(positions[0])
    return {
        "gripper_raw_rm_plus": raw,
        "gripper_open": max(0.0, min(1.0, raw / 1000.0)),
        "gripper_source": "capture_generator_RMPlus_pos_clipped_0_1",
    }


class CaptureUDPReceiver:
    """Exact current capture-generator JSON field/unit parser on an exact local bind."""

    def __init__(self, local_ip: str, port: int, side: str, events: EventBundle004) -> None:
        self.local_ip = local_ip
        self.port = int(port)
        self.side = side
        self.events = events
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest: dict[str, Any] | None = None
        self._error: str | None = None
        self._count = 0
        self._packets = 0
        self._last_top_keys: list[str] = []

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.local_ip, self.port))
        sock.settimeout(0.1)
        self._socket = sock
        self._thread = threading.Thread(target=self._loop, name=f"rr-dep-004-udp-{self.side}")
        self._thread.start()

    def _loop(self) -> None:
        import numpy as np

        assert self._socket is not None
        while not self._stop.is_set():
            try:
                data, peer = self._socket.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self._error = repr(exc)
                break
            try:
                message = json.loads(data.decode("utf-8"))
                if not isinstance(message, dict) or message.get("state") != "realtime_arm_joint_state":
                    continue
                self._packets += 1
                self._last_top_keys = sorted(str(key) for key in message)
                positions = message.get("joint_status", {}).get("joint_position", [])
                waypoint = message.get("waypoint", {})
                rm_plus = message.get("rm_plus_state", "disable")
                gripper = capture_generator_gripper_semantics(rm_plus)
                if gripper is None:
                    continue
                pos = waypoint.get("position") if isinstance(waypoint, dict) else None
                euler = waypoint.get("euler") if isinstance(waypoint, dict) else None
                if len(positions) < 7 or not isinstance(pos, list) or not isinstance(euler, list):
                    continue
                joints_rad = np.deg2rad(np.asarray(positions[:7], dtype=np.float64) / 1000.0)
                pose = np.asarray(
                    [pos[0] / 1_000_000.0, pos[1] / 1_000_000.0, pos[2] / 1_000_000.0,
                     euler[0] / 1000.0, euler[1] / 1000.0, euler[2] / 1000.0],
                    dtype=np.float64,
                )
                if joints_rad.shape != (7,) or pose.shape != (6,) or not np.isfinite(joints_rad).all() or not np.isfinite(pose).all():
                    raise ValueError("nonfinite_or_wrong_shape_capture_state")
                sample = {
                    "side": self.side,
                    "joints_rad": joints_rad.tolist(),
                    **gripper,
                    "eef_pose_m_rad": pose.tolist(),
                    "receive_monotonic_ns": time.monotonic_ns(),
                    "peer_ip": peer[0],
                }
                with self._condition:
                    self._latest = sample
                    self._count += 1
                    self._condition.notify_all()
            except Exception as exc:
                with self._condition:
                    self._error = repr(exc)
                    self._condition.notify_all()
                self.events.emit("safety", "capture_udp_parse_failure", {"side": self.side, "error": repr(exc)})
                break

    def wait_ready(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest is None:
                if self._error:
                    raise MotionRejected(f"capture_udp_failed:{self.side}:{self._error}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MotionRejected(
                        f"capture_udp_timeout:{self.side}:packets={self._packets}:keys={self._last_top_keys}"
                    )
                self._condition.wait(remaining)

    def latest(self, stale_s: float = 0.25) -> dict[str, Any]:
        with self._condition:
            if self._error or self._latest is None:
                raise MotionRejected(f"capture_udp_unavailable:{self.side}:{self._error}")
            sample = dict(self._latest)
        age_ns = time.monotonic_ns() - int(sample["receive_monotonic_ns"])
        if age_ns < 0 or age_ns > int(stale_s * 1e9):
            raise MotionRejected(f"capture_udp_stale:{self.side}:{age_ns / 1e9:.6f}")
        return sample

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class RR004Backend(RealManBackend):
    def __init__(self, config: dict[str, Any], events: EventBundle004) -> None:
        super().__init__(config, events)
        self.raw_errors: dict[str, list[Any]] = {}
        self.udp_receivers: dict[str, CaptureUDPReceiver] = {}
        self.prior_udp_config: dict[str, dict[str, Any]] = {}
        self.udp_configured: set[str] = set()
        self.prior_rm_plus_mode: dict[str, int] = {}
        self.rm_plus_changed: set[str] = set()

    def gather_binding(self, side: str) -> dict[str, Any]:
        result = super().gather_binding(side)
        for method in ("rm_get_realtime_push", "rm_get_gripper_state", "rm_get_rm_plus_mode"):
            try:
                result[method] = json_safe(self._call(side, method))
            except Exception as exc:
                result[method] = {"call_error": repr(exc)}
        state_value = result.get("rm_get_current_arm_state")
        raw = state_value[1].get("err", {}).get("err", []) if isinstance(state_value, list) and len(state_value) == 2 else None
        normalize_zero_error_entries(raw)
        result["raw_state_errors"] = raw
        result["normalized_state_errors"] = []
        return result

    def _set_rm_plus_json(self, side: str, mode: int) -> dict[str, Any]:
        item = self.config["robots"][side]
        command = json.dumps({"command": "set_rm_plus_mode", "mode": int(mode)}) + "\r\n"
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        try:
            sock.connect((str(item["ip"]), int(item["port"])))
            sock.sendall(command.encode("utf-8"))
            response_text = sock.recv(4096).decode("utf-8").strip().splitlines()[0]
            response = json.loads(response_text)
        finally:
            sock.close()
        if not isinstance(response, dict) or response.get("set_state") is not True:
            raise MotionRejected(f"set_rm_plus_mode_failed:{side}:mode={mode}:{response!r}")
        return json_safe(response)

    def read_state(self, side: str, controller_error: int = 0) -> StateSample:
        code, state = self._call(side, "rm_get_current_arm_state")
        if not isinstance(state, dict):
            raise MotionRejected(f"invalid_arm_state:{side}")
        joints = tuple(math.radians(float(value)) for value in state.get("joint", []))
        if len(joints) != 7 or not all(math.isfinite(value) for value in joints):
            raise MotionRejected(f"invalid_joint_state:{side}")
        raw = state.get("err", {}).get("err", [])
        errors = normalize_zero_error_entries(raw)
        self.raw_errors[side] = list(raw)
        return StateSample(
            side=side,
            joints_rad=joints,
            receive_monotonic_ns=time.monotonic_ns(),
            sdk_status=int(code),
            arm_errors=errors,
            controller_error=int(controller_error),
        )

    def _udp_config_from_dict(self, value: Mapping[str, Any]) -> Any:
        custom_value = value.get("custom_config", {})
        custom = self.module.rm_udp_custom_config_t(**{
            key: int(custom_value.get(key, -1))
            for key in ("joint_speed", "lift_state", "expand_state", "arm_current_status", "hand_state", "aloha_state", "plus_base", "plus_state")
        })
        return self.module.rm_realtime_push_config_t(
            cycle=int(value.get("cycle", 1)),
            enable=bool(value.get("enable", False)),
            port=int(value.get("port", 8089)),
            force_coordinate=int(value.get("force_coordinate", 0)),
            ip=str(value.get("ip", "127.0.0.1")),
            custom_config=custom,
        )

    def start_capture_state(self) -> dict[str, Any]:
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
                joint_speed=1, arm_current_status=1, plus_base=1, plus_state=1
            )
            selected = self.module.rm_realtime_push_config_t(
                cycle=1,
                enable=True,
                port=int(item["udp_port"]),
                force_coordinate=0,
                ip=str(item["local_ip"]),
                custom_config=custom,
            )
            status = int(self.arms[side].rm_set_realtime_push(selected))
            if status != 0:
                raise MotionRejected(f"set_realtime_push_failed:{side}:{status}")
            self.udp_configured.add(side)
            evidence[side] = {
                "prior": self.prior_udp_config[side],
                "selected": selected.to_dict(),
                "sdk_status": status,
                "prior_rm_plus_mode": int(prior_mode),
                "selected_rm_plus_mode": 115200,
                "rm_plus_json_response": mode_response,
            }
        for receiver in self.udp_receivers.values():
            receiver.wait_ready()
        evidence["status"] = "pass_exact_capture_generator_fields_units_and_RMPlus_gripper"
        return evidence

    def capture_observation_state(self) -> tuple[Any, int, dict[str, Any]]:
        import numpy as np

        samples = {side: self.udp_receivers[side].latest() for side in ("left", "right")}
        values: list[float] = []
        for side in ("left", "right"):
            sample = samples[side]
            values.extend(sample["joints_rad"])
            values.append(sample["gripper_open"])
            values.extend(sample["eef_pose_m_rad"])
        state = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
        if state.shape != (28,) or not np.isfinite(state).all():
            raise MotionRejected(f"invalid_capture_observation_state:{state.shape}:{state.dtype}")
        timestamp = min(int(sample["receive_monotonic_ns"]) for sample in samples.values())
        return state, timestamp, samples

    def close_capture_state(self) -> None:
        for side in reversed(("left", "right")):
            if side in self.udp_configured:
                try:
                    prior = self._udp_config_from_dict(self.prior_udp_config[side])
                    status = int(self.arms[side].rm_set_realtime_push(prior))
                    self.events.emit("ack", "capture_udp_config_restored", {"side": side, "sdk_status": status})
                    if status != 0:
                        self.events.emit("safety", "capture_udp_restore_failure", {"side": side, "sdk_status": status})
                except Exception as exc:
                    self.events.emit("safety", "capture_udp_restore_failure", {"side": side, "error": repr(exc)})
            if side in self.rm_plus_changed:
                try:
                    response = self._set_rm_plus_json(side, self.prior_rm_plus_mode[side])
                    self.events.emit(
                        "ack", "rm_plus_mode_restored",
                        {"side": side, "mode": self.prior_rm_plus_mode[side], "response": response},
                    )
                except Exception as exc:
                    self.events.emit("safety", "rm_plus_mode_restore_failure", {"side": side, "error": repr(exc)})
        for receiver in self.udp_receivers.values():
            receiver.close()
        self.udp_receivers.clear()

    def close(self) -> None:
        self.close_capture_state()
        super().close()


def _tuple_result(binding: Mapping[str, Any], key: str) -> list[float]:
    value = binding.get(key)
    if not isinstance(value, list) or len(value) != 2 or int(value[0]) != 0:
        raise MotionRejected(f"binding_call_failed:{key}:{value}")
    result = [float(item) for item in value[1]]
    if len(result) != 7 or not all(math.isfinite(item) for item in result):
        raise MotionRejected(f"binding_values_invalid:{key}:{value}")
    return result


def evaluate_robot_gate_004(bindings: Mapping[str, Mapping[str, Any]], stability: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "pass", "arms": {}}
    expected_collision = {"left": 0, "right": 6}
    for side in ("left", "right"):
        binding = bindings[side]
        info_value = binding.get("rm_get_robot_info")
        if not isinstance(info_value, list) or len(info_value) != 2 or int(info_value[0]) != 0:
            raise MotionRejected(f"robot_info_failed:{side}:{info_value}")
        info = info_value[1]
        if info.get("arm_model") != "RM_75" or info.get("force_type") != "6FB" or int(info.get("arm_dof", 0)) != 7:
            raise MotionRejected(f"robot_identity_mismatch:{side}:{info}")
        software = binding.get("rm_get_arm_software_info")
        if not isinstance(software, list) or len(software) != 2 or int(software[0]) != 0:
            raise MotionRejected(f"software_identity_failed:{side}")
        if software[1].get("product_version") != "RM75-6FB":
            raise MotionRejected(f"product_identity_mismatch:{side}")
        lower_deg = _tuple_result(binding, "rm_get_joint_min_pos")
        upper_deg = _tuple_result(binding, "rm_get_joint_max_pos")
        drive_lower_deg = _tuple_result(binding, "rm_get_joint_drive_min_pos")
        drive_upper_deg = _tuple_result(binding, "rm_get_joint_drive_max_pos")
        _tuple_result(binding, "rm_get_joint_max_speed")
        _tuple_result(binding, "rm_get_joint_max_acc")
        lower = [math.radians(max(a, b)) for a, b in zip(lower_deg, drive_lower_deg)]
        upper = [math.radians(min(a, b)) for a, b in zip(upper_deg, drive_upper_deg)]
        initial = list(stability[side]["initial_rad"])
        direction = select_q7_direction(initial, lower, upper)
        enabled = binding.get("rm_get_joint_en_state")
        if enabled != [0, [1] * 7]:
            raise MotionRejected(f"joint_not_enabled:{side}:{enabled}")
        errors = binding.get("rm_get_joint_err_flag")
        if not isinstance(errors, dict) or int(errors.get("return_code", -1)) != 0:
            raise MotionRejected(f"joint_error_query_failed:{side}:{errors}")
        if list(errors.get("err_flag", [])) != [0] * 7 or list(errors.get("brake_state", [])) != [0] * 7:
            raise MotionRejected(f"joint_error_or_brake_changed:{side}:{errors}")
        controller = binding.get("rm_get_controller_state")
        if not isinstance(controller, dict) or int(controller.get("return_code", -1)) != 0 or int(controller.get("system_error", -1)) != 0:
            raise MotionRejected(f"controller_error:{side}:{controller}")
        if binding.get("rm_get_arm_power_state") != [0, 1]:
            raise MotionRejected(f"arm_power_not_on:{side}")
        if binding.get("rm_get_collision_stage") != [0, expected_collision[side]]:
            raise MotionRejected(f"collision_stage_changed:{side}:{binding.get('rm_get_collision_stage')}")
        if binding.get("rm_get_collision_detection") != [-2, 0]:
            raise MotionRejected(f"stationary_collision_getter_changed:{side}:{binding.get('rm_get_collision_detection')}")
        if binding.get("rm_get_self_collision_enable") != [0, False]:
            raise MotionRejected(f"self_collision_value_changed:{side}:{binding.get('rm_get_self_collision_enable')}")
        raw = binding.get("raw_state_errors")
        normalize_zero_error_entries(raw)
        result["arms"][side] = {
            "model": "RM_75",
            "product_version": "RM75-6FB",
            "force_type": "6FB",
            "controller_firmware": software[1].get("ctrl_info", {}).get("version"),
            "sdk_api_version": binding.get("sdk_api_version"),
            "joint_order": [f"q{index}" for index in range(1, 8)],
            "initial_rad": initial,
            "lower_rad": lower,
            "upper_rad": upper,
            "q7_direction": direction,
            "raw_state_errors": raw,
            "normalized_state_errors": [],
            "controller_system_error": 0,
            "joint_error_flags": [0] * 7,
            "brake_state_raw": [0] * 7,
            "collision_stage": expected_collision[side],
            "stationary_collision_getter": [-2, 0],
            "self_collision_enabled": False,
            "watchdog": stability[side],
        }
    return result


def latest_camera_observation(rig: RealSenseRig) -> tuple[dict[str, Any], int, dict[str, Any]]:
    import numpy as np

    with rig._condition:
        if not all(role in rig._latest for role in CAMERA_TO_OBSERVATION):
            raise MotionRejected("camera_observation_not_ready")
        samples = {role: (rig._latest[role][0].copy(), dict(rig._latest[role][1])) for role in CAMERA_TO_OBSERVATION}
    result = {}
    metadata = {}
    for role, (image, item) in samples.items():
        if role != "head":
            image = image[::-1, ::-1].copy()
        converted = resize_with_pad_rgb(image)
        result[CAMERA_TO_OBSERVATION[role]] = np.ascontiguousarray(converted, dtype=np.uint8)
        metadata[role] = item
    capture_ns = min(int(item["receive_monotonic_ns"]) for item in metadata.values())
    if time.monotonic_ns() - capture_ns > 250_000_000:
        raise MotionRejected("camera_observation_stale")
    return result, capture_ns, metadata


def other_robot_command_processes() -> list[str]:
    output = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True, stdout=subprocess.PIPE).stdout
    patterns = ("realman_deployment.client", "run_realman_fastwam_client", "spacemouse_control_arm", "run_rr_dep_003.py")
    current = os.getpid()
    matches = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and int(fields[0]) != current and any(pattern in fields[1] for pattern in patterns):
            matches.append(line.strip())
    return matches


class JointCommandExecutor004:
    def __init__(self, backend: RR004Backend, poller: RobotStatePoller, rig: RealSenseRig, events: EventBundle004) -> None:
        self.backend = backend
        self.poller = poller
        self.rig = rig
        self.events = events
        self.submitted: set[str] = set()
        self.manual_count = 0
        self.model_count = 0
        self.active_side: str | None = None

    def execute(
        self,
        *,
        stage: str,
        side: str,
        kind: str,
        target: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
        envelope_reference: Mapping[str, Sequence[float]],
        unexpected_limit: float,
    ) -> dict[str, Any]:
        command_id = str(uuid.uuid4())
        require_uuid4(command_id, "command_id")
        if command_id in self.submitted:
            raise MotionRejected("duplicate_command_id")
        values = tuple(float(value) for value in target)
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise MotionRejected(f"invalid_command_target:{side}")
        for index, (value, lo, hi) in enumerate(zip(values, lower, upper), start=1):
            if value < float(lo) or value > float(hi):
                raise MotionRejected(f"command_out_of_limit:{side}:q{index}")
        before = {arm: self.poller.latest(arm) for arm in ("left", "right")}
        if any(sample is None for sample in before.values()) or not self.rig.healthy(0.25):
            raise MotionRejected("missing_fresh_feedback_or_camera_before_command")
        now_ns = time.monotonic_ns()
        for arm, sample in before.items():
            assert sample is not None
            if now_ns - sample.receive_monotonic_ns > 250_000_000 or sample.sdk_status or sample.arm_errors or sample.controller_error:
                raise MotionRejected(f"unsafe_state_before_command:{arm}")
        self.submitted.add(command_id)
        if stage == "manual":
            self.manual_count += 1
            if self.manual_count > 6:
                raise MotionRejected("manual_command_budget_exceeded")
            ledger = "manual"
        elif stage == "model":
            self.model_count += 1
            if self.model_count > 4:
                raise MotionRejected("model_command_budget_exceeded")
            ledger = "model"
        else:
            raise MotionRejected(f"unknown_command_stage:{stage}")
        submitted_ns = time.monotonic_ns()
        self.events.emit(ledger, "command_submitting", {"command_id": command_id, "side": side, "kind": kind, "target_rad": values, "speed_percent": 5})
        self.active_side = side
        try:
            status = self.backend.command_absolute(side, values, 5)
        except Exception as exc:
            self.backend.slow_stop(side)
            raise MotionRejected(f"command_acceptance_unknown:{command_id}:{exc!r}") from exc
        self.events.emit(ledger, "sdk_acceptance", {"command_id": command_id, "side": side, "sdk_status": status})
        if status != 0:
            self.backend.slow_stop(side)
            raise MotionRejected(f"command_sdk_rejected:{command_id}:{status}")
        deadline = time.monotonic() + 5.0
        while True:
            if not self.rig.healthy(0.25):
                self.backend.slow_stop(side)
                raise MotionRejected(f"camera_loss_during_command:{command_id}")
            samples = {arm: self.poller.latest(arm) for arm in ("left", "right")}
            now_ns = time.monotonic_ns()
            for arm, sample in samples.items():
                if sample is None or now_ns - sample.receive_monotonic_ns > 250_000_000:
                    self.backend.slow_stop(side)
                    raise MotionRejected(f"feedback_loss_during_command:{command_id}:{arm}")
                if sample.sdk_status or sample.arm_errors or sample.controller_error:
                    self.backend.slow_stop(side)
                    raise MotionRejected(f"alarm_during_command:{command_id}:{arm}")
                reference = envelope_reference[arm]
                for index, (actual, initial) in enumerate(zip(sample.joints_rad, reference), start=1):
                    if abs(actual - float(initial)) > unexpected_limit + 1e-12:
                        self.backend.slow_stop(side)
                        raise MotionRejected(f"unexpected_motion:{command_id}:{arm}:q{index}")
            commanded = samples[side]
            assert commanded is not None
            error = max(abs(actual - expected) for actual, expected in zip(commanded.joints_rad, values))
            if error <= 0.005 and now_ns - submitted_ns >= 250_000_000:
                self.events.emit(ledger, "observed_target", {"command_id": command_id, "side": side, "target_error_rad": error, "observed_rad": commanded.joints_rad})
                self.active_side = None
                return {"command_id": command_id, "side": side, "kind": kind, "sdk_status": status, "target_error_rad": error, "observed_rad": list(commanded.joints_rad)}
            if time.monotonic() >= deadline:
                self.backend.slow_stop(side)
                raise MotionRejected(f"command_timeout:{command_id}")
            time.sleep(0.01)


def select_free_loopback_port(first: int = 18100, last: int = 18109) -> int:
    for port in range(first, last + 1):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            probe.close()
    raise MotionRejected("no_free_authorized_client_tunnel_port")


def start_tunnel(local_port: int, compute_port: int) -> dict[str, Any]:
    if subprocess.run(["tmux", "has-session", "-t", "rr-dep-004-client-tunnel"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise MotionRejected("rr_dep_004_tunnel_session_already_exists")
    command = [
        "tmux", "new-session", "-d", "-s", "rr-dep-004-client-tunnel", "exec", "ssh", "-N",
        "-L", f"127.0.0.1:{local_port}:127.0.0.1:{compute_port}",
        "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3", "xzs_server",
    ]
    subprocess.run(command, check=True)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=0.5):
                return {"status": "running_loopback_only", "local_port": local_port, "compute_port": compute_port, "tmux_session": "rr-dep-004-client-tunnel"}
        except OSError:
            time.sleep(0.1)
    raise MotionRejected("task_tunnel_start_timeout")


def validate_service_metadata(metadata: Mapping[str, Any], local_port: int) -> None:
    expected = {"schema_id": SCHEMA_ID, "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "service_session_id": SERVICE_SESSION_ID}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise MotionRejected(f"server_metadata_mismatch:{key}")
    if metadata.get("bind") != "127.0.0.1:8010" or int(metadata.get("port", -1)) != 8010:
        raise MotionRejected("server_bind_metadata_mismatch")
    if local_port not in range(18100, 18110):
        raise MotionRejected("client_endpoint_outside_authorized_range")


def send_live_request(
    *,
    local_port: int,
    rollout_id: str,
    request_ordinal: int,
    rig: RealSenseRig,
    backend: RR004Backend,
    artifact_root: Path,
    events: EventBundle004,
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    require_uuid4(rollout_id, "rollout_id")
    if request_ordinal not in range(1, 11):
        raise MotionRejected("request_ordinal_outside_budget")
    images, camera_ns, camera_metadata = latest_camera_observation(rig)
    state, state_ns, state_metadata = backend.capture_observation_state()
    capture_ns = min(camera_ns, state_ns)
    send_ns = time.monotonic_ns()
    if send_ns - capture_ns > 250_000_000:
        raise MotionRejected("combined_observation_stale_before_send")
    request_id = str(uuid.uuid4())
    context = {
        "schema_id": SCHEMA_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "rollout_id": rollout_id,
        "service_session_id": SERVICE_SESSION_ID,
        "request_id": request_id,
        "request_ordinal": request_ordinal,
        "client_send_utc": utc_now(),
        "client_send_monotonic_ns": send_ns,
        "observation_capture_monotonic_ns": capture_ns,
    }
    observation = {**images, "state": state, "prompt": PROMPT, "__rr_context__": context}
    module = importlib.import_module("openpi_client.websocket_client_policy")
    policy = module.WebsocketClientPolicy(host="127.0.0.1", port=local_port)
    metadata = policy.get_server_metadata()
    validate_service_metadata(metadata, local_port)
    events.emit("policy", "request_sending", {"context": context, "camera_metadata": camera_metadata, "capture_state": state_metadata, "state_sha256": hashlib.sha256(state.tobytes()).hexdigest()})
    started = time.monotonic_ns()
    response = policy.infer(observation)
    received = time.monotonic_ns()
    if hasattr(policy, "close"):
        policy.close()
    if not isinstance(response, Mapping) or response.get("__rr_context__") != context:
        raise MotionRejected("policy_response_context_mismatch")
    actions = np.asarray(response.get("actions"))
    canonical = map_action28_to_action16(actions)
    full_path = artifact_root / "raw-response-actions.npy"
    with full_path.open("xb") as stream:
        np.save(stream, np.ascontiguousarray(actions, dtype=np.float32), allow_pickle=False)
    response_summary = {
        "status": "pass_complete_no_motion_validation",
        "context": context,
        "source_shape": [50, 28],
        "source_dtype": "float32",
        "source_finite": True,
        "source_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
        "raw_action_file": str(full_path),
        "raw_action_file_sha256": sha256_file(full_path),
        "canonical_shape": [50, 16],
        "server_timing": response.get("server_timing"),
        "rr_result_summary": response.get("rr_result_summary"),
        "client_round_trip_ms": (received - started) / 1_000_000.0,
        "client_receive_monotonic_ns": received,
        "cross_host_timestamp_subtraction_used": False,
    }
    events.emit("policy", "response_validated", response_summary)
    return actions, canonical, response_summary


def prompt_exact(events: EventBundle004, prompt: str, *, expected: str | None, field: str) -> str:
    value = input(prompt).strip()
    events.emit("operator", "operator_input", {"field": field, "value": value})
    if not value or (expected is not None and value != expected):
        raise MotionRejected(f"operator_input_mismatch:{field}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if config.get("identity") != {"task_id": TASK_ID, "run_id": RUN_ID, "service_session_id": SERVICE_SESSION_ID, "compute_port": 8010}:
        raise MotionRejected("config_identity_mismatch")
    if config.get("protocol", {}).get("schema_id") != SCHEMA_ID or config.get("protocol", {}).get("prompt") != PROMPT:
        raise MotionRejected("config_protocol_mismatch")
    return config


def run_live(
    config: dict[str, Any], artifact_root: Path, events: EventBundle004, *, correction_id: str | None = None
) -> dict[str, Any]:
    source_check = verify_frozen_sources(artifact_root / "source-manifest.tsv")
    conflicts = other_robot_command_processes()
    if conflicts:
        raise MotionRejected(f"other_robot_command_source:{conflicts}")
    rollout_id = str(uuid.uuid4())
    events.rollout_id = rollout_id
    backend = RR004Backend(config, events)
    poller: RobotStatePoller | None = None
    rig: RealSenseRig | None = None
    executor: JointCommandExecutor004 | None = None
    try:
        rig = RealSenseRig(config, events)
        rig.enumerate_and_start()
        rig.wait_ready()
        suffix = f"-correction-{correction_id}" if correction_id else ""
        rig.start_recording(artifact_root / "camera-video", f"read-only-gate{suffix}")
        gate_record_start = time.monotonic()
        while time.monotonic() - gate_record_start < 2.05:
            if not rig.healthy(0.25):
                raise MotionRejected("camera_unhealthy_during_read_only_gate")
            time.sleep(0.02)
        camera_recording = rig.stop_recording()
        images, _, _ = latest_camera_observation(rig)
        camera_binding = {
            "status": "pass",
            "binding": rig.binding,
            "active_profiles": rig.profiles,
            "recording": camera_recording,
            "openpi_outputs": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in images.items()},
            "orientation": {"head": "unchanged", "left_wrist": "rotate_180", "right_wrist": "rotate_180"},
        }
        write_yaml(artifact_root / f"camera-binding{suffix}.yaml", camera_binding)

        backend.open()
        bindings = {side: backend.gather_binding(side) for side in ("left", "right")}
        poller = RobotStatePoller(backend, events, hz=20.0)
        poller.start()
        stability = poller.wait_stable(seconds=2.0)
        robot_gate = evaluate_robot_gate_004(bindings, stability)
        udp_evidence = backend.start_capture_state()
        state, state_ns, capture_samples = backend.capture_observation_state()
        if time.monotonic_ns() - state_ns > 250_000_000:
            raise MotionRejected("capture_state_stale_at_gate")
        robot_semantics = {
            "status": "pass",
            "bindings": bindings,
            "gate": robot_gate,
            "capture_udp": udp_evidence,
            "capture_state": {"shape": list(state.shape), "dtype": str(state.dtype), "finite": True, "samples": capture_samples},
            "capture_state_order": ["left_q1_to_q7", "left_gripper_open", "left_eef_xyz_euler", "right_q1_to_q7", "right_gripper_open", "right_eef_xyz_euler"],
            "model_dispatch_order": ["right", "left"],
            "gripper_branch": "hold_unproven_conversion",
            "gripper_reason": "capture uses UDP RMPlus pos while FastWAM physical command path uses rm_set_gripper_position; equivalence is not proven",
            "blocking_semantics": "rm_movej_nonblocking_SDK_acceptance_plus_Client_5s_feedback_block",
            "slow_stop": "rm_set_arm_slow_stop",
        }
        write_yaml(artifact_root / "robot-sdk-semantics.yaml", robot_semantics)

        gate = {
            "status": "awaiting_operator_exact_phrase",
            "source_check": source_check,
            "camera": camera_binding,
            "robot": robot_gate,
            "capture_state": {"shape": [28], "dtype": "float32", "freshness_sec": (time.monotonic_ns() - state_ns) / 1e9},
            "other_robot_command_processes": [],
            "operator_checks": {
                "identity_role": "pending",
                "physically_present": "pending",
                "stop_estop_reachable": "pending",
                "workcell_and_both_swept_volumes_clear": "pending",
                "wrist_tool_cables_slack": "pending",
            },
            "planned_command_bounds": {
                "manual": "left_then_right_each_hold_q7_abs_delta_le_0.01rad_return_six_joint_commands_max",
                "model": "response_row_0_only_each_joint_projected_le_0.01rad_right_then_left_target_and_return_four_joint_commands_max",
                "total_joint_commands_max": 10,
                "speed_percent_max": 5,
                "target_and_return_tolerance_rad": 0.005,
                "stale_timeout_sec": 0.25,
                "command_timeout_sec": 5.0,
                "gripper": "hold_unproven_conversion_zero_commands",
            },
            "required_phrase": OPERATOR_PHRASE,
        }
        print("RR-DEP-004 COMPLETE PRE-MOTION GATE", flush=True)
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
            gate["operator_checks"][field] = "confirmed"
        prompt_exact(events, "MOTION AUTHORIZATION: ", expected=OPERATOR_PHRASE, field="exact_motion_phrase")
        gate["operator_checks"]["identity_role"] = {"identity": identity, "role": role}
        gate["status"] = "pass_task_local_motion_unlock"
        gate["authorization_recorded_at"] = utc_now()
        write_yaml(artifact_root / "motion-unlock-gate.yaml", gate)

        executor = JointCommandExecutor004(backend, poller, rig, events)
        manual_initial = {side: list(poller.wait_stable(seconds=2.0)[side]["initial_rad"]) for side in ("left", "right")}
        manual_results = []
        rig.start_recording(artifact_root / "camera-video", f"manual-{rollout_id}")
        for side in ("left", "right"):
            arm_gate = robot_gate["arms"][side]
            initial = manual_initial[side]
            micro = list(initial)
            micro[6] += int(arm_gate["q7_direction"]) * 0.01
            for kind, target in (("hold_current", initial), ("q7_micro", micro), ("return", initial)):
                manual_results.append(executor.execute(
                    stage="manual", side=side, kind=kind, target=target,
                    lower=arm_gate["lower_rad"], upper=arm_gate["upper_rad"],
                    envelope_reference=manual_initial, unexpected_limit=0.01,
                ))
        manual_recording = rig.stop_recording()
        post_manual = poller.wait_stable(seconds=2.0)
        for side in ("left", "right"):
            if max(abs(a - b) for a, b in zip(post_manual[side]["initial_rad"], manual_initial[side])) > 0.005:
                raise MotionRejected(f"manual_return_tolerance_failed:{side}")
        events.emit("ack", "manual_micro_return_pass", {"commands": manual_results})

        local_port = select_free_loopback_port(int(config["protocol"]["client_port_first"]), int(config["protocol"]["client_port_last"]))
        tunnel = start_tunnel(local_port, int(config["identity"]["compute_port"]))
        events.emit("policy", "tunnel_started", tunnel)
        _raw, canonical, response_summary = send_live_request(
            local_port=local_port, rollout_id=rollout_id, request_ordinal=1,
            rig=rig, backend=backend, artifact_root=artifact_root, events=events,
        )
        response_received_ns = int(response_summary["client_receive_monotonic_ns"])
        pre_model = {side: list(poller.latest(side).joints_rad) for side in ("left", "right")}
        projected = project_model_row(
            canonical[0], pre_model["left"], pre_model["right"],
            {side: robot_gate["arms"][side]["lower_rad"] for side in ("left", "right")},
            {side: robot_gate["arms"][side]["upper_rad"] for side in ("left", "right")},
        )
        if time.monotonic_ns() - response_received_ns > 2_000_000_000:
            raise MotionRejected("response_to_first_command_exceeded_2s")
        model_results = []
        rig.start_recording(artifact_root / "camera-video", f"model-{rollout_id}")
        for side in ("right", "left"):
            model_results.append(executor.execute(
                stage="model", side=side, kind="projected_model_row_0",
                target=projected["projected_joint_targets"][side],
                lower=robot_gate["arms"][side]["lower_rad"], upper=robot_gate["arms"][side]["upper_rad"],
                envelope_reference=pre_model, unexpected_limit=0.015,
            ))
        for side in ("right", "left"):
            model_results.append(executor.execute(
                stage="model", side=side, kind="return_to_pre_model",
                target=pre_model[side], lower=robot_gate["arms"][side]["lower_rad"],
                upper=robot_gate["arms"][side]["upper_rad"], envelope_reference=pre_model,
                unexpected_limit=0.015,
            ))
        model_recording = rig.stop_recording()
        post_model = poller.wait_stable(seconds=2.0)
        for side in ("left", "right"):
            if max(abs(a - b) for a, b in zip(post_model[side]["initial_rad"], pre_model[side])) > 0.005:
                raise MotionRejected(f"model_return_tolerance_failed:{side}")
        write_new_json(artifact_root / "canonical-action16.json", {
            "schema_id": "realman.dual_arm.absolute.v1", "shape": [50, 16],
            "order": ["left_q1_to_q7", "left_gripper", "right_q1_to_q7", "right_gripper"],
            "actions": canonical.tolist(),
        })
        write_new_json(artifact_root / "executed-action.json", {
            "response_row": 0, "projection_abs_delta_max_rad": 0.01,
            "pre_model_joint_state": pre_model, **projected,
        })
        write_new_json(artifact_root / "response-summary.json", response_summary)
        acknowledgement = {
            "status": "pass",
            "rollout_id": rollout_id,
            "operator_identity": identity,
            "operator_role": role,
            "manual_commands": manual_results,
            "model_commands": model_results,
            "manual_return": "within_0.005rad_both",
            "model_return": "within_0.005rad_both",
            "gripper_branch": "hold_unproven_conversion",
            "gripper_commands": 0,
            "joint_command_counts": {"manual": executor.manual_count, "model_including_return": executor.model_count, "total": executor.manual_count + executor.model_count},
            "recordings": {"manual": manual_recording, "model": model_recording},
            "safety_events": "see safety-events.jsonl",
        }
        write_yaml(artifact_root / "physical-acknowledgement.yaml", acknowledgement)
        events.emit("ack", "physical_pipeline_pass", acknowledgement)
        return {
            "status": "completed_bounded_physical_pipeline",
            "rollout_id": rollout_id,
            "local_port": local_port,
            "tunnel": tunnel,
            "manual_command_count": executor.manual_count,
            "model_command_count": executor.model_count,
            "gripper_command_count": 0,
            "response_summary": response_summary,
            "physical_acknowledgement": acknowledgement,
        }
    except Exception as exc:
        if executor is not None and executor.active_side is not None:
            try:
                backend.slow_stop(executor.active_side)
            except Exception as stop_exc:
                events.emit("safety", "slow_stop_failure", {"side": executor.active_side, "error": repr(stop_exc)})
        events.emit("safety", "live_run_failed", {"error": repr(exc), "active_side": executor.active_side if executor else None})
        raise
    finally:
        if rig is not None:
            rig.close()
        if poller is not None:
            poller.close()
        backend.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RR-DEP-004 bounded live runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--authorize-live-bounded-task", action="store_true")
    parser.add_argument("--correction-id", choices=("004", "005"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.artifact_root.resolve()
    if not args.authorize_live_bounded_task:
        print(json.dumps({"status": "rejected", "error": "explicit_bounded_task_authorization_required"}))
        return 2
    if root.name != "client" or root.parent.name != RUN_ID or root.parent.parent.name != TASK_ID:
        print(json.dumps({"status": "rejected", "error": "unexpected_artifact_root"}))
        return 2
    config = load_config(args.config.resolve())
    events = EventBundle004(root)
    try:
        result = run_live(config, root, events, correction_id=args.correction_id)
        suffix = f"-correction-{args.correction_id}" if args.correction_id else ""
        write_new_json(root / f"live-run-result{suffix}.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = {"status": "failed", "error": repr(exc), "captured_at": utc_now()}
        suffix = f"-correction-{args.correction_id}" if args.correction_id else ""
        result_path = root / f"live-run-result{suffix}.json"
        if not result_path.exists():
            write_new_json(result_path, failure)
        print(json.dumps(failure, indent=2, ensure_ascii=False), flush=True)
        return 1
    finally:
        events.close()
