#!/usr/bin/env python3
"""Send the single authorized RR-DEP-002 fixture request and freeze local summaries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_client_runtime.pi05_policy_client import Pi05PolicyClient  # noqa: E402
from robot_client_runtime.rr_dep_002_integration import (  # noqa: E402
    EXPERIMENT_ID,
    PROTOCOL_SCHEMA_ID,
    RRDep002NoMotionIntegration,
    RUN_ID,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-live-one-request", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--service-ready", type=Path, required=True)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--action-output-contract", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.authorize_live_one_request:
        raise RuntimeError("explicit_live_one_request_authorization_required")
    if args.host != "127.0.0.1" or args.port != 18000:
        raise RuntimeError("only_contract_loopback_endpoint_is_allowed")

    import numpy as np

    service_ready = _load_json(args.service_ready)
    service_metadata = _load_json(args.service_metadata)
    action_contract = _load_json(args.action_output_contract)
    input_contract = service_metadata["input_contract"]
    image_shape = tuple(input_contract["image_shape"])
    prompt = input_contract["prompt"]

    policy = Pi05PolicyClient(
        live_enabled=True,
        client_kwargs={"host": args.host, "port": args.port},
    )
    policy.connect()
    integration = RRDep002NoMotionIntegration(
        policy_client=policy,
        service_ready=service_ready,
        expected_server_metadata=service_metadata,
        action_output_contract=action_contract,
    )
    actual_metadata = policy.get_server_metadata()
    integration.validate_server_metadata(actual_metadata)

    client_send_utc = datetime.now(timezone.utc).isoformat()
    client_send_monotonic_ns = time.monotonic_ns()
    context = {
        "schema_id": PROTOCOL_SCHEMA_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "service_session_id": service_ready["service_session_id"],
        "request_id": args.request_id,
        "request_ordinal": 1,
        "client_send_utc": client_send_utc,
        "client_send_monotonic_ns": client_send_monotonic_ns,
    }
    observation = {
        "base_0_rgb": np.zeros(image_shape, dtype=np.uint8),
        "left_wrist_0_rgb": np.zeros(image_shape, dtype=np.uint8),
        "right_wrist_0_rgb": np.zeros(image_shape, dtype=np.uint8),
        "state": np.zeros(tuple(input_contract["state_shape"]), dtype=np.float32),
        "prompt": prompt,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zero_image_sha256 = hashlib.sha256(observation["base_0_rgb"].tobytes()).hexdigest()
    zero_state_sha256 = hashlib.sha256(observation["state"].tobytes()).hexdigest()
    request_record = {
        "schema_version": 1,
        "context": context,
        "transport_endpoint": f"ws://{args.host}:{args.port}",
        "images": {
            key: {"shape": list(observation[key].shape), "dtype": "uint8", "sha256": zero_image_sha256}
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "state": {"shape": list(observation["state"].shape), "dtype": "float32", "sha256": zero_state_sha256},
        "prompt": prompt,
        "raw_dataset_or_camera_input": None,
    }
    _write_json(args.output_dir / "request.json", request_record)

    result = integration.run_one_request(observation, context)
    client_receive_monotonic_ns = time.monotonic_ns()
    client_receive_utc = datetime.now(timezone.utc).isoformat()
    round_trip_ms = (client_receive_monotonic_ns - client_send_monotonic_ns) / 1_000_000
    canonical_actions = result.pop("canonical_actions")
    _write_json(
        args.output_dir / "canonical-action16.json",
        {
            "schema_id": result["canonical_schema_id"],
            "shape": result["canonical_shape"],
            "order": result["canonical_order"],
            "actions": canonical_actions,
            "validation": "pass_shape_order_finiteness_and_explicit_limits_only",
        },
    )
    counters = {
        "policy_connections": policy.policy_connections,
        "requests": policy.requests,
        "responses": policy.responses,
        "retries": policy.retries,
        "reconnects": policy.reconnects,
        "duplicate_executions": policy.duplicate_executions,
        "backend_calls": result["backend_calls"],
        "sdk_calls": result["sdk_calls"],
        "device_opens": result["device_opens"],
        "motion_commands": result["motion_commands"],
    }
    _write_json(
        args.output_dir / "response-summary.json",
        {
            "schema_version": 1,
            "status": "one_response_validated_no_motion",
            "context": result["context"],
            "source_shape": result["source_shape"],
            "source_dtype": result["source_dtype"],
            "canonical_shape": result["canonical_shape"],
            "server_timing": result["server_timing"],
            "client_receive_utc": client_receive_utc,
            "client_receive_monotonic_ns": client_receive_monotonic_ns,
            "client_monotonic_round_trip_ms": round_trip_ms,
            "cross_host_timestamp_subtraction_used": False,
            "counters": counters,
        },
    )
    if counters != {
        "policy_connections": 1,
        "requests": 1,
        "responses": 1,
        "retries": 0,
        "reconnects": 0,
        "duplicate_executions": 0,
        "backend_calls": 0,
        "sdk_calls": 0,
        "device_opens": 0,
        "motion_commands": 0,
    }:
        raise RuntimeError(f"counter_contract_mismatch: {counters!r}")
    print(json.dumps({"status": "pass", "request_id": args.request_id, "round_trip_ms": round_trip_ms}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
