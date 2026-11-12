from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import inspect
import time
import unittest
import uuid

import numpy as np

from robot_client_runtime.rr_dep_003_motion import MotionRejected, StateSample
from robot_client_runtime.rr_dep_003 import CameraRecorder, RealSenseRig
from robot_client_runtime.rr_dep_005 import (
    COMPUTE_RUNTIME_COMMIT,
    EventBundle005,
    EXPERIMENT_ID,
    RUN_ID,
    RR005Backend,
    SCHEMA_ID,
    SERVICE_SESSION_ID,
    build_parser,
    send_live_request,
    settle_after_return,
    validate_args,
    validate_service_metadata,
)


class _FakeRig:
    def __init__(self) -> None:
        now = time.monotonic_ns()
        self._condition = nullcontext()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._latest = {
            role: (frame.copy(), {"receive_monotonic_ns": now, "frame_number": 1})
            for role in ("head", "left_wrist", "right_wrist")
        }

    def healthy(self, _stale: float) -> bool:
        return True


class _FakeCaptureBackend:
    def capture_observation_state(self):
        now = time.monotonic_ns()
        return np.zeros(28, dtype=np.float32), now, {
            "left": {"receive_monotonic_ns": now},
            "right": {"receive_monotonic_ns": now},
        }


class _FakePolicy:
    last_observation = None

    def __init__(self, *, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def get_server_metadata(self):
        return {
            "schema_id": SCHEMA_ID,
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "service_session_id": SERVICE_SESSION_ID,
            "stable_runtime_commit": COMPUTE_RUNTIME_COMMIT,
            "checkpoint_role": "integration_only",
            "checkpoint_selection_status": "not_selected",
            "bind": "127.0.0.1:8020",
            "port": 8020,
        }

    def infer(self, observation):
        type(self).last_observation = observation
        return {
            "actions": np.zeros((50, 28), dtype=np.float32),
            "__rr_context__": observation["__rr_context__"],
            "server_timing": {"infer_ms": 1.0},
        }

    def close(self):
        return None


class _Clock:
    def __init__(self) -> None:
        self.ns = 0

    def monotonic_ns(self) -> int:
        return self.ns

    def sleep(self, seconds: float) -> None:
        self.ns += int(seconds * 1e9)


class _SettlePoller:
    def __init__(self, clock: _Clock, *, unstable: bool = False) -> None:
        self.clock = clock
        self.unstable = unstable

    def latest(self, side: str) -> StateSample:
        value = 0.002 if self.unstable and (self.clock.ns // 50_000_000) % 2 else 0.0
        return StateSample(
            side=side,
            joints_rad=(value,) + (0.0,) * 6,
            receive_monotonic_ns=self.clock.ns,
            sdk_status=0,
            arm_errors=(),
            controller_error=0,
        )


class _SettleEvents:
    def __init__(self) -> None:
        self.records = []

    def emit(self, name, event, payload):
        self.records.append((name, event, payload))


class RRDep005Tests(unittest.TestCase):
    def _args(self):
        args = build_parser().parse_args(["--mode", "dry-run"])
        args.artifact_root = args.artifact_root.resolve()
        args.record_dir = args.record_dir.resolve()
        args.source_manifest = args.source_manifest.resolve()
        args.deltawam_src = args.deltawam_src.resolve()
        return args

    def test_cli_exposes_required_contract_flags_and_rejects_expansion(self):
        help_text = build_parser().format_help()
        for flag in (
            "--mode", "--server-url", "--ssh-alias", "--ssh-local-port", "--compute-port",
            "--experiment-id", "--run-id", "--rollout-id", "--prompt", "--head-serial",
            "--left-wrist-serial", "--right-wrist-serial", "--left-robot-ip", "--right-robot-ip",
            "--robot-port", "--image-width", "--image-height", "--camera-fps", "--model-image-size",
            "--state-dim", "--action-dim", "--action-row", "--joint-delta-limit-rad",
            "--speed-percent", "--state-stale-sec", "--response-to-command-sec",
            "--command-timeout-sec", "--watchdog-hz", "--settle-grace-sec", "--stable-window-sec",
            "--stable-span-threshold-rad", "--settle-timeout-sec", "--max-attempts",
            "--max-policy-requests", "--gripper-mode", "--return-to-start", "--artifact-root",
            "--record-dir", "--operator-phrase", "--return-ack-phrase", "--log-level",
        ):
            self.assertIn(flag, help_text)
        args = self._args()
        validate_args(args)
        for field, bad in (
            ("speed_percent", 6), ("joint_delta_limit_rad", 0.02),
            ("watchdog_hz", 9.0), ("max_attempts", 11), ("max_policy_requests", 11),
        ):
            with self.subTest(field=field):
                candidate = self._args()
                setattr(candidate, field, bad)
                with self.assertRaises(MotionRejected):
                    validate_args(candidate)

    def test_staged_mode_never_treats_phrase_flags_as_authorization(self):
        args = self._args()
        args.mode = "staged-one-batch"
        args.return_to_start = False
        with self.assertRaises(MotionRejected):
            validate_args(args)
        args.return_to_start = True
        args.gripper_mode = "execute-source-verified"
        with self.assertRaises(MotionRejected):
            validate_args(args)

    def test_service_metadata_requires_exact_session_commit_and_role(self):
        args = self._args()
        metadata = _FakePolicy(host="127.0.0.1", port=18200).get_server_metadata()
        validate_service_metadata(metadata, args)
        for key in ("schema_id", "service_session_id", "stable_runtime_commit", "checkpoint_role", "port"):
            with self.subTest(key=key):
                bad = dict(metadata)
                bad[key] = "wrong"
                with self.assertRaises(MotionRejected):
                    validate_service_metadata(bad, args)

    def test_cleanup_backend_is_readback_gated_not_status_only(self):
        self.assertIn("readback_matches", inspect.getsource(RR005Backend.close_capture_state))
        self.assertNotEqual(RR005Backend.close_capture_state, RR005Backend.__mro__[1].close_capture_state)

    def test_camera_capture_does_not_hold_global_recorder_lock_during_pipe_write(self):
        source = inspect.getsource(RealSenseRig._capture_loop)
        self.assertIn("recorder = self._recorder", source)
        self.assertIn("recorder.write(role, frame, **metadata)", source)
        self.assertLess(source.index("recorder = self._recorder"), source.index("recorder.write(role, frame, **metadata)"))

    def test_camera_recorder_decouples_capture_from_encoder_pipe(self):
        write_source = inspect.getsource(CameraRecorder.write)
        worker_source = inspect.getsource(CameraRecorder._writer_loop)
        self.assertIn("put_nowait(frame.copy())", write_source)
        self.assertNotIn("self._writers[role].write(frame)", write_source)
        self.assertIn("self._writers[role].write(item)", worker_source)

    def test_live_recording_starts_after_blocking_robot_preflight(self):
        from robot_client_runtime.rr_dep_005 import run_live
        source = inspect.getsource(run_live)
        self.assertLess(source.index("backend.start_capture_state()"), source.index("start_warmed_recording"))
        self.assertLess(source.index("write_yaml(versioned_path(args.artifact_root, \"robot-sdk-semantics.yaml\""),
                        source.index("start_warmed_recording"))

    def test_fake_camera_state_protocol_context_and_action_mapping(self):
        args = self._args()
        task_cache = args.artifact_root.parents[3] / "cache" / "rr-dep-005-run-001"
        args.artifact_root = task_cache / f"unit-policy-{uuid.uuid4()}"
        args.artifact_root.mkdir(parents=True)
        args.state_stale_sec = 0.25
        args.state_dim = 28
        args.max_policy_requests = 10
        rollout_id = str(uuid.uuid4())
        events = EventBundle005(args.artifact_root, rollout_id)
        fake_module = SimpleNamespace(WebsocketClientPolicy=_FakePolicy)
        try:
            with mock.patch("robot_client_runtime.rr_dep_005.importlib.import_module", return_value=fake_module):
                canonical, summary = send_live_request(
                    args=args, local_port=18200, intent="no_motion_validation",
                    rig=_FakeRig(), backend=_FakeCaptureBackend(), events=events,
                )
        finally:
            events.close()
        self.assertEqual(canonical.shape, (50, 16))
        self.assertEqual(canonical.dtype, np.float32)
        self.assertEqual(summary["context"]["execution_intent"], "no_motion_validation")
        self.assertEqual(summary["context"]["rollout_id"], rollout_id)
        self.assertEqual(_FakePolicy.last_observation["state"].shape, (28,))
        self.assertEqual(_FakePolicy.last_observation["base_0_rgb"].shape, (224, 224, 3))
        self.assertEqual(len(list((args.artifact_root / "raw-actions").glob("*.npy"))), 1)
        self.assertEqual(len(list((args.artifact_root / "canonical-actions").glob("*.npy"))), 1)

    def test_settling_uses_grace_then_full_window_and_times_out_without_expansion(self):
        args = SimpleNamespace(
            state_stale_sec=0.25, target_tolerance_rad=0.005, watchdog_hz=20.0,
            settle_grace_sec=0.1, stable_window_sec=0.2,
            stable_span_threshold_rad=0.001, settle_timeout_sec=0.6,
        )
        clock = _Clock()
        events = _SettleEvents()
        with mock.patch("robot_client_runtime.rr_dep_005.time.monotonic_ns", side_effect=clock.monotonic_ns), \
             mock.patch("robot_client_runtime.rr_dep_005.time.sleep", side_effect=clock.sleep):
            result = settle_after_return(
                args=args, rig=_FakeRig(), poller=_SettlePoller(clock),
                pre_model={"left": [0.0] * 7, "right": [0.0] * 7},
                return_accepted_ns=0, events=events,
            )
        self.assertEqual(result["status"], "pass")
        self.assertGreaterEqual(result["elapsed_sec"], 0.3)
        self.assertTrue(any(event == "stable_window_pass" for _, event, _ in events.records))

        clock = _Clock()
        events = _SettleEvents()
        with mock.patch("robot_client_runtime.rr_dep_005.time.monotonic_ns", side_effect=clock.monotonic_ns), \
             mock.patch("robot_client_runtime.rr_dep_005.time.sleep", side_effect=clock.sleep):
            with self.assertRaisesRegex(MotionRejected, "settle_timeout_hard_stop"):
                settle_after_return(
                    args=args, rig=_FakeRig(), poller=_SettlePoller(clock, unstable=True),
                    pre_model={"left": [0.0] * 7, "right": [0.0] * 7},
                    return_accepted_ns=0, events=events,
                )
        self.assertEqual(args.stable_span_threshold_rad, 0.001)


if __name__ == "__main__":
    unittest.main()
