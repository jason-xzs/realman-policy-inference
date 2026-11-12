from __future__ import annotations

from unittest import mock
import unittest
import uuid

import numpy as np

from robot_client_runtime.rr_dep_003_motion import MotionRejected
from robot_client_runtime.rr_dep_006 import (
    CompactEventSink,
    OPERATOR_PHRASE,
    RETURN_ACK_PHRASE,
    RR006Backend,
    SERVICE_RUNTIME_COMMIT,
    SERVICE_SCHEMA_ID,
    SERVICE_SESSION_ID,
    build_parser,
    ensure_loopback_no_proxy,
    prompt_exact,
    record_request_outcome_resolution,
    validate_attempt_boundary,
    validate_args,
    validate_service_metadata,
)


class _FakeArm:
    def __init__(self, values):
        self.values = iter(values)

    def rm_get_controller_state(self):
        return next(self.values)


class RRDep006Tests(unittest.TestCase):
    def _args(self, mode="dry-run"):
        args = build_parser().parse_args(["--mode", mode])
        args.artifact_root = args.artifact_root.resolve()
        return args

    def test_help_exposes_required_flags_and_record_mode_is_off(self):
        help_text = build_parser().format_help()
        for flag in (
            "--mode", "--record-mode", "--server-url", "--ssh-alias", "--ssh-local-port", "--compute-port",
            "--service-session-id", "--experiment-id", "--run-id", "--rollout-id", "--prompt",
            "--head-serial", "--left-wrist-serial", "--right-wrist-serial", "--left-robot-ip",
            "--right-robot-ip", "--robot-port", "--image-width", "--image-height", "--camera-fps",
            "--model-image-size", "--state-dim", "--action-dim", "--action-row", "--joint-delta-limit-rad",
            "--speed-percent", "--target-tolerance-rad", "--state-stale-sec", "--response-to-command-sec",
            "--command-timeout-sec", "--watchdog-hz", "--controller-query-retries",
            "--controller-query-backoff-sec", "--settle-grace-sec", "--stable-window-sec",
            "--stable-span-threshold-rad", "--settle-timeout-sec", "--max-attempts", "--max-policy-requests",
            "--gripper-mode", "--return-to-start", "--artifact-root", "--operator-phrase",
            "--return-ack-phrase", "--log-level",
        ):
            self.assertIn(flag, help_text)
        args = self._args()
        validate_args(args)
        self.assertEqual(args.record_mode, "off")

    def test_operator_flags_are_exact_expected_text_not_authorization(self):
        args = self._args("staged-one-batch")
        self.assertEqual(args.operator_phrase, OPERATOR_PHRASE)
        self.assertEqual(args.return_ack_phrase, RETURN_ACK_PHRASE)
        args.operator_phrase = "YES"
        with self.assertRaises(MotionRejected):
            validate_args(args)

    def test_limits_and_retry_settings_fail_closed(self):
        for field, value in (
            ("joint_delta_limit_rad", 0.011), ("speed_percent", 6), ("target_tolerance_rad", 0.006),
            ("state_stale_sec", 0.251), ("controller_query_retries", 9),
            ("controller_query_backoff_sec", 0.051), ("stable_span_threshold_rad", 0.0011),
        ):
            args = self._args()
            setattr(args, field, value)
            with self.subTest(field=field), self.assertRaises(MotionRejected):
                validate_args(args)

    def test_compact_event_sink_does_not_retain_frame_state_or_operator_text(self):
        sink = CompactEventSink(str(uuid.uuid4()))
        sink.emit("camera", "frame", {"frame": np.zeros((480, 640, 3), dtype=np.uint8)})
        sink.emit("state", "state_sample", {"joints": list(range(7))})
        sink.emit("operator", "operator_input", {"field": "exact_motion_phrase", "value": OPERATOR_PHRASE})
        compact = sink.compact()
        rendered = repr(compact)
        self.assertNotIn(OPERATOR_PHRASE, rendered)
        self.assertNotIn("joints", rendered)
        self.assertEqual(compact["operator_facts"][0]["present"], True)

    def test_controller_minus_2_retries_read_only_then_recovers(self):
        sink = CompactEventSink(str(uuid.uuid4()))
        backend = RR006Backend({"robots": {}, "sources": {}}, sink, retries=10, backoff_sec=0.05)
        backend.arms = {"right": _FakeArm([
            {"return_code": -2}, {"return_code": -2}, {"return_code": 0, "system_error": 0},
        ])}
        with mock.patch("robot_client_runtime.rr_dep_006.time.sleep") as sleep:
            result = backend._call("right", "rm_get_controller_state")
        self.assertEqual(result["return_code"], 0)
        self.assertEqual(backend.controller_retry_counts["right"], 2)
        self.assertEqual(sleep.call_count, 2)

    def test_controller_retry_is_disabled_after_motion_starts(self):
        sink = CompactEventSink(str(uuid.uuid4()))
        backend = RR006Backend({"robots": {}, "sources": {}}, sink, retries=10, backoff_sec=0.05)
        backend.motion_started = True
        backend.arms = {"left": _FakeArm([{"return_code": -2}])}
        with mock.patch("robot_client_runtime.rr_dep_006.time.sleep") as sleep:
            result = backend._call("left", "rm_get_controller_state")
        self.assertEqual(result["return_code"], -2)
        sleep.assert_not_called()

    def test_capture_configuration_never_retries_unknown_setter_outcome(self):
        source = __import__("inspect").getsource(RR006Backend.start_capture_state)
        self.assertEqual(source.count("rm_set_realtime_push(selected)"), 1)
        self.assertIn("status == -2 and exact", source)
        self.assertIn("set_realtime_push_unresolved", source)
        self.assertIn("readback_exact_selected", source)

    def test_service_metadata_requires_exact_session_and_post_correction_commit(self):
        args = self._args()
        metadata = {
            "schema_id": SERVICE_SCHEMA_ID, "experiment_id": "RR-DEP-006", "run_id": "RR-DEP-006-RUN-001",
            "service_session_id": SERVICE_SESSION_ID, "bind": "127.0.0.1:8020", "port": 8020,
            "checkpoint_role": "integration_only", "checkpoint_selection_status": "not_selected",
            "stable_runtime_commit": SERVICE_RUNTIME_COMMIT, "task_id": "RR-DEP-006",
        }
        validate_service_metadata(metadata, args)
        for field in ("service_session_id", "stable_runtime_commit", "schema_id"):
            bad = dict(metadata)
            bad[field] = "wrong"
            with self.subTest(field=field), self.assertRaises(MotionRejected):
                validate_service_metadata(bad, args)

    def test_loopback_proxy_bypass_is_process_local(self):
        with mock.patch.dict("os.environ", {"HTTP_PROXY": "http://127.0.0.1:7897"}, clear=True):
            result = ensure_loopback_no_proxy()
            self.assertTrue(result["loopback_bypass_present"])
            self.assertIn("127.0.0.1", __import__("os").environ["NO_PROXY"])

    def test_policy_or_command_unknown_outcome_is_never_retried(self):
        args = self._args("no-motion")
        records = [{
            "event": "attempt_terminal", "mode": "no-motion", "status": "failed",
            "events": {"event_counters": {"policy.request_sending": 1}},
        }]
        with mock.patch("robot_client_runtime.rr_dep_006.prior_attempts", return_value=records):
            with self.assertRaisesRegex(MotionRejected, "prior_policy_request"):
                validate_attempt_boundary(args)
        with mock.patch("robot_client_runtime.rr_dep_006.prior_attempts", return_value=records), \
             mock.patch("robot_client_runtime.rr_dep_006.resolved_rejected_rollouts", return_value={"resolved-rollout"}):
            records[0]["rollout_id"] = "resolved-rollout"
            self.assertEqual(validate_attempt_boundary(args), 2)
        records[0]["events"]["event_counters"] = {"model.command_submitting": 1}
        with mock.patch("robot_client_runtime.rr_dep_006.prior_attempts", return_value=records):
            with self.assertRaisesRegex(MotionRejected, "prior_robot_command"):
                validate_attempt_boundary(args)

    def test_only_known_rejected_no_inference_can_be_recorded(self):
        request_id = str(uuid.uuid4())
        rollout_id = str(uuid.uuid4())
        payload = {
            "known": True,
            "request_id": request_id,
            "events": [{
                "event_type": "request_rejection",
                "outcome": "rejected_no_inference",
                "context": {"request_id": request_id, "rollout_id": rollout_id, "request_ordinal": 2,
                            "execution_intent": "one_projected_physical_batch"},
                "detail": {"reason": "rollout mismatch"},
            }],
        }
        with mock.patch("robot_client_runtime.rr_dep_006.resolved_rejected_rollouts", return_value={rollout_id}):
            result = record_request_outcome_resolution(payload)
        self.assertEqual(result["outcome"], "rejected_no_inference")
        bad = dict(payload)
        bad["events"] = [{**payload["events"][0], "outcome": "response_sent"}]
        with self.assertRaises(MotionRejected):
            record_request_outcome_resolution(bad)

    def test_operator_wait_runs_safety_checks_before_and_after_input(self):
        sink = CompactEventSink(str(uuid.uuid4()))
        checks = mock.Mock()
        with mock.patch("robot_client_runtime.rr_dep_006.select.select", side_effect=[([], [], []), ([object()], [], [])]), \
             mock.patch("robot_client_runtime.rr_dep_006.sys.stdin.readline", return_value=OPERATOR_PHRASE + "\n"):
            prompt_exact(
                sink, "AUTH: ", expected=OPERATOR_PHRASE,
                field="exact_motion_phrase", safety_check=checks,
            )
        self.assertEqual(checks.call_count, 3)
        self.assertEqual(sink.compact()["operator_facts"][0]["present"], True)


if __name__ == "__main__":
    unittest.main()
