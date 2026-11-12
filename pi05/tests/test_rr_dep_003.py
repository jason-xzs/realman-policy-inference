from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_client_runtime.rr_dep_003 import (
    resize_with_pad_rgb,
    validate_camera_binding,
    validate_camera_frame,
)
from robot_client_runtime.rr_dep_003_motion import (
    BoundedCommandExecutor,
    CommandSequence,
    MotionRejected,
    MotionThresholds,
    StateSample,
    build_arm_targets,
    check_trajectory_envelope,
    require_fresh_state,
    select_q7_direction,
    target_error,
    validate_absolute_target,
)


LOWER = [-2.0] * 6 + [-6.0]
UPPER = [2.0] * 6 + [6.0]
INITIAL = [0.0] * 7


class _Clock:
    def __init__(self) -> None:
        self.ns = 1_000_000_000

    def monotonic_ns(self) -> int:
        return self.ns

    def sleep(self, seconds: float) -> None:
        self.ns += int(seconds * 1e9)


class _Backend:
    def __init__(self, clock: _Clock, *, status: int = 0, update_target: bool = True) -> None:
        self.clock = clock
        self.status = status
        self.update_target = update_target
        self.calls = []
        self.stops = []
        self.joints = tuple(INITIAL)

    def command_absolute(self, side, target_rad, speed_percent):
        self.calls.append((side, tuple(target_rad), speed_percent))
        if self.update_target:
            self.joints = tuple(target_rad)
        return self.status

    def slow_stop(self, side):
        self.stops.append(side)
        return 0

    def sample(self, side="left"):
        return StateSample(side, self.joints, self.clock.monotonic_ns())


class RRDep003MotionTests(unittest.TestCase):
    def test_exact_command_order_duplicate_and_extra_command_rejection(self):
        sequence = CommandSequence()
        with self.assertRaisesRegex(MotionRejected, "wrong_command_order"):
            sequence.accept("left_q7_micro")
        sequence.accept("left_hold")
        with self.assertRaisesRegex(MotionRejected, "duplicate_command_id"):
            sequence.accept("left_hold")
        for command_id in (
            "left_q7_micro", "left_return", "right_hold", "right_q7_micro", "right_return"
        ):
            sequence.accept(command_id)
        self.assertIsNone(sequence.next_id)
        with self.assertRaisesRegex(MotionRejected, "wrong_command_order"):
            sequence.accept("right_return_again")

    def test_q7_direction_requires_verified_margin_and_uses_larger_margin(self):
        self.assertEqual(select_q7_direction(INITIAL, LOWER, UPPER), 1)
        initial = [0.0] * 6 + [5.99]
        self.assertEqual(select_q7_direction(initial, LOWER, UPPER), -1)
        with self.assertRaisesRegex(MotionRejected, "q7_verified_margin"):
            select_q7_direction([0.0] * 6 + [0.0], [-2.0] * 6 + [-0.01], [2.0] * 6 + [0.01])

    def test_target_builder_and_validation_accept_only_q7_delta(self):
        targets = build_arm_targets(INITIAL, 1)
        self.assertEqual(targets["hold"], tuple(INITIAL))
        self.assertAlmostEqual(targets["micro"][6], 0.01)
        self.assertEqual(targets["return"], tuple(INITIAL))
        validate_absolute_target(
            kind="micro", initial_rad=INITIAL, target_rad=targets["micro"],
            lower_rad=LOWER, upper_rad=UPPER, direction=1,
        )
        extra_joint = list(targets["micro"])
        extra_joint[2] = 0.0001
        with self.assertRaisesRegex(MotionRejected, "extra_joint_target:q3"):
            validate_absolute_target(
                kind="micro", initial_rad=INITIAL, target_rad=extra_joint,
                lower_rad=LOWER, upper_rad=UPPER, direction=1,
            )

    def test_target_rejects_limit_delta_direction_nonfinite_and_speed(self):
        cases = []
        over_delta = [0.0] * 6 + [0.0100001]
        cases.append(("q7_delta_out_of_contract", over_delta, LOWER, UPPER, 1, MotionThresholds()))
        cases.append(("q7_wrong_target_direction", [0.0] * 6 + [-0.01], LOWER, UPPER, 1, MotionThresholds()))
        cases.append(("target_out_of_limit:q7", [0.0] * 6 + [0.01], LOWER, [2.0] * 6 + [0.005], 1, MotionThresholds()))
        cases.append(("target_nonfinite", [0.0] * 6 + [math.nan], LOWER, UPPER, 1, MotionThresholds()))
        cases.append(("speed_above_5_percent", [0.0] * 6 + [0.01], LOWER, UPPER, 1, MotionThresholds(speed_percent=6)))
        for error, target, lower, upper, direction, thresholds in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(MotionRejected, error):
                    validate_absolute_target(
                        kind="micro", initial_rad=INITIAL, target_rad=target,
                        lower_rad=lower, upper_rad=upper, direction=direction,
                        thresholds=thresholds,
                    )

    def test_stale_sdk_alarm_controller_and_nonfinite_state_rejection(self):
        now = 2_000_000_000
        invalid = (
            (StateSample("left", tuple(INITIAL), 1_700_000_000), "stale_state"),
            (StateSample("left", tuple(INITIAL), now, sdk_status=-2), "state_sdk_status"),
            (StateSample("left", tuple(INITIAL), now, arm_errors=("17",)), "arm_alarm"),
            (StateSample("left", tuple(INITIAL), now, controller_error=5), "controller_alarm"),
            (StateSample("left", tuple([0.0] * 6 + [math.inf]), now), "state_nonfinite"),
        )
        for sample, error in invalid:
            with self.subTest(error=error):
                with self.assertRaisesRegex(MotionRejected, error):
                    require_fresh_state(sample, now_ns=now, stale_timeout_s=0.25)

    def test_trajectory_envelope_hold_non_q7_q7_and_wrong_direction(self):
        hold = [0.0] * 7
        hold[0] = 0.0030001
        with self.assertRaisesRegex(MotionRejected, "hold_motion_exceeded:q1"):
            check_trajectory_envelope(hold, INITIAL, kind="hold", direction=1)
        non_q7 = [0.0] * 7
        non_q7[4] = 0.0050001
        with self.assertRaisesRegex(MotionRejected, "unexpected_non_q7_motion:q5"):
            check_trajectory_envelope(non_q7, INITIAL, kind="micro", direction=1)
        with self.assertRaisesRegex(MotionRejected, "q7_displacement_exceeded"):
            check_trajectory_envelope([0.0] * 6 + [0.0150001], INITIAL, kind="micro", direction=1)
        with self.assertRaisesRegex(MotionRejected, "q7_wrong_observed_direction"):
            check_trajectory_envelope([0.0] * 6 + [-0.001], INITIAL, kind="micro", direction=1)

    def test_target_and_return_tolerance_is_max_all_joint_error(self):
        self.assertEqual(target_error([0.005] * 7, INITIAL), 0.005)
        self.assertGreater(target_error([0.0] * 6 + [0.005001], INITIAL), 0.005)

    def test_executor_submits_once_at_5_percent_and_observes_hold(self):
        clock = _Clock()
        backend = _Backend(clock)
        events = []
        executor = BoundedCommandExecutor(
            backend, backend.sample, lambda: True,
            monotonic_ns=clock.monotonic_ns, sleep=clock.sleep,
            event_sink=lambda kind, payload: events.append((kind, payload)),
        )
        executor.execute(
            command_id="left_hold", side="left", kind="hold", initial_rad=INITIAL,
            target_rad=INITIAL, lower_rad=LOWER, upper_rad=UPPER, direction=1,
        )
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][2], 5)
        self.assertIn("sdk_acceptance", [kind for kind, _ in events])
        self.assertIn("observed_target", [kind for kind, _ in events])

    def test_executor_sdk_rejection_timeout_camera_loss_and_invalid_feedback_stop(self):
        scenarios = []
        clock = _Clock()
        scenarios.append((
            "command_sdk_rejection", clock,
            _Backend(clock, status=1), lambda backend: backend.sample(), lambda: True,
            "left_hold", "hold", INITIAL, False,
        ))
        clock = _Clock()
        scenarios.append((
            "command_timeout", clock,
            _Backend(clock, update_target=False), lambda backend: StateSample("left", backend.joints, backend.clock.monotonic_ns()), lambda: True,
            "left_q7_micro", "micro", [0.0] * 6 + [0.01], True,
        ))
        clock = _Clock()
        camera_calls = {"count": 0}
        def camera_health():
            camera_calls["count"] += 1
            return camera_calls["count"] < 3
        scenarios.append((
            "camera_loss", clock,
            _Backend(clock, update_target=False), lambda backend: StateSample("left", backend.joints, backend.clock.monotonic_ns()), camera_health,
            "left_hold", "hold", INITIAL, False,
        ))
        clock = _Clock()
        feedback_calls = {"count": 0}
        def feedback_with_alarm(backend):
            feedback_calls["count"] += 1
            return StateSample(
                "left", backend.joints, backend.clock.monotonic_ns(),
                controller_error=9 if feedback_calls["count"] > 1 else 0,
            )
        scenarios.append((
            "controller_alarm", clock,
            _Backend(clock), feedback_with_alarm, lambda: True,
            "left_hold", "hold", INITIAL, False,
        ))
        for error, clock, backend, sample_factory, camera, command_id, kind, target, preadvance in scenarios:
            with self.subTest(error=error):
                executor = BoundedCommandExecutor(
                    backend, lambda _side, b=backend, f=sample_factory: f(b), camera,
                    monotonic_ns=clock.monotonic_ns, sleep=clock.sleep,
                )
                if preadvance:
                    executor.sequence.accept("left_hold")
                with self.assertRaisesRegex(MotionRejected, error):
                    executor.execute(
                        command_id=command_id, side="left", kind=kind, initial_rad=INITIAL,
                        target_rad=target, lower_rad=LOWER, upper_rad=UPPER, direction=1,
                    )
                self.assertLessEqual(len(backend.calls), 1)
                self.assertEqual(backend.stops, ["left"])

    def test_executor_rejects_missing_state_and_unhealthy_camera_before_submit(self):
        clock = _Clock()
        backend = _Backend(clock)
        for latest, healthy, error in (
            (lambda _side: None, lambda: True, "missing_state_before_command"),
            (backend.sample, lambda: False, "camera_unhealthy_before_command"),
        ):
            with self.subTest(error=error):
                executor = BoundedCommandExecutor(
                    backend, latest, healthy, monotonic_ns=clock.monotonic_ns, sleep=clock.sleep,
                )
                with self.assertRaisesRegex(MotionRejected, error):
                    executor.execute(
                        command_id="left_hold", side="left", kind="hold", initial_rad=INITIAL,
                        target_rad=INITIAL, lower_rad=LOWER, upper_rad=UPPER, direction=1,
                    )
        self.assertEqual(backend.calls, [])

    def test_resize_with_pad_is_rgb_uint8_224(self):
        import numpy as np

        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr[:, :, 0] = 11
        bgr[:, :, 1] = 22
        bgr[:, :, 2] = 33
        result = resize_with_pad_rgb(bgr)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result[112, 112].tolist(), [33, 22, 11])
        self.assertEqual(result[0, 0].tolist(), [0, 0, 0])

    def test_fake_camera_binding_frame_timestamp_and_native_gap_rules(self):
        import numpy as np

        configured = {"head": "h", "left_wrist": "l", "right_wrist": "r"}
        enumerated = [
            {"serial": "r", "name": "right"},
            {"serial": "h", "name": "head"},
            {"serial": "l", "name": "left"},
        ]
        self.assertEqual(validate_camera_binding(configured, enumerated), configured)
        with self.assertRaisesRegex(MotionRejected, "configured_camera_serials_not_enumerated"):
            validate_camera_binding(configured, enumerated[:2])
        with self.assertRaisesRegex(MotionRejected, "camera_binding_roles_must_be_exact"):
            validate_camera_binding({"head": "h"}, enumerated)
        with self.assertRaisesRegex(MotionRejected, "camera_binding_serials_must_be_distinct"):
            validate_camera_binding(
                {"head": "h", "left_wrist": "h", "right_wrist": "r"}, enumerated
            )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        validate_camera_frame(
            "head", frame, frame_number=11, sdk_timestamp_ms=1.5,
            receive_monotonic_ns=200, previous_frame_number=10,
            previous_receive_monotonic_ns=100, require_contiguous=True,
        )
        invalid_cases = (
            ({"frame": frame[:, :-1], "frame_number": 11, "sdk_timestamp_ms": 1.5,
              "receive_monotonic_ns": 200}, "invalid_live_frame"),
            ({"frame": frame, "frame_number": 12, "sdk_timestamp_ms": 1.5,
              "receive_monotonic_ns": 200, "previous_frame_number": 10,
              "previous_receive_monotonic_ns": 100, "require_contiguous": True}, "native_frame_gap"),
            ({"frame": frame, "frame_number": 11, "sdk_timestamp_ms": math.nan,
              "receive_monotonic_ns": 200}, "invalid_camera_timestamp_or_number"),
            ({"frame": frame, "frame_number": 11, "sdk_timestamp_ms": 1.5,
              "receive_monotonic_ns": 100, "previous_receive_monotonic_ns": 100}, "nonmonotonic"),
        )
        for kwargs, error in invalid_cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(MotionRejected, error):
                    validate_camera_frame("head", **kwargs)

    def test_live_module_has_no_policy_or_gripper_execution_api(self):
        import robot_client_runtime.rr_dep_003 as live

        source = Path(live.__file__).read_text(encoding="utf-8")
        self.assertNotIn("WebsocketClientPolicy", source)
        self.assertNotIn("rm_set_gripper", source)
        self.assertNotIn("infer(", source)


if __name__ == "__main__":
    unittest.main()
