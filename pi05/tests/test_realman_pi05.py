from __future__ import annotations

from contextlib import nullcontext
import copy
import io
import inspect
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest import mock
import threading
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_client_runtime.realman_pi05 import (
    DEFAULT_CONFIG,
    DEFAULT_PRESET,
    PresetBackend,
    RuntimeEvents,
    UserExit,
    adapt_action_chunk,
    adapt_observation_state,
    build_parser,
    capture_observation,
    execute_dual_joint_targets,
    execute_joint_target,
    gripper_targets_from_row,
    load_config,
    main,
    operator_gate,
    run_live,
    validate_service_metadata,
)
from robot_client_runtime.rr_dep_003_motion import MotionRejected, StateSample


class _Clock:
    def __init__(self) -> None:
        self.ns = time.monotonic_ns()

    def monotonic_ns(self) -> int:
        return self.ns

    def monotonic(self) -> float:
        return self.ns / 1e9

    def sleep(self, seconds: float) -> None:
        self.ns += int(seconds * 1e9)


class _Rig:
    def healthy(self, _stale: float) -> bool:
        return True


class _Poller:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.joints = {"left": (0.0,) * 7, "right": (0.0,) * 7}
        self._condition = nullcontext()
        self._errors = {}

    def latest(self, side: str) -> StateSample:
        return StateSample(
            side=side,
            joints_rad=self.joints[side],
            receive_monotonic_ns=self.clock.ns,
            sdk_status=0,
            arm_errors=(),
            controller_error=0,
        )


class _Backend:
    def __init__(self, poller: _Poller, *, fail: bool = False) -> None:
        self.poller = poller
        self.fail = fail
        self.command_calls = 0
        self.dual_command_calls = 0
        self.stop_calls = 0
        self.gripper_calls = []

    def command_absolute(self, side, target, _speed):
        self.command_calls += 1
        if self.fail:
            raise RuntimeError("unknown")
        self.poller.joints[side] = tuple(target)
        return 0

    def command_dual_absolute(self, targets, _speed):
        self.dual_command_calls += 1
        if self.fail:
            raise RuntimeError("unknown")
        for side in ("left", "right"):
            self.poller.joints[side] = tuple(targets[side])
        return {"right": 0, "left": 0}

    def slow_stop(self, _side):
        self.stop_calls += 1
        return 0

    def command_gripper(self, side, position):
        self.gripper_calls.append((side, position))
        return 0


class _Policy:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.infer_calls = 0
        self.close_calls = 0

    def infer(self, _observation):
        self.infer_calls += 1
        if self.fail:
            raise RuntimeError("unknown")
        return {"actions": np.zeros((50, 28), dtype=np.float32)}

    def close(self):
        self.close_calls += 1


class RealManPi05Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(DEFAULT_CONFIG, DEFAULT_PRESET)
        # Preserve coverage for the supported 28D adapter without exposing a
        # stale historical preset through the production registry.
        self.config["observation"].update(
            state_dimension=28,
            state_order="left_joints_gripper_eef_then_right_joints_gripper_eef",
        )
        self.config["action"]["allowed_source_dimensions"] = [28]
        self.config_16d = load_config(DEFAULT_CONFIG, "pi05-organize-15000")

    def test_active_cli_only_selects_config_and_preset(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.preset, DEFAULT_PRESET)
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
            if option not in ("-h", "--help")
        }
        self.assertEqual(
            options,
            {"--config", "--preset"},
        )
        for removed in ("--execute", "--no-motion", "--max-replans", "--prompt", "--log-level"):
            with self.subTest(removed=removed), self.assertRaises(SystemExit):
                parser.parse_args([removed])

    def test_main_runs_selected_preset_without_cli_execution_flags(self):
        with mock.patch(
            "robot_client_runtime.realman_pi05.run_live", return_value={"status": "mock_completed"}
        ) as live:
            self.assertEqual(main([]), 0)
        live.assert_called_once()

    def test_ctrl_c_is_reported_as_normal_clean_exit(self):
        with mock.patch("robot_client_runtime.realman_pi05.run_live", side_effect=KeyboardInterrupt):
            self.assertEqual(main([]), 0)

    def test_preset_contains_hardware_action_sync_motion_and_gripper_settings(self):
        self.assertEqual(self.config["_preset"], "pi05-sugar-15000")
        self.assertEqual(
            (
                self.config["cameras"]["width"],
                self.config["cameras"]["height"],
                self.config["cameras"]["fps"],
                self.config["cameras"]["format"],
            ),
            (640, 480, 30, "bgr8"),
        )
        self.assertEqual(self.config["execution"], {
            "mode": "execute", "max_replans": 1000, "confirmation": "enter", "log_level": "INFO",
        })
        self.assertEqual(self.config["motion"]["speed_percent"], 20)
        self.assertEqual(self.config["motion"]["command_api"], "rm_movej_canfd")
        self.assertEqual(self.config["motion"]["trajectory_mode"], 2)
        self.assertEqual(self.config["motion"]["smoothing"], 50)
        self.assertEqual(self.config["motion"]["max_joint_step_rad"], 2.0)
        self.assertEqual(self.config["motion"]["target_tolerance_rad"], 0.01)
        self.assertEqual(self.config["runtime"]["state_stale_sec"], 1)
        self.assertEqual(self.config["runtime"]["max_sync_skew_sec"], 1)
        self.assertEqual(self.config["action"]["row_index"], 0)
        self.assertEqual(self.config["action"]["actions_per_chunk"], 20)
        self.assertEqual(self.config["action"]["target_mode"], "direct_absolute")
        self.assertEqual(self.config["action"]["command_order"], ["right", "left"])
        self.assertIs(self.config["motion"]["software_joint_step_limit_enabled"], True)
        self.assertEqual(self.config["gripper"], {
            "mode": "binary_threshold",
            "commands_enabled": True,
            "close_threshold": 0.4,
            "open_threshold": 0.7,
            "value_tolerance": 0.05,
            "close_position": 1,
            "open_position": 1000,
        })

    def test_gripper_threshold_maps_normalized_left_first_values(self):
        row = np.zeros(16, dtype=np.float32)
        row[7] = 0.3
        row[15] = 0.7
        targets = gripper_targets_from_row(row, self.config_16d["gripper"])
        self.assertEqual(targets["left"], {"state": "closed", "position": 1})
        self.assertEqual(targets["right"], {"state": "open", "position": 1000})
        row[7] = 0.5
        row[15] = 0.5
        held = gripper_targets_from_row(row, self.config_16d["gripper"], targets)
        self.assertEqual(held, targets)
        row[7] = -0.009575845
        row[15] = 1.01
        targets = gripper_targets_from_row(row, self.config_16d["gripper"])
        self.assertEqual(targets["left"], {"state": "closed", "position": 1})
        self.assertEqual(targets["right"], {"state": "open", "position": 1000})
        row[7] = -0.051
        with self.assertRaisesRegex(MotionRejected, "gripper_value_out_of_normalized_range"):
            gripper_targets_from_row(row, self.config_16d["gripper"])

    def test_preset_backend_switches_protocol_before_gripper_command(self):
        backend = object.__new__(PresetBackend)
        backend.locks = {"left": threading.Lock(), "right": threading.Lock()}
        backend.arms = {"left": mock.Mock(), "right": mock.Mock()}
        backend.sdk_global_lock = threading.Lock()
        backend.config = {"runtime": {"sdk_lock_timeout_sec": 0.01}}
        backend.events = mock.Mock()
        backend.motion_started = False
        backend.arms["left"].rm_set_tool_rs485_mode.return_value = 0
        backend.arms["left"].rm_set_gripper_position.return_value = 0

        self.assertEqual(backend.command_gripper("left", 1), 0)
        backend.arms["left"].rm_set_tool_rs485_mode.assert_called_once_with(2, 115200)
        backend.arms["left"].rm_set_gripper_position.assert_called_once_with(1, False, 0)
        self.assertIs(backend.motion_started, True)

        backend.arms["right"].rm_set_tool_rs485_mode.return_value = -4
        with self.assertRaisesRegex(MotionRejected, "gripper_protocol_mode_rejected:right:-4"):
            backend.command_gripper("right", 1000)
        backend.arms["right"].rm_set_gripper_position.assert_not_called()

    def test_preset_backend_sdk_lock_wait_is_bounded(self):
        backend = object.__new__(PresetBackend)
        backend.locks = {"left": threading.Lock(), "right": threading.Lock()}
        backend.arms = {"left": mock.Mock(), "right": mock.Mock()}
        backend.sdk_global_lock = threading.Lock()
        backend.config = {"runtime": {"sdk_lock_timeout_sec": 0.01}}
        backend.events = mock.Mock()
        backend.motion_started = False
        backend.locks["left"].acquire()
        try:
            with self.assertRaisesRegex(
                MotionRejected, r"robot_sdk_side_lock_timeout:joint_command:left:0.01"
            ):
                backend.command_absolute("left", [0.0] * 7, 5)
        finally:
            backend.locks["left"].release()
        backend.arms["left"].rm_movej.assert_not_called()

    def test_preset_backend_serializes_reads_and_commands_globally(self):
        backend = object.__new__(PresetBackend)
        backend.locks = {"left": threading.Lock(), "right": threading.Lock()}
        backend.sdk_global_lock = threading.Lock()
        backend.config = {"runtime": {"sdk_lock_timeout_sec": 0.01}}
        backend.events = mock.Mock()
        backend.motion_started = False
        backend.arms = {"left": mock.Mock(), "right": mock.Mock()}
        backend.sdk_global_lock.acquire()
        try:
            with self.assertRaisesRegex(
                MotionRejected, r"robot_sdk_global_lock_timeout:joint_command:left:0.01"
            ):
                backend.command_absolute("left", [0.0] * 7, 5)
        finally:
            backend.sdk_global_lock.release()
        backend.arms["left"].rm_movej.assert_not_called()

    def test_preset_backend_reuses_fastwam_canfd_command(self):
        backend = object.__new__(PresetBackend)
        backend.locks = {"left": threading.Lock(), "right": threading.Lock()}
        backend.sdk_global_lock = threading.Lock()
        backend.config = {
            "runtime": {"sdk_lock_timeout_sec": 0.01},
            "motion": {"follow": False, "trajectory_mode": 2, "smoothing": 50},
        }
        backend.events = mock.Mock()
        backend.motion_started = False
        backend.arms = {"left": mock.Mock(), "right": mock.Mock()}
        backend.arms["left"].rm_movej_canfd.return_value = 0
        self.assertEqual(backend.command_absolute("left", [0.0] * 7, 20), 0)
        backend.arms["left"].rm_movej_canfd.assert_called_once_with(
            [0.0] * 7, False, trajectory_mode=2, radio=50
        )
        backend.arms["left"].rm_movej.assert_not_called()

    def test_preset_backend_submits_dual_canfd_pair_without_read_interleave(self):
        backend = object.__new__(PresetBackend)
        backend.locks = {"left": threading.Lock(), "right": threading.Lock()}
        backend.sdk_global_lock = threading.Lock()
        backend.config = {
            "runtime": {"sdk_lock_timeout_sec": 0.01},
            "motion": {"follow": False, "trajectory_mode": 2, "smoothing": 50},
            "action": {"command_order": ["right", "left"]},
        }
        backend.events = mock.Mock()
        backend.motion_started = False
        backend.arms = {"left": mock.Mock(), "right": mock.Mock()}
        calls = mock.Mock()
        calls.attach_mock(backend.arms["right"].rm_movej_canfd, "right")
        calls.attach_mock(backend.arms["left"].rm_movej_canfd, "left")
        backend.arms["right"].rm_movej_canfd.return_value = 0
        backend.arms["left"].rm_movej_canfd.return_value = 0
        statuses = backend.command_dual_absolute(
            {"left": [0.0] * 7, "right": [0.0] * 7}, 20
        )
        self.assertEqual(statuses, {"right": 0, "left": 0})
        self.assertEqual(
            calls.mock_calls,
            [
                mock.call.right([0.0] * 7, False, trajectory_mode=2, radio=50),
                mock.call.left([0.0] * 7, False, trajectory_mode=2, radio=50),
            ],
        )

        backend.arms["right"].rm_movej_canfd.reset_mock()
        backend.arms["left"].rm_movej_canfd.reset_mock()
        backend.arms["right"].rm_movej_canfd.return_value = 1
        statuses = backend.command_dual_absolute(
            {"left": [0.0] * 7, "right": [0.0] * 7}, 20
        )
        self.assertEqual(statuses, {"right": 1})
        backend.arms["left"].rm_movej_canfd.assert_not_called()

    def test_16d_preset_inherits_defaults_and_restricts_native_left_first_output(self):
        self.assertEqual(self.config_16d["cameras"], self.config["cameras"])
        self.assertEqual(self.config_16d["robots"], self.config["robots"])
        self.assertEqual(self.config_16d["policy"]["host"], "127.0.0.1")
        self.assertEqual(self.config_16d["policy"]["port"], 18200)
        self.assertEqual(
            self.config_16d["policy"]["expected_config"],
            "pi05_realman_joint_16d_step15000_ema",
        )
        self.assertEqual(
            self.config_16d["policy"]["expected_checkpoint"],
            "/share/jhk/phywam-mope-real/checkpoints/finetuned/pi05/"
            "RR-BL-008/RR-BL-008-RUN-001/15000",
        )
        self.assertEqual(self.config_16d["action"]["allowed_source_dimensions"], [16])
        self.assertEqual(self.config_16d["observation"]["state_dimension"], 16)
        self.assertEqual(self.config["action"]["allowed_source_dimensions"], [28])
        self.assertEqual(self.config["observation"]["state_dimension"], 28)
        self.assertEqual(
            self.config_16d["action"]["canonical_order"],
            "left_q1_q7_gripper_then_right_q1_q7_gripper",
        )

        native = np.zeros((50, 16), dtype=np.float32)
        np.testing.assert_array_equal(
            adapt_action_chunk(native, self.config_16d["action"]), native
        )
        with self.assertRaises(MotionRejected):
            adapt_action_chunk(np.zeros((50, 28), dtype=np.float32), self.config_16d["action"])

    def test_16d_preset_maps_captured_state_before_policy_request(self):
        captured = np.arange(28, dtype=np.float32)
        expected = captured[list(range(8)) + list(range(14, 22))]
        mapped = adapt_observation_state(captured, self.config_16d["observation"])
        np.testing.assert_array_equal(mapped, expected)
        self.assertEqual(mapped.shape, (16,))
        self.assertEqual(mapped.dtype, np.float32)
        self.assertTrue(mapped.flags.c_contiguous)

        unchanged = adapt_observation_state(captured, self.config["observation"])
        np.testing.assert_array_equal(unchanged, captured)

    def test_unknown_preset_is_rejected_before_live_access(self):
        with mock.patch("robot_client_runtime.realman_pi05.run_live") as live:
            self.assertEqual(main(["--preset", "missing"]), 1)
        live.assert_not_called()

    def test_enter_confirms_and_ctrl_c_exits_confirmation(self):
        safety_check = mock.Mock()
        with mock.patch("robot_client_runtime.realman_pi05.select.select", return_value=([object()], [], [])), \
             mock.patch("robot_client_runtime.realman_pi05.sys.stdin", io.StringIO("\n")):
            operator_gate(stop_event=threading.Event(), safety_check=safety_check)
        self.assertEqual(safety_check.call_count, 2)

        stopped = threading.Event()
        stopped.set()
        with self.assertRaises(UserExit):
            operator_gate(stop_event=stopped, safety_check=mock.Mock())

    def test_action_mapping_accepts_only_finite_float32_16d_or_28d(self):
        source = np.arange(2 * 28, dtype=np.float32).reshape(2, 28)
        mapped = adapt_action_chunk(source, self.config["action"])
        self.assertEqual(mapped.shape, (2, 16))
        np.testing.assert_array_equal(mapped[0], source[0, list(range(8)) + list(range(14, 22))])
        native = np.zeros((3, 16), dtype=np.float32)
        self.assertEqual(adapt_action_chunk(native, self.config_16d["action"]).shape, (3, 16))
        for invalid in (
            np.zeros((1, 32), dtype=np.float32),
            np.zeros((1, 28), dtype=np.float64),
            np.full((1, 28), np.nan, dtype=np.float32),
        ):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(MotionRejected):
                adapt_action_chunk(invalid, self.config["action"])

    def test_server_identity_and_request_budget_are_validated(self):
        metadata = {
            "config": self.config["policy"]["expected_config"],
            "checkpoint": self.config["policy"]["expected_checkpoint"],
            "maximum_unique_matching_Client_requests": 3,
            "state_dimension": 28,
            "action_dimension": 28,
        }
        result = validate_service_metadata(metadata, self.config, request_budget=3)
        self.assertEqual(result["maximum_requests"], 3)
        for field in ("config", "checkpoint"):
            bad = dict(metadata)
            bad[field] = "wrong"
            with self.subTest(field=field), self.assertRaises(MotionRejected):
                validate_service_metadata(bad, self.config, request_budget=1)
        with self.assertRaisesRegex(MotionRejected, "server_request_budget_too_small"):
            validate_service_metadata(metadata, self.config, request_budget=4)
        invalid_dimension = dict(metadata, action_dimension="unknown")
        with self.assertRaisesRegex(MotionRejected, "server_action_dimension_invalid"):
            validate_service_metadata(invalid_dimension, self.config, request_budget=1)
        wrong_state_dimension = dict(metadata, state_dimension=16)
        with self.assertRaisesRegex(MotionRejected, "server_state_dimension_mismatch"):
            validate_service_metadata(wrong_state_dimension, self.config, request_budget=1)

        metadata_16d = {
            "config": self.config_16d["policy"]["expected_config"],
            "checkpoint": self.config_16d["policy"]["expected_checkpoint"],
            "state_dimension": 16,
            "action_dimension": 16,
        }
        result_16d = validate_service_metadata(metadata_16d, self.config_16d, request_budget=1)
        self.assertEqual(result_16d["state_dimension"], 16)
        self.assertEqual(result_16d["action_dimension"], 16)

    def test_observation_sync_skew_is_checked_across_all_five_sources(self):
        now = time.monotonic_ns()
        images = {
            key: np.zeros((224, 224, 3), dtype=np.uint8)
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        }
        camera_metadata = {
            role: {"receive_monotonic_ns": now}
            for role in ("head", "left_wrist", "right_wrist")
        }
        backend = mock.Mock()
        backend.capture_observation_state.return_value = (
            np.zeros(28, dtype=np.float32),
            now,
            {
                "left": {"receive_monotonic_ns": now},
                "right": {"receive_monotonic_ns": now + 110_000_000},
            },
        )
        config = copy.deepcopy(self.config)
        config["runtime"]["max_sync_skew_sec"] = 0.1
        with mock.patch(
            "robot_client_runtime.realman_pi05.latest_camera_observation",
            return_value=(images, now, camera_metadata),
        ), self.assertRaisesRegex(MotionRejected, "observation_sync_skew_exceeded"):
            capture_observation(_Rig(), backend, config, SimpleNamespace(
                model_image_size=224, state_stale_sec=0.25,
            ), "test")

    def test_capture_observation_sends_16d_state_for_16d_preset(self):
        now = time.monotonic_ns()
        images = {
            key: np.zeros((224, 224, 3), dtype=np.uint8)
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        }
        camera_metadata = {
            role: {"receive_monotonic_ns": now}
            for role in ("head", "left_wrist", "right_wrist")
        }
        captured = np.arange(28, dtype=np.float32)
        backend = mock.Mock()
        backend.capture_observation_state.return_value = (
            captured,
            now,
            {
                "left": {"receive_monotonic_ns": now},
                "right": {"receive_monotonic_ns": now},
            },
        )
        with mock.patch(
            "robot_client_runtime.realman_pi05.latest_camera_observation",
            return_value=(images, now, camera_metadata),
        ):
            observation, _ = capture_observation(
                _Rig(),
                backend,
                self.config_16d,
                SimpleNamespace(model_image_size=224, state_stale_sec=100),
                "test",
            )
        np.testing.assert_array_equal(
            observation["state"], captured[list(range(8)) + list(range(14, 22))]
        )

    def test_robot_state_is_frozen_before_camera_image_preprocessing(self):
        now = time.monotonic_ns()
        order = []
        images = {
            key: np.zeros((224, 224, 3), dtype=np.uint8)
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        }
        camera_metadata = {
            role: {"receive_monotonic_ns": now}
            for role in ("head", "left_wrist", "right_wrist")
        }
        backend = mock.Mock()

        def capture_state():
            order.append("robot_state")
            return (
                np.zeros(28, dtype=np.float32),
                now,
                {
                    "left": {"receive_monotonic_ns": now},
                    "right": {"receive_monotonic_ns": now},
                },
            )

        def process_images(_rig, _args):
            order.append("camera_snapshot_and_preprocess")
            return images, now, camera_metadata

        backend.capture_observation_state.side_effect = capture_state
        with mock.patch(
            "robot_client_runtime.realman_pi05.latest_camera_observation",
            side_effect=process_images,
        ):
            capture_observation(
                _Rig(),
                backend,
                self.config_16d,
                SimpleNamespace(model_image_size=224, state_stale_sec=100),
                "test",
            )
        self.assertEqual(order, ["robot_state", "camera_snapshot_and_preprocess"])

    def _args(self):
        return SimpleNamespace(
            state_stale_sec=0.25,
            target_tolerance_rad=0.005,
            speed_percent=5,
            command_timeout_sec=5.0,
            max_joint_step_rad=0.15,
        )

    def test_target_already_in_tolerance_sends_no_command(self):
        clock = _Clock()
        poller = _Poller(clock)
        backend = _Backend(poller)
        result = execute_joint_target(
            backend=backend,
            poller=poller,
            rig=_Rig(),
            events=RuntimeEvents(),
            args=self._args(),
            stop_event=threading.Event(),
            side="left",
            target=[0.001] * 7,
            lower=[-1.0] * 7,
            upper=[1.0] * 7,
        )
        self.assertEqual(result["status"], "already_within_tolerance")
        self.assertEqual(backend.command_calls, 0)

    def test_command_is_submitted_once_and_observed(self):
        clock = _Clock()
        poller = _Poller(clock)
        backend = _Backend(poller)
        with mock.patch("robot_client_runtime.realman_pi05.time.monotonic_ns", side_effect=clock.monotonic_ns), \
             mock.patch("robot_client_runtime.realman_pi05.time.monotonic", side_effect=clock.monotonic), \
             mock.patch("robot_client_runtime.realman_pi05.time.sleep", side_effect=clock.sleep):
            result = execute_joint_target(
                backend=backend,
                poller=poller,
                rig=_Rig(),
                events=RuntimeEvents(),
                args=self._args(),
                stop_event=threading.Event(),
                side="right",
                target=[0.01] * 7,
                lower=[-1.0] * 7,
                upper=[1.0] * 7,
            )
        self.assertEqual(result["status"], "observed")
        self.assertEqual(backend.command_calls, 1)
        self.assertEqual(backend.stop_calls, 0)

    def test_dual_command_is_submitted_as_one_pair_and_both_targets_are_observed(self):
        clock = _Clock()
        poller = _Poller(clock)
        backend = _Backend(poller)
        limits = {
            side: {"lower": [-1.0] * 7, "upper": [1.0] * 7}
            for side in ("left", "right")
        }
        with mock.patch("robot_client_runtime.realman_pi05.time.monotonic_ns", side_effect=clock.monotonic_ns), \
             mock.patch("robot_client_runtime.realman_pi05.time.monotonic", side_effect=clock.monotonic), \
             mock.patch("robot_client_runtime.realman_pi05.time.sleep", side_effect=clock.sleep):
            result = execute_dual_joint_targets(
                backend=backend,
                poller=poller,
                rig=_Rig(),
                events=RuntimeEvents(),
                args=self._args(),
                stop_event=threading.Event(),
                targets={"left": [0.01] * 7, "right": [0.02] * 7},
                limits=limits,
                command_order=["right", "left"],
            )
        self.assertEqual(backend.dual_command_calls, 1)
        self.assertEqual([item["side"] for item in result], ["right", "left"])
        self.assertEqual(poller.joints["left"], (0.01,) * 7)
        self.assertEqual(poller.joints["right"], (0.02,) * 7)

    def test_direct_target_is_not_projected_but_is_step_limited(self):
        clock = _Clock()
        poller = _Poller(clock)
        backend = _Backend(poller)
        with mock.patch("robot_client_runtime.realman_pi05.time.monotonic_ns", side_effect=clock.monotonic_ns), \
             mock.patch("robot_client_runtime.realman_pi05.time.monotonic", side_effect=clock.monotonic), \
             mock.patch("robot_client_runtime.realman_pi05.time.sleep", side_effect=clock.sleep):
            with self.assertRaisesRegex(MotionRejected, "pre_send_joint_step_exceeded:left"):
                execute_joint_target(
                    backend=backend,
                    poller=poller,
                    rig=_Rig(),
                    events=RuntimeEvents(),
                    args=self._args(),
                    stop_event=threading.Event(),
                    side="left",
                    target=[0.5] * 7,
                    lower=[-1.0] * 7,
                    upper=[1.0] * 7,
                )
        self.assertEqual(backend.command_calls, 0)

    def test_unknown_command_outcome_is_not_retried(self):
        clock = _Clock()
        poller = _Poller(clock)
        backend = _Backend(poller, fail=True)
        with self.assertRaisesRegex(MotionRejected, "command_acceptance_unknown"):
            execute_joint_target(
                backend=backend,
                poller=poller,
                rig=_Rig(),
                events=RuntimeEvents(),
                args=self._args(),
                stop_event=threading.Event(),
                side="right",
                target=[0.01] * 7,
                lower=[-1.0] * 7,
                upper=[1.0] * 7,
            )
        self.assertEqual(backend.command_calls, 1)
        self.assertEqual(backend.stop_calls, 1)

    def test_no_motion_live_path_uses_one_request_and_zero_commands(self):
        config = copy.deepcopy(self.config)
        config["execution"]["mode"] = "no_motion"
        clock = _Clock()
        poller = _Poller(clock)
        backend = _Backend(poller)
        policy = _Policy()
        gate = {"arms": {}}
        with mock.patch(
            "robot_client_runtime.realman_pi05.open_live_stack",
            return_value=(backend, _Rig(), poller, gate),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.open_policy",
            return_value=(policy, {"config": "test"}),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.capture_observation",
            return_value=({"state": np.zeros(28, dtype=np.float32)}, {"skew_sec": 0.0}),
        ), mock.patch("robot_client_runtime.realman_pi05.close_live_stack"):
            result = run_live(
                config,
                stop_event=threading.Event(),
            )
        self.assertEqual(result["status"], "no_motion_pass")
        self.assertEqual(policy.infer_calls, 1)
        self.assertEqual(backend.command_calls, 0)
        self.assertEqual(policy.close_calls, 1)

    def test_unknown_policy_outcome_is_not_retried_and_stops_both(self):
        config = copy.deepcopy(self.config)
        config["execution"]["mode"] = "no_motion"
        clock = _Clock()
        poller = _Poller(clock)
        backend = _Backend(poller)
        policy = _Policy(fail=True)
        with mock.patch(
            "robot_client_runtime.realman_pi05.open_live_stack",
            return_value=(backend, _Rig(), poller, {"arms": {}}),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.open_policy",
            return_value=(policy, {"config": "test"}),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.capture_observation",
            return_value=({"state": np.zeros(28, dtype=np.float32)}, {"skew_sec": 0.0}),
        ), mock.patch("robot_client_runtime.realman_pi05.close_live_stack"):
            with self.assertRaisesRegex(MotionRejected, "policy_request_outcome_unknown"):
                run_live(
                    config,
                    stop_event=threading.Event(),
                )
        self.assertEqual(policy.infer_calls, 1)
        self.assertEqual(backend.stop_calls, 2)

    def test_explicit_server_rejection_preserves_reason_and_stops_both(self):
        config = copy.deepcopy(self.config_16d)
        config["execution"]["mode"] = "no_motion"
        clock = _Clock()
        poller = _Poller(clock)
        backend = _Backend(poller)
        policy = _Policy()
        policy.infer = mock.Mock(side_effect=RuntimeError(
            'Error in inference server:\n'
            '{"status":"rejected","error_type":"RequestRejected",'
            '"reason":"state must have shape [16]"}'
        ))
        with mock.patch(
            "robot_client_runtime.realman_pi05.open_live_stack",
            return_value=(backend, _Rig(), poller, {"arms": {}}),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.open_policy",
            return_value=(policy, {"config": "test"}),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.capture_observation",
            return_value=({"state": np.zeros(16, dtype=np.float32)}, {"skew_sec": 0.0}),
        ), mock.patch("robot_client_runtime.realman_pi05.close_live_stack"):
            with self.assertRaisesRegex(
                MotionRejected,
                r"policy_request_rejected:1:RequestRejected:state must have shape \[16\]",
            ):
                run_live(config, stop_event=threading.Event())
        self.assertEqual(policy.infer.call_count, 1)
        self.assertEqual(backend.stop_calls, 2)

    def test_execute_path_uses_one_fresh_response_and_two_arm_targets_per_replan(self):
        config = copy.deepcopy(self.config)
        config["execution"]["max_replans"] = 1
        config["action"]["actions_per_chunk"] = 1
        clock = _Clock()
        poller = _Poller(clock)
        backend = _Backend(poller)
        policy = _Policy()
        gate = {
            "arms": {
                side: {"lower_rad": [-1.0] * 7, "upper_rad": [1.0] * 7}
                for side in ("left", "right")
            }
        }
        with mock.patch(
            "robot_client_runtime.realman_pi05.open_live_stack",
            return_value=(backend, _Rig(), poller, gate),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.open_policy",
            return_value=(policy, {"config": "test"}),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.operator_gate",
        ) as operator, mock.patch(
            "robot_client_runtime.realman_pi05.capture_observation",
            return_value=({"state": np.zeros(28, dtype=np.float32)}, {"skew_sec": 0.0}),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.execute_dual_joint_targets",
            return_value=[
                {"side": "right", "status": "observed"},
                {"side": "left", "status": "observed"},
            ],
        ) as execute, mock.patch("robot_client_runtime.realman_pi05.close_live_stack"):
            result = run_live(
                config,
                stop_event=threading.Event(),
            )
        self.assertEqual(result["status"], "completed_bounded_inference")
        self.assertEqual(policy.infer_calls, 1)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(backend.gripper_calls, [("right", 1), ("left", 1)])
        self.assertEqual(
            [item["side"] for item in result["commands"]], ["right", "left"]
        )
        operator.assert_called_once()

    def test_execute_path_consumes_configured_consecutive_action_rows(self):
        config = copy.deepcopy(self.config)
        config["execution"]["max_replans"] = 1
        config["action"]["row_index"] = 1
        config["action"]["actions_per_chunk"] = 3
        clock = _Clock()
        poller = _Poller(clock)
        backend = _Backend(poller)
        policy = _Policy()
        gate = {
            "arms": {
                side: {"lower_rad": [-1.0] * 7, "upper_rad": [1.0] * 7}
                for side in ("left", "right")
            }
        }
        with mock.patch(
            "robot_client_runtime.realman_pi05.open_live_stack",
            return_value=(backend, _Rig(), poller, gate),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.open_policy",
            return_value=(policy, {"config": "test"}),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.operator_gate",
        ), mock.patch(
            "robot_client_runtime.realman_pi05.capture_observation",
            return_value=({"state": np.zeros(28, dtype=np.float32)}, {"skew_sec": 0.0}),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.execute_dual_joint_targets",
            return_value=[
                {"side": "right", "status": "observed"},
                {"side": "left", "status": "observed"},
            ],
        ) as execute, mock.patch("robot_client_runtime.realman_pi05.close_live_stack"):
            result = run_live(config, stop_event=threading.Event())
        self.assertEqual(execute.call_count, 3)
        self.assertEqual(
            [(item["action_row"], item["side"]) for item in result["commands"]],
            [
                (1, "right"), (1, "left"),
                (2, "right"), (2, "left"),
                (3, "right"), (3, "left"),
            ],
        )
        self.assertEqual(result["actions_per_chunk"], 3)

    def test_gripper_commands_are_one_shot_per_threshold_state_change(self):
        config = copy.deepcopy(self.config_16d)
        config["execution"]["max_replans"] = 1
        config["action"]["actions_per_chunk"] = 3
        actions = np.zeros((50, 16), dtype=np.float32)
        actions[:, 15] = 1.0
        actions[2, 7] = 1.0
        actions[2, 15] = 0.0
        clock = _Clock()
        poller = _Poller(clock)
        backend = _Backend(poller)
        policy = _Policy()
        policy.infer = mock.Mock(return_value={"actions": actions})
        gate = {
            "arms": {
                side: {"lower_rad": [-1.0] * 7, "upper_rad": [1.0] * 7}
                for side in ("left", "right")
            }
        }
        with mock.patch(
            "robot_client_runtime.realman_pi05.open_live_stack",
            return_value=(backend, _Rig(), poller, gate),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.open_policy",
            return_value=(policy, {"config": "test"}),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.operator_gate",
        ), mock.patch(
            "robot_client_runtime.realman_pi05.capture_observation",
            return_value=({"state": np.zeros(16, dtype=np.float32)}, {"skew_sec": 0.0}),
        ), mock.patch(
            "robot_client_runtime.realman_pi05.execute_dual_joint_targets",
            return_value=[
                {"side": "right", "status": "observed"},
                {"side": "left", "status": "observed"},
            ],
        ), mock.patch("robot_client_runtime.realman_pi05.close_live_stack"):
            result = run_live(config, stop_event=threading.Event())
        self.assertEqual(
            backend.gripper_calls,
            [("right", 1000), ("left", 1), ("right", 1), ("left", 1000)],
        )
        self.assertEqual(result["gripper_commands"], 4)

    def test_active_runner_has_no_historical_finalize_or_artifact_flow(self):
        source = inspect.getsource(__import__(
            "robot_client_runtime.realman_pi05", fromlist=["realman_pi05"]
        ))
        self.assertNotIn("request_compute_finalize", source)
        self.assertNotIn("fetch_compute_bundle", source)
        self.assertNotIn("client-receipt.yaml", source)
        self.assertNotIn("artifact_root", source)
        self.assertNotIn("project_model_row", source)
        self.assertNotIn("joint_delta_limit_rad", source)
        self.assertNotIn("exact_runtime_provenance", source)
        self.assertNotIn("deployment_readiness", source)


if __name__ == "__main__":
    unittest.main()
