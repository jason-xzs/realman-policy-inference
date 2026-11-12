from __future__ import annotations

import copy
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_client_runtime.production_pi05 import (
    CAMERA_TO_OPENPI,
    DEFAULT_PRESET,
    FlatCameraVideoRecorder,
    build_parser,
    canonical_row_to_hardware_action,
    main,
    move_to_initial_pose,
    prepare_openpi_observation,
    prepare_recording_destination,
    prompt_return_to_initial_pose,
    run_session,
    supervise,
    worker_entry,
    MotionRejected,
    _format_worker_message,
    _install_arm_state_retries,
)
from robot_client_runtime.realman_pi05 import (
    DEFAULT_CONFIG,
    gripper_targets_from_row,
    load_config,
)


PRESET = "pi05-organize-15000"


def _spawn_fixture_worker(config, authorize_event, return_home_event, stop_event, status_queue):
    del config, return_home_event, stop_event
    status_queue.put({"event": "phase", "phase": "startup", "timeout_sec": 2.0})
    status_queue.put({"event": "ready", "service": {"config": "fixture"}, "gate": {}})
    if authorize_event.wait(2.0):
        status_queue.put({"event": "result", "result": {"status": "no_motion_pass"}})


def _spawn_hanging_worker(config, authorize_event, return_home_event, stop_event, status_queue):
    del config, authorize_event, return_home_event, stop_event
    status_queue.put({"event": "phase", "phase": "fixture_hang", "timeout_sec": 0.1})
    time.sleep(5.0)


def _spawn_stop_aware_worker(config, authorize_event, return_home_event, stop_event, status_queue):
    del config, return_home_event
    status_queue.put({"event": "phase", "phase": "startup", "timeout_sec": 2.0})
    status_queue.put({"event": "ready", "service": {"config": "fixture"}, "gate": {}})
    authorize_event.wait(2.0)
    stop_event.wait(2.0)
    status_queue.put({
        "event": "result",
        "result": {
            "status": "stopped",
            "reason": "operator_stop",
            "recording_outputs": ["high.mp4", "left.mp4", "right.mp4"],
        },
    })


def _spawn_stop_cleanup_error_worker(
    config, authorize_event, return_home_event, stop_event, status_queue
):
    del config, return_home_event
    status_queue.put({"event": "phase", "phase": "startup", "timeout_sec": 2.0})
    status_queue.put({"event": "ready", "service": {"config": "fixture"}, "gate": {}})
    authorize_event.wait(2.0)
    stop_event.wait(2.0)
    status_queue.put({
        "event": "error",
        "error_type": "WorkerCleanupError",
        "reason": "recording_writer_errors",
    })


def _spawn_home_decision_worker(
    config, authorize_event, return_home_event, stop_event, status_queue
):
    del config, stop_event
    status_queue.put({"event": "phase", "phase": "startup", "timeout_sec": 2.0})
    status_queue.put({"event": "ready", "service": {"config": "fixture"}, "gate": {}})
    if authorize_event.wait(2.0):
        status_queue.put({
            "event": "result",
            "result": {
                "status": "completed",
                "initial_pose": "reached" if return_home_event.is_set() else "skipped",
            },
        })


def _spawn_home_hanging_worker(
    config, authorize_event, return_home_event, stop_event, status_queue
):
    del config, return_home_event, stop_event
    status_queue.put({"event": "phase", "phase": "startup", "timeout_sec": 2.0})
    status_queue.put({"event": "ready", "service": {"config": "fixture"}, "gate": {}})
    if authorize_event.wait(2.0):
        status_queue.put({"event": "phase", "phase": "initial_pose", "timeout_sec": 0.1})
        time.sleep(5.0)


class _Hardware:
    def __init__(self) -> None:
        self.actions = []
        self.observe_calls = 0
        self.closed = False
        self.stopped = False
        self.left_joints = np.zeros(7, dtype=np.float64)
        self.right_joints = np.zeros(7, dtype=np.float64)

    def observe(self):
        self.observe_calls += 1
        images = {
            name: np.zeros((480, 640, 3), dtype=np.uint8)
            for name in CAMERA_TO_OPENPI
        }
        state = np.arange(28, dtype=np.float32)
        state[7] = 1.0
        state[21] = 1.0
        return images, state.tolist()

    def execute_action(self, action, *, dry_run=False):
        if dry_run:
            raise AssertionError("production execute session must not pass dry_run=True")
        self.actions.append(list(action))
        self.right_joints[:] = action[0:7]
        self.left_joints[:] = action[8:15]

    def current_joints(self):
        return self.left_joints.copy(), self.right_joints.copy()

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _NonSettlingHardware(_Hardware):
    def execute_action(self, action, *, dry_run=False):
        if dry_run:
            raise AssertionError("production execute session must not pass dry_run=True")
        self.actions.append(list(action))


class _InitialPoseArm:
    def __init__(self, hardware, side) -> None:
        self.hardware = hardware
        self.side = side
        self.movej_calls = []
        self.gripper_calls = []

    def rm_movej(self, joints_deg, speed, radius, connect, block):
        self.hardware.commands.append(("joint", self.side))
        self.movej_calls.append((list(joints_deg), speed, radius, connect, block))
        setattr(
            self.hardware,
            f"{self.side}_joints",
            np.deg2rad(np.asarray(joints_deg, dtype=np.float64)),
        )
        return 0

    def rm_set_gripper_position(self, position, block, timeout):
        self.hardware.commands.append(("gripper", self.side))
        self.gripper_calls.append((position, block, timeout))
        self.hardware.grippers[self.side] = float(position)
        return 0


class _InitialPoseHardware:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_grippers = {"left": None, "right": None}
        self.commands = []
        self.left_joints = np.zeros(7, dtype=np.float64)
        self.right_joints = np.zeros(7, dtype=np.float64)
        self.grippers = {"left": 1.0, "right": 1.0}
        self.left_arm = _InitialPoseArm(self, "left")
        self.right_arm = _InitialPoseArm(self, "right")

    def current_joints(self):
        return self.left_joints.copy(), self.right_joints.copy()

    def current_grippers(self):
        return dict(self.grippers)


class _StartRecorder:
    def __init__(self, order=None) -> None:
        self.order = [] if order is None else order
        self.output_files = []
        self.start_calls = 0
        self.close_calls = 0

    def start(self):
        self.start_calls += 1
        self.order.append("recording")
        return 123

    def close(self):
        self.close_calls += 1


class _Policy:
    def __init__(self, actions) -> None:
        self.actions = np.asarray(actions, dtype=np.float32)
        self.calls = 0
        self.closed = False

    def infer(self, observation):
        self.calls += 1
        self.observation = observation
        return {"actions": self.actions.copy()}

    def close(self):
        self.closed = True


class _FakeRecorder:
    CAMERA_NAMES = ("high_rgb", "left_wrist_rgb", "right_wrist_rgb")

    def __init__(self, root_dir, **_kwargs):
        self.run_dir = Path(root_dir) / "client_fixture"
        self.run_dir.mkdir(parents=True)
        self._errors = {}
        self.submissions = []

    def submit(self, name, _image, *, sequence, timestamp_ns):
        self.submissions.append((name, sequence, timestamp_ns))

    def close(self):
        for name in self.CAMERA_NAMES:
            (self.run_dir / f"{name}.mp4").write_bytes(f"video:{name}".encode())
        (self.run_dir / "camera_frames.jsonl").write_text("{}\n")


class ProductionPi05Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(DEFAULT_CONFIG, PRESET)

    def test_production_defaults_are_configured_and_supervised(self):
        self.assertEqual(DEFAULT_PRESET, "pi05-sugar-15000")
        self.assertEqual(build_parser().parse_args([]).preset, "pi05-sugar-15000")
        self.assertEqual(self.config["observation"]["state_dimension"], 16)
        self.assertEqual(self.config["action"]["actions_per_chunk"], 20)
        self.assertEqual(self.config["action"]["action_hz"], 30.0)
        self.assertEqual(self.config["runtime"]["max_sync_skew_sec"], 1)
        self.assertEqual(self.config["runtime"]["state_stale_sec"], 1)
        self.assertEqual(self.config["runtime"]["action_call_timeout_sec"], 5.0)
        self.assertEqual(self.config["runtime"]["chunk_settle_timeout_sec"], 5.0)
        self.assertEqual(self.config["runtime"]["chunk_stable_window_sec"], 0.25)
        self.assertEqual(self.config["runtime"]["chunk_settle_poll_hz"], 20.0)
        self.assertEqual(self.config["runtime"]["shutdown_grace_sec"], 5.0)
        with DEFAULT_CONFIG.open("rb") as stream:
            registry = tomllib.load(stream)
        self.assertFalse(registry["default"]["recording"]["enabled"])
        self.assertEqual(self.config["recording"]["root_dir"], "/home/robot2/jhk/phywam-mope-real/real-records")
        self.assertEqual(
            self.config["initial_pose"]["left_joint_deg"],
            [-0.328, -0.396, -0.420, 43.949, 1.288, 116.485, -27.627],
        )
        self.assertEqual(
            self.config["initial_pose"]["right_joint_deg"],
            [-0.265, -0.506, -0.157, 43.954, 1.220, 121.749, -0.186],
        )
        self.assertEqual(self.config["initial_pose"]["gripper_open_position"], 1000)
        self.assertEqual(self.config["initial_pose"]["left_speed_percent"], 80)
        self.assertEqual(self.config["initial_pose"]["right_speed_percent"], 10)
        self.assertNotIn("speed_percent", self.config["initial_pose"])

    def test_initial_pose_config_rejects_wrong_joint_dimension(self):
        source = DEFAULT_CONFIG.read_text()
        malformed = source.replace(
            "left_joint_deg = [-0.328, -0.396, -0.420, 43.949, 1.288, 116.485, -27.627]",
            "left_joint_deg = [0.0]",
            1,
        )
        self.assertNotEqual(source, malformed)
        with tempfile.TemporaryDirectory() as temporary_root:
            path = Path(temporary_root) / "presets.toml"
            path.write_text(malformed)
            with self.assertRaisesRegex(
                MotionRejected,
                "initial_pose_joints_invalid:left",
            ):
                load_config(path, PRESET)

    def test_initial_pose_config_rejects_invalid_side_specific_speed(self):
        source = DEFAULT_CONFIG.read_text()
        malformed = source.replace("right_speed_percent = 10", "right_speed_percent = 0", 1)
        self.assertNotEqual(source, malformed)
        with tempfile.TemporaryDirectory() as temporary_root:
            path = Path(temporary_root) / "presets.toml"
            path.write_text(malformed)
            with self.assertRaisesRegex(
                MotionRejected,
                "initial_pose_speed_percent_invalid:right:0",
            ):
                load_config(path, PRESET)

    def test_arm_state_sdk_minus_2_recovers_without_blocking(self):
        class SequencedArm:
            def __init__(self, statuses):
                self.statuses = list(statuses)
                self.calls = 0

            def rm_get_current_arm_state(self):
                status = self.statuses[min(self.calls, len(self.statuses) - 1)]
                self.calls += 1
                return status, {"joint": [0.0] * 7}

        hardware = mock.Mock()
        hardware.left_arm = SequencedArm([0])
        hardware.right_arm = SequencedArm([-2, -2, 0])
        config = copy.deepcopy(self.config)
        config["runtime"].update(
            {"controller_query_retries": 2, "controller_query_backoff_sec": 0.05}
        )

        _install_arm_state_retries(hardware, config)
        with mock.patch("robot_client_runtime.production_pi05.time.sleep") as sleep_mock:
            code, state = hardware.right_arm.rm_get_current_arm_state()

        self.assertEqual(code, 0)
        self.assertEqual(state["joint"], [0.0] * 7)
        self.assertEqual(hardware.right_arm._arm.calls, 3)
        self.assertEqual(sleep_mock.call_args_list, [mock.call(0.05), mock.call(0.05)])

    def test_arm_state_retry_does_not_hide_persistent_or_other_sdk_errors(self):
        class ConstantArm:
            def __init__(self, status):
                self.status = status
                self.calls = 0

            def rm_get_current_arm_state(self):
                self.calls += 1
                return self.status, {}

        config = copy.deepcopy(self.config)
        config["runtime"].update(
            {"controller_query_retries": 2, "controller_query_backoff_sec": 0.0}
        )
        hardware = mock.Mock()
        hardware.left_arm = ConstantArm(-2)
        hardware.right_arm = ConstantArm(-1)

        _install_arm_state_retries(hardware, config)
        self.assertEqual(hardware.left_arm.rm_get_current_arm_state()[0], -2)
        self.assertEqual(hardware.right_arm.rm_get_current_arm_state()[0], -1)
        self.assertEqual(hardware.left_arm._arm.calls, 3)
        self.assertEqual(hardware.right_arm._arm.calls, 1)

    def test_worker_progress_is_compact_and_includes_raw_gripper_values(self):
        self.assertEqual(
            _format_worker_message(
                {
                    "event": "phase",
                    "phase": "action",
                    "replan": 39,
                    "action_row": 17,
                    "gripper": {"right": 0.742, "left": 0.287},
                    "timeout_sec": 5.03,
                }
            ),
            "event: phase, phase: action, replan: 39, action_row: 17, gripper: 0.29 & 0.74",
        )
        self.assertEqual(
            _format_worker_message(
                {
                    "event": "phase",
                    "phase": "action",
                    "replan": 40,
                    "action_row": 0,
                    "gripper": {"left": -0.004, "right": 1.04},
                }
            ),
            "event: phase, phase: action, replan: 40, action_row: 0, gripper: 0.00 & 1.00",
        )
        self.assertEqual(
            _format_worker_message({"event": "action_done", "gripper": {"left": -0.0}}),
            "event: action_done, gripper: 0.00",
        )

    def test_gripper_hysteresis_closes_opens_and_holds_middle_range(self):
        row = np.zeros(16, dtype=np.float32)
        row[7] = 0.3
        row[15] = 0.7
        targets = gripper_targets_from_row(row, self.config["gripper"])
        self.assertEqual(targets["left"], {"state": "closed", "position": 1})
        self.assertEqual(targets["right"], {"state": "open", "position": 1000})

        row[7] = 0.5
        row[15] = 0.5
        self.assertEqual(
            gripper_targets_from_row(row, self.config["gripper"], targets),
            targets,
        )

    def test_preset_gripper_override_replaces_default_recursively(self):
        organize = load_config(DEFAULT_CONFIG, "pi05-organize-15000")
        sugar = load_config(DEFAULT_CONFIG, "pi05-sugar-15000")
        self.assertEqual(organize["gripper"]["close_threshold"], 0.4)
        self.assertEqual(sugar["gripper"]["close_threshold"], 0.4)
        self.assertEqual(sugar["gripper"]["open_threshold"], 0.7)
        self.assertEqual(sugar["gripper"]["open_position"], 1000)

    def test_recording_overwrite_gate_declines_or_clears_exact_target(self):
        config = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root) / "real-records"
            target = root / "episode_001"
            target.mkdir(parents=True)
            marker = target / "old.txt"
            marker.write_text("old")
            config["recording"].update(
                {
                    "enabled": True,
                    "root_dir": str(root),
                    "cache_dir": str(root / ".cache"),
                    "subdirectory": "episode_001",
                }
            )
            prompts = []
            self.assertFalse(
                prepare_recording_destination(
                    config,
                    input_fn=lambda prompt: prompts.append(prompt) or "",
                )
            )
            self.assertEqual(prompts, ["目录非空，覆盖[y/N]:"])
            self.assertTrue(marker.is_file())

            self.assertTrue(
                prepare_recording_destination(config, input_fn=lambda _prompt: "y")
            )
            self.assertEqual(list(target.iterdir()), [])
            self.assertTrue((root / ".cache").is_dir())

    def test_flat_recording_publishes_only_three_named_mp4_files(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root) / "real-records"
            target = root / "episode_002"
            cache = root / ".cache"
            target.mkdir(parents=True)
            cache.mkdir()
            recorder = FlatCameraVideoRecorder(
                target_dir=target,
                cache_dir=cache,
                basename="episode_002",
                fps=30.0,
                codec="h264_vaapi",
                queue_size=8,
                recorder_type=_FakeRecorder,
            )
            recorder.close()
            self.assertEqual(
                sorted(path.name for path in target.iterdir()),
                [
                    "episode_002_high.mp4",
                    "episode_002_left.mp4",
                    "episode_002_right.mp4",
                ],
            )
            self.assertEqual(
                [Path(path).name for path in recorder.output_files],
                [
                    "episode_002_high.mp4",
                    "episode_002_left.mp4",
                    "episode_002_right.mp4",
                ],
            )
            self.assertEqual(list(cache.iterdir()), [])

    def test_recording_stays_paused_until_started_and_excludes_older_frames(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root) / "real-records"
            target = root / "episode_paused"
            cache = root / ".cache"
            target.mkdir(parents=True)
            cache.mkdir()
            recorder = FlatCameraVideoRecorder(
                target_dir=target,
                cache_dir=cache,
                basename="episode_paused",
                fps=30.0,
                codec="h264_vaapi",
                queue_size=8,
                recorder_type=_FakeRecorder,
                start_paused=True,
            )
            recorder.submit("high_rgb", None, sequence=1, timestamp_ns=time.monotonic_ns())
            self.assertEqual(recorder._delegate.submissions, [])
            started_ns = recorder.start()
            recorder.submit("high_rgb", None, sequence=2, timestamp_ns=started_ns - 1)
            recorder.submit("high_rgb", None, sequence=3, timestamp_ns=started_ns)
            self.assertEqual(
                recorder._delegate.submissions,
                [("high_rgb", 3, started_ns)],
            )
            recorder.close()

    def test_never_started_recording_is_aborted_without_published_files(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root) / "real-records"
            target = root / "episode_aborted"
            cache = root / ".cache"
            target.mkdir(parents=True)
            cache.mkdir()
            recorder = FlatCameraVideoRecorder(
                target_dir=target,
                cache_dir=cache,
                basename="episode_aborted",
                fps=30.0,
                codec="h264_vaapi",
                queue_size=8,
                recorder_type=_FakeRecorder,
                start_paused=True,
            )
            recorder.close()
            self.assertEqual(recorder.output_files, [])
            self.assertEqual(list(target.iterdir()), [])
            self.assertEqual(list(cache.iterdir()), [])

    def test_declined_recording_overwrite_exits_before_supervise(self):
        config = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root) / "real-records"
            target = root / "episode_003"
            target.mkdir(parents=True)
            (target / "existing.mp4").write_bytes(b"old")
            config["recording"].update(
                {
                    "enabled": True,
                    "root_dir": str(root),
                    "cache_dir": str(root / ".cache"),
                    "subdirectory": "episode_003",
                }
            )
            with mock.patch(
                "robot_client_runtime.production_pi05.load_config",
                return_value=config,
            ), mock.patch(
                "builtins.input",
                return_value="",
            ), mock.patch(
                "robot_client_runtime.production_pi05.supervise"
            ) as supervise_mock:
                self.assertEqual(main(["--preset", PRESET]), 0)
            supervise_mock.assert_not_called()

    def test_observation_is_rgb_224_and_native_left_first_16d(self):
        hardware = _Hardware()
        images, state = hardware.observe()
        observation = prepare_openpi_observation(images, state, self.config)
        self.assertEqual(
            set(observation),
            {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "prompt"},
        )
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            self.assertEqual(observation[key].shape, (224, 224, 3))
            self.assertEqual(observation[key].dtype, np.uint8)
        np.testing.assert_array_equal(
            observation["state"],
            np.asarray(state, dtype=np.float32)[list(range(8)) + list(range(14, 22))],
        )

    def test_left_first_row_is_converted_to_fastwam_right_first_with_binary_grippers(self):
        row = [0.1] * 7 + [0.2] + [0.3] * 7 + [0.8]
        action = canonical_row_to_hardware_action(row, self.config)
        self.assertEqual(action, [0.3] * 7 + [1000.0] + [0.1] * 7 + [1.0])

    def test_initial_pose_moves_both_arms_opens_grippers_and_verifies_feedback(self):
        config = copy.deepcopy(self.config)
        config["initial_pose"].update(
            {
                "settle_timeout_sec": 1.0,
                "stable_window_sec": 0.001,
                "poll_hz": 1000.0,
                "left_speed_percent": 17,
                "right_speed_percent": 23,
            }
        )
        hardware = _InitialPoseHardware()
        result = move_to_initial_pose(
            hardware=hardware,
            config=config,
            stop_event=threading.Event(),
        )
        self.assertEqual(
            hardware.commands,
            [
                ("joint", "right"),
                ("joint", "left"),
                ("gripper", "right"),
                ("gripper", "left"),
            ],
        )
        self.assertEqual(
            hardware.left_arm.movej_calls,
            [(config["initial_pose"]["left_joint_deg"], 17, 0, 0, 0)],
        )
        self.assertEqual(
            hardware.right_arm.movej_calls,
            [(config["initial_pose"]["right_joint_deg"], 23, 0, 0, 0)],
        )
        self.assertEqual(hardware.left_arm.gripper_calls, [(1000, False, 0)])
        self.assertEqual(hardware.right_arm.gripper_calls, [(1000, False, 0)])
        self.assertLessEqual(result["joint_error_rad"], 1e-12)
        self.assertEqual(result["gripper_error"], 0.0)
        self.assertEqual(result["speed_percent_by_side"], {"left": 17, "right": 23})
        self.assertEqual(hardware._last_grippers, {"left": 1000, "right": 1000})

    def test_execute_session_consumes_configured_chunk_at_action_frequency(self):
        config = copy.deepcopy(self.config)
        config["execution"]["max_replans"] = 1
        config["action"]["actions_per_chunk"] = 3
        config["action"]["action_hz"] = 10000.0
        actions = []
        for index in range(3):
            actions.append(
                [0.01 * (index + 1)] * 7
                + [0.2]
                + [0.02 * (index + 1)] * 7
                + [0.8]
            )
        hardware = _Hardware()
        policy = _Policy(actions)
        events = []
        result = run_session(
            config=config,
            hardware=hardware,
            policy=policy,
            stop_event=threading.Event(),
            emit=events.append,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["robot_commands"], 3)
        self.assertEqual(result["gripper_commands"], 1)
        self.assertEqual(policy.calls, 1)
        self.assertEqual(len(hardware.actions), 3)
        self.assertEqual(
            [event["event"] for event in events].count("chunk_settle_done"), 1
        )
        np.testing.assert_allclose(
            hardware.actions[0], [0.02] * 7 + [1000.0] + [0.01] * 7 + [1.0]
        )
        self.assertEqual(
            [event["event"] for event in events].count("action_done"), 3
        )

    def test_no_motion_session_runs_one_request_and_zero_commands(self):
        config = copy.deepcopy(self.config)
        config["execution"]["mode"] = "no_motion"
        hardware = _Hardware()
        policy = _Policy([[0.0] * 16])
        result = run_session(
            config=config,
            hardware=hardware,
            policy=policy,
            stop_event=threading.Event(),
            emit=lambda _message: None,
        )
        self.assertEqual(result["status"], "no_motion_pass")
        self.assertEqual(result["robot_commands"], 0)
        self.assertEqual(result["gripper_commands"], 0)
        self.assertEqual(hardware.actions, [])

    def test_chunk_settle_timeout_blocks_before_next_inference(self):
        config = copy.deepcopy(self.config)
        config["execution"]["max_replans"] = 2
        config["action"]["actions_per_chunk"] = 1
        config["action"]["action_hz"] = 10000.0
        config["runtime"]["chunk_settle_timeout_sec"] = 0.03
        config["runtime"]["chunk_stable_window_sec"] = 0.01
        config["runtime"]["chunk_settle_poll_hz"] = 1000.0
        hardware = _NonSettlingHardware()
        policy = _Policy([[0.1] * 7 + [0.2] + [0.1] * 7 + [0.8]])
        events = []
        with self.assertRaisesRegex(MotionRejected, "chunk_settle_timeout"):
            run_session(
                config=config,
                hardware=hardware,
                policy=policy,
                stop_event=threading.Event(),
                emit=events.append,
            )
        self.assertEqual(policy.calls, 1)
        self.assertEqual(len(hardware.actions), 1)
        self.assertIn("chunk_settle", [event.get("phase") for event in events if event["event"] == "phase"])

    def test_repository_script_dispatches_only_to_production_runner(self):
        source = (Path(__file__).resolve().parents[1] / "scripts/run_pi05_realman.py").read_text()
        self.assertIn("robot_client_runtime.production_pi05 import main", source)
        self.assertNotIn("robot_client_runtime.realman_pi05 import main", source)

    def test_worker_owns_and_closes_hardware_and_policy(self):
        config = copy.deepcopy(self.config)
        config["execution"]["mode"] = "no_motion"
        hardware = _Hardware()
        policy = _Policy([[0.0] * 16])
        authorize = threading.Event()
        authorize.set()
        return_home = threading.Event()
        stop = threading.Event()
        messages = queue.Queue()
        with mock.patch(
            "robot_client_runtime.production_pi05._load_fastwam_hardware",
            return_value=hardware,
        ), mock.patch(
            "robot_client_runtime.production_pi05._validate_hardware_ready",
            return_value={"left": {"state": "ready"}, "right": {"state": "ready"}},
        ), mock.patch(
            "robot_client_runtime.production_pi05.open_policy",
            return_value=(policy, {"config": "test"}),
        ):
            worker_entry(config, authorize, return_home, stop, messages)
        collected = []
        while not messages.empty():
            collected.append(messages.get_nowait())
        self.assertIn("ready", [item["event"] for item in collected])
        results = [item["result"] for item in collected if item["event"] == "result"]
        self.assertEqual(results[0]["status"], "no_motion_pass")
        self.assertTrue(hardware.closed)
        self.assertTrue(policy.closed)

    def test_worker_homes_before_recording_and_inference_or_skips_without_commands(self):
        for requested, expected_status, expected_order in (
            (True, "reached", ["home", "recording", "session"]),
            (False, "skipped", ["recording", "session"]),
        ):
            with self.subTest(requested=requested):
                config = copy.deepcopy(self.config)
                config["execution"]["mode"] = "execute"
                order = []
                recorder = _StartRecorder(order)
                hardware = _Hardware()
                hardware._production_frame_recorder = recorder
                policy = _Policy([[0.0] * 16])
                authorize = threading.Event()
                authorize.set()
                return_home = threading.Event()
                if requested:
                    return_home.set()
                stop = threading.Event()
                messages = queue.Queue()

                def fake_home(**_kwargs):
                    order.append("home")
                    return {
                        "joint_error_rad": 0.0,
                        "gripper_error": 0.0,
                        "feedback_span_rad": 0.0,
                    }

                def fake_session(**_kwargs):
                    order.append("session")
                    return {"status": "completed"}

                with mock.patch(
                    "robot_client_runtime.production_pi05._load_fastwam_hardware",
                    return_value=hardware,
                ), mock.patch(
                    "robot_client_runtime.production_pi05._validate_hardware_ready",
                    return_value={"left": {"state": "ready"}, "right": {"state": "ready"}},
                ), mock.patch(
                    "robot_client_runtime.production_pi05.open_policy",
                    return_value=(policy, {"config": "test"}),
                ), mock.patch(
                    "robot_client_runtime.production_pi05.move_to_initial_pose",
                    side_effect=fake_home,
                ) as home_mock, mock.patch(
                    "robot_client_runtime.production_pi05.run_session",
                    side_effect=fake_session,
                ):
                    worker_entry(config, authorize, return_home, stop, messages)

                self.assertEqual(order, expected_order)
                self.assertEqual(home_mock.call_count, int(requested))
                results = []
                while not messages.empty():
                    message = messages.get_nowait()
                    if message["event"] == "result":
                        results.append(message["result"])
                self.assertEqual(results[0]["initial_pose"], expected_status)

    def test_initial_pose_failure_blocks_recording_and_inference(self):
        config = copy.deepcopy(self.config)
        config["execution"]["mode"] = "execute"
        recorder = _StartRecorder()
        hardware = _Hardware()
        hardware._production_frame_recorder = recorder
        policy = _Policy([[0.0] * 16])
        authorize = threading.Event()
        authorize.set()
        return_home = threading.Event()
        return_home.set()
        stop = threading.Event()
        messages = queue.Queue()
        with mock.patch(
            "robot_client_runtime.production_pi05._load_fastwam_hardware",
            return_value=hardware,
        ), mock.patch(
            "robot_client_runtime.production_pi05._validate_hardware_ready",
            return_value={"left": {"state": "ready"}, "right": {"state": "ready"}},
        ), mock.patch(
            "robot_client_runtime.production_pi05.open_policy",
            return_value=(policy, {"config": "test"}),
        ), mock.patch(
            "robot_client_runtime.production_pi05.move_to_initial_pose",
            side_effect=MotionRejected("initial_pose_settle_timeout"),
        ), mock.patch(
            "robot_client_runtime.production_pi05.run_session",
        ) as session_mock:
            worker_entry(config, authorize, return_home, stop, messages)
        session_mock.assert_not_called()
        self.assertEqual(recorder.start_calls, 0)
        self.assertTrue(hardware.stopped)
        errors = []
        while not messages.empty():
            message = messages.get_nowait()
            if message["event"] == "error":
                errors.append(message)
        self.assertIn("initial_pose_settle_timeout", errors[0]["reason"])

    def test_supervisor_spawn_round_trip_without_hardware(self):
        config = copy.deepcopy(self.config)
        config["execution"]["mode"] = "no_motion"
        result = supervise(config, worker_target=_spawn_fixture_worker)
        self.assertEqual(result["status"], "no_motion_pass")
        self.assertEqual(result["preset"], PRESET)

    def test_supervisor_prompts_for_home_after_execute_confirmation(self):
        for answer, expected in (("", "reached"), ("n", "skipped"), ("N", "skipped")):
            with self.subTest(answer=answer):
                config = copy.deepcopy(self.config)
                config["execution"]["mode"] = "execute"
                replies = iter(("", answer))
                prompts = []

                def answer_prompt(prompt):
                    prompts.append(prompt)
                    return next(replies)

                result = supervise(
                    config,
                    worker_target=_spawn_home_decision_worker,
                    input_fn=answer_prompt,
                )
                self.assertEqual(result["initial_pose"], expected)
                self.assertEqual(
                    prompts,
                    [
                        "请回车确认，Ctrl+C 退出：",
                        "是否回到初始位姿？回车同意，n/N拒绝：",
                    ],
                )

    def test_home_prompt_repeats_until_enter_or_n(self):
        replies = iter(("yes", "N"))
        prompts = []
        with mock.patch("builtins.print") as print_mock:
            result = prompt_return_to_initial_pose(
                lambda prompt: prompts.append(prompt) or next(replies)
            )
        self.assertFalse(result)
        self.assertEqual(len(prompts), 2)
        print_mock.assert_called_once_with("请输入回车同意，或 n/N 拒绝。")

    def test_supervisor_terminates_a_stuck_worker(self):
        config = copy.deepcopy(self.config)
        config["execution"]["mode"] = "no_motion"
        config["recording"]["enabled"] = False
        config["runtime"]["shutdown_grace_sec"] = 0.1
        started = time.monotonic()
        result = supervise(config, worker_target=_spawn_hanging_worker)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("worker_phase_timeout:fixture_hang", result["reason"])
        self.assertLess(time.monotonic() - started, 2.0)

    def test_initial_pose_timeout_is_motion_started_and_runs_emergency_stop(self):
        config = copy.deepcopy(self.config)
        config["execution"]["mode"] = "execute"
        config["recording"]["enabled"] = False
        config["runtime"]["shutdown_grace_sec"] = 0.1
        replies = iter(("", ""))
        with mock.patch(
            "robot_client_runtime.production_pi05._run_emergency_stop",
            return_value={"status": "completed", "sdk": {"left": 0, "right": 0}},
        ) as emergency_stop:
            result = supervise(
                config,
                worker_target=_spawn_home_hanging_worker,
                input_fn=lambda _prompt: next(replies),
            )
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["motion_started"])
        self.assertIn("worker_phase_timeout:initial_pose", result["reason"])
        emergency_stop.assert_called_once()

    def test_ctrl_c_requests_graceful_worker_stop(self):
        config = copy.deepcopy(self.config)
        config["execution"]["mode"] = "execute"
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            result = supervise(config, worker_target=_spawn_stop_aware_worker)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["reason"], "operator_stop")
        self.assertEqual(
            result["recording_outputs"],
            ["high.mp4", "left.mp4", "right.mp4"],
        )

    def test_ctrl_c_recording_cleanup_error_remains_blocked(self):
        config = copy.deepcopy(self.config)
        config["execution"]["mode"] = "execute"
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            result = supervise(config, worker_target=_spawn_stop_cleanup_error_worker)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("recording_writer_errors", result["reason"])


if __name__ == "__main__":
    unittest.main()
