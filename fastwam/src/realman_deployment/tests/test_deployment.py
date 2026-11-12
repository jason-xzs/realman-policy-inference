from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

from realman_deployment.client import (
    BackgroundCameraVideoRecorder, DEFAULT_INSTRUCTION, RemoteInferenceClient, align_action_chunk)
from realman_deployment.hardware import (
    BufferedCameraRig,
    BinaryGripperHysteresis,
    HardwareError,
    MockRealManHardware,
    RealManHardware,
    SafetyError,
    TimedFrame,
    orient_deployment_images,
    select_synchronized_frames,
    validate_joint_step,
)
from realman_deployment.policy import MockPolicy
from realman_deployment.protocol import ProtocolError, make_infer_request, validate_actions
from realman_deployment.server import build_action_diagnostics, create_server
from realman_deployment.simulate import run_simulation


class ServerFixture(unittest.TestCase):
    token = "test-secret"

    def setUp(self):
        self.policy = MockPolicy(action_horizon=5)
        self.server = create_server("127.0.0.1", 0, self.policy, api_token=self.token, max_actions=3)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_health_and_inference(self):
        remote = RemoteInferenceClient(self.url, timeout=3, api_token=self.token)
        self.assertTrue(remote.health()["ready"])
        hardware = MockRealManHardware(height=32, width=48)
        images, state = hardware.observe()
        actions, metadata = remote.infer(images, state, DEFAULT_INSTRUCTION, jpeg_quality=70)
        self.assertEqual(len(actions), 3)
        self.assertTrue(metadata["mock"])
        self.assertEqual(len(actions[0]), 16)

    def test_authentication_rejected(self):
        request = urllib.request.Request(self.url + "/v1/reset", data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=3)
        self.assertEqual(caught.exception.code, 401)

    def test_bad_request_rejected(self):
        request = urllib.request.Request(
            self.url + "/v1/infer", data=json.dumps({"version": 1}).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=3)
        self.assertEqual(caught.exception.code, 400)


class ProtocolAndLoopTests(unittest.TestCase):
    def test_server_action_diagnostics(self):
        state = [0.0] * 28
        state[7] = 0.5
        state[21] = 0.75
        current = state[14:21] + [750.0] + state[0:7] + [500.0]
        actions = []
        for step in range(4):
            action = current.copy()
            action[0] += 0.01 * (step + 1)
            actions.append(action)
        diagnostics = build_action_diagnostics(state, actions, 2, clipped_values=3)
        self.assertEqual(diagnostics["predicted_horizon"], 4)
        self.assertEqual(diagnostics["returned_count"], 2)
        self.assertEqual(diagnostics["returned_step_joint_max_abs_rad"], [0.01, 0.01])
        self.assertEqual(diagnostics["returned_endpoint_joint_max_abs_rad"], 0.02)
        self.assertEqual(diagnostics["full_endpoint_joint_max_abs_rad"], 0.04)
        self.assertEqual(diagnostics["returned_grippers"]["right"], [750.0, 750.0])
        self.assertEqual(diagnostics["clipped_values"], 3)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[3] /
         "runs/realman_demo_organize_finetune_3cam_384_1e-5/2026-08-17_00-39-16/"
         "checkpoints/weights/step_005000.pt").is_file(),
        "Optional local training checkpoint is not distributed with this repository",
    )
    def test_requested_checkpoint_training_contract(self):
        project = Path(__file__).resolve().parents[3]
        run = project / "runs/realman_demo_organize_finetune_3cam_384_1e-5/2026-08-17_00-39-16"
        self.assertTrue((run / "checkpoints/weights/step_005000.pt").is_file())
        self.assertTrue((run / "dataset_stats.json").is_file())
        cfg = OmegaConf.load(run / "config.yaml")
        self.assertEqual([entry.key for entry in cfg.data.train.shape_meta.images],
                         ["high_rgb", "left_wrist_rgb", "right_wrist_rgb"])
        self.assertEqual(cfg.data.train.shape_meta.state[0].raw_shape, 28)
        self.assertEqual(cfg.data.train.shape_meta.state[0].shape, 16)
        self.assertEqual(cfg.data.train.shape_meta.action[0].shape, 16)
        self.assertEqual(cfg.data.train.concat_multi_camera, "robotwin")
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(run / "dataset_stats.json")))
        transformed = processor.action_state_transform({
            "state": {"default": torch.zeros(1, 28, dtype=torch.float32)}
        })
        normalized = processor.normalizer.forward(transformed)["state"]["default"]
        self.assertEqual(tuple(normalized.shape), (1, 16))
        self.assertTrue(bool(torch.isfinite(normalized).all()))

    def test_protocol_rejects_wrong_state_and_action(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ProtocolError):
            make_infer_request({name: image for name in ("high_rgb", "left_wrist_rgb", "right_wrist_rgb")},
                               [0.0] * 16, "task")
        with self.assertRaises(ProtocolError):
            validate_actions([[0.0] * 15])

    def test_joint_jump_safety_gate(self):
        safe = [0.0] * 16
        safe[0] = 0.1
        self.assertAlmostEqual(validate_joint_step([0.0] * 7, [0.0] * 7, safe, 0.15), 0.1)
        unsafe = safe.copy()
        unsafe[8] = 0.151
        with self.assertRaises(SafetyError):
            validate_joint_step([0.0] * 7, [0.0] * 7, unsafe, 0.15)
    def test_action_chunk_alignment_selects_closest_limited_action(self):
        actions = []
        for value in (0.0, 0.1, 0.2, 0.3, 0.4):
            action = [value] * 7 + [1000.0] + [value] * 7 + [1000.0]
            actions.append(action)
        aligned, diagnostics = align_action_chunk(
            actions, [0.21] * 7, [0.21] * 7,
            max_skip=3, min_improvement_rad=0.01,
        )
        self.assertEqual(diagnostics["selected_index"], 2)
        self.assertEqual(aligned, actions[2:])
        self.assertAlmostEqual(diagnostics["selected_error_rad"], 0.01)

        limited, limited_diagnostics = align_action_chunk(
            actions, [0.31] * 7, [0.31] * 7,
            max_skip=1, min_improvement_rad=0.01,
        )
        self.assertEqual(limited_diagnostics["selected_index"], 1)
        self.assertEqual(limited, actions[1:])

    def test_tracking_gate_repeats_target_until_reached(self):
        class TrackingArm:
            def __init__(self):
                self.joints_deg = np.zeros(7, dtype=np.float64)
                self.commands = 0

            def rm_get_current_arm_state(self):
                return 0, {"joint": self.joints_deg.tolist()}

            def rm_movej_canfd(self, target_deg, *_args, **_kwargs):
                target = np.asarray(target_deg, dtype=np.float64)
                self.joints_deg += (target - self.joints_deg) * 0.5
                self.commands += 1
                return 0

        hardware = RealManHardware.__new__(RealManHardware)
        hardware.left_arm = TrackingArm()
        hardware.right_arm = TrackingArm()
        hardware._lock = threading.Lock()
        hardware.trajectory_mode = 2
        hardware.smoothing = 50
        action = [0.1] * 7 + [1000.0] + [0.1] * 7 + [1000.0]
        result = hardware.wait_for_joint_target(
            action, tolerance_rad=0.02, timeout_s=1.0, resend_hz=1000,
        )
        self.assertLessEqual(result["final_error_rad"], 0.02)
        self.assertGreaterEqual(result["resends"], 3)
        self.assertEqual(hardware.left_arm.commands, result["resends"])
        self.assertEqual(hardware.right_arm.commands, result["resends"])

    def test_tracking_gate_times_out_when_robot_does_not_move(self):
        class StaticArm:
            def rm_get_current_arm_state(self):
                return 0, {"joint": [0.0] * 7}

            def rm_movej_canfd(self, *_args, **_kwargs):
                return 0

        hardware = RealManHardware.__new__(RealManHardware)
        hardware.left_arm = StaticArm()
        hardware.right_arm = StaticArm()
        hardware._lock = threading.Lock()
        hardware.trajectory_mode = 2
        hardware.smoothing = 50
        action = [0.1] * 7 + [1000.0] + [0.1] * 7 + [1000.0]
        with self.assertRaises(SafetyError):
            hardware.wait_for_joint_target(
                action, tolerance_rad=0.01, timeout_s=0.01, resend_hz=1000,
            )

    def test_action_chunk_alignment_requires_minimum_improvement(self):
        actions = [
            [0.0] * 7 + [1000.0] + [0.0] * 7 + [1000.0],
            [0.02] * 7 + [1000.0] + [0.02] * 7 + [1000.0],
        ]
        aligned, diagnostics = align_action_chunk(
            actions, [0.02] * 7, [0.02] * 7,
            max_skip=1, min_improvement_rad=0.03,
        )
        self.assertEqual(diagnostics["best_index"], 1)
        self.assertEqual(diagnostics["selected_index"], 0)
        self.assertEqual(aligned, actions)

    def test_binary_gripper_hysteresis_by_actions(self):
        controller = BinaryGripperHysteresis(
            close_threshold=300, open_threshold=700, confirm_actions=3, confirm_seconds=0
        )
        self.assertIsNone(controller.update("right", 250, now=0.0))
        self.assertIsNone(controller.update("right", 200, now=0.1))
        self.assertEqual(controller.update("right", 100, now=0.2), 1)
        self.assertIsNone(controller.update("right", 0, now=0.3))
        self.assertIsNone(controller.update("right", 500, now=0.4))
        self.assertIsNone(controller.update("right", 800, now=0.5))
        self.assertIsNone(controller.update("right", 900, now=0.6))
        self.assertEqual(controller.update("right", 1000, now=0.7), 1000)

    def test_binary_gripper_hysteresis_by_time_and_per_side(self):
        controller = BinaryGripperHysteresis(
            close_threshold=300, open_threshold=700, confirm_actions=0, confirm_seconds=0.2
        )
        self.assertIsNone(controller.update("left", 100, now=1.0))
        self.assertIsNone(controller.update("right", 900, now=1.1))
        self.assertEqual(controller.update("left", 100, now=1.21), 1)
        self.assertEqual(controller.update("right", 900, now=1.31), 1000)

    def test_binary_gripper_hysteresis_supports_per_side_overrides(self):
        controller = BinaryGripperHysteresis(
            close_threshold={"left": 500, "right": 200},
            open_threshold=900,
            confirm_actions={"left": 1, "right": 2},
            confirm_seconds=0,
        )
        self.assertEqual(controller.update("left", 400, now=0.0), 1)
        self.assertIsNone(controller.update("right", 400, now=0.0))
        self.assertIsNone(controller.update("right", 150, now=0.1))
        self.assertEqual(controller.update("right", 150, now=0.2), 1)
        self.assertEqual(controller.close_threshold, {"left": 500.0, "right": 200.0})
        self.assertEqual(controller.confirm_actions, {"left": 1, "right": 2})

    def test_gripper_before_arm_waits_for_grippers_first(self):
        class OrderedArm:
            def __init__(self, side, events):
                self.side = side
                self.events = events

            def rm_get_current_arm_state(self):
                return 0, {"joint": [0.0] * 7}

            def rm_set_gripper_position(self, position, block, timeout):
                self.events.append((self.side, "gripper", position, block, timeout))
                return 0

            def rm_movej_canfd(self, *_args, **_kwargs):
                self.events.append((self.side, "arm"))
                return 0

        events = []
        hardware = RealManHardware.__new__(RealManHardware)
        hardware.left_arm = OrderedArm("left", events)
        hardware.right_arm = OrderedArm("right", events)
        hardware._lock = threading.Lock()
        hardware.max_joint_step_rad = 0.5
        hardware.trajectory_mode = 2
        hardware.smoothing = 50
        hardware.gripper_threshold = 5
        hardware.binary_grippers = None
        hardware._last_grippers = {"left": None, "right": None}
        hardware._executed_action_count = 0
        hardware.gripper_before_arm = True
        hardware.gripper_blocking_timeout_s = 5.0
        action = [0.0] * 7 + [500.0] + [0.0] * 7 + [600.0]
        hardware.execute_action(action)
        self.assertEqual(events, [
            ("right", "gripper", 500, True, 5),
            ("left", "gripper", 600, True, 5),
            ("right", "arm"),
            ("left", "arm"),
        ])

    def test_synchronized_frame_selection(self):
        buffers = {
            "high_rgb": [TimedFrame(0, 100_000_000, "old-high"), TimedFrame(1, 201_000_000, "high")],
            "left_wrist_rgb": [TimedFrame(0, 107_000_000, "old-left"), TimedFrame(1, 198_000_000, "left")],
            "right_wrist_rgb": [TimedFrame(0, 95_000_000, "old-right"), TimedFrame(1, 205_000_000, "right")],
        }
        images, metadata = select_synchronized_frames(buffers, 200_000_000, 10_000_000)
        self.assertEqual(images, {"high_rgb": "high", "left_wrist_rgb": "left", "right_wrist_rgb": "right"})
        self.assertEqual(metadata["skew_ms"], 7.0)
        with self.assertRaises(HardwareError):
            select_synchronized_frames(buffers, 200_000_000, 4_000_000)

    def test_deployment_camera_orientation(self):
        images = {
            name: np.arange(18, dtype=np.uint8).reshape(2, 3, 3) + index * 20
            for index, name in enumerate(("high_rgb", "left_wrist_rgb", "right_wrist_rgb"))
        }
        oriented = orient_deployment_images(images)
        np.testing.assert_array_equal(oriented["high_rgb"], images["high_rgb"])
        np.testing.assert_array_equal(oriented["left_wrist_rgb"], images["left_wrist_rgb"][::-1, ::-1])
        np.testing.assert_array_equal(oriented["right_wrist_rgb"], images["right_wrist_rgb"][::-1, ::-1])
        self.assertTrue(oriented["left_wrist_rgb"].flags.c_contiguous)
        self.assertTrue(oriented["right_wrist_rgb"].flags.c_contiguous)

    def test_buffered_camera_rig(self):
        class FakeCamera:
            def __init__(self, value):
                self.value = value

            def get(self):
                time.sleep(0.003)
                return np.full((4, 5, 3), self.value, dtype=np.uint8), None

        rig = BufferedCameraRig(
            {
                "high_rgb": FakeCamera(1),
                "left_wrist_rgb": FakeCamera(2),
                "right_wrist_rgb": FakeCamera(3),
            },
            max_skew_ms=20,
            timeout_s=1,
        )
        try:
            images, metadata = rig.capture_near(time.monotonic_ns())
            rig.wait_until_ready(timeout_s=1)
            self.assertLessEqual(metadata["skew_ms"], 20)
            self.assertEqual(int(images["high_rgb"][0, 0, 0]), 1)
            self.assertEqual(int(images["left_wrist_rgb"][0, 0, 0]), 2)
            self.assertEqual(int(images["right_wrist_rgb"][0, 0, 0]), 3)
        finally:
            rig.close()

    def test_full_control_loop(self):
        result = run_simulation(replans=2, actions_per_replan=3)
        self.assertEqual(result["executed_actions"], 6)
        self.assertEqual(result["requests"], 2)

    def test_background_camera_video_recording(self):
        import cv2

        with tempfile.TemporaryDirectory() as root:
            recorder = BackgroundCameraVideoRecorder(root, fps=30, codec="MJPG")
            run_dir = recorder.run_dir
            for sequence in range(3):
                for index, name in enumerate(
                    ("high_rgb", "left_wrist_rgb", "right_wrist_rgb")
                ):
                    frame = np.full((24, 32, 3), sequence * 50 + index * 20, dtype=np.uint8)
                    recorder.submit(name, frame, sequence=sequence, timestamp_ns=sequence * 1_000_000)
            recorder.close()
            for name in ("high_rgb", "left_wrist_rgb", "right_wrist_rgb"):
                video = run_dir / f"{name}.avi"
                self.assertTrue(video.is_file())
                self.assertGreater(video.stat().st_size, 0)
                capture = cv2.VideoCapture(str(video))
                self.assertTrue(capture.isOpened())
                frame_count = 0
                while capture.read()[0]:
                    frame_count += 1
                capture.release()
                self.assertEqual(frame_count, 3)
            metadata = [json.loads(line) for line in (run_dir / "camera_frames.jsonl").read_text().splitlines()]
            self.assertEqual(len(metadata), 9)
            self.assertEqual(sorted({row["sequence"] for row in metadata}), [0, 1, 2])

    def test_raw_state_and_action_layout(self):
        hardware = MockRealManHardware(height=8, width=8)
        hardware.left_joints[:] = np.arange(7) + 10
        hardware.right_joints[:] = np.arange(7) + 20
        _, state = hardware.observe()
        request = make_infer_request(hardware.images, state, "task")
        actions, _ = MockPolicy(action_horizon=1, offset_rad=0).infer(request)
        self.assertEqual(actions[0][0:7], hardware.right_joints.tolist())
        self.assertEqual(actions[0][8:15], hardware.left_joints.tolist())
        self.assertEqual(actions[0][7], 1000.0)
        self.assertEqual(actions[0][15], 1000.0)


if __name__ == "__main__":
    unittest.main()
