from __future__ import annotations

import math
import socket
import unittest
import uuid

import numpy as np

from robot_client_runtime.rr_dep_003_motion import MotionRejected
from robot_client_runtime.rr_dep_004 import (
    capture_generator_gripper_semantics,
    evaluate_robot_gate_004,
    map_action28_to_action16,
    normalize_zero_error_entries,
    project_model_row,
    require_uuid4,
    resolve_gripper_branch,
    select_free_loopback_port,
    validate_service_metadata,
)


def _binding(side: str) -> dict:
    min_pos = [-178, -130, -178, -135, -178, -130, -360]
    max_pos = [178, 130, 178, 135, 178, 130, 360]
    drive_min = [-183, -135, -183, -140, -360, -133, -365]
    drive_max = [183, 135, 183, 140, 360, 133, 365]
    return {
        "sdk_api_version": "1.1.6",
        "rm_get_robot_info": [0, {"arm_dof": 7, "arm_model": "RM_75", "force_type": "6FB"}],
        "rm_get_arm_software_info": [0, {"product_version": "RM75-6FB", "ctrl_info": {"version": "V1.7.0-1a610b9"}}],
        "rm_get_joint_min_pos": [0, min_pos],
        "rm_get_joint_max_pos": [0, max_pos],
        "rm_get_joint_drive_min_pos": [0, drive_min],
        "rm_get_joint_drive_max_pos": [0, drive_max],
        "rm_get_joint_max_speed": [0, [30] * 7],
        "rm_get_joint_max_acc": [0, [600] * 7],
        "rm_get_joint_en_state": [0, [1] * 7],
        "rm_get_joint_err_flag": {"return_code": 0, "err_flag": [0] * 7, "brake_state": [0] * 7},
        "rm_get_controller_state": {"return_code": 0, "system_error": 0},
        "rm_get_arm_power_state": [0, 1],
        "rm_get_collision_stage": [0, 0 if side == "left" else 6],
        "rm_get_collision_detection": [-2, 0],
        "rm_get_self_collision_enable": [0, False],
        "raw_state_errors": ["0"],
    }


class RRDep004Tests(unittest.TestCase):
    def test_zero_error_normalization_accepts_only_numeric_zero(self):
        self.assertEqual(normalize_zero_error_entries(["0", 0, 0.0, "0.0"]), ())
        for invalid in (["1"], [-1], [float("nan")], [True], ["alarm"]):
            with self.subTest(invalid=invalid), self.assertRaises(MotionRejected):
                normalize_zero_error_entries(invalid)

    def test_action_mapping_is_exact_left_first(self):
        actions = np.arange(50 * 28, dtype=np.float32).reshape(50, 28)
        mapped = map_action28_to_action16(actions)
        self.assertEqual(mapped.shape, (50, 16))
        self.assertEqual(mapped.dtype, np.float32)
        self.assertEqual(mapped[0].tolist(), list(range(8)) + list(range(14, 22)))
        with self.assertRaises(MotionRejected):
            map_action28_to_action16(actions.astype(np.float64))

    def test_projection_clips_each_joint_and_holds_unproven_grippers(self):
        row = [0.1] * 7 + [0.25] + [-0.1] * 7 + [0.75]
        limits = {"left": [-1.0] * 7, "right": [-1.0] * 7}
        upper = {"left": [1.0] * 7, "right": [1.0] * 7}
        result = project_model_row(row, [0.0] * 7, [0.0] * 7, limits, upper)
        self.assertTrue(all(math.isclose(value, 0.01) for value in result["projected_joint_targets"]["left"]))
        self.assertTrue(all(math.isclose(value, -0.01) for value in result["projected_joint_targets"]["right"]))
        self.assertEqual(result["gripper_branch"], "hold_unproven_conversion")
        self.assertEqual(result["gripper_commands"], 0)

    def test_gripper_execute_and_hold_branches_are_explicit(self):
        hold = resolve_gripper_branch({"left": 0.2, "right": 0.8}, capture_to_fastwam_conversion_proven=False)
        self.assertEqual(hold, {"gripper_branch": "hold_unproven_conversion", "commands": {}})
        execute = resolve_gripper_branch({"left": 0.2, "right": 0.8}, capture_to_fastwam_conversion_proven=True)
        self.assertEqual(execute["commands"], {"left": 200, "right": 800})
        with self.assertRaises(MotionRejected):
            resolve_gripper_branch({"left": 2.0, "right": 0.5}, capture_to_fastwam_conversion_proven=True)

    def test_capture_generator_offline_and_clip_semantics(self):
        self.assertIsNone(capture_generator_gripper_semantics("disable"))
        offline = capture_generator_gripper_semantics("offline")
        self.assertEqual(offline["gripper_open"], 0.0)
        self.assertIsNone(offline["gripper_raw_rm_plus"])
        online = capture_generator_gripper_semantics({"pos": [1001, 500]})
        self.assertEqual(online["gripper_open"], 1.0)
        self.assertEqual(online["gripper_raw_rm_plus"], 1001)
        low = capture_generator_gripper_semantics({"pos": [-2]})
        self.assertEqual(low["gripper_open"], 0.0)

    def test_gate_accepts_only_frozen_known_zero_semantics(self):
        stability = {
            side: {"initial_rad": [0.0] * 7, "samples": 41, "measured_hz": 20.0, "max_joint_span_rad": 0.0}
            for side in ("left", "right")
        }
        result = evaluate_robot_gate_004({side: _binding(side) for side in ("left", "right")}, stability)
        self.assertEqual(result["status"], "pass")
        bad = {side: _binding(side) for side in ("left", "right")}
        bad["right"]["raw_state_errors"] = ["3"]
        with self.assertRaises(MotionRejected):
            evaluate_robot_gate_004(bad, stability)

    def test_uuid_and_protocol_identity_rejection(self):
        value = str(uuid.uuid4())
        self.assertEqual(require_uuid4(value, "request_id"), value)
        with self.assertRaises(MotionRejected):
            require_uuid4("not-a-uuid", "request_id")
        metadata = {
            "schema_id": "rr_dep_openpi_robot_live_v1",
            "experiment_id": "RR-DEP-004",
            "run_id": "RR-DEP-004-RUN-001",
            "service_session_id": "495c2511-265c-4bec-b814-891d9468d1b7",
            "bind": "127.0.0.1:8010",
            "port": 8010,
        }
        validate_service_metadata(metadata, 18100)
        metadata["run_id"] = "wrong"
        with self.assertRaises(MotionRejected):
            validate_service_metadata(metadata, 18100)

    def test_first_free_port_skips_occupied_candidate(self):
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        try:
            self.assertEqual(select_free_loopback_port(port, port + 1), port + 1)
        finally:
            occupied.close()


if __name__ == "__main__":
    unittest.main()
