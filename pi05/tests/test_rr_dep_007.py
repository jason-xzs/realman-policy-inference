from __future__ import annotations

import inspect
from unittest import mock
import unittest
import uuid

from robot_client_runtime.rr_dep_003_motion import MotionRejected
from robot_client_runtime.rr_dep_007 import (
    PRESET,
    ProjectedCommandExecutor,
    build_parser,
    preset_args,
    validate_service_metadata,
)


class RRDep007Tests(unittest.TestCase):
    def test_researcher_cli_exposes_only_preset(self):
        parser = build_parser()
        args = parser.parse_args(["--preset", PRESET])
        self.assertEqual(args.preset, PRESET)
        option_strings = {
            value
            for action in parser._actions
            for value in action.option_strings
            if value not in ("-h", "--help")
        }
        self.assertEqual(option_strings, {"--preset"})

    def test_preset_freezes_motion_camera_and_mapping_limits(self):
        args = preset_args()
        self.assertEqual((args.image_width, args.image_height, args.camera_fps), (640, 480, 30))
        self.assertEqual(args.model_image_size, 224)
        self.assertEqual(args.joint_delta_limit_rad, 0.01)
        self.assertEqual(args.speed_percent, 5)
        self.assertEqual(args.target_tolerance_rad, 0.005)
        self.assertEqual(args.state_stale_sec, 0.25)

    def test_service_context_is_discovered_and_exact(self):
        session_id = str(uuid.uuid4())
        metadata = {
            "task_id": "RR-DEP-007",
            "experiment_id": "RR-DEP-007",
            "run_id": "RR-DEP-007-RUN-001",
            "schema_id": "rr_dep_openpi_robot_live_v3",
            "bind": "127.0.0.1:8020",
            "port": 8020,
            "checkpoint_role": "integration_only",
            "checkpoint_selection_status": "not_selected",
            "maximum_unique_matching_Client_requests": 2,
            "execution_intents": ["no_motion_validation", "one_projected_physical_batch"],
            "service_session_id": session_id,
            "stable_runtime_commit": "a" * 40,
            "stable_runtime_clean": True,
            "OpenPI_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
            "config": "pi05_realman_joint_201_step30000_ema",
            "checkpoint": "/share/jhk/phywam-mope-real/checkpoints/finetuned/pi05/RR-BL-003/RR-BL-003-RUN-001/attempt-001/30000",
            "artifact_root": "/share/jhk/phywam-mope-real/runs/RR-DEP-007/RR-DEP-007-RUN-001/compute",
        }
        result = validate_service_metadata(metadata)
        self.assertEqual(result["service_session_id"], session_id)
        for field in ("task_id", "schema_id", "maximum_unique_matching_Client_requests"):
            bad = dict(metadata)
            bad[field] = "wrong"
            with self.subTest(field=field), self.assertRaises(MotionRejected):
                validate_service_metadata(bad)

    def test_executor_enforces_projection_before_send_not_raw_feedback_envelope(self):
        source = inspect.getsource(ProjectedCommandExecutor.execute)
        self.assertIn("pre_send_projection_exceeded", source)
        self.assertNotIn("unexpected_motion_envelope", source)
        self.assertNotIn("abs(actual - initial)", source)
        self.assertIn("target_tolerance_rad", source)

    def test_help_has_no_live_import_or_access(self):
        with mock.patch("robot_client_runtime.rr_dep_007.open_live_stack") as live:
            with self.assertRaises(SystemExit) as caught:
                build_parser().parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)
        live.assert_not_called()


if __name__ == "__main__":
    unittest.main()
