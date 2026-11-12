from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
import sys
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_client_runtime.action_validator import ActionValidator
from robot_client_runtime.chunk_scheduler import ChunkScheduler
from robot_client_runtime.cli import ZERO_LIVE_ACCESS_COUNTERS, build_fixture_dry_run_result, main
from robot_client_runtime.observation_adapter import ObservationAdapter
from robot_client_runtime.pi05_action_adapter import Pi05ActionAdapter
from robot_client_runtime.pi05_policy_client import Pi05PolicyClient
from robot_client_runtime.realman_adapter import RealManDualArmAdapter
from robot_client_runtime.rr_dep_002_integration import RRDep002NoMotionIntegration
from robot_client_runtime.schemas import IntegrationNotEnabledError, ValidationError


class _FakeBackend:
    def __init__(self) -> None:
        self.calls = []

    def execute_targets(self, first_target, second_target, *, order):
        record = (list(first_target), list(second_target), order)
        self.calls.append(record)
        return record


class RuntimeContractTests(unittest.TestCase):
    def test_action28_to_action16_exact_indices(self):
        action = [float(index) for index in range(28)]
        self.assertEqual(
            Pi05ActionAdapter().adapt_action(action),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
             14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
        )

    def test_action16_identity(self):
        action = [index / 10 for index in range(16)]
        adapted = Pi05ActionAdapter().adapt_action(action)
        self.assertEqual(adapted, action)
        self.assertIsNot(adapted, action)

    def test_action_chunks_preserve_horizon_and_left_first_order(self):
        chunk = [[float(index + offset) for index in range(28)] for offset in (0, 100)]
        adapted = Pi05ActionAdapter().adapt_chunk(chunk)
        self.assertEqual(len(adapted), 2)
        self.assertEqual(adapted[0][0:8], chunk[0][0:8])
        self.assertEqual(adapted[0][8:16], chunk[0][14:22])
        self.assertEqual(adapted[1][0:8], chunk[1][0:8])
        self.assertEqual(adapted[1][8:16], chunk[1][14:22])

    def test_wrong_dimension_and_nonfinite_rejection(self):
        adapter = Pi05ActionAdapter()
        invalid_actions = (
            [],
            [0.0] * 15,
            [0.0] * 17,
            [0.0] * 27,
            [0.0] * 29,
            [0.0] * 15 + [True],
            [0.0] * 15 + ["1"],
            [0.0] * 15 + [math.nan],
            [0.0] * 15 + [math.inf],
        )
        for action in invalid_actions:
            with self.subTest(action_length=len(action)):
                with self.assertRaises(ValidationError):
                    adapter.adapt_action(action)
        with self.assertRaisesRegex(ValidationError, "empty_chunk"):
            adapter.adapt_chunk([])
        with self.assertRaisesRegex(ValidationError, "empty_row"):
            adapter.adapt_chunk([[0.0] * 28, []])
        with self.assertRaisesRegex(ValidationError, "ragged_chunk"):
            adapter.adapt_chunk([[0.0] * 28, [0.0] * 16])

    def test_32D_unpadding_disabled_without_confirmed_mode(self):
        with self.assertRaisesRegex(ValidationError, "unconfirmed_32d_layout"):
            Pi05ActionAdapter().adapt_action([0.0] * 32)

    def test_32D_confirmed_mode_extracts_first_28_then_maps_to_16(self):
        action = [float(index) for index in range(32)]
        adapter = Pi05ActionAdapter(server_first_28_native_last_4_padding_confirmed=True)
        self.assertEqual(
            adapter.adapt_action(action),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
             14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
        )

    def test_no_invented_gripper_scale_or_clamp(self):
        action = [0.0] * 28
        action[7] = -123.5
        action[21] = 9876.5
        adapted = Pi05ActionAdapter().adapt_action(action)
        self.assertEqual(adapted[7], -123.5)
        self.assertEqual(adapted[15], 9876.5)

    def test_validator_uses_only_explicit_configured_limits(self):
        action = [9999.0] * 16
        default_validator = ActionValidator()
        self.assertEqual(default_validator.validate(action), action)
        self.assertEqual(default_validator.rejection_records, [])
        self.assertEqual(default_validator.execution_acknowledgements, [])

        bounded = ActionValidator(limits={0: (-1.0, 1.0)})
        with self.assertRaisesRegex(ValidationError, "explicit_limit_rejection"):
            bounded.validate(action)
        self.assertEqual(len(bounded.rejection_records), 1)
        self.assertEqual(bounded.execution_acknowledgements, [])

    def test_chunk_scheduler_order_and_empty_behavior(self):
        scheduler = ChunkScheduler()
        self.assertIsNone(scheduler.pop_next())
        self.assertEqual(scheduler.enqueue([]), 0)
        self.assertEqual(scheduler.enqueue([[1.0] * 16, [2.0] * 16]), 2)
        self.assertEqual(scheduler.pending, 2)
        self.assertEqual(scheduler.pop_next(), [1.0] * 16)
        self.assertEqual(scheduler.pop_next(), [2.0] * 16)
        self.assertIsNone(scheduler.pop_next())

    def test_OpenPI_wrapper_lazy_import_and_dry_run_connection_rejection(self):
        client = Pi05PolicyClient()
        with mock.patch("importlib.import_module") as importer:
            with self.assertRaisesRegex(IntegrationNotEnabledError, "integration_not_enabled"):
                client.connect()
            importer.assert_not_called()
        self.assertEqual(client.policy_connections, 0)

    def test_RealMan_fake_backend_split_or_internal_reorder(self):
        action = [float(index) for index in range(16)]
        left_backend = _FakeBackend()
        left_adapter = RealManDualArmAdapter(left_backend, backend_order="left_first")
        left_adapter.dispatch(action)
        self.assertEqual(left_backend.calls[0], (action[:8], action[8:], "left_first"))

        right_backend = _FakeBackend()
        right_adapter = RealManDualArmAdapter(right_backend, backend_order="right_first")
        right_adapter.dispatch(action)
        self.assertEqual(right_backend.calls[0], (action[8:], action[:8], "right_first"))

        with self.assertRaisesRegex(IntegrationNotEnabledError, "integration_not_enabled"):
            RealManDualArmAdapter().dispatch(action)

    def test_mock_dry_run_makes_zero_network_device_or_motion_calls(self):
        result = build_fixture_dry_run_result()
        self.assertEqual(result["status"], "fixture_mock_dry_run_pass")
        for counter, expected in ZERO_LIVE_ACCESS_COUNTERS.items():
            self.assertEqual(result[counter], expected)
        self.assertEqual(result["backend_fixture_calls"], 1)

    def test_observation_adapter_maps_roles_without_transform(self):
        images = {
            "high_rgb": object(),
            "left_wrist_rgb": object(),
            "right_wrist_rgb": object(),
        }
        result = ObservationAdapter().adapt(
            state=[0.0] * 28,
            images=images,
            prompt="task",
        )
        self.assertIs(result["base_0_rgb"], images["high_rgb"])
        self.assertIs(result["left_wrist_0_rgb"], images["left_wrist_rgb"])
        self.assertIs(result["right_wrist_0_rgb"], images["right_wrist_rgb"])

    def test_cli_rejects_live_flags_before_side_effects(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["--connect-policy", "--execute-motion"])
        self.assertEqual(exit_code, 2)
        self.assertIn('"error": "integration_not_enabled"', output.getvalue())
        for counter in ZERO_LIVE_ACCESS_COUNTERS:
            self.assertIn(f'"{counter}": 0', output.getvalue())

    def test_policy_wrapper_allows_only_one_connection_and_one_request(self):
        fake_client = mock.Mock()
        fake_client.get_server_metadata.return_value = {"server": "metadata"}
        fake_client.infer.return_value = {"response": 1}
        fake_module = mock.Mock(WebsocketClientPolicy=mock.Mock(return_value=fake_client))
        client = Pi05PolicyClient(live_enabled=True, client_kwargs={"host": "127.0.0.1", "port": 18000})
        with mock.patch("importlib.import_module", return_value=fake_module):
            client.connect()
        self.assertEqual(client.get_server_metadata(), {"server": "metadata"})
        self.assertEqual(client.infer({"fixture": True}), {"response": 1})
        with self.assertRaisesRegex(RuntimeError, "second_request_forbidden"):
            client.infer({"fixture": True})
        self.assertEqual(client.policy_connections, 1)
        self.assertEqual(client.requests, 1)
        self.assertEqual(client.responses, 1)
        self.assertEqual(client.duplicate_executions, 1)

    def test_RR_DEP_002_D28_response_validates_without_backend(self):
        import numpy as np

        context = {
            "schema_id": "rr_dep_openpi_ws_v1",
            "experiment_id": "RR-DEP-002",
            "run_id": "RR-DEP-002-RUN-001",
            "service_session_id": "fixture-session",
            "request_id": "fixture-request",
            "request_ordinal": 1,
            "client_send_utc": "2026-08-25T00:00:00+00:00",
            "client_send_monotonic_ns": 1,
        }
        service_ready = {
            "schema_id": "rr_dep_openpi_ws_v1",
            "run_id": "RR-DEP-002-RUN-001",
            "status": "compute_service_ready_for_exactly_one_Client_request",
            "bind": "127.0.0.1:8000",
            "non_loopback_listener": False,
            "service_session_id": "fixture-session",
            "action_output_contract": {"shape": [2, 28], "dtype": "float32"},
        }
        action_contract = {
            "schema_id": "rr_dep_openpi_ws_v1",
            "experiment_id": "RR-DEP-002",
            "run_id": "RR-DEP-002-RUN-001",
            "output": {
                "actions": {
                    "shape": [2, 28],
                    "dtype": "float32",
                    "finite_required": True,
                    "order": "native_left_first",
                }
            },
        }
        metadata = {
            "schema_id": "rr_dep_openpi_ws_v1",
            "experiment_id": "RR-DEP-002",
            "run_id": "RR-DEP-002-RUN-001",
            "service_session_id": "fixture-session",
            "action_output_contract": {"shape": [2, 28], "dtype": "float32"},
        }
        response = {
            "actions": np.arange(56, dtype=np.float32).reshape(2, 28),
            "server_timing": {"infer_ms": 1.5},
            "__rr_context__": context,
        }
        policy = mock.Mock()
        policy.infer.return_value = response
        integration = RRDep002NoMotionIntegration(
            policy_client=policy,
            service_ready=service_ready,
            expected_server_metadata=metadata,
            action_output_contract=action_contract,
        )
        integration.validate_server_metadata(metadata)
        result = integration.run_one_request({"fixture": True}, context)
        self.assertEqual(result["canonical_shape"], [2, 16])
        self.assertEqual(result["canonical_actions"][0], [*range(8), *range(14, 22)])
        self.assertEqual(result["backend_calls"], 0)
        self.assertEqual(result["sdk_calls"], 0)
        self.assertEqual(result["device_opens"], 0)
        self.assertEqual(result["motion_commands"], 0)

    def test_RR_DEP_002_rejects_metadata_mismatch_before_request(self):
        service_ready = {
            "schema_id": "rr_dep_openpi_ws_v1",
            "run_id": "RR-DEP-002-RUN-001",
            "status": "compute_service_ready_for_exactly_one_Client_request",
            "bind": "127.0.0.1:8000",
            "non_loopback_listener": False,
            "service_session_id": "fixture-session",
            "action_output_contract": {"shape": [2, 28], "dtype": "float32"},
        }
        action_contract = {
            "schema_id": "rr_dep_openpi_ws_v1",
            "experiment_id": "RR-DEP-002",
            "run_id": "RR-DEP-002-RUN-001",
            "output": {
                "actions": {
                    "shape": [2, 28],
                    "dtype": "float32",
                    "finite_required": True,
                    "order": "native_left_first",
                }
            },
        }
        integration = RRDep002NoMotionIntegration(
            policy_client=mock.Mock(),
            service_ready=service_ready,
            expected_server_metadata={"service_session_id": "fixture-session"},
            action_output_contract=action_contract,
        )
        with self.assertRaisesRegex(ValidationError, "server_metadata_mismatch"):
            integration.validate_server_metadata({"service_session_id": "wrong-session"})


if __name__ == "__main__":
    unittest.main()
