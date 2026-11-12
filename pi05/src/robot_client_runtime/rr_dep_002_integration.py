"""RR-DEP-002 one-request, no-device WebSocket integration validation."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any

from .action_validator import ActionValidator
from .pi05_action_adapter import Pi05ActionAdapter
from .schemas import ACTION_ORDER, ACTION_SCHEMA_ID, ValidationError


PROTOCOL_SCHEMA_ID = "rr_dep_openpi_ws_v1"
EXPERIMENT_ID = "RR-DEP-002"
RUN_ID = "RR-DEP-002-RUN-001"
RESERVED_CONTEXT_KEY = "__rr_context__"
REQUIRED_CONTEXT_FIELDS = (
    "schema_id",
    "experiment_id",
    "run_id",
    "service_session_id",
    "request_id",
    "request_ordinal",
    "client_send_utc",
    "client_send_monotonic_ns",
)

ZERO_PHYSICAL_ACCESS_COUNTERS = {
    "backend_calls": 0,
    "sdk_calls": 0,
    "device_opens": 0,
    "motion_commands": 0,
}


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError("invalid_mapping", f"{field} must be a mapping")
    return value


def _require_exact(value: Any, expected: Any, *, field: str) -> None:
    if value != expected:
        raise ValidationError("contract_mismatch", f"{field} expected {expected!r}, received {value!r}")


class RRDep002NoMotionIntegration:
    """Validates one official OpenPI response without importing any robot backend."""

    def __init__(
        self,
        *,
        policy_client: Any,
        service_ready: Mapping[str, Any],
        expected_server_metadata: Mapping[str, Any],
        action_output_contract: Mapping[str, Any],
        explicit_limits: Mapping[int, tuple[float, float]] | None = None,
    ) -> None:
        self.policy_client = policy_client
        self.service_ready = dict(service_ready)
        self.expected_server_metadata = dict(expected_server_metadata)
        self.action_output_contract = dict(action_output_contract)
        self.validator = ActionValidator(limits=explicit_limits)
        self.source_dimension, self.horizon, self.confirmed_32d_padding = self._validate_static_contracts()

    def _validate_static_contracts(self) -> tuple[int, int, bool]:
        _require_exact(self.service_ready.get("schema_id"), PROTOCOL_SCHEMA_ID, field="service_ready.schema_id")
        _require_exact(self.service_ready.get("run_id"), RUN_ID, field="service_ready.run_id")
        _require_exact(
            self.service_ready.get("status"),
            "compute_service_ready_for_exactly_one_Client_request",
            field="service_ready.status",
        )
        _require_exact(self.service_ready.get("bind"), "127.0.0.1:8000", field="service_ready.bind")
        _require_exact(self.service_ready.get("non_loopback_listener"), False, field="non_loopback_listener")

        _require_exact(self.action_output_contract.get("schema_id"), PROTOCOL_SCHEMA_ID, field="contract.schema_id")
        _require_exact(self.action_output_contract.get("experiment_id"), EXPERIMENT_ID, field="contract.experiment_id")
        _require_exact(self.action_output_contract.get("run_id"), RUN_ID, field="contract.run_id")
        output = _require_mapping(self.action_output_contract.get("output"), field="contract.output")
        actions = _require_mapping(output.get("actions"), field="contract.output.actions")
        shape = actions.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValidationError("invalid_action_shape", "contract action shape must be [H,D]")
        horizon, dimension = shape
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValidationError("invalid_horizon", "contract horizon must be a positive integer")
        if dimension not in {16, 28, 32}:
            raise ValidationError("unknown_action_dimension", f"contract serves unsupported D={dimension!r}")
        _require_exact(actions.get("dtype"), "float32", field="contract.output.actions.dtype")
        _require_exact(actions.get("finite_required"), True, field="contract.output.actions.finite_required")
        _require_exact(actions.get("order"), "native_left_first", field="contract.output.actions.order")

        ready_action = _require_mapping(self.service_ready.get("action_output_contract"), field="ready.action")
        _require_exact(ready_action.get("shape"), [horizon, dimension], field="ready.action.shape")
        _require_exact(ready_action.get("dtype"), "float32", field="ready.action.dtype")

        confirmed_32d_padding = False
        if dimension == 32:
            evidence = _require_mapping(
                self.action_output_contract.get("dimension_evidence"), field="contract.dimension_evidence"
            )
            _require_exact(evidence.get("native_action_dim"), 28, field="dimension.native_action_dim")
            _require_exact(evidence.get("model_action_dim"), 32, field="dimension.model_action_dim")
            _require_exact(evidence.get("model_padding_indices"), [28, 29, 30, 31], field="dimension.padding")
            _require_exact(
                evidence.get("32D_server_mode"),
                "first_28_native_last_4_padding",
                field="dimension.32D_server_mode",
            )
            confirmed_32d_padding = True
        return dimension, horizon, confirmed_32d_padding

    def validate_server_metadata(self, actual: Mapping[str, Any]) -> None:
        actual_mapping = _require_mapping(actual, field="server_metadata")
        if dict(actual_mapping) != self.expected_server_metadata:
            raise ValidationError("server_metadata_mismatch", "actual metadata differs from frozen Compute metadata")
        _require_exact(actual_mapping.get("schema_id"), PROTOCOL_SCHEMA_ID, field="metadata.schema_id")
        _require_exact(actual_mapping.get("experiment_id"), EXPERIMENT_ID, field="metadata.experiment_id")
        _require_exact(actual_mapping.get("run_id"), RUN_ID, field="metadata.run_id")
        _require_exact(
            actual_mapping.get("service_session_id"),
            self.service_ready.get("service_session_id"),
            field="metadata.service_session_id",
        )
        metadata_action = _require_mapping(actual_mapping.get("action_output_contract"), field="metadata.action")
        _require_exact(
            metadata_action.get("shape"),
            [self.horizon, self.source_dimension],
            field="metadata.action.shape",
        )
        _require_exact(metadata_action.get("dtype"), "float32", field="metadata.action.dtype")

    @staticmethod
    def validate_context(context: Mapping[str, Any]) -> dict[str, Any]:
        context_mapping = _require_mapping(context, field="request_context")
        if set(context_mapping) != set(REQUIRED_CONTEXT_FIELDS):
            raise ValidationError("context_fields_mismatch", "request context fields do not match protocol")
        _require_exact(context_mapping.get("schema_id"), PROTOCOL_SCHEMA_ID, field="context.schema_id")
        _require_exact(context_mapping.get("experiment_id"), EXPERIMENT_ID, field="context.experiment_id")
        _require_exact(context_mapping.get("run_id"), RUN_ID, field="context.run_id")
        _require_exact(context_mapping.get("request_ordinal"), 1, field="context.request_ordinal")
        if not isinstance(context_mapping.get("request_id"), str):
            raise ValidationError("invalid_request_id", "request_id must be a string")
        if not isinstance(context_mapping.get("service_session_id"), str):
            raise ValidationError("invalid_session_id", "service_session_id must be a string")
        monotonic_ns = context_mapping.get("client_send_monotonic_ns")
        if isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int) or monotonic_ns < 0:
            raise ValidationError("invalid_monotonic_time", "client_send_monotonic_ns must be nonnegative int")
        if not isinstance(context_mapping.get("client_send_utc"), str):
            raise ValidationError("invalid_utc_time", "client_send_utc must be a string")
        return dict(context_mapping)

    def run_one_request(self, observation: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
        context_dict = self.validate_context(context)
        _require_exact(
            context_dict["service_session_id"],
            self.service_ready.get("service_session_id"),
            field="context.service_session_id",
        )
        request = dict(_require_mapping(observation, field="observation"))
        if RESERVED_CONTEXT_KEY in request:
            raise ValidationError("reserved_context_collision", RESERVED_CONTEXT_KEY)
        request[RESERVED_CONTEXT_KEY] = context_dict

        response = _require_mapping(self.policy_client.infer(request), field="response")
        if not {"actions", "server_timing", RESERVED_CONTEXT_KEY}.issubset(response):
            raise ValidationError("missing_response_field", "response lacks required protocol fields")
        _require_exact(response[RESERVED_CONTEXT_KEY], context_dict, field="response.__rr_context__")

        import numpy as np

        actions = response["actions"]
        if not isinstance(actions, np.ndarray):
            raise ValidationError("invalid_actions_type", "response actions must be a NumPy array")
        if list(actions.shape) != [self.horizon, self.source_dimension]:
            raise ValidationError("wrong_action_shape", f"received {list(actions.shape)!r}")
        if actions.dtype != np.dtype("float32"):
            raise ValidationError("wrong_action_dtype", f"received {actions.dtype}")
        if not np.isfinite(actions).all():
            raise ValidationError("nonfinite_actions", "response actions must be finite")

        timing = _require_mapping(response["server_timing"], field="response.server_timing")
        infer_ms = timing.get("infer_ms")
        if isinstance(infer_ms, bool) or not isinstance(infer_ms, Real):
            raise ValidationError("invalid_server_timing", "infer_ms must be numeric")
        infer_ms_float = float(infer_ms)
        if not math.isfinite(infer_ms_float) or infer_ms_float < 0:
            raise ValidationError("invalid_server_timing", "infer_ms must be finite and nonnegative")

        adapter = Pi05ActionAdapter(
            server_first_28_native_last_4_padding_confirmed=self.confirmed_32d_padding
        )
        canonical_chunk = adapter.adapt_chunk(actions.tolist())
        validated_chunk = [self.validator.validate(row, schema_id=ACTION_SCHEMA_ID) for row in canonical_chunk]
        return {
            "context": context_dict,
            "server_timing": dict(timing),
            "source_shape": [self.horizon, self.source_dimension],
            "source_dtype": "float32",
            "canonical_shape": [self.horizon, 16],
            "canonical_schema_id": ACTION_SCHEMA_ID,
            "canonical_order": list(ACTION_ORDER),
            "canonical_actions": validated_chunk,
            **ZERO_PHYSICAL_ACCESS_COUNTERS,
        }
