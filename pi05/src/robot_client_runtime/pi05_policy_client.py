"""Lazy optional wrapper for OpenPI's built-in WebSocket client."""

from __future__ import annotations

import importlib
from typing import Any

from .schemas import IntegrationNotEnabledError


class Pi05PolicyClient:
    transport_contract = "openpi_client.WebsocketClientPolicy"

    def __init__(self, *, live_enabled: bool = False, client_kwargs: dict[str, Any] | None = None):
        if not isinstance(live_enabled, bool):
            raise TypeError("live_enabled must be an explicit boolean")
        self.live_enabled = live_enabled
        self.client_kwargs = dict(client_kwargs or {})
        self._client: Any | None = None
        self.policy_connections = 0
        self.requests = 0
        self.responses = 0
        self.retries = 0
        self.reconnects = 0
        self.duplicate_executions = 0

    def _require_live_enabled(self) -> None:
        if not self.live_enabled:
            raise IntegrationNotEnabledError(
                "OpenPI WebSocket live methods are disabled for RR-DEP-001"
            )

    def connect(self) -> Any:
        self._require_live_enabled()
        if self._client is not None or self.policy_connections != 0:
            self.duplicate_executions += 1
            raise RuntimeError("duplicate_policy_connection_forbidden")
        module = importlib.import_module("openpi_client")
        policy_class = getattr(module, "WebsocketClientPolicy")
        self._client = policy_class(**self.client_kwargs)
        self.policy_connections += 1
        return self._client

    def get_server_metadata(self) -> dict[str, Any]:
        self._require_live_enabled()
        if self._client is None:
            raise RuntimeError("policy_not_connected")
        metadata = self._client.get_server_metadata()
        if not isinstance(metadata, dict):
            raise RuntimeError("invalid_server_metadata")
        return metadata

    def infer(self, observation: dict[str, Any]) -> Any:
        self._require_live_enabled()
        if self._client is None:
            raise RuntimeError("policy_not_connected")
        if self.requests != 0:
            self.duplicate_executions += 1
            raise RuntimeError("second_request_forbidden")
        self.requests += 1
        response = self._client.infer(observation)
        self.responses += 1
        return response

    def close(self) -> None:
        self._require_live_enabled()
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
        self._client = None
