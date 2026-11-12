"""Fixture/mock-only command line interface for RR-DEP-001."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from .action_validator import ActionValidator
from .chunk_scheduler import ChunkScheduler
from .observation_adapter import ObservationAdapter
from .pi05_action_adapter import Pi05ActionAdapter
from .realman_adapter import RealManDualArmAdapter


ZERO_LIVE_ACCESS_COUNTERS = {
    "policy_connections": 0,
    "network_calls": 0,
    "device_opens": 0,
    "sdk_calls": 0,
    "motion_commands": 0,
}


class _FixtureBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute_targets(
        self,
        first_target: list[float],
        second_target: list[float],
        *,
        order: str,
    ) -> dict[str, Any]:
        call = {
            "order": order,
            "first_target": list(first_target),
            "second_target": list(second_target),
            "fixture_only": True,
        }
        self.calls.append(call)
        return call


def build_fixture_dry_run_result() -> dict[str, Any]:
    source_action = [float(index) for index in range(28)]
    action_adapter = Pi05ActionAdapter()
    canonical = action_adapter.adapt_action(source_action)
    validated = ActionValidator().validate(canonical)

    scheduler = ChunkScheduler()
    scheduler.enqueue([validated])
    scheduled = scheduler.pop_next()

    fixture_backend = _FixtureBackend()
    robot_adapter = RealManDualArmAdapter(fixture_backend, backend_order="right_first")
    dispatch_record = robot_adapter.dispatch(scheduled)

    image_markers = {
        "high_rgb": "fixture-high",
        "left_wrist_rgb": "fixture-left-wrist",
        "right_wrist_rgb": "fixture-right-wrist",
    }
    observation = ObservationAdapter().adapt(
        state=[float(index) for index in range(28)],
        images=image_markers,
        prompt="fixture prompt",
    )

    return {
        "status": "fixture_mock_dry_run_pass",
        "schema_id": "realman.dual_arm.absolute.v1",
        "action28_to_action16": canonical,
        "backend_fixture_calls": len(fixture_backend.calls),
        "backend_dispatch_order": dispatch_record["order"],
        "observation_keys": sorted(observation),
        **ZERO_LIVE_ACCESS_COUNTERS,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true", help="run the only enabled mode")
    parser.add_argument("--live-sensing", action="store_true")
    parser.add_argument("--connect-policy", action="store_true")
    parser.add_argument("--open-device", action="store_true")
    parser.add_argument("--enable-sdk", action="store_true")
    parser.add_argument("--execute-motion", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    forbidden_requested = any(
        (
            args.live_sensing,
            args.connect_policy,
            args.open_device,
            args.enable_sdk,
            args.execute_motion,
        )
    )
    if forbidden_requested:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "error": "integration_not_enabled",
                    **ZERO_LIVE_ACCESS_COUNTERS,
                },
                sort_keys=True,
            )
        )
        return 2
    if not args.fixture:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "error": "integration_not_enabled",
                    "detail": "RR-DEP-001 requires --fixture",
                    **ZERO_LIVE_ACCESS_COUNTERS,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(build_fixture_dry_run_result(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
