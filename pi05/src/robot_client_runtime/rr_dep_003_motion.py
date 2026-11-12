"""Fail-closed motion primitives for the bounded RR-DEP-003 Q7 test.

This module contains no camera, network, policy, gripper, or vendor-SDK import.
The live runner injects state and command backends after all pre-motion gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Protocol, Sequence


COMMAND_IDS = (
    "left_hold",
    "left_q7_micro",
    "left_return",
    "right_hold",
    "right_q7_micro",
    "right_return",
)


class MotionRejected(RuntimeError):
    """A command or observed state violated the bounded motion contract."""


@dataclass(frozen=True)
class StateSample:
    side: str
    joints_rad: tuple[float, ...]
    receive_monotonic_ns: int
    sdk_status: int = 0
    arm_errors: tuple[str, ...] = ()
    controller_error: int = 0


@dataclass(frozen=True)
class MotionThresholds:
    delta_abs_max_rad: float = 0.01
    verified_margin_rad: float = 0.02
    hold_max_change_rad: float = 0.003
    target_tolerance_rad: float = 0.005
    non_q7_stop_rad: float = 0.005
    q7_stop_rad: float = 0.015
    stale_timeout_s: float = 0.25
    command_timeout_s: float = 5.0
    speed_percent: int = 5


class MotionBackend(Protocol):
    def command_absolute(self, side: str, target_rad: Sequence[float], speed_percent: int) -> int:
        ...

    def slow_stop(self, side: str) -> int:
        ...


class CommandSequence:
    """Accept each approved command ID once, in exact contract order."""

    def __init__(self) -> None:
        self._next_index = 0
        self._submitted: set[str] = set()

    @property
    def submitted(self) -> tuple[str, ...]:
        return tuple(COMMAND_IDS[: self._next_index])

    @property
    def next_id(self) -> str | None:
        if self._next_index >= len(COMMAND_IDS):
            return None
        return COMMAND_IDS[self._next_index]

    def accept(self, command_id: str) -> None:
        if command_id in self._submitted:
            raise MotionRejected(f"duplicate_command_id:{command_id}")
        expected = self.next_id
        if command_id != expected:
            raise MotionRejected(f"wrong_command_order:expected={expected}:received={command_id}")
        self._submitted.add(command_id)
        self._next_index += 1


def _finite_joints(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != 7:
        raise MotionRejected(f"{name}_must_have_7_joints")
    if not all(math.isfinite(value) for value in result):
        raise MotionRejected(f"{name}_nonfinite")
    return result


def select_q7_direction(
    initial_rad: Sequence[float],
    lower_rad: Sequence[float],
    upper_rad: Sequence[float],
    *,
    margin_rad: float = 0.02,
) -> int:
    initial = _finite_joints(initial_rad, "initial")
    lower = _finite_joints(lower_rad, "lower_limit")
    upper = _finite_joints(upper_rad, "upper_limit")
    if any(lo >= hi for lo, hi in zip(lower, upper)):
        raise MotionRejected("invalid_joint_limits")
    plus_margin = upper[6] - initial[6]
    minus_margin = initial[6] - lower[6]
    if plus_margin >= margin_rad and plus_margin >= minus_margin:
        return 1
    if minus_margin >= margin_rad:
        return -1
    if plus_margin >= margin_rad:
        return 1
    raise MotionRejected(
        f"q7_verified_margin_below_{margin_rad:.6f}:plus={plus_margin:.6f}:minus={minus_margin:.6f}"
    )


def build_arm_targets(
    initial_rad: Sequence[float], direction: int, *, delta_rad: float = 0.01
) -> dict[str, tuple[float, ...]]:
    initial = _finite_joints(initial_rad, "initial")
    if direction not in (-1, 1):
        raise MotionRejected("q7_direction_must_be_plus_or_minus_one")
    if not math.isfinite(delta_rad) or not 0.0 < abs(delta_rad) <= 0.01:
        raise MotionRejected("q7_delta_out_of_contract")
    micro = list(initial)
    micro[6] += direction * abs(delta_rad)
    return {"hold": initial, "micro": tuple(micro), "return": initial}


def validate_absolute_target(
    *,
    kind: str,
    initial_rad: Sequence[float],
    target_rad: Sequence[float],
    lower_rad: Sequence[float],
    upper_rad: Sequence[float],
    direction: int,
    thresholds: MotionThresholds = MotionThresholds(),
) -> tuple[float, ...]:
    initial = _finite_joints(initial_rad, "initial")
    target = _finite_joints(target_rad, "target")
    lower = _finite_joints(lower_rad, "lower_limit")
    upper = _finite_joints(upper_rad, "upper_limit")
    if kind not in ("hold", "micro", "return"):
        raise MotionRejected(f"unknown_command_kind:{kind}")
    if direction not in (-1, 1):
        raise MotionRejected("unverified_q7_direction")
    for index, (value, lo, hi) in enumerate(zip(target, lower, upper), start=1):
        if lo >= hi:
            raise MotionRejected(f"invalid_joint_limit:q{index}")
        if value < lo or value > hi:
            raise MotionRejected(f"target_out_of_limit:q{index}")
    for index in range(6):
        if target[index] != initial[index]:
            raise MotionRejected(f"extra_joint_target:q{index + 1}")
    delta = target[6] - initial[6]
    if kind in ("hold", "return") and delta != 0.0:
        raise MotionRejected(f"{kind}_q7_not_initial")
    if kind == "micro":
        if not 0.0 < abs(delta) <= thresholds.delta_abs_max_rad:
            raise MotionRejected("q7_delta_out_of_contract")
        if math.copysign(1.0, delta) != float(direction):
            raise MotionRejected("q7_wrong_target_direction")
    if thresholds.speed_percent > 5:
        raise MotionRejected("speed_above_5_percent")
    return target


def require_fresh_state(sample: StateSample, *, now_ns: int, stale_timeout_s: float) -> None:
    _finite_joints(sample.joints_rad, "state")
    if sample.sdk_status != 0:
        raise MotionRejected(f"state_sdk_status:{sample.sdk_status}")
    if sample.arm_errors:
        raise MotionRejected(f"arm_alarm:{','.join(sample.arm_errors)}")
    if sample.controller_error != 0:
        raise MotionRejected(f"controller_alarm:{sample.controller_error}")
    age_ns = now_ns - int(sample.receive_monotonic_ns)
    if age_ns < 0 or age_ns > int(stale_timeout_s * 1_000_000_000):
        raise MotionRejected(f"stale_state:{age_ns / 1_000_000_000.0:.6f}s")


def check_trajectory_envelope(
    sample_rad: Sequence[float],
    initial_rad: Sequence[float],
    *,
    kind: str,
    direction: int,
    thresholds: MotionThresholds = MotionThresholds(),
) -> None:
    sample = _finite_joints(sample_rad, "observed_state")
    initial = _finite_joints(initial_rad, "initial")
    for index in range(6):
        change = abs(sample[index] - initial[index])
        if change > thresholds.non_q7_stop_rad:
            raise MotionRejected(f"unexpected_non_q7_motion:q{index + 1}:{change:.6f}")
        if kind == "hold" and change > thresholds.hold_max_change_rad:
            raise MotionRejected(f"hold_motion_exceeded:q{index + 1}:{change:.6f}")
    q7_delta = sample[6] - initial[6]
    if abs(q7_delta) > thresholds.q7_stop_rad:
        raise MotionRejected(f"q7_displacement_exceeded:{q7_delta:.6f}")
    if kind == "hold" and abs(q7_delta) > thresholds.hold_max_change_rad:
        raise MotionRejected(f"hold_motion_exceeded:q7:{q7_delta:.6f}")
    if kind == "micro" and abs(q7_delta) > 0.0005:
        if math.copysign(1.0, q7_delta) != float(direction):
            raise MotionRejected(f"q7_wrong_observed_direction:{q7_delta:.6f}")


def target_error(sample_rad: Sequence[float], target_rad: Sequence[float]) -> float:
    sample = _finite_joints(sample_rad, "observed_state")
    target = _finite_joints(target_rad, "target")
    return max(abs(actual - expected) for actual, expected in zip(sample, target))


class BoundedCommandExecutor:
    """Submit one command once and block on fresh feedback for at most five seconds."""

    def __init__(
        self,
        backend: MotionBackend,
        latest_state: Callable[[str], StateSample | None],
        camera_healthy: Callable[[], bool],
        *,
        thresholds: MotionThresholds = MotionThresholds(),
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        event_sink: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.backend = backend
        self.latest_state = latest_state
        self.camera_healthy = camera_healthy
        self.thresholds = thresholds
        self.monotonic_ns = monotonic_ns
        self.sleep = sleep
        self.event_sink = event_sink or (lambda _kind, _payload: None)
        self.sequence = CommandSequence()

    def _stop_and_raise(self, side: str, reason: str) -> None:
        stop_status = self.backend.slow_stop(side)
        self.event_sink("safety_stop", {"side": side, "reason": reason, "stop_status": stop_status})
        raise MotionRejected(reason)

    def execute(
        self,
        *,
        command_id: str,
        side: str,
        kind: str,
        initial_rad: Sequence[float],
        target_rad: Sequence[float],
        lower_rad: Sequence[float],
        upper_rad: Sequence[float],
        direction: int,
        minimum_observe_s: float = 0.25,
    ) -> StateSample:
        if side not in ("left", "right"):
            raise MotionRejected(f"unknown_side:{side}")
        target = validate_absolute_target(
            kind=kind,
            initial_rad=initial_rad,
            target_rad=target_rad,
            lower_rad=lower_rad,
            upper_rad=upper_rad,
            direction=direction,
            thresholds=self.thresholds,
        )
        if not self.camera_healthy():
            raise MotionRejected("camera_unhealthy_before_command")
        before = self.latest_state(side)
        if before is None:
            raise MotionRejected("missing_state_before_command")
        require_fresh_state(
            before,
            now_ns=self.monotonic_ns(),
            stale_timeout_s=self.thresholds.stale_timeout_s,
        )
        check_trajectory_envelope(
            before.joints_rad,
            initial_rad,
            kind=kind,
            direction=direction,
            thresholds=self.thresholds,
        )
        self.sequence.accept(command_id)
        self.event_sink(
            "command_submitting",
            {"command_id": command_id, "side": side, "kind": kind, "target_rad": list(target)},
        )
        sdk_status = self.backend.command_absolute(side, target, self.thresholds.speed_percent)
        self.event_sink(
            "sdk_acceptance",
            {"command_id": command_id, "side": side, "sdk_status": sdk_status},
        )
        if sdk_status != 0:
            self._stop_and_raise(side, f"command_sdk_rejection:{command_id}:{sdk_status}")

        started_ns = self.monotonic_ns()
        deadline_ns = started_ns + int(self.thresholds.command_timeout_s * 1_000_000_000)
        observe_until_ns = started_ns + int(minimum_observe_s * 1_000_000_000)
        while True:
            now_ns = self.monotonic_ns()
            if not self.camera_healthy():
                self._stop_and_raise(side, f"camera_loss_during:{command_id}")
            sample = self.latest_state(side)
            if sample is None:
                self._stop_and_raise(side, f"missing_feedback:{command_id}")
            try:
                require_fresh_state(
                    sample,
                    now_ns=now_ns,
                    stale_timeout_s=self.thresholds.stale_timeout_s,
                )
                check_trajectory_envelope(
                    sample.joints_rad,
                    initial_rad,
                    kind=kind,
                    direction=direction,
                    thresholds=self.thresholds,
                )
            except MotionRejected as exc:
                self._stop_and_raise(side, f"{command_id}:{exc}")
            error = target_error(sample.joints_rad, target)
            if error <= self.thresholds.target_tolerance_rad and now_ns >= observe_until_ns:
                self.event_sink(
                    "observed_target",
                    {
                        "command_id": command_id,
                        "side": side,
                        "target_error_rad": error,
                        "observed_rad": list(sample.joints_rad),
                    },
                )
                return sample
            if now_ns >= deadline_ns:
                self._stop_and_raise(side, f"command_timeout:{command_id}")
            self.sleep(0.01)
