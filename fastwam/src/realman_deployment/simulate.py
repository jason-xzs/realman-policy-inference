"""One-command end-to-end simulation of HTTP transport and control loop."""

from __future__ import annotations

import argparse
import logging
import threading

from .client import DEFAULT_INSTRUCTION, RemoteInferenceClient, run_control_loop
from .hardware import MockRealManHardware
from .policy import MockPolicy
from .server import create_server


def run_simulation(replans: int = 3, actions_per_replan: int = 4) -> dict:
    policy = MockPolicy(action_horizon=actions_per_replan)
    server = create_server("127.0.0.1", 0, policy, max_actions=actions_per_replan)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    hardware = MockRealManHardware(height=96, width=128)
    try:
        remote = RemoteInferenceClient(f"http://127.0.0.1:{server.server_port}", timeout=5)
        completed = run_control_loop(
            hardware, remote, instruction=DEFAULT_INSTRUCTION, action_hz=200,
            jpeg_quality=75, max_replans=replans, dry_run=False,
        )
        result = {
            "ok": True,
            "replans": completed,
            "requests": policy.calls,
            "executed_actions": len(hardware.executed),
            "final_right_joint_1_rad": float(hardware.right_joints[0]),
        }
        expected = replans * actions_per_replan
        if completed != replans or policy.calls != replans or len(hardware.executed) != expected:
            raise RuntimeError(f"simulation count mismatch: {result}, expected actions={expected}")
        return result
    finally:
        hardware.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replans", type=int, default=3)
    parser.add_argument("--actions-per-replan", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(run_simulation(args.replans, args.actions_per_replan))


if __name__ == "__main__":
    main()
