#!/usr/bin/env python3
"""Repository-local entrypoint for the RR-DEP-006 low-latency Client."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from robot_client_runtime.rr_dep_006 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
