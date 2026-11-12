#!/usr/bin/env python3
"""Repository-local entrypoint for reusable RealMan π0.5 inference."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from robot_client_runtime.production_pi05 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
