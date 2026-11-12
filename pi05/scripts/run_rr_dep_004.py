#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_client_runtime.rr_dep_004 import main

if __name__ == "__main__":
    raise SystemExit(main())
