from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_client_runtime.environment_check import (
    _drift_status,
    _nearest_existing_parent,
    _sha256,
    build_parser,
    main,
)


class EnvironmentCheckTest(unittest.TestCase):
    def test_sha256_reads_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(b"client-runtime")
            self.assertEqual(_sha256(path), hashlib.sha256(b"client-runtime").hexdigest())

    def test_nearest_existing_parent_does_not_create_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "one" / "two"
            self.assertEqual(_nearest_existing_parent(missing), root)
            self.assertFalse((root / "one").exists())

    def test_version_drift_is_strict_by_default(self) -> None:
        self.assertEqual(_drift_status(False), "fail")
        self.assertEqual(_drift_status(True), "warn")

    def test_parser_defaults_to_offline_check(self) -> None:
        args = build_parser().parse_args([])
        self.assertFalse(args.allow_version_drift)
        self.assertFalse(args.json)

    def test_missing_config_fails_without_device_access(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["--config", "/definitely/missing.toml", "--json"])
        self.assertEqual(status, 1)
        self.assertIn('"status": "failed"', output.getvalue())
        self.assertIn('"camera_devices": "not_opened"', output.getvalue())
        self.assertIn('"motion": "not_run"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
