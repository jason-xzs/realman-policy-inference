"""Offline-only environment checks for the production RealMan client.

This module does not connect to the policy service, enumerate/open cameras, or
load/connect the RealMan SDK. It verifies the local Python and file layout that
must be in place before a separately authorized hardware validation.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib
from typing import Any, Mapping

from .realman_pi05 import DEFAULT_CONFIG, MotionRejected, load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = REPOSITORY_ROOT / "environment" / "known-good-client.toml"
DEFAULT_PRESET = "pi05-sugar-15000"
REQUIRED_IMPORTS = (
    "numpy",
    "PIL",
    "cv2",
    "yaml",
    "pyrealsense2",
    "openpi_client.websocket_client_policy",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check the local π0.5 client environment without opening devices "
            "or network connections"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="report package/file fingerprint drift as warnings instead of failures",
    )
    parser.add_argument("--json", action="store_true", help="emit only JSON")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _result(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _drift_status(allow_version_drift: bool) -> str:
    return "warn" if allow_version_drift else "fail"


def _load_lock(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("rb") as stream:
        lock = tomllib.load(stream)
    if lock.get("schema_version") != 1:
        raise MotionRejected(f"environment_lock_schema_unsupported:{lock.get('schema_version')!r}")
    return lock


def _check_packages(
    lock: Mapping[str, Any], *, allow_version_drift: bool
) -> list[dict[str, str]]:
    checks = []
    for item in lock.get("python_packages", []):
        distribution = str(item["distribution"])
        expected = str(item["version"])
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            checks.append(_result(f"package:{distribution}", "fail", "not installed"))
            continue
        status = "pass" if actual == expected else _drift_status(allow_version_drift)
        checks.append(
            _result(f"package:{distribution}", status, f"installed={actual}, locked={expected}")
        )
    for module in REQUIRED_IMPORTS:
        try:
            available = util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            checks.append(_result(f"import:{module}", "fail", f"{type(exc).__name__}:{exc}"))
        else:
            checks.append(
                _result(
                    f"import:{module}",
                    "pass" if available else "fail",
                    "available" if available else "not found",
                )
            )
    return checks


def _check_repository_files(
    lock: Mapping[str, Any], *, allow_version_drift: bool
) -> list[dict[str, str]]:
    checks = []
    for item in lock.get("repository_files", []):
        relative = Path(str(item["path"]))
        path = REPOSITORY_ROOT / relative
        name = f"repository-file:{relative.as_posix()}"
        if not path.is_file():
            checks.append(_result(name, "fail", f"missing: {path}"))
            continue
        actual = _sha256(path)
        expected = str(item["sha256"])
        status = "pass" if actual == expected else _drift_status(allow_version_drift)
        checks.append(_result(name, status, f"sha256={actual}, locked={expected}"))
    return checks


def _check_external_sources(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    allow_version_drift: bool,
) -> list[dict[str, str]]:
    checks = []
    source_root = Path(str(config["sources"]["deltawam_src"])).expanduser().resolve()
    expected_sources = lock.get("external_sources", {})
    revision = _git_revision(source_root.parent)
    expected_revision = str(expected_sources.get("deltawam_git_commit", ""))
    if revision is None:
        checks.append(
            _result(
                "source:deltawam-git",
                "fail",
                f"not a Git checkout: {source_root.parent}",
            )
        )
    else:
        status = "pass" if revision == expected_revision else _drift_status(allow_version_drift)
        checks.append(
            _result(
                "source:deltawam-git",
                status,
                f"actual={revision}, locked={expected_revision}",
            )
        )

    r3kit_root = source_root / "realman_deployment" / "r3kit"
    r3kit_revision = _git_revision(r3kit_root)
    expected_r3kit = str(expected_sources.get("r3kit_git_commit", ""))
    if r3kit_revision is None:
        checks.append(_result("source:r3kit-git", "fail", f"not a Git checkout: {r3kit_root}"))
    else:
        status = "pass" if r3kit_revision == expected_r3kit else _drift_status(allow_version_drift)
        checks.append(
            _result(
                "source:r3kit-git",
                status,
                f"actual={r3kit_revision}, locked={expected_r3kit}",
            )
        )

    for item in lock.get("external_files", []):
        relative = Path(str(item["path"]))
        path = source_root / relative
        name = f"file:{relative.as_posix()}"
        if not path.is_file():
            checks.append(_result(name, "fail", f"missing: {path}"))
            continue
        actual = _sha256(path)
        expected = str(item["sha256"])
        status = "pass" if actual == expected else _drift_status(allow_version_drift)
        checks.append(_result(name, status, f"sha256={actual}, locked={expected}"))
    return checks


def _check_recording(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    allow_version_drift: bool,
) -> list[dict[str, str]]:
    if not config["recording"]["enabled"]:
        return [_result("recording", "pass", "disabled for selected preset")]

    checks = []
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        checks.append(_result("recording:ffmpeg", "fail", "ffmpeg not found on PATH"))
    else:
        try:
            version = subprocess.run(
                [ffmpeg, "-version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            ).stdout.splitlines()[0]
            encoders = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            checks.append(_result("recording:ffmpeg", "fail", f"{type(exc).__name__}:{exc}"))
        else:
            expected = str(lock.get("system", {}).get("ffmpeg", ""))
            detail = f"{version}; locked={expected}"
            status = (
                "pass"
                if expected and expected in version
                else _drift_status(allow_version_drift)
            )
            checks.append(_result("recording:ffmpeg", status, detail))
            checks.append(
                _result(
                    "recording:h264_vaapi",
                    "pass" if "h264_vaapi" in encoders else "fail",
                    "encoder available" if "h264_vaapi" in encoders else "encoder missing",
                )
            )

    render_node = Path(str(lock.get("system", {}).get("render_node", "/dev/dri/renderD128")))
    render_ready = render_node.exists() and os.access(render_node, os.R_OK | os.W_OK)
    checks.append(
        _result(
            "recording:render-node",
            "pass" if render_ready else "fail",
            f"readable+writable={render_ready}: {render_node}",
        )
    )
    root = Path(str(config["recording"]["root_dir"])).expanduser().resolve()
    parent = _nearest_existing_parent(root)
    writable = parent.is_dir() and os.access(parent, os.W_OK | os.X_OK)
    checks.append(
        _result(
            "recording:destination",
            "pass" if writable else "fail",
            f"nearest_existing_parent={parent}, writable={writable}",
        )
    )
    return checks


def build_report(
    *,
    config_path: Path,
    preset: str,
    lock_path: Path,
    allow_version_drift: bool,
) -> dict[str, Any]:
    lock = _load_lock(lock_path)
    config = load_config(config_path, preset)
    checks = []
    system_lock = lock.get("system", {})
    expected_python = str(lock.get("system", {}).get("python", ""))
    actual_python = platform.python_version()
    python_status = (
        "pass" if actual_python == expected_python else _drift_status(allow_version_drift)
    )
    checks.append(
        _result(
            "python",
            python_status,
            f"actual={actual_python}, locked={expected_python}",
        )
    )

    try:
        actual_os = platform.freedesktop_os_release().get("PRETTY_NAME", "")
    except OSError:
        actual_os = platform.platform()
    expected_os = str(system_lock.get("os", ""))
    os_status = "pass" if actual_os == expected_os else _drift_status(allow_version_drift)
    checks.append(
        _result(
            "operating-system",
            os_status,
            f"actual={actual_os}, locked={expected_os}",
        )
    )

    actual_kernel = f"{platform.system()} {platform.release()}"
    expected_kernel = str(system_lock.get("kernel", ""))
    kernel_status = (
        "pass" if actual_kernel == expected_kernel else _drift_status(allow_version_drift)
    )
    checks.append(
        _result(
            "kernel",
            kernel_status,
            f"actual={actual_kernel}, locked={expected_kernel}",
        )
    )

    expected_arch = str(system_lock.get("architecture", ""))
    actual_arch = platform.machine()
    architecture_status = "pass" if actual_arch == expected_arch else "fail"
    checks.append(
        _result(
            "architecture",
            architecture_status,
            f"actual={actual_arch}, locked={expected_arch}",
        )
    )
    checks.append(_result("config", "pass", f"{config['_path']} preset={config['_preset']}"))
    checks.extend(_check_repository_files(lock, allow_version_drift=allow_version_drift))
    checks.extend(_check_packages(lock, allow_version_drift=allow_version_drift))
    checks.extend(
        _check_external_sources(config, lock, allow_version_drift=allow_version_drift)
    )
    checks.extend(
        _check_recording(config, lock, allow_version_drift=allow_version_drift)
    )

    failures = sum(item["status"] == "fail" for item in checks)
    warnings = sum(item["status"] == "warn" for item in checks)
    status = "failed" if failures else ("passed_with_warnings" if warnings else "passed")
    return {
        "status": status,
        "preset": preset,
        "checks": checks,
        "summary": {
            "passed": len(checks) - failures - warnings,
            "warnings": warnings,
            "failed": failures,
        },
        "validation_scope": {
            "filesystem_and_imports": "checked",
            "network_connections": "not_run",
            "policy_service": "not_run",
            "camera_devices": "not_opened",
            "robot_sdk": "not_loaded_or_connected",
            "motion": "not_run",
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(
            config_path=args.config,
            preset=args.preset,
            lock_path=args.lock,
            allow_version_drift=args.allow_version_drift,
        )
    except BaseException as exc:
        report = {
            "status": "failed",
            "reason": f"{type(exc).__name__}:{exc}",
            "validation_scope": {
                "network_connections": "not_run",
                "camera_devices": "not_opened",
                "robot_sdk": "not_loaded_or_connected",
                "motion": "not_run",
            },
        }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for item in report.get("checks", []):
            print(f"[{item['status'].upper():4}] {item['name']}: {item['detail']}")
        summary = {key: value for key, value in report.items() if key != "checks"}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if report.get("status") in ("passed", "passed_with_warnings") else 1


if __name__ == "__main__":
    raise SystemExit(main())
