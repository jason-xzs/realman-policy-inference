"""Wire protocol and validation shared by the deployment client/server."""

from __future__ import annotations

import base64
import json
import math
import time
import uuid
from typing import Any


PROTOCOL_VERSION = 1
CAMERA_NAMES = ("high_rgb", "left_wrist_rgb", "right_wrist_rgb")
RAW_STATE_DIM = 28
ACTION_DIM = 16
MAX_REQUEST_BYTES = 16 * 1024 * 1024


class ProtocolError(ValueError):
    pass


def _finite_vector(value: Any, size: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ProtocolError(f"{name} must be a list with {size} elements")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ProtocolError(f"{name} contains NaN or infinity")
    return result


def encode_jpeg(image: Any, quality: int = 90) -> str:
    """Encode a BGR uint8 numpy image produced by OpenCV/r3kit."""
    import cv2
    import numpy as np

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ProtocolError(f"camera image must be uint8 HxWx3, got {array.shape}/{array.dtype}")
    ok, encoded = cv2.imencode(".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ProtocolError("OpenCV failed to JPEG-encode a camera frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def make_infer_request(
    images: dict[str, Any],
    state: list[float],
    instruction: str,
    *,
    jpeg_quality: int = 90,
    request_id: str | None = None,
) -> dict[str, Any]:
    if set(images) != set(CAMERA_NAMES):
        raise ProtocolError(f"images must contain exactly {list(CAMERA_NAMES)}")
    return {
        "version": PROTOCOL_VERSION,
        "request_id": request_id or uuid.uuid4().hex,
        "timestamp_ns": time.time_ns(),
        "instruction": str(instruction),
        "state": _finite_vector(state, RAW_STATE_DIM, "state"),
        "images": {name: encode_jpeg(images[name], jpeg_quality) for name in CAMERA_NAMES},
    }


def validate_infer_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProtocolError("request body must be a JSON object")
    if payload.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {payload.get('version')!r}")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise ProtocolError("request_id must be a non-empty string of at most 128 characters")
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4096:
        raise ProtocolError("instruction must be a non-empty string of at most 4096 characters")
    images = payload.get("images")
    if not isinstance(images, dict) or set(images) != set(CAMERA_NAMES):
        raise ProtocolError(f"images must contain exactly {list(CAMERA_NAMES)}")
    for name, encoded in images.items():
        if not isinstance(encoded, str) or not encoded:
            raise ProtocolError(f"images.{name} must be non-empty base64 JPEG data")
    payload["state"] = _finite_vector(payload.get("state"), RAW_STATE_DIM, "state")
    return payload


def validate_actions(actions: Any) -> list[list[float]]:
    if not isinstance(actions, list) or not actions:
        raise ProtocolError("actions must be a non-empty list")
    return [_finite_vector(action, ACTION_DIM, f"actions[{index}]") for index, action in enumerate(actions)]


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

