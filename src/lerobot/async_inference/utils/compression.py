from typing import Any

import cv2  # type: ignore
import numpy as np

def _is_uint8_hwc3_image(x: Any) -> bool:
    if not isinstance(x, np.ndarray):
        return False
    if x.dtype != np.uint8:
        return False
    if x.ndim != 3:
        return False
    h, w, c = x.shape
    if h <= 0 or w <= 0:
        return False
    return c == 3


def encode_images_for_transport(
    observation: Any,
    jpeg_quality: int,
) -> tuple[Any, dict[str, int]]:
    """Recursively JPEG-encode uint8 HWC3 images inside an observation structure."""
    stats = {"images_encoded": 0, "raw_bytes_total": 0, "encoded_bytes_total": 0}

    def _encode_any(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _encode_any(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_encode_any(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_encode_any(v) for v in x)

        if not _is_uint8_hwc3_image(x):
            return x

        bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("OpenCV failed to JPEG-encode image for transport")

        payload = bytes(buf)
        stats["images_encoded"] += 1
        stats["raw_bytes_total"] += int(x.nbytes)
        stats["encoded_bytes_total"] += len(payload)
        return {"__lerobot_image_encoding__": "jpeg", "quality": int(jpeg_quality), "data": payload}

    return _encode_any(observation), stats

def decode_images_from_transport(observation: Any) -> tuple[Any, dict[str, int]]:
    """Recursively decode JPEG-marked images back into uint8 HWC3 RGB numpy arrays."""
    stats = {"images_decoded": 0, "raw_bytes_total": 0, "encoded_bytes_total": 0}

    def _maybe_decode_payload(x: Any) -> Any:
        if isinstance(x, dict) and x.get("__lerobot_image_encoding__") == "jpeg":
            data = x.get("data")
            if not isinstance(data, (bytes, bytearray)):
                raise TypeError("JPEG payload missing bytes 'data'")

            buf = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("OpenCV failed to decode JPEG payload")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            stats["images_decoded"] += 1
            stats["encoded_bytes_total"] += len(data)
            stats["raw_bytes_total"] += int(rgb.nbytes)
            return rgb

        if isinstance(x, dict):
            return {k: _maybe_decode_payload(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_maybe_decode_payload(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_maybe_decode_payload(v) for v in x)
        return x

    return _maybe_decode_payload(observation), stats