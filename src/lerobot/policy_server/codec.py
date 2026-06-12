# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MessagePack codecs for the remote-inference wire schema.

Encoding rules:
- Tensors are raw little-endian bytes + dtype + shape (msgpack's ``bin``
  type), so decoding is a zero-parse ``np.frombuffer``.
- Images are JPEG by default (``jpeg_quality=0`` sends raw bytes).  The
  in-memory convention on both ends is **RGB** uint8 HWC; the OpenCV
  BGR↔RGB conversion happens inside this module only.
- Decoders are tolerant: unknown keys are ignored, missing optional keys
  take dataclass defaults — schema evolution is additive-only.
- No pickle anywhere: nothing in this codec can carry code.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    import msgpack
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Remote inference requires the 'async' extra: pip install 'lerobot[async]' (eclipse-zenoh + msgpack)"
    ) from e

from .schema import (
    IMAGE_CODEC_JPEG,
    IMAGE_CODEC_RAW,
    ActionChunkMsg,
    ObservationMsg,
    ResetAckMsg,
    ResetMsg,
    SessionAckMsg,
    SessionCloseMsg,
    SessionOpenMsg,
    StatusMsg,
)

# ---------------------------------------------------------------------------
# Tensor codec
# ---------------------------------------------------------------------------


def _to_little_endian(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    return np.ascontiguousarray(arr)


def encode_tensor(arr: np.ndarray | None) -> dict[str, Any] | None:
    """Encode an ndarray as raw little-endian bytes + dtype + shape."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    # Record the shape before ascontiguousarray, which promotes 0-d to 1-d.
    shape = list(arr.shape)
    arr = _to_little_endian(arr)
    return {"dtype": arr.dtype.str, "shape": shape, "data": arr.tobytes()}


def decode_tensor(obj: dict[str, Any] | None) -> np.ndarray | None:
    if obj is None:
        return None
    dtype = np.dtype(obj["dtype"])
    if dtype.hasobject:
        raise ValueError(f"Refusing object dtype {dtype} on the wire")
    arr = np.frombuffer(obj["data"], dtype=dtype).reshape(obj["shape"])
    # frombuffer returns a read-only view; copy so downstream torch.from_numpy works.
    return arr.copy()


# ---------------------------------------------------------------------------
# Image codec (RGB uint8 HWC on both ends)
# ---------------------------------------------------------------------------


def encode_image(img: np.ndarray, jpeg_quality: int = 90) -> dict[str, Any]:
    """Encode an RGB uint8 HWC image; ``jpeg_quality=0`` keeps it raw."""
    img = np.asarray(img)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected uint8 HWC RGB image, got dtype={img.dtype} shape={img.shape}")
    if jpeg_quality <= 0:
        return {"codec": IMAGE_CODEC_RAW, "shape": list(img.shape), "data": _to_little_endian(img).tobytes()}
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        raise ValueError("JPEG encoding failed")
    return {"codec": IMAGE_CODEC_JPEG, "data": buf.tobytes()}


def decode_image(obj: dict[str, Any]) -> np.ndarray:
    """Decode to an RGB uint8 HWC image."""
    codec = obj.get("codec", IMAGE_CODEC_JPEG)
    if codec == IMAGE_CODEC_RAW:
        return np.frombuffer(obj["data"], dtype=np.uint8).reshape(obj["shape"]).copy()
    if codec == IMAGE_CODEC_JPEG:
        bgr = cv2.imdecode(np.frombuffer(obj["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("JPEG decoding failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unknown image codec: {codec!r}")


# ---------------------------------------------------------------------------
# msgpack helpers
# ---------------------------------------------------------------------------


def _packb(obj: dict[str, Any]) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def _unpackb(data: bytes) -> dict[str, Any]:
    return msgpack.unpackb(data, raw=False)


def decode_raw(data: bytes) -> dict[str, Any]:
    """Decode a body to a plain dict (e.g. to peek a control-plane ``op``)."""
    return _unpackb(data)


# ---------------------------------------------------------------------------
# Data-plane messages
# ---------------------------------------------------------------------------


def encode_observation(msg: ObservationMsg) -> bytes:
    return _packb(
        {
            "state": encode_tensor(msg.state),
            "images": {name: encode_image(img, msg.jpeg_quality) for name, img in msg.images.items()},
            "task": msg.task,
            "inference_delay_steps": int(msg.inference_delay_steps),
            "prefix_model": encode_tensor(msg.prefix_model),
            "prefix_robot": encode_tensor(msg.prefix_robot),
            "episode_start": bool(msg.episode_start),
        }
    )


def decode_observation(data: bytes) -> ObservationMsg:
    obj = _unpackb(data)
    return ObservationMsg(
        state=decode_tensor(obj.get("state")),
        images={name: decode_image(img) for name, img in obj.get("images", {}).items()},
        task=obj.get("task", ""),
        inference_delay_steps=obj.get("inference_delay_steps", 0),
        prefix_model=decode_tensor(obj.get("prefix_model")),
        prefix_robot=decode_tensor(obj.get("prefix_robot")),
        episode_start=obj.get("episode_start", False),
    )


def encode_action_chunk(msg: ActionChunkMsg) -> bytes:
    return _packb(
        {
            "seq_id_echo": int(msg.seq_id_echo),
            "client_mono_ns_echo": int(msg.client_mono_ns_echo),
            "episode_id_echo": int(msg.episode_id_echo),
            "chunk_model": encode_tensor(msg.chunk_model),
            "chunk_robot": encode_tensor(msg.chunk_robot),
            "queue_wait_ms": float(msg.queue_wait_ms),
            "inference_ms": float(msg.inference_ms),
            "superseded_seqs": int(msg.superseded_seqs),
            "server_load": float(msg.server_load),
        }
    )


def decode_action_chunk(data: bytes) -> ActionChunkMsg:
    obj = _unpackb(data)
    return ActionChunkMsg(
        seq_id_echo=obj.get("seq_id_echo", 0),
        client_mono_ns_echo=obj.get("client_mono_ns_echo", 0),
        episode_id_echo=obj.get("episode_id_echo", 0),
        chunk_model=decode_tensor(obj.get("chunk_model")),
        chunk_robot=decode_tensor(obj.get("chunk_robot")),
        queue_wait_ms=obj.get("queue_wait_ms", 0.0),
        inference_ms=obj.get("inference_ms", 0.0),
        superseded_seqs=obj.get("superseded_seqs", 0),
        server_load=obj.get("server_load", 0.0),
    )


# ---------------------------------------------------------------------------
# Control-plane messages (flat scalar/list/dict fields → generic codec)
# ---------------------------------------------------------------------------


def _encode_flat(msg: Any) -> bytes:
    return _packb(dict(vars(msg).items()))


def _decode_flat(cls: type, data: bytes) -> Any:
    obj = _unpackb(data)
    known = set(cls.__dataclass_fields__)
    return cls(**{k: v for k, v in obj.items() if k in known})


def encode_session_open(msg: SessionOpenMsg) -> bytes:
    return _encode_flat(msg)


def decode_session_open(data: bytes) -> SessionOpenMsg:
    return _decode_flat(SessionOpenMsg, data)


def encode_session_ack(msg: SessionAckMsg) -> bytes:
    return _encode_flat(msg)


def decode_session_ack(data: bytes) -> SessionAckMsg:
    return _decode_flat(SessionAckMsg, data)


def encode_status(msg: StatusMsg) -> bytes:
    return _encode_flat(msg)


def decode_status(data: bytes) -> StatusMsg:
    return _decode_flat(StatusMsg, data)


def encode_reset(msg: ResetMsg) -> bytes:
    return _encode_flat(msg)


def decode_reset(data: bytes) -> ResetMsg:
    return _decode_flat(ResetMsg, data)


def encode_reset_ack(msg: ResetAckMsg) -> bytes:
    return _encode_flat(msg)


def decode_reset_ack(data: bytes) -> ResetAckMsg:
    return _decode_flat(ResetAckMsg, data)


def encode_session_close(msg: SessionCloseMsg) -> bytes:
    return _encode_flat(msg)


def decode_session_close(data: bytes) -> SessionCloseMsg:
    return _decode_flat(SessionCloseMsg, data)
