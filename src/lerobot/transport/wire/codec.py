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

"""Bounded MessagePack values and explicit tensor/RGB records.

Arrays decode into owned writable NumPy storage. The caller chooses which named visual
features are RGB; JPEG is never inferred from an array's shape. CPU snapshots are the
caller's responsibility, before an asynchronous encoder accesses reused sensor buffers.
"""

import io
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

from lerobot.utils.import_utils import _msgpack_available, require_package

from .protocol import Envelope, ErrorCode, ProtocolError

if TYPE_CHECKING or _msgpack_available:
    import msgpack


@dataclass(frozen=True)
class CodecLimits:
    """Allocation and parsing bounds applied independently of feature negotiation."""

    max_payload_bytes: int = 32 * 1024 * 1024
    max_decoded_bytes: int = 64 * 1024 * 1024
    max_tensor_bytes: int = 32 * 1024 * 1024
    max_image_pixels: int = 4096 * 4096
    max_dimension: int = 16384
    max_rank: int = 8
    max_string_bytes: int = 16384
    max_container_items: int = 4096
    max_nodes: int = 32768
    max_depth: int = 16

    def __post_init__(self) -> None:
        """Reject limits that would disable a bound accidentally."""
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("Codec limits must be positive integers")


@dataclass(frozen=True)
class RGBImage:
    """An explicitly RGB, HWC uint8 image; raw transport is lossless."""

    array: np.ndarray
    encoding: Literal["raw", "jpeg"] = "raw"
    quality: int = 90


_DTYPES: dict[str, np.dtype] = {
    np.dtype(name).name: np.dtype(name)
    for name in (
        "bool",
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float16",
        "float32",
        "float64",
    )
}
_MARKER = "__lerobot_type__"


def _malformed(message: str) -> ProtocolError:
    return ProtocolError(ErrorCode.MALFORMED, message)


class _Budget:
    def __init__(self, limits: CodecLimits, tensor_converter: Callable | None = None):
        self.tensor_converter = tensor_converter
        self.limits = limits
        self.nodes = 0
        self.decoded_bytes = 0

    def visit(self, depth: int) -> None:
        self.nodes += 1
        if depth > self.limits.max_depth or self.nodes > self.limits.max_nodes:
            raise _malformed("Message nesting or node count exceeds limit")

    def allocate(self, size: int) -> None:
        self.decoded_bytes += size
        if self.decoded_bytes > self.limits.max_decoded_bytes:
            raise _malformed("Total decoded tensor/image bytes exceed limit")


def _string(value: str, limits: CodecLimits) -> str:
    if len(value.encode("utf-8")) > limits.max_string_bytes:
        raise _malformed("String exceeds byte limit")
    return value


def _shape(shape: Any, limits: CodecLimits) -> tuple[int, ...]:
    if not isinstance(shape, (tuple, list)) or len(shape) > limits.max_rank:
        raise _malformed("Invalid tensor rank")
    if any(type(dim) is not int or not 0 < dim <= limits.max_dimension for dim in shape):
        raise _malformed("Invalid tensor dimensions")
    return tuple(shape)


def _tensor_meta(
    dtype: Any, shape: Any, endian: Any, limits: CodecLimits
) -> tuple[np.dtype, tuple[int, ...], int]:
    if not isinstance(dtype, str) or dtype not in _DTYPES or endian not in ("little", "big"):
        raise _malformed("Unsupported tensor dtype or endianness")
    resolved = _DTYPES[dtype].newbyteorder("<" if endian == "little" else ">")
    dimensions = _shape(shape, limits)
    expected = math.prod(dimensions) * resolved.itemsize
    if expected > limits.max_tensor_bytes:
        raise _malformed("Tensor exceeds decoded byte limit")
    return resolved, dimensions, expected


def _rgb_shape(shape: Any, limits: CodecLimits) -> tuple[int, ...]:
    dimensions = _shape(shape, limits)
    if len(dimensions) != 3 or dimensions[2] != 3:
        raise _malformed("RGB requires HWC with three channels")
    if dimensions[0] * dimensions[1] > limits.max_image_pixels:
        raise _malformed("RGB image exceeds decoded pixel limit")
    return dimensions


def _encode_value(value: Any, budget: _Budget, depth: int = 0) -> Any:
    budget.visit(depth)
    limits = budget.limits
    if isinstance(value, RGBImage):
        array = value.array
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8:
            raise _malformed("RGB requires a uint8 NumPy array")
        shape = _rgb_shape(array.shape, limits)
        budget.allocate(math.prod(shape))
        if value.encoding == "raw":
            data = array.tobytes(order="C")
        elif value.encoding == "jpeg":
            if type(value.quality) is not int or not 1 <= value.quality <= 100:
                raise _malformed("JPEG quality must be in [1, 100]")
            stream = io.BytesIO()
            Image.fromarray(array).save(stream, format="JPEG", quality=value.quality)
            data = stream.getvalue()
        else:
            raise _malformed("Unsupported RGB encoding")
        if len(data) > limits.max_payload_bytes:
            raise _malformed("Encoded image exceeds byte limit")
        return {
            _MARKER: "rgb",
            "encoding": value.encoding,
            "shape": list(shape),
            "dtype": "uint8",
            "channel_order": "RGB",
            "data": data,
        }
    if budget.tensor_converter is not None:
        value = budget.tensor_converter(value)
    if isinstance(value, np.ndarray):
        dtype, shape, size = _tensor_meta(value.dtype.name, value.shape, "little", limits)
        budget.allocate(size)
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise _malformed("Tensor contains non-finite values")
        # NumPy bool arrays may contain any nonzero backing byte for True. Emit
        # canonical 0/1 bytes even for such valid local arrays or strided views.
        data = (
            np.where(value, np.uint8(1), np.uint8(0)).tobytes(order="C")
            if dtype.kind == "b"
            else value.astype(dtype, copy=False).tobytes(order="C")
        )
        return {
            _MARKER: "tensor",
            "dtype": dtype.name,
            "shape": list(shape),
            "endianness": "little",
            "data": data,
        }
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise _malformed("Non-finite scalar")
        return float(value)
    if isinstance(value, (int, np.integer)):
        if not -(2**63) <= value < 2**64:
            raise _malformed("Integer outside MessagePack range")
        return int(value)
    if isinstance(value, str):
        return _string(value, limits)
    if isinstance(value, bytes):
        if len(value) > limits.max_payload_bytes:
            raise _malformed("Byte string exceeds limit")
        return value
    if isinstance(value, (tuple, list)):
        if len(value) > limits.max_container_items:
            raise _malformed("Sequence exceeds item limit")
        return [_encode_value(item, budget, depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > limits.max_container_items or _MARKER in value:
            raise _malformed("Invalid map size or reserved codec marker")
        if any(not isinstance(key, str) for key in value):
            raise _malformed("Map keys must be strings")
        return {_string(key, limits): _encode_value(item, budget, depth + 1) for key, item in value.items()}
    raise _malformed(f"Unsupported value type: {type(value).__name__}")


def _decode_record(record: dict[str, Any], budget: _Budget) -> np.ndarray:
    limits = budget.limits
    data = record.get("data")
    if not isinstance(data, bytes) or len(data) > limits.max_payload_bytes:
        raise _malformed("Invalid tensor/image byte payload")
    if record[_MARKER] == "tensor":
        dtype, shape, size = _tensor_meta(
            record.get("dtype"), record.get("shape"), record.get("endianness"), limits
        )
        if len(data) != size:
            raise _malformed("Tensor payload size does not match dtype and shape")
        budget.allocate(size)
        if dtype.kind == "b" and np.any(np.frombuffer(data, dtype=np.uint8) > 1):
            raise _malformed("Boolean tensor bytes must be 0 or 1")
        array = np.frombuffer(data, dtype=dtype).reshape(shape)
        if dtype.kind == "f" and not np.isfinite(array).all():
            raise _malformed("Tensor contains non-finite values")
        return array.astype(dtype.newbyteorder("="), copy=True)
    if record[_MARKER] != "rgb":
        raise _malformed("Unknown typed record")
    shape = _rgb_shape(record.get("shape"), limits)
    if record.get("dtype") != "uint8" or record.get("channel_order") != "RGB":
        raise _malformed("Invalid RGB dtype/channel order")
    budget.allocate(math.prod(shape))
    if record.get("encoding") == "raw":
        if len(data) != math.prod(shape):
            raise _malformed("Raw RGB byte count mismatch")
        return np.frombuffer(data, dtype=np.uint8).reshape(shape).copy()
    if record.get("encoding") != "jpeg":
        raise _malformed("Unsupported RGB encoding")
    try:
        with Image.open(io.BytesIO(data)) as decoded:
            # Check the actual decoder header before allocating/decompressing pixels.
            if decoded.format != "JPEG" or decoded.mode != "RGB" or decoded.size != (shape[1], shape[0]):
                raise _malformed("JPEG header does not match the declared RGB dimensions")
            return np.array(decoded, dtype=np.uint8, copy=True)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise _malformed("Malformed JPEG data") from exc


def _decode_value(value: Any, budget: _Budget, depth: int = 0) -> Any:
    budget.visit(depth)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise _malformed("Map keys must be strings")
        if _MARKER in value:
            return _decode_record(value, budget)
        return {key: _decode_value(item, budget, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item, budget, depth + 1) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise _malformed("Non-finite scalar")
    if value is not None and not isinstance(value, (str, bytes, bool, int, float)):
        raise _malformed("Unsupported MessagePack value")
    return value


def _unique_map(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise _malformed("Invalid or duplicate map key")
        result[key] = value
    return result


class _UnpackBudget:
    """Stop MessagePack as containers complete, before building the whole tree.

    Each child container has already been counted by its own hook. Scalar values
    are counted by their parent; map keys remain bounded by the per-map and string
    limits. At most one bounded container is materialized before its hook runs.
    """

    def __init__(self, limits: CodecLimits):
        self.limits = limits
        self.nodes = 0

    def _count(self, values: Iterable[Any]) -> None:
        self.nodes += 1 + sum(not isinstance(value, (list, dict)) for value in values)
        if self.nodes > self.limits.max_nodes:
            raise _malformed("Message node count exceeds limit during parsing")

    def sequence(self, items: list[Any]) -> list[Any]:
        self._count(items)
        return items

    def mapping(self, pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
        self._count(value for _, value in pairs)
        return _unique_map(pairs)


def _reject_extension(code: int, data: bytes) -> None:
    raise _malformed("MessagePack extensions are unsupported")


def encode_message(
    envelope: Envelope, limits: CodecLimits = CodecLimits(), *, tensor_converter: Callable | None = None
) -> bytes:
    """Encode only supported, finite values into an explicitly typed envelope."""
    require_package("msgpack", "remote")
    envelope.__post_init__()
    record = {
        "version": envelope.version,
        "message_type": str(envelope.message_type),
        "instance_id": envelope.instance_id,
        "session_id": envelope.session_id,
        "generation": envelope.generation,
        "request_id": envelope.request_id,
        "body": envelope.body,
    }
    encoded = msgpack.packb(_encode_value(record, _Budget(limits, tensor_converter)), use_bin_type=True)
    if len(encoded) > limits.max_payload_bytes:
        raise _malformed("Encoded message exceeds byte limit")
    return encoded


def _unpack_envelope(payload: bytes, limits: CodecLimits) -> tuple[Envelope, Any]:
    require_package("msgpack", "remote")
    if not isinstance(payload, bytes) or len(payload) > limits.max_payload_bytes:
        raise _malformed("Encoded message exceeds byte limit or is not bytes")
    try:
        budget = _UnpackBudget(limits)
        record = msgpack.unpackb(
            payload,
            raw=False,
            strict_map_key=True,
            object_pairs_hook=budget.mapping,
            list_hook=budget.sequence,
            ext_hook=_reject_extension,
            max_str_len=limits.max_string_bytes,
            max_bin_len=limits.max_payload_bytes,
            max_array_len=limits.max_container_items,
            max_map_len=limits.max_container_items,
            max_ext_len=0,
        )
        if not isinstance(record, dict):
            raise _malformed("Expected message envelope map")
        # Validate correlation/version before allocating tensor or image storage.
        required = (
            "version",
            "message_type",
            "instance_id",
            "session_id",
            "generation",
            "request_id",
            "body",
        )
        if any(key not in record for key in required):
            raise _malformed("Incomplete message envelope")
        return Envelope(**{key: record[key] for key in required if key != "body"}), record["body"]
    except ProtocolError:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError, msgpack.StackError) as exc:
        raise _malformed("Malformed MessagePack envelope") from exc


def peek_envelope(payload: bytes, limits: CodecLimits = CodecLimits()) -> Envelope:
    """Validate headers with an empty body, without decoding tensors or image data.

    Clients correlate this header before full decoding so an obsolete reply's malformed
    tensor/image body cannot fault a healthy active request. MessagePack parsing is still
    bounded, and invalid MessagePack cannot reliably supply a correlation identity.
    """
    envelope, _ = _unpack_envelope(payload, limits)
    return envelope


def decode_message(payload: bytes, limits: CodecLimits = CodecLimits()) -> Envelope:
    """Validate the envelope and allocation bounds before materializing tensors/images."""
    envelope, body = _unpack_envelope(payload, limits)
    if not isinstance(body, dict):
        raise _malformed("Message body must be a map")
    return Envelope(
        envelope.message_type,
        envelope.instance_id,
        envelope.session_id,
        envelope.generation,
        envelope.request_id,
        _decode_value(body, _Budget(limits)),
        envelope.version,
    )
