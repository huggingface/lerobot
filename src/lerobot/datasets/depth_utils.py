#!/usr/bin/env python

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
"""
Depth encoding/decoding helpers for :class:`VideoEncoderConfig`.
"""

import math
from typing import Literal

import av
import numpy as np
import torch
from numpy.typing import NDArray

from lerobot.configs.video import (
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_DEPTH_SHIFT,
    DEFAULT_DEPTH_USE_LOG,
    DEPTH_QMAX,
)

from .pyav_utils import write_u16_plane

_MM_PER_METRE = 1000.0
_UINT16_MAX = 65535

# Pixel format supported by the depth encode/decode helpers.
DEPTH_PIX_FMT: str = "gray12le"


def _validate_log_quant_params(depth_min: float, shift: float) -> None:
    """Ensure ``log(depth_min + shift)`` is finite."""
    if depth_min + shift <= 0:
        raise ValueError(
            f"depth_min + shift must be positive for logarithmic quantization, "
            f"got depth_min={depth_min} + shift={shift} = {depth_min + shift}"
        )


def _depth_input_to_float32_and_unit(
    depth: NDArray[np.integer] | NDArray[np.floating],
    input_unit: Literal["auto", "m", "mm"],
) -> tuple[NDArray[np.float32], Literal["m", "mm"]]:
    """Convert depth to float32 in the chosen unit, and return the resolved unit."""
    resolved_unit = (
        ("m" if np.issubdtype(depth.dtype, np.floating) else "mm") if input_unit == "auto" else input_unit
    )
    return depth.astype(np.float32, order="K"), resolved_unit


def quantize_depth(
    depth: NDArray[np.uint16] | NDArray[np.float32] | torch.Tensor,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
    use_log: bool = DEFAULT_DEPTH_USE_LOG,
    video_backend: str | None = "pyav",
    input_unit: Literal["auto", "m", "mm"] = "auto",
) -> NDArray[np.uint16] | av.VideoFrame:
    """Quantize depth to 12-bit codes (``uint16``, values ``0…DEPTH_QMAX``).

    Depth maps are packed into 12-bit integer frames so they fit in standard
    high-bit-depth pixel formats (e.g. ``yuv420p12le`` / ``gray12le``)
    and can be encoded by widely supported video codecs (HEVC Main 12, ffv1).
    Logarithmic quantization is the default because it allocates more quanta
    to near-range depth, which matches the (1/depth) error profile of typical
    depth sensors. Math is ported from BEHAVIOR-1K's ``obs_utils.py``.

    **Input units**:

    - ``input_unit="auto"`` (default): infer from dtype (floating = m, non-floating = mm).
    - ``input_unit="mm"``: interpret input values as millimetres.
    - ``input_unit="m"``: interpret input values as metres.

    Quantization math runs in the **resolved input unit**.

    ``depth_min``, ``depth_max``, and ``shift`` are always in **metres**.

    Args:
        depth: Depth map; ``torch.Tensor`` is moved to CPU for conversion.
        depth_min: Depth (metres) at quantum ``0``.
        depth_max: Depth (metres) at quantum :data:`DEPTH_QMAX`.
        shift: Depth shift (metres); used in log mode. Must satisfy ``depth_min + shift > 0``.
        use_log: If ``True`` (default), quantize in log space.
        video_backend: Video backend to use for encoding. Defaults to "pyav".
        input_unit: Input unit policy (``"auto"``, ``"mm"``, ``"m"``).

    Returns:
        ``numpy.ndarray``, ``dtype=uint16``, same shape as ``depth``, values in
        ``[0, DEPTH_QMAX]``.

    Raises:
        ValueError: If ``input_unit`` is not ``"auto"``, ``"mm"``, or ``"m"``.
        ValueError: If ``use_log=True`` and ``depth_min + shift <= 0``.
    """
    if input_unit not in ("auto", "m", "mm"):
        raise ValueError(f"input_unit must be 'auto', 'm', or 'mm', got {input_unit!r}")

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    # Squeeze single-channel dim: (H, W, 1) or (1, H, W) → (H, W)
    if depth.ndim == 3 and (depth.shape[-1] == 1 or depth.shape[0] == 1):
        depth = depth.squeeze()

    depth_f, resolved_unit = _depth_input_to_float32_and_unit(depth, input_unit=input_unit)

    # Convert depth_min, depth_max, and shift to the resolved input unit.
    depth_min_u = np.float32(depth_min) if resolved_unit == "m" else np.float32(depth_min * _MM_PER_METRE)
    depth_max_u = np.float32(depth_max) if resolved_unit == "m" else np.float32(depth_max * _MM_PER_METRE)
    shift_u = np.float32(shift) if resolved_unit == "m" else np.float32(shift * _MM_PER_METRE)

    # Normalization and quantization is performed in the resolved input unit.
    if use_log:
        _validate_log_quant_params(depth_min, shift)
        log_min = math.log(float(depth_min_u + shift_u))
        log_max = math.log(float(depth_max_u + shift_u))
        norm = (np.log(depth_f + shift_u) - log_min) / (log_max - log_min)
    else:
        norm = (depth_f - depth_min_u) / (depth_max_u - depth_min_u)

    quantized = np.rint(norm * DEPTH_QMAX).clip(0, DEPTH_QMAX).astype(np.uint16, copy=False)

    if video_backend == "pyav":
        frame = av.VideoFrame.from_ndarray(quantized, format=DEPTH_PIX_FMT)
        write_u16_plane(frame.planes[0], quantized)
        return frame
    else:
        return quantized


def dequantize_depth(
    quantized: NDArray[np.uint16] | av.VideoFrame,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
    use_log: bool = DEFAULT_DEPTH_USE_LOG,
    output_unit: Literal["m", "mm"] = "mm",
    output_tensor: bool = False,
) -> NDArray[np.uint16] | NDArray[np.float32] | torch.Tensor:
    """Inverse of :func:`quantize_depth`.

    Tuning arguments **must match** :func:`quantize_depth`.

    Decoding inverts the same normalized code mapping as :func:`quantize_depth`
    using ``depth_min`` / ``depth_max`` / ``shift`` (in metres), then returns
    the requested output unit.

    Args:
        quantized: 12-bit codes ``[0, DEPTH_QMAX]``, ``dtype=uint16``.
        depth_min, depth_max, shift, use_log: Same as :func:`quantize_depth` (metres).
        output_unit: ``\"mm\"`` returns ``uint16`` millimetres (``rint``, clip
            ``[0, 65535]``). ``\"m\"`` returns ``float32`` metres in
            ``[depth_min, depth_max]``.
        output_tensor: If True, return a torch.Tensor instead of a numpy array.

    Returns:
        Depth map in the requested unit and dtype.

    Raises:
        ValueError: If ``use_log=True`` and ``depth_min + shift <= 0``.
        ValueError: If ``output_unit`` is not ``\"m\"`` or ``\"mm\"``.
    """
    if output_unit not in ("m", "mm"):
        raise ValueError(f"output_unit must be 'm' or 'mm', got {output_unit!r}")

    if isinstance(quantized, av.VideoFrame):
        quantized = quantized.to_ndarray(format=DEPTH_PIX_FMT)

    norm = np.asarray(quantized, dtype=np.float32, order="K") / DEPTH_QMAX

    depth_min_m = np.float32(depth_min)
    depth_max_m = np.float32(depth_max)
    shift_m = np.float32(shift)

    # The de-normalization and de-quantization is performed in meters (convenience choice).
    if use_log:
        _validate_log_quant_params(depth_min, shift)
        log_min = math.log(float(depth_min_m + shift_m))
        log_max = math.log(float(depth_max_m + shift_m))
        depth_m = np.exp(norm * (log_max - log_min) + log_min) - shift_m
    else:
        depth_m = norm * (depth_max_m - depth_min_m) + depth_min_m
    depth_m = np.clip(depth_m, depth_min_m, depth_max_m).astype(np.float32, copy=False)

    # Add single-channel dim: (H, W) → (H, W, 1)
    if depth_m.ndim == 2:
        depth_m = depth_m[..., np.newaxis]

    # Return depth as float32 meters.
    if output_unit == "m":
        return torch.from_numpy(depth_m) if output_tensor else depth_m

    # Return depth as uint16 millimeters.
    mm = np.rint(depth_m * _MM_PER_METRE).clip(0, _UINT16_MAX).astype(np.uint16, copy=False)
    if output_tensor:
        # torch.uint16 support is very limited, we convert to float32 instead.
        return torch.from_numpy(mm.astype(np.float32))
    else:
        return mm
