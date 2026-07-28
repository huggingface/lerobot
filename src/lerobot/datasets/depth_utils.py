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
Depth encoding/decoding helpers for :class:`DepthEncoderConfig`.
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
    DEFAULT_DEPTH_PIX_FMT,
    DEFAULT_DEPTH_SHIFT,
    DEFAULT_DEPTH_USE_LOG,
    DEPTH_METER_UNIT,
    DEPTH_MILLIMETER_UNIT,
    DEPTH_QMAX,
    infer_depth_unit,
)

from .image_writer import squeeze_single_channel
from .pyav_utils import write_u16_plane

MM_PER_METRE = 1000.0
_UINT16_MAX = 65535


def _validate_log_quant_params(depth_min: float, shift: float) -> None:
    """Ensure ``log(depth_min + shift)`` is finite."""
    if depth_min + shift <= 0:
        raise ValueError(
            f"depth_min + shift must be positive for logarithmic quantization, "
            f"got depth_min={depth_min} + shift={shift} = {depth_min + shift}"
        )


def _depth_input_to_float32_and_unit(
    depth: NDArray[np.integer] | NDArray[np.floating],
    input_unit: Literal["auto", DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT],
) -> tuple[NDArray[np.float32], Literal[DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT]]:
    """Convert depth to float32 in the chosen unit, and return the resolved unit."""
    resolved_unit = infer_depth_unit(depth.dtype) if input_unit == "auto" else input_unit
    return depth.astype(np.float32, order="K"), resolved_unit


def quantize_depth(
    depth: NDArray[np.uint16] | NDArray[np.float32] | torch.Tensor,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
    use_log: bool = DEFAULT_DEPTH_USE_LOG,
    pix_fmt: str = DEFAULT_DEPTH_PIX_FMT,
    video_backend: str | None = "pyav",
    input_unit: Literal["auto", DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT] = "auto",
) -> NDArray[np.uint16] | av.VideoFrame:
    """Quantize depth to 12-bit codes (``uint16``, values ``0…DEPTH_QMAX``).

    Depth maps are packed into 12-bit integer frames so they fit in standard
    high-bit-depth pixel formats (e.g. ``yuv420p12le`` / ``gray12le``)
    and can be encoded by widely supported video codecs (e.g. HEVC Main 12).
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
    if input_unit not in ("auto", DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT):
        raise ValueError(
            f"input_unit must be 'auto', '{DEPTH_METER_UNIT}', or '{DEPTH_MILLIMETER_UNIT}', got {input_unit!r}"
        )

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    # Squeeze single-channel dim: (H, W, 1) or (1, H, W) → (H, W)
    depth = squeeze_single_channel(depth)

    depth_f, resolved_unit = _depth_input_to_float32_and_unit(depth, input_unit=input_unit)

    # Convert depth_min, depth_max, and shift to the resolved input unit.
    depth_min_u = (
        np.float32(depth_min) if resolved_unit == DEPTH_METER_UNIT else np.float32(depth_min * MM_PER_METRE)
    )
    depth_max_u = (
        np.float32(depth_max) if resolved_unit == DEPTH_METER_UNIT else np.float32(depth_max * MM_PER_METRE)
    )
    shift_u = np.float32(shift) if resolved_unit == DEPTH_METER_UNIT else np.float32(shift * MM_PER_METRE)

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
        frame = av.VideoFrame.from_ndarray(quantized, format=pix_fmt)
        write_u16_plane(frame.planes[0], quantized)
        return frame
    else:
        return quantized


def dequantize_depth(
    quantized: NDArray[np.uint16] | av.VideoFrame | torch.Tensor,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
    use_log: bool = DEFAULT_DEPTH_USE_LOG,
    pix_fmt: str = DEFAULT_DEPTH_PIX_FMT,
    output_unit: Literal[DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT] = DEPTH_MILLIMETER_UNIT,
    output_tensor: bool = True,
    output_channel_last: bool = False,
) -> NDArray[np.uint16] | NDArray[np.float32] | torch.Tensor:
    """Inverse of :func:`quantize_depth`.

    Decoding inverts the same normalized code mapping as :func:`quantize_depth`
    using ``depth_min`` / ``depth_max`` / ``shift`` (in metres), then returns
    the requested output unit. Tuning arguments **must match** :func:`quantize_depth`.

    Accepted input layouts :

    - ``(H, W, 1)`` or ``(H, W)`` — single frame with channel-last.
    - ``(..., 1, H, W)`` — batched frames with channel-first.
    - ``(..., H, W, 1)`` — batched frames with channel-last.
    Output layout is determined by ``output_channel_last``.

    Args:
        quantized: 12-bit codes in ``[0, DEPTH_QMAX]``. ``np.ndarray``,
            ``av.VideoFrame``, or ``torch.Tensor`` (any integer or float dtype).
        depth_min, depth_max, shift, use_log: Same as :func:`quantize_depth` (metres).
        pix_fmt: Pixel format used to extract the plane from an ``av.VideoFrame``.
        output_unit: ``"mm"`` returns ``uint16`` millimetres (rint, clip
            ``[0, 65535]``) when returning a numpy array, or ``float32`` mm when
            ``output_tensor=True``. ``"m"`` returns ``float32`` metres in
            ``[depth_min, depth_max]``.
        output_tensor: If True, return a ``torch.Tensor`` instead of a numpy array.

    Returns:
        Depth map in the requested unit and dtype.

    Raises:
        ValueError: If ``output_unit`` is not ``"m"`` or ``"mm"``.
        ValueError: If ``use_log=True`` and ``depth_min + shift <= 0``.
    """
    if output_unit not in (DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT):
        raise ValueError(
            f"output_unit must be '{DEPTH_METER_UNIT}' or '{DEPTH_MILLIMETER_UNIT}', got {output_unit!r}"
        )
    if use_log:
        _validate_log_quant_params(depth_min, shift)

    if isinstance(quantized, av.VideoFrame):
        quantized = quantized.to_ndarray(format=pix_fmt)

    # Compute the scale and offset first.
    depth_min_m = float(depth_min)
    depth_max_m = float(depth_max)
    shift_m = float(shift)
    if use_log:
        log_min = math.log(depth_min_m + shift_m)
        log_max = math.log(depth_max_m + shift_m)
        scale = (log_max - log_min) / DEPTH_QMAX
        offset = log_min
    else:
        scale = (depth_max_m - depth_min_m) / DEPTH_QMAX
        offset = depth_min_m

    # ── Torch path: stay on the input device, single fp32 allocation. ────────
    if isinstance(quantized, torch.Tensor):
        if quantized.ndim >= 3:
            # Drop the single-channel dimension so the math runs on (..., H, W).
            quantized = quantized.squeeze(-3) if quantized.shape[-3] == 1 else quantized.squeeze(-1)

        # Single allocation we own; everything else is in-place.
        buf = quantized.to(dtype=torch.float32, copy=True)
        buf.mul_(scale).add_(offset)
        if use_log:
            buf.exp_().sub_(shift_m)
        buf.clamp_(depth_min_m, depth_max_m)
        buf.unsqueeze_(-1) if output_channel_last else buf.unsqueeze_(-3)

        if output_unit == DEPTH_METER_UNIT:
            return buf if output_tensor else buf.cpu().numpy()

        # mm path: round + clamp in float32, skipping the uint16 round-trip
        # when returning a tensor (torch.uint16 is poorly supported).
        buf.mul_(MM_PER_METRE).round_().clamp_(0.0, _UINT16_MAX)
        if output_tensor:
            return buf
        return buf.cpu().numpy().astype(np.uint16, copy=False)

    # ── NumPy path: single fp32 allocation, ``out=`` for in-place math. ─────
    arr = np.asarray(quantized)
    if arr.ndim >= 3:
        # Drop the single-channel dimension so the math runs on (..., H, W).
        arr = np.squeeze(arr, axis=-3) if arr.shape[-3] == 1 else np.squeeze(arr, axis=-1)

    buf = np.empty(arr.shape, dtype=np.float32)
    np.multiply(arr, scale, out=buf)
    np.add(buf, offset, out=buf)
    if use_log:
        np.exp(buf, out=buf)
        np.subtract(buf, shift_m, out=buf)
    np.clip(buf, depth_min_m, depth_max_m, out=buf)
    buf = np.expand_dims(buf, axis=-1) if output_channel_last else np.expand_dims(buf, axis=-3)

    if output_unit == DEPTH_METER_UNIT:
        return torch.from_numpy(buf) if output_tensor else buf

    np.multiply(buf, MM_PER_METRE, out=buf)
    np.rint(buf, out=buf)
    np.clip(buf, 0.0, _UINT16_MAX, out=buf)
    if output_tensor:
        # torch.uint16 support is very limited; return float32 millimetres.
        return torch.from_numpy(buf)
    return buf.astype(np.uint16, copy=False)
