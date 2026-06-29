# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Training-free observation editing: translate the masked cube within a frame.

This is the *image-edit baseline* for overhead time-advance: erase the cube from
its current location (filling the hole with the surrounding background color) and
re-paste it shifted by ``offset_px``, so the policy sees the cube where it will be
after the inference latency. A latent-space variant can later replace this module
without touching the RTC integration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def shift_cube_in_frame(
    frame: NDArray,
    mask: NDArray[np.bool_],
    offset_px: tuple[float, float],
) -> NDArray:
    """Return a copy of ``frame`` with the masked region translated by ``offset_px``.

    Args:
        frame: ``HxWx3`` RGB image (any dtype; returned dtype matches input).
        mask: ``HxW`` boolean cube mask (same spatial size as ``frame``).
        offset_px: ``(dx, dy)`` translation in pixels (``dx`` is columns/x, ``dy``
            is rows/y). Rounded to the nearest integer.

    The hole left by the erased cube is filled with the median color of the
    non-cube pixels (a good approximation for a uniform conveyor belt). Pixels
    that would land outside the frame are dropped.
    """
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Expected an RGB image with shape HxWx3, got {frame.shape}")
    if mask.shape != frame.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match frame {frame.shape[:2]}")

    dx = int(round(offset_px[0]))
    dy = int(round(offset_px[1]))
    if (dx == 0 and dy == 0) or not mask.any():
        return frame

    h, w = mask.shape
    out = frame.copy()

    # Background fill for the erased cube: median of the non-cube pixels.
    bg = frame[~mask]
    fill = np.median(bg, axis=0) if bg.size else frame.reshape(-1, frame.shape[2]).mean(axis=0)
    out[mask] = fill.astype(frame.dtype)

    # Re-paste the cube pixels at the shifted location, dropping out-of-bounds.
    ys, xs = np.nonzero(mask)
    tys = ys + dy
    txs = xs + dx
    inside = (tys >= 0) & (tys < h) & (txs >= 0) & (txs < w)
    out[tys[inside], txs[inside]] = frame[ys[inside], xs[inside]]
    return out
