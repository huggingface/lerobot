# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Shared weight-loading, dtype, timestep and camera-grid helpers."""

import contextlib
import math
import os
from collections.abc import Iterator

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import hf_hub_download
from torch import Tensor


def resolve_weights(spec: str, default_filename: str) -> str:
    """A local path is returned as is; otherwise use ``repo_id[:filename][@revision]``."""
    if os.path.exists(spec):
        return spec
    body, _, revision = spec.partition("@")
    repo_id, _, filename = body.partition(":")
    return hf_hub_download(repo_id, filename or default_filename, revision=revision or None)


@contextlib.contextmanager
def default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Temporarily set the default tensor dtype and restore it on exit."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def rational_time_shift(t: Tensor, shift: float) -> Tensor:
    """Apply a rational shift to timesteps in [0, 1]."""
    return shift * t / (1.0 + (shift - 1.0) * t)


def grid_shape(n_cams: int) -> tuple[int, int]:
    """Near-square ``(rows, cols)``, wider than tall: 1 -> 1x1, 2 -> 1x2, 3 and 4 -> 2x2, 5 and 6 -> 2x3."""
    cols = math.ceil(math.sqrt(n_cams))
    return math.ceil(n_cams / cols), cols


def compose_grid(cams: Tensor, canvas_hw: tuple[int, int]) -> Tensor:
    """Resize cameras into a row-major grid; unused cells stay black.

    Maps ``(n_cams, T, C, H, W)`` to ``(T, C, canvas_h, canvas_w)``.
    """
    n_cams, t, c = cams.shape[:3]
    ch, cw = canvas_hw
    rows, cols = grid_shape(n_cams)
    cell_h, cell_w = ch // rows, cw // cols
    canvas = cams.new_zeros(t, c, ch, cw)
    for i, cam in enumerate(cams.unbind(0)):
        r, k = divmod(i, cols)
        cell = F.interpolate(cam, size=(cell_h, cell_w), mode="bilinear", align_corners=False, antialias=True)
        canvas[:, :, r * cell_h : (r + 1) * cell_h, k * cell_w : (k + 1) * cell_w] = cell
    return canvas
