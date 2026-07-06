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

"""Single-track multi-camera tiling.

aiortc carries **one** video track; to stream N cameras we vertically stack their frames
into one tall frame on the robot (``tile``) and slice them back on the cloud (``untile``).
Both ends derive the **same** layout from the camera specs — sorted by name, so the order
is deterministic — which means no per-frame metadata is needed and the existing
seq-in-pts pairing is untouched (still one frame per capture seq).

Layout: cameras stacked top→bottom in name order; each keeps its own ``(h, w)``; narrower
ones are left-aligned and the row padded to the max width. **One camera is the identity**
(the combined frame *is* that frame), so single-camera behaviour is byte-for-byte
unchanged. This trades per-camera bitrate control for a single encoder (cheapest on
aiortc's Python/GIL encode path) — the right call for a few same-ish robot cameras; for
many/high-res streams use the LiveKit backend (DESIGN §11.1).
"""

from __future__ import annotations

import numpy as np

# A camera spec in the canonical tiling order: (name, height, width).
Spec = tuple[str, int, int]


def ordered_specs(cameras: dict[str, tuple[int, int]]) -> list[Spec]:
    """``{name: (h, w)}`` -> ``[(name, h, w), ...]`` sorted by name (the canonical order)."""
    return [(name, cameras[name][0], cameras[name][1]) for name in sorted(cameras)]


def tiled_size(specs: list[Spec]) -> tuple[int, int]:
    """``(height, width)`` of the combined frame: stacked heights × max width."""
    return sum(h for _, h, _ in specs), max((w for _, _, w in specs), default=0)


def tile(frames: dict[str, np.ndarray], specs: list[Spec]) -> np.ndarray:
    """Stack per-camera frames into one ``(ΣH, maxW, 3)`` uint8 frame (name order).

    Each ``frames[name]`` must already be exactly ``(h, w, 3)`` uint8 for its spec.
    """
    height, width = tiled_size(specs)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    y = 0
    for name, h, w in specs:
        out[y : y + h, 0:w] = frames[name]
        y += h
    return out


def untile(combined: np.ndarray, specs: list[Spec]) -> dict[str, np.ndarray]:
    """Inverse of :func:`tile`: slice the combined frame back into ``{name: (h, w, 3)}``.

    Returns views into ``combined``; copy/contiguous-ify downstream if needed.
    """
    out: dict[str, np.ndarray] = {}
    y = 0
    for name, h, w in specs:
        out[name] = combined[y : y + h, 0:w]
        y += h
    return out
