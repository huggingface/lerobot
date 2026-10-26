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

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def canonical_observation(
    obs: Mapping,
    *,
    flip_images: bool = False,
) -> dict[str, np.ndarray]:
    """Map the native wrappers' common values to canonical LeRobot observations."""
    result = {}
    pixels = obs.get("pixels")
    if pixels is not None:
        images = pixels if isinstance(pixels, dict) else {None: pixels}
        for name, image in images.items():
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("Canonical RGB requires HWC uint8")
            if flip_images:
                image = image[::-1, ::-1]
            key = f"observation.images.{name}" if name is not None else "observation.image"
            result[key] = np.ascontiguousarray(image)
    for raw, canonical in (
        ("agent_pos", "observation.state"),
        ("environment_state", "observation.environment_state"),
    ):
        if raw in obs:
            result[canonical] = np.asarray(obs[raw], dtype=np.float32)
    handled = {"pixels", "agent_pos", "environment_state"}
    for key, value in obs.items():
        if key not in handled and isinstance(value, np.ndarray):
            target = key if key.startswith("observation.") else f"observation.{key}"
            result[target] = np.asarray(value, dtype=np.float32)
    return result


def parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize ``camera_name`` into a non-empty list of strings.

    Accepts a comma-separated string (``"cam_a,cam_b"``) or a sequence of
    strings (tuples/lists). Whitespace is stripped; empty entries are
    dropped. Raises ``TypeError`` for unsupported input types and
    ``ValueError`` when the normalized list is empty.
    """
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams
