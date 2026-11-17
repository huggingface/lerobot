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

"""Helpers shared by simulator integrations."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR

# Same as lerobot.lerobot_types.RobotObservation, redefined here so integrations don't import torch.
RobotObservation = dict[str, Any]


def canonical_observation(
    obs: Mapping,
    *,
    flip_images: bool = False,
) -> dict[str, np.ndarray]:
    """Map a simulator observation to LeRobot keys; images must be HWC `uint8` with three channels.

    With `flip_images`, images are rotated by 180 degrees, as LIBERO needs.
    """
    result = {}
    pixels = obs.get("pixels")
    if pixels is not None:
        images = pixels if isinstance(pixels, dict) else {None: pixels}
        for name, image in images.items():
            key = f"{OBS_IMAGES}.{name}" if name is not None else OBS_IMAGE
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(
                    f"Feature {key!r}: images must be HWC uint8 with three channels, "
                    f"got {image.shape}, {image.dtype}"
                )
            if flip_images:
                image = image[::-1, ::-1]
            result[key] = np.ascontiguousarray(image)
    for raw, canonical in (("agent_pos", OBS_STATE), ("environment_state", OBS_ENV_STATE)):
        if raw in obs:
            result[canonical] = np.asarray(obs[raw], dtype=np.float32)
    handled = {"pixels", "agent_pos", "environment_state"}
    for key, value in obs.items():
        if key not in handled and isinstance(value, np.ndarray):
            target = key if key.startswith(f"{OBS_STR}.") else f"{OBS_STR}.{key}"
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
