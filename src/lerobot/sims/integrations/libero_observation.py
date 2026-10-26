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

"""LIBERO-native observation conventions, without importing the simulator."""

from collections.abc import Mapping

import numpy as np

from .utils import canonical_observation as _canonical_observation


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert LIBERO's xyzw quaternion with its legacy numerical convention."""
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[-1:] != (4,):
        raise ValueError("Quaternion must end in four xyzw components")
    w = np.clip(quat[..., 3], -1, 1)
    den = np.sqrt(np.maximum(1 - w * w, 0))
    scale = np.divide(2 * np.arccos(w), den, out=np.zeros_like(den), where=den > 1e-10)
    return (quat[..., :3] * scale[..., None]).astype(np.float32)


def canonical_observation(obs: Mapping) -> dict[str, np.ndarray]:
    """Apply LIBERO camera and robot-state conventions at the integration boundary."""
    adapted = dict(obs)
    state = adapted.pop("robot_state", None)
    if state is not None:
        adapted["agent_pos"] = np.concatenate(
            (state["eef"]["pos"], _quat_to_axisangle(state["eef"]["quat"]), state["gripper"]["qpos"]),
            axis=-1,
        ).astype(np.float32)
    return _canonical_observation(adapted, flip_images=True)
