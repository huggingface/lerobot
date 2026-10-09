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

"""Native simulator observations to canonical dataset values, using NumPy only."""

from collections.abc import Mapping

import numpy as np


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions with the legacy LIBERO numerical convention."""
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[-1:] != (4,):
        raise ValueError("Quaternion must end in four xyzw components")
    w = np.clip(quat[..., 3], -1, 1)
    den = np.sqrt(np.maximum(1 - w * w, 0))
    scale = np.divide(2 * np.arccos(w), den, out=np.zeros_like(den), where=den > 1e-10)
    return (quat[..., :3] * scale[..., None]).astype(np.float32)


def canonical_observation(obs: Mapping, sim_type: str) -> dict[str, np.ndarray]:
    """Emit canonical dataset values without model preprocessing or normalization."""
    result = {}
    pixels = obs.get("pixels")
    if pixels is not None:
        images = pixels if isinstance(pixels, dict) else {None: pixels}
        for name, image in images.items():
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("Canonical RGB requires HWC uint8")
            if sim_type in {"libero", "libero_plus", "robocerebra"}:
                image = image[::-1, ::-1]
            key = f"observation.images.{name}" if name is not None else "observation.image"
            result[key] = np.ascontiguousarray(image)
    if "robot_state" in obs:
        state = obs["robot_state"]
        result["observation.state"] = np.concatenate(
            (state["eef"]["pos"], quat_to_axisangle(state["eef"]["quat"]), state["gripper"]["qpos"]), axis=-1
        ).astype(np.float32)
    for raw, canonical in (
        ("agent_pos", "observation.state"),
        ("environment_state", "observation.environment_state"),
    ):
        if raw in obs:
            result[canonical] = np.asarray(obs[raw], dtype=np.float32)
    handled = {"pixels", "robot_state", "agent_pos", "environment_state"}
    for key, value in obs.items():
        if key not in handled and isinstance(value, np.ndarray):
            target = key if key.startswith("observation.") else f"observation.{key}"
            result[target] = np.asarray(value, dtype=np.float32)
    return result


def hold_action(action: np.ndarray, control: str, gripper_indices: tuple[int, ...]) -> np.ndarray:
    """Retain position targets or zero motion while preserving declared gripper components."""
    result = action.copy()
    if control in {"delta", "velocity"}:
        result.fill(0)
        result[..., list(gripper_indices)] = action[..., list(gripper_indices)]
    elif control not in {"position", "eef_pose"}:
        raise ValueError(f"Unsupported hold control: {control}")
    return result
