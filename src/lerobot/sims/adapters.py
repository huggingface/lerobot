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

"""Simulator-independent observation conversion and explicit episode boundaries."""

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_OBSERVATION_KEYS = {"agent_pos": "observation.state", "environment_state": "observation.environment_state"}


def env_attribute(env: Any, name: str, default: Any = None) -> Any:
    """Read native declarations through Gymnasium wrappers as well as plain environments."""
    if isinstance(env, gym.Wrapper):
        try:
            return env.get_wrapper_attr(name)
        except AttributeError:
            return default
    return getattr(env, name, default)


def episode_limit(env: Any) -> int | None:
    """Return the declared horizon, or None; an unknown horizon must not change the task."""
    limit = env_attribute(env, "_max_episode_steps")
    if limit is None:
        limit = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, Integral) or limit <= 0):
        raise ValueError("Simulator episode limit must be a positive integer or None")
    return int(limit) if limit is not None else None


def canonical_observation(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Rename declared arrays; native integrations own robot-specific conversions."""
    result = {}
    for key, value in obs.items():
        if key == "pixels":
            images = value if isinstance(value, Mapping) else {None: value}
            for name, image in images.items():
                image = np.asarray(image)
                if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError("Canonical RGB requires HWC uint8")
                target = f"observation.images.{name}" if name is not None else "observation.image"
                result[target] = np.ascontiguousarray(image)
            continue
        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"Observation {key!r} must be an array; flatten native observations in the integration"
            )
        target = _OBSERVATION_KEYS.get(key, key if key.startswith("observation.") else f"observation.{key}")
        if target == "observation.image" or target.startswith("observation.images."):
            if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
                raise ValueError("Canonical RGB requires HWC uint8")
            result[target] = np.ascontiguousarray(value)
        else:
            result[target] = np.asarray(value, dtype=np.float32)
    return result


def canonical_observation_space(space: spaces.Dict) -> spaces.Dict:
    """Describe the same conversion using spaces, without sampling or rendering."""
    result = {}
    for key, value in space.spaces.items():
        if key == "pixels":
            images = value.spaces if isinstance(value, spaces.Dict) else {None: value}
            for name, image in images.items():
                if (
                    not isinstance(image, spaces.Box)
                    or len(image.shape) != 3
                    or image.shape[-1] != 3
                    or image.dtype != np.uint8
                ):
                    raise ValueError("Canonical RGB requires HWC uint8")
                target = f"observation.images.{name}" if name is not None else "observation.image"
                result[target] = image
            continue
        if not isinstance(value, spaces.Box):
            raise ValueError(f"Observation {key!r} requires a Box space; flatten it in the integration")
        target = _OBSERVATION_KEYS.get(key, key if key.startswith("observation.") else f"observation.{key}")
        if target == "observation.image" or target.startswith("observation.images."):
            if len(value.shape) != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
                raise ValueError("Canonical RGB requires HWC uint8")
            result[target] = value
        else:
            result[target] = spaces.Box(-np.inf, np.inf, value.shape, np.float32)
    return spaces.Dict(result)


def check_explicit_reset(env: Any) -> None:
    """Reject vector or automatic-reset environments before acquiring an episode."""
    if isinstance(env, gym.vector.VectorEnv):
        raise ValueError("SimRobot requires one scalar environment, not a vector environment")
    current = env
    if env_attribute(env, "is_vector_env", False):
        raise ValueError("SimRobot requires one scalar environment, not a native vector environment")
    while isinstance(current, gym.Wrapper):
        if isinstance(current, gym.wrappers.Autoreset):
            raise ValueError("Automatic reset is unsupported; supply an environment with explicit reset")
        current = current.env


def check_step_info(info: dict[str, Any]) -> None:
    """Never mistake an automatically reset observation for the terminal observation."""
    if any(key in info for key in ("final_observation", "final_obs", "terminal_observation", "final_info")):
        raise ValueError("Automatic-reset step detected; disable autoreset in the simulator integration")


def hold_action(action: np.ndarray, control: str, gripper_indices: tuple[int, ...]) -> np.ndarray:
    """Retain position targets or zero motion while preserving declared gripper components."""
    result = action.copy()
    if control in {"delta", "velocity"}:
        result.fill(0)
        result[..., list(gripper_indices)] = action[..., list(gripper_indices)]
    elif control not in {"position", "eef_pose"}:
        raise ValueError(f"Unsupported hold control: {control}")
    return result
