#!/usr/bin/env python

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
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.envs.lazy_vec_env import LazyVectorEnv

# Action layout (flat 12D, normalized to [-1, 1]):
#   [0:3]   end_effector_position  (delta x, y, z)
#   [3:6]   end_effector_rotation  (delta roll, pitch, yaw)
#   [6:7]   gripper_close          (open=-1, close=+1)
#   [7:11]  base_motion            (x, y, theta, torso_height)
#   [11:12] control_mode           (arm=-1, base=+1)
ACTION_DIM = 12
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# Proprioceptive state layout (flat 16D):
#   [0:2]   gripper_qpos
#   [2:5]   base_position
#   [5:9]   base_rotation (quaternion)
#   [9:12]  end_effector_position_relative
#   [12:16] end_effector_rotation_relative (quaternion)
STATE_DIM = 16

# Obs dict keys from RoboCasaGymEnv.get_observation()
_CAM_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
_STATE_KEYS_ORDERED = (
    "state.gripper_qpos",           # (2,)
    "state.base_position",          # (3,)
    "state.base_rotation",          # (4,)
    "state.end_effector_position_relative",   # (3,)
    "state.end_effector_rotation_relative",   # (4,)
)

# Mapping from video.* key → short image name used in features_map
CAM_KEY_TO_NAME = {
    "video.robot0_agentview_left":  "agentview_left",
    "video.robot0_agentview_right": "agentview_right",
    "video.robot0_eye_in_hand":     "eye_in_hand",
}


def _flat_to_action_dict(flat: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a 12D flat action array to the Dict format expected by RoboCasaGymEnv."""
    return {
        "action.end_effector_position": flat[0:3],
        "action.end_effector_rotation": flat[3:6],
        "action.gripper_close":         flat[6:7],
        "action.base_motion":           flat[7:11],
        "action.control_mode":          flat[11:12],
    }


class RoboCasaEnv(gym.Env):
    """Thin wrapper around RoboCasaGymEnv that provides a flat Box action space
    and a structured observation dict compatible with LeRobot policies.

    Observations returned by step/reset:
        {
            "pixels": {
                "agentview_left":  (H, W, 3) uint8,
                "agentview_right": (H, W, 3) uint8,
                "eye_in_hand":     (H, W, 3) uint8,
            },
            "robot_state": (16,) float32,
        }

    Actions: flat float32 ndarray of shape (12,), normalized to [-1, 1].
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str,
        split: str = "target",
        image_size: int = 128,
        render_mode: str = "rgb_array",
        episode_length: int = 500,
        **gym_kwargs: Any,
    ):
        super().__init__()
        # Lazy import — robocasa is optional
        import robocasa.environments  # noqa: F401 — registers all gym envs

        self.task = task
        self.render_mode = render_mode
        self.image_size = image_size
        self._max_episode_steps = episode_length
        self._step_count = 0

        self._env = gym.make(
            f"robocasa/{task}",
            split=split,
            camera_widths=image_size,
            camera_heights=image_size,
            **gym_kwargs,
        )

        # Flat 12D Box action space
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

        images = {
            name: spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8)
            for name in CAM_KEY_TO_NAME.values()
        }
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(images),
                "robot_state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
                ),
            }
        )

    def _format_obs(self, raw_obs: dict) -> dict:
        pixels = {
            CAM_KEY_TO_NAME[k]: raw_obs[k]
            for k in _CAM_KEYS
            if k in raw_obs
        }
        state_parts = [
            np.asarray(raw_obs[k], dtype=np.float32)
            for k in _STATE_KEYS_ORDERED
            if k in raw_obs
        ]
        robot_state = np.concatenate(state_parts) if state_parts else np.zeros(STATE_DIM, dtype=np.float32)
        return {"pixels": pixels, "robot_state": robot_state}

    def reset(self, seed: int | None = None, **kwargs) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._step_count = 0
        raw_obs, info = self._env.reset(seed=seed)
        info.setdefault("is_success", False)
        info["task"] = self.task
        return self._format_obs(raw_obs), info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        if action.ndim != 1 or action.shape[0] != ACTION_DIM:
            raise ValueError(
                f"Expected 1-D action of shape ({ACTION_DIM},), got {action.shape}"
            )
        action_dict = _flat_to_action_dict(action)
        raw_obs, reward, terminated, truncated, info = self._env.step(action_dict)
        self._step_count += 1

        is_success = bool(info.get("success", False))
        terminated = terminated or is_success
        if self._step_count >= self._max_episode_steps:
            truncated = True

        info.update({"task": self.task, "is_success": is_success})
        obs = self._format_obs(raw_obs)

        if terminated or truncated:
            info["final_info"] = {"task": self.task, "is_success": is_success}

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._env.render()
        return None

    def close(self) -> None:
        self._env.close()


def _make_env_fns(
    *,
    task: str,
    n_envs: int,
    image_size: int,
    split: str,
    episode_length: int,
    gym_kwargs: dict[str, Any],
) -> list[Callable[[], RoboCasaEnv]]:
    """Build n_envs factory callables for a single task."""
    def _make(episode_index: int) -> RoboCasaEnv:  # noqa: ARG001
        return RoboCasaEnv(
            task=task,
            split=split,
            image_size=image_size,
            episode_length=episode_length,
            **gym_kwargs,
        )

    return [partial(_make, i) for i in range(n_envs)]


def create_robocasa_envs(
    tasks: str | Sequence[str],
    n_envs: int,
    image_size: int = 128,
    split: str = "target",
    episode_length: int = 500,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """Create vectorized RoboCasa environments.

    Args:
        tasks: A single task name or list of task names (without "robocasa/" prefix).
            E.g. "PickPlaceCounterToCabinet" or ["BoilPot", "PrepareCoffee"].
        n_envs: Number of parallel envs per task.
        image_size: Square image resolution for all cameras.
        split: RoboCasa dataset split — "pretrain" or "target".
        episode_length: Max steps per episode before truncation.
        gym_kwargs: Extra kwargs forwarded to each RoboCasaEnv.
        env_cls: Callable to wrap list of factory fns (SyncVectorEnv or AsyncVectorEnv).

    Returns:
        dict[task_name][task_id=0] -> vec_env
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable wrapping a list of env factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    if isinstance(tasks, str):
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    else:
        task_list = [str(t).strip() for t in tasks if str(t).strip()]
    if not task_list:
        raise ValueError("`tasks` must contain at least one task name.")

    gym_kwargs = dict(gym_kwargs or {})
    out: dict[str, dict[int, Any]] = defaultdict(dict)
    total_tasks = len(task_list)
    lazy = total_tasks > 50

    print(f"Creating RoboCasa envs | tasks={task_list} | n_envs(per task)={n_envs} | split={split}")
    if lazy:
        print(f"Using lazy env creation for {total_tasks} tasks (envs created on demand)")
    for task in task_list:
        fns = _make_env_fns(
            task=task,
            n_envs=n_envs,
            image_size=image_size,
            split=split,
            episode_length=episode_length,
            gym_kwargs=gym_kwargs,
        )
        out["robocasa"][len(out["robocasa"])] = LazyVectorEnv(env_cls, fns) if lazy else env_cls(fns)
        if not lazy:
            print(f"  Built vec env | task={task} | n_envs={n_envs}")

    return {suite: dict(task_map) for suite, task_map in out.items()}
