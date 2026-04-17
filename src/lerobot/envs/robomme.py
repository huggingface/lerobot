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

"""RoboMME environment wrapper for LeRobot evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROBOMME_TASKS = [
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]


class RoboMMEGymEnv(gym.Env):
    """Thin Gymnasium wrapper around a single RoboMME episode env."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        task: str = "PickXtimes",
        action_space_type: str = "joint_angle",
        dataset: str = "test",
        episode_idx: int = 0,
        max_steps: int = 300,
    ):
        super().__init__()
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        self._builder = BenchmarkEnvBuilder(
            env_id=task,
            dataset=dataset,
            action_space=action_space_type,
            gui_render=False,
            max_steps=max_steps,
        )
        self._max_episode_steps = max_steps
        self._episode_idx = episode_idx
        self._max_steps = max_steps
        self._env = None
        self._last_raw_obs: dict | None = None

        action_dim = 8 if action_space_type == "joint_angle" else 7
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8),
                "wrist_image": spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env = self._builder.make_env_for_episode(
            episode_idx=self._episode_idx,
            max_steps=self._max_steps,
        )
        obs, info = self._env.reset()
        self._last_raw_obs = obs
        return self._convert_obs(obs), self._convert_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._last_raw_obs = obs

        terminated_bool = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
        truncated_bool = bool(truncated.item()) if hasattr(truncated, "item") else bool(truncated)

        status = info.get("status", "ongoing")
        conv_info = self._convert_info(info)
        conv_info["is_success"] = status == "success"

        return self._convert_obs(obs), float(reward), terminated_bool, truncated_bool, conv_info

    def render(self) -> np.ndarray | None:
        if self._last_raw_obs is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        front = self._last_raw_obs.get("front_rgb_list")
        if front is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        frame = front[-1] if isinstance(front, list) else front
        return np.asarray(frame, dtype=np.uint8)

    def _convert_obs(self, obs: dict) -> dict:
        front_rgb = (
            obs["front_rgb_list"][-1] if isinstance(obs["front_rgb_list"], list) else obs["front_rgb_list"]
        )
        wrist_rgb = (
            obs["wrist_rgb_list"][-1] if isinstance(obs["wrist_rgb_list"], list) else obs["wrist_rgb_list"]
        )
        joint_state = (
            obs["joint_state_list"][-1]
            if isinstance(obs["joint_state_list"], list)
            else obs["joint_state_list"]
        )
        gripper_state = (
            obs["gripper_state_list"][-1]
            if isinstance(obs["gripper_state_list"], list)
            else obs["gripper_state_list"]
        )

        joint = np.asarray(joint_state, dtype=np.float32).flatten()[:7]
        gripper = np.asarray(gripper_state, dtype=np.float32).flatten()[:1]
        state = np.concatenate([joint, gripper])

        return {
            "image": np.asarray(front_rgb, dtype=np.uint8),
            "wrist_image": np.asarray(wrist_rgb, dtype=np.uint8),
            "state": state,
        }

    def _convert_info(self, info: dict) -> dict:
        return {
            "status": info.get("status", "ongoing"),
            "task_goal": info.get("task_goal", ""),
        }


def _make_env_fns(
    *,
    task: str,
    n_envs: int,
    action_space_type: str,
    dataset: str,
    episode_length: int,
    task_id: int,
) -> list[Callable[[], RoboMMEGymEnv]]:
    def _make_one(episode_index: int) -> RoboMMEGymEnv:
        return RoboMMEGymEnv(
            task=task,
            action_space_type=action_space_type,
            dataset=dataset,
            episode_idx=episode_index,
            max_steps=episode_length,
        )

    return [partial(_make_one, task_id + i) for i in range(n_envs)]


def create_robomme_envs(
    task: str,
    n_envs: int = 1,
    action_space_type: str = "joint_angle",
    dataset: str = "test",
    episode_length: int = 300,
    task_ids: list[int] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create vectorized RoboMME environments for evaluation."""
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of env factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    if task_ids is None:
        task_ids = [0]

    task_names = [t.strip() for t in task.split(",") if t.strip()]
    out: dict[str, dict[int, gym.vector.VectorEnv]] = {}
    for task_name in task_names:
        envs_by_task: dict[int, gym.vector.VectorEnv] = {}
        for task_id in task_ids:
            fns = _make_env_fns(
                task=task_name,
                n_envs=n_envs,
                action_space_type=action_space_type,
                dataset=dataset,
                episode_length=episode_length,
                task_id=task_id,
            )
            envs_by_task[task_id] = env_cls(fns)
        out[task_name] = envs_by_task
    return out
