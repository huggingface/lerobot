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

from lerobot.types import RobotObservation

from .utils import _LazyAsyncVectorEnv

# Dimensions for the flat action/state vectors used by the LeRobot wrapper.
# These correspond to the PandaOmron robot in RoboCasa365.
OBS_STATE_DIM = 16  # base_pos(3) + base_quat(4) + ee_pos_rel(3) + ee_quat_rel(4) + gripper_qpos(2)
ACTION_DIM = 12  # base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# Default cameras for the PandaOmron robot.
DEFAULT_CAMERAS = [
    "robot0_agentview_left",
    "robot0_eye_in_hand",
    "robot0_agentview_right",
]

# Map raw RoboCasa camera names to LeRobot convention names.
DEFAULT_CAMERA_NAME_MAPPING = {
    "robot0_agentview_left": "image",
    "robot0_eye_in_hand": "image2",
    "robot0_agentview_right": "image3",
}


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def convert_state(raw_obs: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate RoboCasa robot state dict into a flat (16,) vector.

    Layout: base_pos(3) + base_quat(4) + ee_pos_rel(3) + ee_quat_rel(4) + gripper_qpos(2)
    """
    return np.concatenate(
        [
            raw_obs["robot0_base_pos"],  # (3,)
            raw_obs["robot0_base_quat"],  # (4,)
            raw_obs["robot0_base_to_eef_pos"],  # (3,)
            raw_obs["robot0_base_to_eef_quat"],  # (4,)
            raw_obs["robot0_gripper_qpos"],  # (2,)
        ],
        axis=-1,
    ).astype(np.float32)


def convert_action(flat_action: np.ndarray) -> dict[str, Any]:
    """Split a flat (12,) action vector into a RoboCasa action dict.

    Layout: base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1)
    """
    return {
        "action.base_motion": flat_action[0:4],
        "action.control_mode": flat_action[4:5],
        "action.end_effector_position": flat_action[5:8],
        "action.end_effector_rotation": flat_action[8:11],
        "action.gripper_close": flat_action[11:12],
    }


class RoboCasaEnv(gym.Env):
    """LeRobot gym.Env wrapper for RoboCasa365 kitchen environments.

    Wraps the RoboCasaGymEnv from the robocasa package and converts its
    dict-based observations and actions into flat arrays compatible with
    the LeRobot evaluation pipeline.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str,
        camera_name: str | Sequence[str] = ",".join(DEFAULT_CAMERAS),
        camera_name_mapping: dict[str, str] | None = None,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        split: str | None = None,
        episode_length: int | None = None,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.split = split

        self.camera_name = _parse_camera_names(camera_name)
        if camera_name_mapping is None:
            camera_name_mapping = dict(DEFAULT_CAMERA_NAME_MAPPING)
        self.camera_name_mapping = camera_name_mapping

        self._max_episode_steps = episode_length if episode_length is not None else 1000

        # Deferred — created on first reset() inside the worker subprocess
        # to avoid inheriting stale GPU/EGL contexts across fork().
        self._env = None
        self.task_description = ""

        # Build observation space
        images = {}
        for cam in self.camera_name:
            mapped = self.camera_name_mapping.get(cam, cam)
            images[mapped] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": spaces.Dict(images)})
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "agent_pos": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(OBS_STATE_DIM,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type '{self.obs_type}'. Use 'pixels' or 'pixels_agent_pos'.")

        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    def _ensure_env(self) -> None:
        """Create the underlying RoboCasaGymEnv on first use.

        Called inside the worker subprocess after fork(), so each worker gets
        its own clean rendering context rather than inheriting a stale one from
        the parent process (which causes crashes with AsyncVectorEnv).
        """
        if self._env is not None:
            return
        from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

        # RoboCasaGymEnv has a broken default split="test" (invalid for create_env
        # which only accepts None/"all"/"pretrain"/"target"). Always pass a valid
        # value so we don't hit that default.
        kwargs: dict[str, Any] = {
            "env_name": self.task,
            "camera_widths": self.observation_width,
            "camera_heights": self.observation_height,
            "split": self.split if self.split is not None else "all",
        }

        self._env = RoboCasaGymEnv(**kwargs)

        # Extract task description from environment metadata
        assert self._env is not None
        ep_meta = self._env.env.get_ep_meta()
        self.task_description = ep_meta.get("lang", self.task)

    def _format_raw_obs(self, raw_obs: dict) -> RobotObservation:
        """Convert RoboCasaGymEnv observation dict to LeRobot format."""
        # Extract camera images (RoboCasaGymEnv provides "video.<cam>" keys)
        images = {}
        for cam in self.camera_name:
            video_key = f"video.{cam}"
            if video_key in raw_obs:
                mapped = self.camera_name_mapping.get(cam, cam)
                images[mapped] = raw_obs[video_key]

        if self.obs_type == "pixels":
            return {"pixels": images}

        # Extract state from raw_obs (state.* keys from PandaOmronKeyConverter)
        agent_pos = np.concatenate(
            [
                raw_obs.get("state.base_position", np.zeros(3)),
                raw_obs.get("state.base_rotation", np.zeros(4)),
                raw_obs.get("state.end_effector_position_relative", np.zeros(3)),
                raw_obs.get("state.end_effector_rotation_relative", np.zeros(4)),
                raw_obs.get("state.gripper_qpos", np.zeros(2)),
            ],
            axis=-1,
        ).astype(np.float32)

        return {
            "pixels": images,
            "agent_pos": agent_pos,
        }

    def render(self) -> np.ndarray:
        self._ensure_env()
        assert self._env is not None
        return self._env.render()

    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        assert self._env is not None
        super().reset(seed=seed)
        raw_obs, info = self._env.reset(seed=seed)

        # Update task description on each reset (may change per episode)
        ep_meta = self._env.env.get_ep_meta()
        self.task_description = ep_meta.get("lang", self.task)

        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        assert self._env is not None
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )

        # Convert flat action to RoboCasa dict format
        action_dict = convert_action(action)
        raw_obs, reward, done, truncated, info = self._env.step(action_dict)

        is_success = bool(info.get("success", False))
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "done": done,
                "is_success": is_success,
            }
        )

        observation = self._format_raw_obs(raw_obs)
        if terminated:
            self.reset()

        return observation, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()


# ---- Main API ----------------------------------------------------------------


def _make_env_fns(
    *,
    task: str,
    n_envs: int,
    camera_names: list[str],
    camera_name_mapping: dict[str, str] | None,
    obs_type: str,
    render_mode: str,
    observation_width: int,
    observation_height: int,
    split: str | None,
    episode_length: int | None,
) -> list[Callable[[], RoboCasaEnv]]:
    """Build n_envs factory callables for a single task."""

    def _make_env(**kwargs) -> RoboCasaEnv:
        return RoboCasaEnv(
            task=task,
            camera_name=camera_names,
            camera_name_mapping=camera_name_mapping,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            split=split,
            episode_length=episode_length,
            **kwargs,
        )

    return [partial(_make_env) for _ in range(n_envs)]


def create_robocasa_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = ",".join(DEFAULT_CAMERAS),
    camera_name_mapping: dict[str, str] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    episode_length: int | None = None,
) -> dict[str, dict[int, Any]]:
    """Create vectorized RoboCasa365 environments with a consistent return shape.

    Returns:
        dict[task_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (parallel environments).
        - `task` can be a single task or a comma-separated list of tasks.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    obs_type = gym_kwargs.pop("obs_type", "pixels_agent_pos")
    render_mode = gym_kwargs.pop("render_mode", "rgb_array")
    observation_width = gym_kwargs.pop("observation_width", 256)
    observation_height = gym_kwargs.pop("observation_height", 256)
    split = gym_kwargs.pop("split", None)

    camera_names = _parse_camera_names(camera_name)
    task_names = [t.strip() for t in str(task).split(",") if t.strip()]
    if not task_names:
        raise ValueError("`task` must contain at least one RoboCasa task name.")

    print(f"Creating RoboCasa envs | tasks={task_names} | n_envs(per task)={n_envs}")

    is_async = env_cls is gym.vector.AsyncVectorEnv

    cached_obs_space: spaces.Space | None = None
    cached_act_space: spaces.Space | None = None
    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for _tid, task_name in enumerate(task_names):
        fns = _make_env_fns(
            task=task_name,
            n_envs=n_envs,
            camera_names=camera_names,
            camera_name_mapping=camera_name_mapping,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            split=split,
            episode_length=episode_length,
        )

        if is_async:
            lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space)
            if cached_obs_space is None:
                cached_obs_space = lazy.observation_space
                cached_act_space = lazy.action_space
            out[task_name][0] = lazy
        else:
            out[task_name][0] = env_cls(fns)
        print(f"Built vec env | task={task_name} | n_envs={n_envs}")

    return {name: dict(task_map) for name, task_map in out.items()}
