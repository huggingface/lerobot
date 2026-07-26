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

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.types import RobotObservation

from .utils import _LazyAsyncVectorEnv, parse_camera_names

logger = logging.getLogger(__name__)

# Dimensions for the flat action/state vectors used by the LeRobot wrapper.
# These correspond to the PandaOmron robot in RoboCasa365.
OBS_STATE_DIM = 16  # base_pos(3) + base_quat(4) + ee_pos_rel(3) + ee_quat_rel(4) + gripper_qpos(2)
ACTION_DIM = 12  # base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# Default PandaOmron cameras. We surface these raw names directly as
# `observation.images.<name>` so the LeRobot dataset/policy keys match
# RoboCasa's native convention (no implicit renaming).
DEFAULT_CAMERAS = [
    "robot0_agentview_left",
    "robot0_eye_in_hand",
    "robot0_agentview_right",
]

# Object-mesh registries to sample from. RoboCasa's upstream default is
# ("objaverse", "lightwheel"), but the objaverse pack is huge (~30GB) and
# most users — including our CI image — only download the lightwheel pack
# (`--type objs_lw` in `download_kitchen_assets`). When a sampled object
# category has zero candidates in every registry, robocasa crashes with
# `ValueError: Probabilities contain NaN` (0/0 divide in the probability
# normalization). Restricting to registries that are actually on disk
# avoids the NaN and matches what the asset download provides.
DEFAULT_OBJ_REGISTRIES: tuple[str, ...] = ("lightwheel",)

# Task-group shortcuts accepted as `--env.task`. When the user passes one of
# these names, we expand it to the upstream RoboCasa task list and auto-set
# the dataset split. Individual task names (optionally comma-separated) still
# take precedence; this only triggers on an exact group-name match.
_TASK_GROUP_SPLITS = {
    "atomic_seen": "target",
    "composite_seen": "target",
    "composite_unseen": "target",
    "pretrain50": "pretrain",
    "pretrain100": "pretrain",
    "pretrain200": "pretrain",
    "pretrain300": "pretrain",
}


def _resolve_tasks(task: str) -> tuple[list[str], str | None]:
    """Resolve a `--env.task` value to (task_names, split_override).

    If `task` is a known task-group name (e.g. `atomic_seen`, `pretrain100`),
    expand it via `robocasa.utils.dataset_registry.{TARGET,PRETRAINING}_TASKS`
    and return the matching split. Otherwise treat `task` as a single task or
    comma-separated list and leave the split untouched (None).
    """
    key = task.strip()
    if key in _TASK_GROUP_SPLITS:
        from robocasa.utils.dataset_registry import PRETRAINING_TASKS, TARGET_TASKS

        combined = {**TARGET_TASKS, **PRETRAINING_TASKS}
        if key not in combined:
            raise ValueError(
                f"Task group '{key}' is not available in this version of robocasa. "
                f"Known groups: {sorted(combined.keys())}."
            )
        return list(combined[key]), _TASK_GROUP_SPLITS[key]

    names = [t.strip() for t in task.split(",") if t.strip()]
    if not names:
        raise ValueError("`task` must contain at least one RoboCasa task name.")
    return names, None


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

    Wraps RoboCasaGymEnv from the robocasa package and converts its
    dict-based observations and actions into the flat arrays LeRobot expects.
    Raw RoboCasa camera names are preserved verbatim under `pixels/<cam>`.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str,
        camera_name: str | Sequence[str] = ",".join(DEFAULT_CAMERAS),
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 512,
        visualization_height: int = 512,
        split: str | None = None,
        episode_length: int | None = None,
        obj_registries: Sequence[str] = DEFAULT_OBJ_REGISTRIES,
        episode_index: int = 0,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.split = split
        self.obj_registries = tuple(obj_registries)
        # Per-worker index (0..n_envs-1) used to spread the user-provided
        # seed across factories so each sub-env explores a distinct layout
        # even when the same seed is passed to `reset()`.
        self.episode_index = int(episode_index)

        self.camera_name = parse_camera_names(camera_name)

        self._max_episode_steps = episode_length if episode_length is not None else 1000

        # Deferred — created on first reset() inside the worker subprocess
        # to avoid inheriting stale GPU/EGL contexts across fork().
        self._env: Any = None
        self.task_description = ""

        images = {
            cam: spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
            for cam in self.camera_name
        }

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

        # RoboCasaGymEnv defaults split="test", which create_env rejects
        # (only None/"all"/"pretrain"/"target" are valid). Always pass a
        # valid value so we don't hit that default. Extra kwargs are
        # forwarded to the underlying kitchen env via create_env/robosuite.make.
        self._env = RoboCasaGymEnv(
            env_name=self.task,
            camera_widths=self.observation_width,
            camera_heights=self.observation_height,
            split=self.split if self.split is not None else "all",
            obj_registries=self.obj_registries,
        )

        ep_meta = self._env.env.get_ep_meta()
        self.task_description = ep_meta.get("lang", self.task)

    def _format_raw_obs(self, raw_obs: dict) -> RobotObservation:
        """Convert RoboCasaGymEnv observation dict to LeRobot format."""
        # RoboCasaGymEnv emits camera frames under "video.<cam>".
        images = {cam: raw_obs[f"video.{cam}"] for cam in self.camera_name if f"video.{cam}" in raw_obs}

        if self.obs_type == "pixels":
            return {"pixels": images}

        # `state.*` keys come from PandaOmronKeyConverter inside the wrapper.
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

        return {"pixels": images, "agent_pos": agent_pos}

    def render(self) -> np.ndarray:
        self._ensure_env()
        assert self._env is not None
        return self._env.render()

    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        assert self._env is not None
        super().reset(seed=seed)
        # Spread the seed across workers so n_envs factories don't all
        # roll the same scene. With an explicit user seed we shift it by
        # episode_index; with no seed we fall back to episode_index so
        # each worker is still distinct rather than inheriting the same
        # global RNG state.
        worker_seed = seed + self.episode_index if seed is not None else self.episode_index
        raw_obs, info = self._env.reset(seed=worker_seed)

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

        action_dict = convert_action(action)
        raw_obs, reward, done, truncated, info = self._env.step(action_dict)

        is_success = bool(info.get("success", False))
        terminated = done or is_success
        info.update({"task": self.task, "done": done, "is_success": is_success})

        observation = self._format_raw_obs(raw_obs)
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "done": bool(done),
                "is_success": bool(is_success),
            }
            self.reset()

        return observation, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()


def _make_env_fns(
    *,
    task: str,
    n_envs: int,
    camera_names: list[str],
    obs_type: str,
    render_mode: str,
    observation_width: int,
    observation_height: int,
    visualization_width: int,
    visualization_height: int,
    split: str | None,
    episode_length: int | None,
    obj_registries: Sequence[str],
) -> list[Callable[[], RoboCasaEnv]]:
    """Build n_envs factory callables for a single task.

    Each factory carries a distinct ``episode_index`` (``0..n_envs-1``) so
    ``RoboCasaEnv.reset()`` can derive a per-worker seed series from the
    user-provided seed.
    """

    def _make_env(episode_index: int) -> RoboCasaEnv:
        return RoboCasaEnv(
            task=task,
            camera_name=camera_names,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
            split=split,
            episode_length=episode_length,
            obj_registries=obj_registries,
            episode_index=episode_index,
        )

    return [partial(_make_env, i) for i in range(n_envs)]


def create_robocasa_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = ",".join(DEFAULT_CAMERAS),
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    episode_length: int | None = None,
    obj_registries: Sequence[str] = DEFAULT_OBJ_REGISTRIES,
) -> dict[str, dict[int, Any]]:
    """Create vectorized RoboCasa365 environments with a consistent return shape.

    Returns:
        dict[task_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)

    `task` can be:
      - a single task name (e.g. `CloseFridge`)
      - a comma-separated list of task names (e.g. `CloseFridge,PickPlaceCoffee`)
      - a benchmark-group shortcut (`atomic_seen`, `composite_seen`,
        `composite_unseen`, `pretrain50`, `pretrain100`, `pretrain200`,
        `pretrain300`), which auto-expands to the upstream task list and
        auto-sets the dataset `split` ("target" or "pretrain").
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
    visualization_width = gym_kwargs.pop("visualization_width", 512)
    visualization_height = gym_kwargs.pop("visualization_height", 512)
    split = gym_kwargs.pop("split", None)

    camera_names = parse_camera_names(camera_name)
    task_names, group_split = _resolve_tasks(str(task))
    if group_split is not None and split is None:
        split = group_split

    logger.info(
        "Creating RoboCasa envs | tasks=%s | split=%s | n_envs(per task)=%d",
        task_names,
        split,
        n_envs,
    )

    is_async = env_cls is gym.vector.AsyncVectorEnv

    cached_obs_space: spaces.Space | None = None
    cached_act_space: spaces.Space | None = None
    cached_metadata: dict[str, Any] | None = None
    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for task_name in task_names:
        fns = _make_env_fns(
            task=task_name,
            n_envs=n_envs,
            camera_names=camera_names,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
            split=split,
            episode_length=episode_length,
            obj_registries=obj_registries,
        )

        if is_async:
            lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
            if cached_obs_space is None:
                cached_obs_space = lazy.observation_space
                cached_act_space = lazy.action_space
                cached_metadata = lazy.metadata
            out[task_name][0] = lazy
        else:
            out[task_name][0] = env_cls(fns)
        logger.info("Built vec env | task=%s | n_envs=%d", task_name, n_envs)

    return {name: dict(task_map) for name, task_map in out.items()}
