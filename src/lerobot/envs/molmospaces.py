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
import os
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .utils import _LazyAsyncVectorEnv, parse_camera_names

ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DEFAULT_EPISODE_LENGTH = 500
DEFAULT_CAMERA_H = 256
DEFAULT_CAMERA_W = 256

logger = logging.getLogger(__name__)

MOLMO_SPACES_TASKS = (
    "pick",
    "pick_place",
    "open",
    "close",
    "push",
    "pour",
    "rearrange",
    "navigation",
)

MOLMO_SPACES_BENCHMARKS = (
    "molmospaces_bench_v1",
    "molmospaces_bench_v2",
)


class MolmoSpacesEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str = "pick",
        camera_name: str = "front",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        max_episode_steps: int = 500,
        benchmark_name: str = "molmospaces_bench_v1",
        episode_index: int = 0,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        parsed = parse_camera_names(camera_name)
        self.camera_name = parsed[0] if parsed else "front"
        self.max_episode_steps = max_episode_steps
        self.benchmark_name = benchmark_name
        self.episode_index = episode_index
        self._env = None
        self._sim = None
        self._episode_step = 0

        self._setup_spaces()

    def _init_env(self) -> None:
        if self._env is not None:
            return

        try:
            from molmo_spaces.envs.episode import load_benchmark
            from molmo_spaces.simulators.mujoco import MuJoCoSimulator
        except ImportError as e:
            raise ImportError(
                'MolmoSpaces is not installed. Please install it with:\npip install "lerobot[molmospaces]"'
            ) from e

        benchmark_path = os.environ.get("MLSPACES_ASSETS_DIR", ".")
        benchmark_path = os.path.join(benchmark_path, "bench", f"{self.benchmark_name}.json")

        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(
                f"Benchmark file not found: {benchmark_path}\n"
                "Please set MLSPACES_ASSETS_DIR or download assets with:\n"
                "python -m molmo_spaces.molmo_spaces_constants"
            )

        self._benchmark = load_benchmark(benchmark_path)

        task_configs = [tc for tc in self._benchmark.task_configs if tc.task_name == self.task]
        if not task_configs:
            available = [tc.task_name for tc in self._benchmark.task_configs]
            raise ValueError(f"Task '{self.task}' not found in benchmark. Available tasks: {available}")

        self._task_config = task_configs[0]

        self._sim = MuJoCoSimulator(
            scene_config=self._task_config.scene,
            robot_config=self._task_config.robot,
        )
        self._env = self._sim

    def _setup_spaces(self) -> None:
        if self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            self.camera_name: spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            )
                        }
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            self.camera_name: spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            )
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(14,),
                        dtype=np.float64,
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._episode_step = 0

        if self._sim is None:
            self._init_env()

        self._sim.reset(seed=seed)

        obs = self._get_obs()
        info = {"is_success": False, "task": self.task}
        return obs, info

    def step(self, action: np.ndarray):
        self._episode_step += 1

        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )

        if self._sim is None:
            self._init_env()

        raw_obs, reward, done, info = self._sim.step(action)
        info = dict(info)

        is_success = info.get("success", False)
        terminated = done or is_success
        truncated = self._episode_step >= self.max_episode_steps

        info["is_success"] = is_success
        info["task"] = self.task

        if terminated or truncated:
            final_obs = self._get_obs()
            obs, reset_info = self.reset()
            info["final_observation"] = final_obs
            info["reset_info"] = reset_info
        else:
            obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        if self._sim._scene is None:
            return self._get_dummy_obs()

        try:
            self._sim.render()
        except Exception as e:
            logger.warning(f"Render failed: {e}, returning dummy observation")
            return self._get_dummy_obs()

        height, width = self.observation_height, self.observation_width
        pixels = np.zeros((height, width, 3), dtype=np.uint8)

        try:
            if hasattr(self._sim, "_renderer") and self._sim._renderer is not None:
                img = self._sim._renderer.render()
                if img is not None:
                    if img.shape[0] != height or img.shape[1] != width:
                        import cv2

                        img = cv2.resize(img, (width, height))
                    pixels = img
        except Exception as e:
            logger.warning(f"Image capture failed: {e}, returning black frame")

        if self.obs_type == "pixels":
            return {"pixels": {self.camera_name: pixels}}

        data = self._sim._data
        qpos = data.qpos.copy()[:7] if data.qpos is not None and len(data.qpos) >= 7 else np.zeros(7)
        qvel = data.qvel.copy()[:7] if data.qvel is not None and len(data.qvel) >= 7 else np.zeros(7)

        end_effector_pos = np.zeros(3)
        end_effector_quat = np.zeros(4)
        gripper_qpos = np.zeros(1)

        try:
            if hasattr(self._sim, "_robot"):
                robot = self._sim._robot
                if hasattr(robot, "ee_pos"):
                    end_effector_pos = np.array(robot.ee_pos) if robot.ee_pos is not None else np.zeros(3)
                if hasattr(robot, "ee_quat"):
                    end_effector_quat = np.array(robot.ee_quat) if robot.ee_quat is not None else np.zeros(4)
                if hasattr(robot, "gripper_qpos"):
                    gripper_qpos = np.array([robot.gripper_qpos])
        except Exception:
            pass

        agent_pos = np.concatenate(
            [
                qpos,
                qvel,
                end_effector_pos,
                end_effector_quat[:3] if len(end_effector_quat) >= 3 else np.zeros(3),
                gripper_qpos,
            ]
        )

        if len(agent_pos) < 14:
            agent_pos = np.zeros(14)

        return {
            "pixels": {self.camera_name: pixels},
            "agent_pos": agent_pos.astype(np.float64),
        }

    def _get_dummy_obs(self) -> dict:
        pixels = np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)
        agent_pos = np.zeros(14, dtype=np.float64)

        if self.obs_type == "pixels":
            return {"pixels": {self.camera_name: pixels}}
        return {"pixels": {self.camera_name: pixels}, "agent_pos": agent_pos}

    def render(self):
        return self._get_obs()["pixels"]

    def close(self):
        if self._sim is not None:
            try:
                self._sim.close()
            except Exception:
                pass


def _make_env_fns(
    *,
    task: str,
    n_envs: int,
    camera_name: str,
    gym_kwargs: dict[str, Any],
    benchmark_name: str,
) -> list[Callable[[], MolmoSpacesEnv]]:
    """Build n_envs factory callables for a single task."""

    def _make_env(episode_index: int) -> MolmoSpacesEnv:
        return MolmoSpacesEnv(
            task=task,
            camera_name=camera_name,
            episode_index=episode_index,
            benchmark_name=benchmark_name,
            **gym_kwargs,
        )

    fns: list[Callable[[], MolmoSpacesEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index))
    return fns


def create_molmospaces_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str = "front",
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    benchmark_name: str = "molmospaces_bench_v1",
) -> dict[str, dict[int, Any]]:
    """Create vectorized MolmoSpaces environments.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})

    print(f"Creating MolmoSpaces envs | task={task} | n_envs={n_envs} | benchmark={benchmark_name}")

    is_async = env_cls is gym.vector.AsyncVectorEnv

    fns = _make_env_fns(
        task=task,
        n_envs=n_envs,
        camera_name=camera_name,
        gym_kwargs=gym_kwargs,
        benchmark_name=benchmark_name,
    )

    if is_async:
        vec_env = _LazyAsyncVectorEnv(fns)
    else:
        vec_env = env_cls(fns)

    return {benchmark_name: {0: vec_env}}
