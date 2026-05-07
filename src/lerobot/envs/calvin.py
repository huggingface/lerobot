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
"""CALVIN env wrapper (gymnasium API) for closed-loop eval of pi05/pi05_d.

Builds on top of `calvin_env.envs.play_table_env.PlayTableSimEnv` (PyBullet) and
the official task oracle (`calvin_env/conf/tasks/new_playtable_tasks.yaml`).

Differences vs the official `evaluate_policy.py`:
  - Single task per env (vs sequence of 5). Sequence eval can be built on top
    by chaining N CalvinEnv resets; we keep the leaf simple.
  - Always headless (use_egl=true).
  - Tactile camera disabled by default (no tacto submodule installed; we don't
    use tactile in pi05_d either).
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf


# === Canonical CALVIN tasks ===================================================
# 37 tasks from calvin_env/conf/tasks/new_playtable_tasks.yaml.
# Keep this list in sync with that YAML; tested via _CALVIN_TASKS_PARITY below.
CALVIN_TASKS: tuple[str, ...] = (
    # rotation (6)
    "rotate_red_block_right", "rotate_red_block_left",
    "rotate_blue_block_right", "rotate_blue_block_left",
    "rotate_pink_block_right", "rotate_pink_block_left",
    # pushing (6)
    "push_red_block_right", "push_red_block_left",
    "push_blue_block_right", "push_blue_block_left",
    "push_pink_block_right", "push_pink_block_left",
    # open/close (4)
    "move_slider_left", "move_slider_right",
    "open_drawer", "close_drawer",
    # lifting (9)
    "lift_red_block_table", "lift_red_block_slider", "lift_red_block_drawer",
    "lift_blue_block_table", "lift_blue_block_slider", "lift_blue_block_drawer",
    "lift_pink_block_table", "lift_pink_block_slider", "lift_pink_block_drawer",
    # placing (2)
    "place_in_slider", "place_in_drawer",
    # stacking (2)
    "stack_block", "unstack_block",
    # lights (4)
    "turn_on_lightbulb", "turn_off_lightbulb",
    "turn_on_led", "turn_off_led",
    # pushing into drawer (1)
    "push_into_drawer",
)
assert len(CALVIN_TASKS) == 34, f"Expected 34 unique task names, got {len(CALVIN_TASKS)}"

ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DEFAULT_EPISODE_LENGTH = 360  # CALVIN paper uses 360 max steps per subtask


def _calvin_env_module_path() -> Path:
    """Return Path to calvin_env repo root (parent of `calvin_env` package)."""
    import calvin_env
    return Path(calvin_env.__file__).resolve().parents[1]


def _load_task_oracle():
    """Instantiate the official `Tasks` oracle from new_playtable_tasks.yaml."""
    yaml_path = _calvin_env_module_path() / "conf" / "tasks" / "new_playtable_tasks.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Task oracle YAML not found at {yaml_path}. Is calvin_env installed correctly?"
        )
    cfg = OmegaConf.load(yaml_path)
    return hydra.utils.instantiate(cfg)


class CalvinEnv(gym.Env):
    """Gymnasium wrapper around CALVIN's PlayTableSimEnv for single-task eval.

    Observation format mirrors LiberoEnv for consistency:
      obs_type='pixels':           {"pixels": {"image": rgb_static, ["image2": rgb_gripper]}}
      obs_type='pixels_agent_pos': above + {"robot_state": {"eef": ..., "gripper": ..., "joints": ...}}

    Depth is exposed under `obs["depth"]` when `include_depth=True`, separate
    from `pixels` so processors that don't know about depth ignore it cleanly.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str,
        dataset_path: str | Path,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        observation_height: int = 200,
        observation_width: int = 200,
        gripper_observation_height: int = 84,
        gripper_observation_width: int = 84,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        include_depth: bool = True,
        include_gripper_camera: bool = True,
        control_mode: str = "relative",
        seed: int = 0,
    ):
        super().__init__()
        if task not in CALVIN_TASKS:
            raise ValueError(
                f"Unknown CALVIN task '{task}'. Available ({len(CALVIN_TASKS)}): "
                f"{', '.join(CALVIN_TASKS)}"
            )
        if obs_type not in ("pixels", "pixels_agent_pos"):
            raise ValueError(f"obs_type must be 'pixels' or 'pixels_agent_pos', got '{obs_type}'")
        if control_mode not in ("relative", "absolute"):
            raise ValueError(f"control_mode must be 'relative' or 'absolute', got '{control_mode}'")

        self.task = task
        self.dataset_path = Path(dataset_path)
        if not (self.dataset_path / ".hydra" / "merged_config.yaml").exists():
            raise FileNotFoundError(
                f"{self.dataset_path}/.hydra/merged_config.yaml missing — "
                f"calvin_env.get_env() needs the dataset's hydra config to build the scene."
            )
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.gripper_observation_height = gripper_observation_height
        self.gripper_observation_width = gripper_observation_width
        self.include_depth = include_depth
        self.include_gripper_camera = include_gripper_camera
        self.control_mode = control_mode
        self.episode_length = episode_length
        self._step_count = 0
        self._start_info: dict[str, Any] | None = None

        # Lazily import to avoid hard dep when wrapper file is just imported.
        from calvin_env.envs.play_table_env import get_env as _calvin_get_env

        # Build obs filter: tells get_env() to drop the tactile camera (which
        # would require the tacto submodule we don't install) and any cameras
        # we don't want.
        rgb_keys = ["rgb_static"]
        depth_keys: list[str] = ["depth_static"] if include_depth else []
        if include_gripper_camera:
            rgb_keys.append("rgb_gripper")
            if include_depth:
                depth_keys.append("depth_gripper")
        self._raw_obs_space = {"rgb_obs": rgb_keys, "depth_obs": depth_keys}

        self._env = _calvin_get_env(
            str(self.dataset_path),
            obs_space=self._raw_obs_space,
            show_gui=False,
        )
        self._env.seed(seed)

        # Task oracle: shared instance is fine for single-process use.
        self._task_oracle = _load_task_oracle()

        # Spaces
        self.observation_space = self._build_observation_space()
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    # --- helpers ---------------------------------------------------------------

    def _build_observation_space(self) -> spaces.Dict:
        images = {
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            ),
        }
        if self.include_gripper_camera:
            images["image2"] = spaces.Box(
                low=0, high=255,
                shape=(self.gripper_observation_height, self.gripper_observation_width, 3),
                dtype=np.uint8,
            )

        space_dict: dict[str, Any] = {"pixels": spaces.Dict(images)}

        if self.include_depth:
            depth_dict = {
                "depth_static": spaces.Box(
                    low=0.0, high=np.inf,
                    shape=(self.observation_height, self.observation_width),
                    dtype=np.float32,
                ),
            }
            if self.include_gripper_camera:
                depth_dict["depth_gripper"] = spaces.Box(
                    low=0.0, high=np.inf,
                    shape=(self.gripper_observation_height, self.gripper_observation_width),
                    dtype=np.float32,
                )
            space_dict["depth"] = spaces.Dict(depth_dict)

        if self.obs_type == "pixels_agent_pos":
            # robot_obs layout (15-d): [tcp_pos(3), tcp_orn_euler(3), gripper_width(1),
            #                          arm_joints(7), gripper_action(1)]
            space_dict["robot_state"] = spaces.Dict({
                "eef": spaces.Dict({
                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                    "orn_euler": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                }),
                "gripper": spaces.Dict({
                    "width": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64),
                    "action": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
                }),
                "joints": spaces.Dict({
                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                }),
            })

        return spaces.Dict(space_dict)

    def _format_obs(self, raw_obs: dict) -> dict:
        rgb = raw_obs["rgb_obs"]
        out: dict[str, Any] = {
            "pixels": {"image": rgb["rgb_static"]},
        }
        if self.include_gripper_camera and "rgb_gripper" in rgb:
            out["pixels"]["image2"] = rgb["rgb_gripper"]

        if self.include_depth:
            depth = raw_obs["depth_obs"]
            out["depth"] = {"depth_static": depth["depth_static"].astype(np.float32)}
            if self.include_gripper_camera and "depth_gripper" in depth:
                out["depth"]["depth_gripper"] = depth["depth_gripper"].astype(np.float32)

        if self.obs_type == "pixels_agent_pos":
            robot_obs = raw_obs["robot_obs"]
            out["robot_state"] = {
                "eef": {
                    "pos": np.asarray(robot_obs[0:3], dtype=np.float64),
                    "orn_euler": np.asarray(robot_obs[3:6], dtype=np.float64),
                },
                "gripper": {
                    "width": np.asarray(robot_obs[6:7], dtype=np.float64),
                    "action": np.asarray(robot_obs[14:15], dtype=np.float64),
                },
                "joints": {
                    "pos": np.asarray(robot_obs[7:14], dtype=np.float64),
                },
            }
        return out

    # --- gymnasium API ---------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None,
              robot_obs: np.ndarray | None = None, scene_obs: np.ndarray | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.seed(seed)
        raw_obs = self._env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self._start_info = self._env.get_info()
        self._step_count = 0
        observation = self._format_obs(raw_obs)
        info = {
            "is_success": False,
            "task": self.task,
            "step_count": 0,
        }
        return observation, info

    def step(self, action: np.ndarray):
        if action.ndim != 1 or action.shape[0] != ACTION_DIM:
            raise ValueError(
                f"Expected action shape ({ACTION_DIM},), got {action.shape}"
            )
        if self._start_info is None:
            raise RuntimeError("step() called before reset()")
        # NB: PlayTableSimEnv.step always returns done=False, reward=0.0 — eval
        # signal comes from the task oracle, not the env.
        raw_obs, _, _, _ = self._env.step(action)
        self._step_count += 1
        current_info = self._env.get_info()

        completed = self._task_oracle.get_task_info_for_set(
            self._start_info, current_info, {self.task}
        )
        is_success = self.task in completed
        terminated = bool(is_success)
        truncated = self._step_count >= self.episode_length

        observation = self._format_obs(raw_obs)
        info = {
            "is_success": is_success,
            "task": self.task,
            "step_count": self._step_count,
            "completed_tasks": list(completed),
        }
        if terminated or truncated:
            info["final_info"] = {
                "task": self.task,
                "is_success": is_success,
                "step_count": self._step_count,
            }
        reward = 1.0 if is_success else 0.0
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            raise NotImplementedError(f"render_mode='{self.render_mode}' not supported")
        rgb_obs, _ = self._env.get_camera_obs()
        return rgb_obs["rgb_static"]

    def close(self):
        self._env.close()


# === Vectorized factory (LiberoEnv pattern) ===================================


def _make_env_fns(
    *,
    task: str,
    dataset_path: str | Path,
    n_envs: int,
    gym_kwargs: dict[str, Any],
) -> list[Callable[[], CalvinEnv]]:
    """Build n_envs factory callables for a single CALVIN task."""
    def _make_env(episode_index: int, **kwargs) -> CalvinEnv:
        local_kwargs = dict(kwargs)
        # Use episode_index as seed shift so each parallel env has a distinct
        # initial scene state (CALVIN has no init_states bank like LIBERO).
        seed = local_kwargs.pop("seed", 0) + episode_index
        return CalvinEnv(task=task, dataset_path=dataset_path, seed=seed, **local_kwargs)

    fns: list[Callable[[], CalvinEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


def create_calvin_envs(
    task: str | Sequence[str],
    dataset_path: str | Path,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, Any]:
    """Build vectorized CALVIN envs, one vec-env per task.

    Args:
        task: single task name or comma-separated string ("open_drawer,close_drawer")
            or sequence. Use CalvinEnv.CALVIN_TASKS for the canonical list.
        dataset_path: path containing `.hydra/merged_config.yaml` (e.g.
            ~/Prometheus/calvin/dataset/task_ABC_D/validation).
        n_envs: number of parallel rollouts per task.
        gym_kwargs: forwarded to CalvinEnv (obs_type, episode_length, etc.).
        env_cls: gym vec-env class wrapping a list of factory callables
            (e.g. gym.vector.SyncVectorEnv).

    Returns:
        dict[task_name] -> vec_env
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of env factories.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})

    if isinstance(task, str):
        tasks = [t.strip() for t in task.split(",") if t.strip()]
    else:
        tasks = [str(t).strip() for t in task if str(t).strip()]
    if not tasks:
        raise ValueError("`task` resolved to empty list.")
    unknown = [t for t in tasks if t not in CALVIN_TASKS]
    if unknown:
        raise ValueError(f"Unknown CALVIN task(s): {unknown}")

    print(f"Creating CALVIN envs | tasks={tasks} | n_envs(per task)={n_envs}")
    out: dict[str, Any] = {}
    for t in tasks:
        fns = _make_env_fns(
            task=t, dataset_path=dataset_path, n_envs=n_envs, gym_kwargs=gym_kwargs
        )
        out[t] = env_cls(fns)
        print(f"Built vec env | task={t} | n_envs={n_envs}")
    return out
