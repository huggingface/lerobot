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
from collections import defaultdict
from collections.abc import Callable, Sequence, Mapping
from functools import partial
from typing import Any
from gymnasium import spaces

import numpy as np

from lerobot.types import RobotObservation
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS, TARGET_TASKS, PRETRAINING_TASKS

OBS_STATE_DIM = 16
ACTION_DIM = 12
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

def convert_state(dict_state): 
    """
    Converts input state (dict) to format expected LeRobot (np.array)
    """
    dict_state = dict_state.copy()
    final_state = np.concat([
        dict_state["state.base_position"],
        dict_state["state.base_rotation"],
        dict_state["state.end_effector_position_relative"],
        dict_state["state.end_effector_rotation_relative"],
        dict_state["state.gripper_qpos"],
    ], axis=0)
    
    return final_state

def convert_action(action):
    """
    Converts input action (np.array) to format expected by gym env (dict)
    """
    action = action.copy()
    output_action = {
        "action.base_motion": action[0:4],
        "action.control_mode": action[4:5],
        "action.end_effector_position": action[5:8],
        "action.end_effector_rotation": action[8:11],
        "action.gripper_close": action[11:12],
    }
    return output_action

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


class RoboCasaEnv(RoboCasaGymEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str,
        camera_name: Sequence[str] = ["robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"],
        render_mode: str = "rgb_array",
        obs_type: str = "pixels_agent_pos",
        observation_width: int = 256,
        observation_height: int = 256,
        split: str | None = None, # {None, "all", "pretrain", "target"}
        **kwargs
    ):
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.kwargs = kwargs
        self.split = split
        self.task = task

        meta_info = {**ATOMIC_TASK_DATASETS, **COMPOSITE_TASK_DATASETS}
        try:
            self._max_episode_steps = meta_info[task]['horizon']
        except KeyError:
            raise ValueError(f"Unknown task '{task}'. Valid tasks are: {list(meta_info.keys())}")
        
        super().__init__(
            task,
            camera_names=camera_name,
            camera_widths=observation_width,
            camera_heights=observation_height,
            enable_render=(render_mode is not None),
            split=split,
            **kwargs
        )
    
    def _create_obs_and_action_space(self):
        images = {}
        for cam in self.camera_names:
            images[cam] = spaces.Box(
                low=0,
                high=255,
                shape=(self.camera_heights, self.camera_widths, 3),
                dtype=np.uint8,
            )
        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in RoboCasaEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "agent_pos": spaces.Box(
                        low=-1000,
                        high=1000,
                        shape=(OBS_STATE_DIM,),
                        dtype=np.float64,
                    ),
                }
            )
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")

        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32
        )

    def reset(
        self,
        seed: int | None = None,
        **kwargs,
    ) -> tuple[RobotObservation, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (Optional[int]): Random seed for environment initialization.

        Returns:
            observation (RobotObservation): The initial formatted observation.
            info (Dict[str, Any]): Additional info about the reset state.
        """
        self.unwrapped.sim._render_context_offscreen.gl_ctx.free()
        observation, info = super().reset(seed, **kwargs)
        new_obs = self._format_raw_obs(observation)
        return new_obs, info 
    
    def _format_raw_obs(self, raw_obs:dict):
        new_obs = {}
        if self.obs_type == "pixels_agent_pos":
            new_obs["agent_pos"] = convert_state(raw_obs)
        new_obs["pixels"] = {}
        for k, v in raw_obs.items():
            if "video." in k:
                new_obs["pixels"][k.replace("video.", "")] = v
        return new_obs

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        """
        Perform one environment step.

        Args:
            action (np.ndarray): The action to execute, must be 1-D with shape (action_dim,).

        Returns:
            observation (RobotObservation): The formatted observation after the step.
            reward (float): The scalar reward for this step.
            terminated (bool): Whether the episode terminated successfully.
            truncated (bool): Whether the episode was truncated due to a time limit.
            info (Dict[str, Any]): Additional environment info.
        """
        self.unwrapped.sim._render_context_offscreen.gl_ctx.make_current()
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        action_dict = convert_action(action)
        observation, reward, done, truncated, info = super().step(action_dict)
        new_obs = self._format_raw_obs(observation)

        # Determine whether the task was successful
        is_success = bool(info.get("success", 0))
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "done": done,
                "is_success": is_success,
            }
        )
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "done": bool(done),
                "is_success": bool(is_success),
            }
            self.reset()

        return new_obs, reward, terminated, truncated, info


def _make_env_fns(
    *,
    task_name: str,
    n_envs: int,
    camera_names: list[str],
    gym_kwargs: Mapping[str, Any]
) -> list[Callable[[], RoboCasaEnv]]:
    """Build n_envs factory callables for an environment."""

    def _make_env(episode_index: int, **kwargs) -> RoboCasaEnv:
        # Ensure each environment gets a different seed if not explicitly provided
        seed = kwargs.pop("seed", episode_index)
        
        return RoboCasaEnv(
            task=task_name,
            camera_name=camera_names,
            seed=seed,
            **kwargs,
        )

    fns: list[Callable[[], RoboCasaEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------


def create_robocasa_envs(
    task_name: str,
    n_envs: int,
    gym_kwargs: dict[str, Any],
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized RoboCasa environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Args:
        task_name:  Name of the task (e.g. "CloseFridge"), 
                    or list of tasks separated by commas (e.g. "CloseFridge,PrepareCoffee"), 
                    or benchmark name (i.e.: atomic_seen, composite_seen, composite_unseen, pretrain50, pretrain100, pretrain200, pretrain300).
        n_envs: Number of environments to create per task.
        gym_kwargs: Additional arguments to pass to RoboCasaEnv.
        env_cls: Callable that wraps a list of environment factory callables (for vectorization)
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    parsed_camera_names = _parse_camera_names(gym_kwargs.pop("camera_name"))

    combined_tasks = {**TARGET_TASKS, **PRETRAINING_TASKS}

    if task_name in combined_tasks:
        print(f"Creating RoboCasa {task_name} benchmark")
        task_names = combined_tasks[task_name]
        gym_kwargs["split"] = "target" if task_name in TARGET_TASKS else "pretrain"
    else:
        task_names = [t.strip() for t in task_name.split(",")]

    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for task in task_names:
        print(f"Building vec env | task = {task} | n_envs (per task) = {n_envs}")

        fns = _make_env_fns(
            task_name=task,
            n_envs=n_envs,
            camera_names=parsed_camera_names,
            gym_kwargs=gym_kwargs
        )

        out[task][0] = env_cls(fns)

    # return plain dicts for predictability
    return {suite: dict(task_map) for suite, task_map in out.items()}