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
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.task import Task
from rlbench.gym import RLBenchEnv as RLBenchGymEnv
from rlbench.tasks import FS10_V1, FS25_V1, FS50_V1, FS95_V1, MT15_V1, MT30_V1, MT55_V1, MT100_V1


class AbsoluteJointPositionActionMode(MoveArmThenGripper):
    """Absolute joint position action mode for arm and gripper.

    The arm action is first applied, followed by the gripper action.
    """

    def __init__(self):
        # Call super
        super().__init__(
            JointPosition(absolute_mode=True),  # Arm in absolute joint positions
            Discrete(),  # Gripper in discrete open/close (<0.5 → close, >=0.5 → open)
        )

    def action_bounds(self):
        """Returns the min and max of the action mode.

        Range is [- 2*pi, 2*pi] for each joint and [0.0, 1.0] for the gripper.
        """
        return np.array([-2 * np.pi] * 7 + [0.0]), np.array([2 * np.pi] * 7 + [1.0])


# ---- Load configuration data from the external JSON file ----
CONFIG_PATH = Path(__file__).parent / "rlbench_config.json"
try:
    with open(CONFIG_PATH) as f:
        data = json.load(f)
except FileNotFoundError as err:
    raise FileNotFoundError(
        "Could not find 'rlbench_config.json'. "
        "Please ensure the configuration file is in the same directory as the script."
    ) from err
except json.JSONDecodeError as err:
    raise ValueError(
        "Failed to decode 'rlbench_config.json'. Please ensure it is a valid JSON file."
    ) from err

# ---- Process the loaded data ----

# extract and type-check top-level dicts
TASK_DESCRIPTIONS: dict[str, str] = data.get("TASK_DESCRIPTIONS", {})
TASK_NAME_TO_ID: dict[str, int] = data.get("TASK_NAME_TO_ID", {})

ACTION_DIM = 8  # 7 joints + 1 gripper  # NOTE: RLBench supports also EEF pose+gripper (dim=8, [x,y,z,rx,ry,rz,gripper])
OBS_DIM = 8  # 7 joints + 1 gripper


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


def _get_suite(name: str) -> dict[str, list[Task]]:
    """Instantiate a RLBench suite by name with clear validation."""

    suites = {
        "FS10_V1": FS10_V1,
        "FS25_V1": FS25_V1,
        "FS50_V1": FS50_V1,
        "FS95_V1": FS95_V1,
        "MT15_V1": MT15_V1,
        "MT30_V1": MT30_V1,
        "MT55_V1": MT55_V1,
        "MT100_V1": MT100_V1,
    }

    if name not in suites:
        raise ValueError(f"Unknown RLBench suite '{name}'. Available: {', '.join(sorted(suites.keys()))}")
    suite = suites[name]

    if not suite.get("train", None) and not suite.get("test", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """Validate/normalize task ids. If None → all tasks."""
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


class RLBenchEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: Task | None = None,
        task_suite: dict[str, list[Task]] | None = None,
        camera_name: str | Sequence[str] = "front_rgb,wrist_rgb",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        camera_name_mapping: dict[str, str] | None = None,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.camera_name = _parse_camera_names(camera_name)

        # Map raw camera names to "image1" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        if camera_name_mapping is None:
            camera_name_mapping = {
                "front_rgb": "image",
                "wrist_rgb": "image2",
            }
        self.camera_name_mapping = camera_name_mapping

        self._env = self._make_envs_task(self.task)
        self._max_episode_steps = 500  # TODO: make configurable depending on task suite?
        task_name = self.task.get_name() if self.task is not None else ""
        self.task_description = TASK_DESCRIPTIONS.get(task_name, "")

        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if (
            self.obs_type == "state"
        ):  # TODO: This can be implemented in RLBench, because the observation contains joint positions and gripper pose
            raise ValueError(
                "The 'state' observation type is not supported in RLBench. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(OBS_DIM,),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(ACTION_DIM,), dtype=np.float32)

    def render(self) -> np.ndarray:
        """
        Render the current environment frame.

        Returns:
            np.ndarray: The rendered RGB image from the environment.
        """
        return self._env.render()

    def _make_envs_task(self, task: Task):
        return RLBenchGymEnv(
            task,
            observation_mode="vision",
            render_mode=self.render_mode,
            action_mode=AbsoluteJointPositionActionMode(),
        )

    def _format_raw_obs(self, raw_obs: dict) -> dict[str, Any]:
        images = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            image = image[::-1, ::-1]  # rotate 180 degrees
            images[self.camera_name_mapping[camera_name]] = image

        agent_pos = np.concatenate(
            raw_obs["joint_positions"],
            [raw_obs["gripper_open"]],
        )

        if (
            self.obs_type == "state"
        ):  # TODO: this can be implemented in RLBench, because the observation contains joint positions and gripper pose
            raise NotImplementedError(
                "'state' obs_type not implemented for RLBench. Use pixel modes instead."
            )

        if self.obs_type == "pixels":
            obs = {"pixels": image.copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": image.copy(),
                "agent_pos": agent_pos,
            }
        else:
            raise NotImplementedError(
                f"The observation type '{self.obs_type}' is not supported in RLBench. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )
        return obs

    def reset(
        self,
        seed: int | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (Optional[int]): Random seed for environment initialization.

        Returns:
            observation (Dict[str, Any]): The initial formatted observation.
            info (Dict[str, Any]): Additional info about the reset state.
        """
        super().reset(seed=seed)

        raw_obs, info = self._env.reset(seed=seed)

        observation = self._format_raw_obs(raw_obs)

        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Perform one environment step.

        Args:
            action (np.ndarray): The action to execute, must be 1-D with shape (action_dim,).

        Returns:
            observation (Dict[str, Any]): The formatted observation after the step.
            reward (float): The scalar reward for this step.
            terminated (bool): Whether the episode terminated successfully.
            truncated (bool): Whether the episode was truncated due to a time limit.
            info (Dict[str, Any]): Additional environment info.
        """
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, done, truncated, info = self._env.step(action)

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

        # Format the raw observation into the expected structure
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
        self._env.close()


def _make_env_fns(
    *,
    suite: dict[str, list[Task]],
    task: Task,
    n_envs: int,
    camera_names: list[str],
    gym_kwargs: Mapping[str, Any],
) -> list[Callable[[], RLBenchEnv]]:
    """Build n_envs factory callables for a single (suite, task)."""

    def _make_env(**kwargs) -> RLBenchEnv:
        local_kwargs = dict(kwargs)
        return RLBenchEnv(
            task=task,
            task_suite=suite,
            camera_name=camera_names,
            **local_kwargs,
        )

    fns: list[Callable[[], RLBenchEnv]] = []
    for _ in range(n_envs):
        fns.append(partial(_make_env, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------


def create_rlbench_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "front_rgb,wrist_rgb",
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized RLBench environments with a consistent return shape.

    Returns:
        dict[task_group][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single suite or a comma-separated list of suites.
        - You may pass `task_names` (list[str]) inside `gym_kwargs` to restrict tasks per suite.
    """

    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks

    camera_names = _parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one RLBench suite name.")

    print(f"Creating RLBench envs | task_groups={suite_names} | n_envs(per task)={n_envs}")
    if task_ids_filter is not None:
        print(f"Restricting to task_ids={task_ids_filter}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite["train"])
        selected = _select_task_ids(total, task_ids_filter)

        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

        for tid in selected:  # FIXME: this breaks!
            fns = _make_env_fns(
                suite=suite,
                task=suite["train"][tid],
                n_envs=n_envs,
                camera_names=camera_names,
                gym_kwargs=gym_kwargs,
            )
            out[suite_name][tid] = env_cls(fns)
            print(f"Built vec env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

    # return plain dicts for predictability
    return {group: dict(task_map) for group, task_map in out.items()}
