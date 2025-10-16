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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import metaworld
import metaworld.policies as policies
import numpy as np
from gymnasium import spaces

# ---- Load configuration data from the external JSON file ----
CONFIG_PATH = Path(__file__).parent / "metaworld_config.json"
try:
    with open(CONFIG_PATH) as f:
        data = json.load(f)
except FileNotFoundError as err:
    raise FileNotFoundError(
        "Could not find 'metaworld_config.json'. "
        "Please ensure the configuration file is in the same directory as the script."
    ) from err
except json.JSONDecodeError as err:
    raise ValueError(
        "Failed to decode 'metaworld_config.json'. Please ensure it is a valid JSON file."
    ) from err

# ---- Process the loaded data ----

# extract and type-check top-level dicts
task_descriptions_obj = data.get("TASK_DESCRIPTIONS")
if not isinstance(task_descriptions_obj, dict):
    raise TypeError("Expected TASK_DESCRIPTIONS to be a dict[str, str]")
TASK_DESCRIPTIONS: dict[str, str] = task_descriptions_obj

task_name_to_id_obj = data.get("TASK_NAME_TO_ID")
if not isinstance(task_name_to_id_obj, dict):
    raise TypeError("Expected TASK_NAME_TO_ID to be a dict[str, int]")
TASK_NAME_TO_ID: dict[str, int] = task_name_to_id_obj

# difficulty -> tasks mapping
difficulty_to_tasks = data.get("DIFFICULTY_TO_TASKS")
if not isinstance(difficulty_to_tasks, dict):
    raise TypeError("Expected 'DIFFICULTY_TO_TASKS' to be a dict[str, list[str]]")
DIFFICULTY_TO_TASKS: dict[str, list[str]] = difficulty_to_tasks

# convert policy strings -> actual policy classes
task_policy_mapping = data.get("TASK_POLICY_MAPPING")
if not isinstance(task_policy_mapping, dict):
    raise TypeError("Expected 'TASK_POLICY_MAPPING' to be a dict[str, str]")
TASK_POLICY_MAPPING: dict[str, Any] = {
    task_name: getattr(policies, policy_class_name)
    for task_name, policy_class_name in task_policy_mapping.items()
}
ACTION_DIM = 4
OBS_DIM = 4


class MetaworldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task,
        camera_name="corner2",
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=480,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task.replace("metaworld-", "")
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.camera_name = camera_name

        self._env = self._make_envs_task(self.task)
        self._max_episode_steps = self._env.max_path_length
        self.task_description = TASK_DESCRIPTIONS[self.task]

        self.expert_policy = TASK_POLICY_MAPPING[self.task]()

        if self.obs_type == "state":
            raise NotImplementedError()
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
        image = self._env.render()
        if self.camera_name == "corner2":
            # Images from this camera are flipped — correct them
            image = np.flip(image, (0, 1))
        return image

    def _make_envs_task(self, env_name: str):
        mt1 = metaworld.MT1(env_name, seed=42)
        env = mt1.train_classes[env_name](render_mode="rgb_array", camera_name=self.camera_name)
        env.set_task(mt1.train_tasks[0])
        if self.camera_name == "corner2":
            env.model.cam_pos[2] = [
                0.75,
                0.075,
                0.7,
            ]  # corner2 position, similar to https://arxiv.org/pdf/2206.14244
        env.reset()
        env._freeze_rand_vec = False  # otherwise no randomization
        return env

    def _format_raw_obs(self, raw_obs: np.ndarray) -> dict[str, Any]:
        image = None
        if self._env is not None:
            image = self._env.render()
            if self.camera_name == "corner2":
                # NOTE: The "corner2" camera in MetaWorld environments outputs images with both axes inverted.
                image = np.flip(image, (0, 1))
        agent_pos = raw_obs[:4]
        if self.obs_type == "state":
            raise NotImplementedError(
                "'state' obs_type not implemented for MetaWorld. Use pixel modes instead."
            )

        elif self.obs_type in ("pixels", "pixels_agent_pos"):
            assert image is not None, (
                "Expected `image` to be rendered before constructing pixel-based observations. "
                "This likely means `env.render()` returned None or the environment was not provided."
            )

            if self.obs_type == "pixels":
                obs = {"pixels": image.copy()}

            else:  # pixels_agent_pos
                obs = {
                    "pixels": image.copy(),
                    "agent_pos": agent_pos,
                }
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")
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


# ---- Main API ----------------------------------------------------------------


def create_metaworld_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized Meta-World environments with a consistent return shape.

    Returns:
        dict[task_group][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single difficulty group (e.g., "easy", "medium", "hard") or a comma-separated list.
        - If a task name is not in DIFFICULTY_TO_TASKS, we treat it as a single custom task.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_groups = [t.strip() for t in task.split(",") if t.strip()]
    if not task_groups:
        raise ValueError("`task` must contain at least one Meta-World task or difficulty group.")

    print(f"Creating Meta-World envs | task_groups={task_groups} | n_envs(per task)={n_envs}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for group in task_groups:
        # if not in difficulty presets, treat it as a single custom task
        tasks = DIFFICULTY_TO_TASKS.get(group, [group])

        for tid, task_name in enumerate(tasks):
            print(f"Building vec env | group={group} | task_id={tid} | task={task_name}")

            # build n_envs factories
            fns = [(lambda tn=task_name: MetaworldEnv(task=tn, **gym_kwargs)) for _ in range(n_envs)]

            out[group][tid] = env_cls(fns)

    # return a plain dict for consistency
    return {group: dict(task_map) for group, task_map in out.items()}
