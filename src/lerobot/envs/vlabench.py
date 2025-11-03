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

from collections.abc import Callable, Iterable, Mapping, Sequence
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from VLABench.envs import load_env

VALID_OBS_KEYS = [
    "q_state",
    "q_velocity", 
    "q_acceleration",
    "rgb",
    "depth",
    "segmentation",
    "robot_mask",
    "instrinsic",
    "extrinsic",
    "masked_point_cloud",
    "point_cloud",
    "ee_state",
    "grasped_obj_name"
]

ACTION_DIM = 7
JOINT_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0


class VLABenchEnv(gym.Env):
    """
    The environment class for VLABench environments.
    Args: 
        task: The name of the task to load.
        robot: The type of robot to use.
        config: The additional configuration for the environment, such as camera settings, task settings, and engine settings.
                If None, the default configuration will be used， without any specification.
        time_limit: The time limit for each episode.
        reset_wait_step: The steps to wait when init the episode. This is useful to avoid collision at the beginning of the episode.
        episode_config: A config dict including task-specific parameters, such as object instances, positions, success conditions, etc.
                If None, the environment will be initialized within a default random range.
        obs_keys: A list of keys to include in the observation. If empty, all keys will be included.        
    """
    def __init__(
        self,
        task,
        robot="franka",
        config=None,
        time_limit=500,
        reset_wait_step=0,
        episode_config=None,
        obs_keys=['rgb', 'q_state', 'ee_state'],
        render_resolution=(480, 480),
        **kwargs,
    ):
        super().__init__()
        self.task_name = task
        assert len(obs_keys) >= 3, "obs_keys should at least include 'rgb', 'q_state' and 'ee_state'"
        assert all([key in VALID_OBS_KEYS for key in obs_keys]), f"obs_keys should be in {VALID_OBS_KEYS}"
        self.obs_keys = obs_keys
        self._env = load_env(
            task,
            robot,
            config,
            time_limit,
            reset_wait_step,
            episode_config,
            render_resolution=render_resolution,
            **kwargs,
        )
        # define observation spaces and action spaces
        n_cam = self._env.physics.model.ncam
        self._build_observation_space(n_cam, render_resolution)
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )
    @property
    def name(self):
        return self.task_name
    
    @property
    def task(self):
        return self._env.task
    
    @property
    def robot(self):
        return self._env.robot
    
    def render(self):
        return self._env.render()
    
    def reset(self):
        timestep = self._env.reset()
        obs = timestep.observation
        obs = self._filter_observation(obs)
        info = {"is_success": False}
        return obs, info
    
    def _filter_observation(self, observation):
        if not self.obs_keys:
            return observation
        filtered_obs = {key: observation[key] for key in self.obs_keys if key in observation}
        return filtered_obs
    
    def step(self, action: np.ndarray):
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        timestep = self._env.step(action)
        obs, reward, done, info = timestep.observation, timestep.reward, timestep.last(), {}
        obs = self._filter_observation(obs)
        is_success = self._env.task.should_terminate_episode()
        terminated = done or is_success
        if done:
            self.reset()
            info.update(
                {
                    "task": self.task_name,
                    "episode_config": self._env.save(),
                    "done": done,
                    "is_success": is_success,
                }
            )
        truncated = False
        return obs, reward, terminated, truncated, info
    
    def _build_observation_space(self, n_cam, render_resolution):
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    low=0, 
                    high=255, 
                    shape=(n_cam, 3, render_resolution[0], render_resolution[1]), 
                    dtype=np.uint8
                ),
                "q_state": spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(7,), 
                    dtype=np.float32
                ),
                "ee_state": spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(7,), 
                    dtype=np.float32
                )
            }
        )
        if "depth" in self.obs_keys:
            self.observation_space.spaces.update(
                {
                    "depth": spaces.Box(
                        low=0.0, 
                        high=1.0, 
                        shape=(n_cam, 1, render_resolution[0], render_resolution[1]), 
                        dtype=np.float32
                    )
                }
            )
        if "segmentation" in self.obs_keys:
            self.observation_space.spaces.update(
                {
                    "segmentation": spaces.Box(
                        low=0, 
                        high=255, 
                        shape=(n_cam, 1, render_resolution[0], render_resolution[1]), 
                        dtype=np.uint8
                    )
                }
            )
        if "q_velocity" in self.obs_keys:
            self.observation_space.spaces.update(
                {
                    "q_velocity": spaces.Box(
                        low=-np.inf, 
                        high=np.inf, 
                        shape=(self._env.action_spec().shape[0],), 
                        dtype=np.float32
                    )
                }
            )
        if "q_acceleration" in self.obs_keys:
            self.observation_space.spaces.update(
                {
                    "q_acceleration": spaces.Box(
                        low=-np.inf, 
                        high=np.inf, 
                        shape=(self._env.action_spec().shape[0],), 
                        dtype=np.float32
                    )
                }
            )
    
    def close(self):
        self._env.close()
        
# ---- Main API ----------------------------------------------------------------

def create_vlabench_envs(
    task,
    n_envs: int,
    robot="franka",
    configs: Sequence[dict] | None = None,
    episode_configs: Sequence[dict] | None = None,
):
    """
    Create vectorized VLABench environments.
    Args:
        task: The name of the task to load.
        n_envs: The number of environments to create.
        robot: The type of robot to use.
        configs: A sequence of configuration dicts for each environment. 
            If None, the default configuration will be used for all environments.
        episode_configs: A sequence of episode configuration dicts for each environment. 
            If None, the environment will be initialized within a default random range.
    Notes:
        - The difference between configs and episode_configs is that configs are used to configure the environment itself such as camera settings, task settings, and engine settings;
        while episode_configs are used to configure the specific episode within the environment instead of random initialization.
        - To get the episode configuration, please refer to `VLABench/VLABench/evaluation/evaluator/base.py`
    Returns:
        List of VLABenchEnv instances.
    """
    if configs is not None :
        assert len(configs) == n_envs, "Length of configs should match n_envs."
    if episode_configs is not None :
        assert len(episode_configs) == n_envs, "Length of episode_configs should match n_envs."
    # Create the environments
    envs = {
        f"{task}":{}
    }
    for i in range(n_envs):
        env = VLABenchEnv(
            task=task,
            robot=robot,
            config=configs[i] if configs is not None else None,
            time_limit=configs[i]['evaluation'].get("max_episode_length", 500),
            episode_config=episode_configs[i] if episode_configs is not None else None,
        )
        envs[f"{task}"][f"{i}"] = env
    return envs