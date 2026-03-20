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
from collections.abc import Callable, Sequence, Mapping
from functools import partial
from typing import Any
from gymnasium import spaces

import numpy as np

from lerobot.types import RobotObservation
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
from robocasa.utils.env_utils import create_env, convert_action
from robosuite.controllers.composite.composite_controller import HybridMobileBase

OBS_STATE_DIM = 16
ACTION_DIM = 12
AGENT_POS_LOW = -1000.0
AGENT_POS_HIGH = 1000.0
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DEFAULT_MAX_EPISODE_STEPS = 1000
DEFAULT_MAX_EPISODE_STEPS_BY_TASK = {
    # single_stage tasks
    "CloseDoubleDoor": 474,
    "CloseDrawer": 227,
    "CloseSingleDoor": 322,
    "CoffeePressButton": 156,
    "CoffeeServeMug": 433,
    "CoffeeSetupMug": 376,
    "NavigateKitchen": 322,
    "OpenDoubleDoor": 889,
    "OpenDrawer": 260,
    "OpenSingleDoor": 414,
    "PnPCabToCounter": 477,
    "PnPCounterToCab": 364,
    "PnPCounterToMicrowave": 509,
    "PnPCounterToSink": 680,
    "PnPCounterToStove": 404,
    "PnPMicrowaveToCounter": 430,
    "PnPSinkToCounter": 351,
    "PnPStoveToCounter": 417,
    "TurnOffMicrowave": 318,
    "TurnOffSinkFaucet": 336,
    "TurnOffStove": 338,
    "TurnOnMicrowave": 279,
    "TurnOnSinkFaucet": 342,
    "TurnOnStove": 349,
    "TurnSinkSpout": 187,
    # multi_stage tasks
    "ArrangeVegetables": 1132,
    "MicrowaveThawing": 906,
    "PreSoakPan": 1439,
    "PrepareCoffee": 980,
    "RestockPantry": 925,
}

CAMERA_NAME_MAPPING = {
        "robot0_agentview_left_image": "robot0_agentview_left",
        "robot0_agentview_right_image": "robot0_agentview_right",
        "robot0_eye_in_hand_image": "robot0_eye_in_hand",
    }
def convert_state(dict_state): 
    """
    Converts input state (dict) to format expected LeRobot (np.array)
    """
    dict_state = dict_state.copy()
    final_state = np.concat([
        dict_state["state.end_effector_position_relative"],
        dict_state["state.end_effector_rotation_relative"],
        dict_state["state.gripper_qpos"],
        dict_state["state.base_position"],
        dict_state["state.base_rotation"],
    ], axis=0)
    
    return final_state


class RoboCasaEnv(RoboCasaGymEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task:str,
        camera_name: str | Sequence[str] = "robot0_agentview_left_image,robot0_eye_in_hand,robot0_agentview_right_image",
        render_mode="rgb_array",
        obs_type: str = "pixels_agent_pos",
        observation_width=480,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
        split=None, # {None, "all", "pretrain", "target"}
        ep_meta: dict| None = None,
        **kwargs
    ):
        kwargs["style_ids"] = ep_meta.get("style_ids", [-1]) if ep_meta is not None else [-1]
        kwargs["layout_ids"] = ep_meta.get("layout_ids", [-1]) if ep_meta is not None else [-1]
        self.obs_type = obs_type
        self.camera_name = camera_name
        self.camera_name_mapping = CAMERA_NAME_MAPPING
        self.max_episode_steps = DEFAULT_MAX_EPISODE_STEPS_BY_TASK.get(
            task.replace("PickPlace", "PnP"), DEFAULT_MAX_EPISODE_STEPS
        )
        self._max_episode_steps = (
            self.max_episode_steps
        )  # Required by gymnasium for env.call("_max_episode_steps")
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.kwargs = kwargs
        self.split = split
        self.task = task
        super().__init__(
            task,
            camera_names=camera_name,
            camera_widths=observation_width,
            camera_heights=observation_height,
            enable_render=(render_mode is not None),
            split=split,
            **kwargs
        )
        if ep_meta is not None:
            self.env.set_ep_meta(ep_meta)
    
    def _create_obs_and_action_space(self):
        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping.get(cam, cam)] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
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
                        low=AGENT_POS_LOW,
                        high=AGENT_POS_HIGH,
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

    def render(self) -> np.ndarray:
        """
        Render the current environment frame.

        Returns:
            np.ndarray: The rendered RGB image from the environment.
        """
        return super().render()

    def _make_envs_task(self, task: Any):
        env = create_env(
            task,
            camera_name=self.camera_name,
            camera_width=self.observation_width,
            camera_height=self.observation_height,
            enable_render=True,
            split=self.split,
            **self.kwargs
        )
        env.reset()
        return env


    def _format_raw_obs(self, raw_obs: np.ndarray) -> RobotObservation:
        return super().get_observation(raw_obs)

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
        observation, info = super().reset(seed, **kwargs)
        new_obs = self.get_obs(observation)
        return new_obs, info 
    
    def get_obs(self, raw_obs:dict):
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
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        action_dict = convert_action(action)
        observation, reward, done, truncated, info = super().step(action_dict)
        new_obs = self.get_obs(observation)

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
    gym_kwargs: Mapping[str, Any],
    ep_metas: list[dict[str, Any]] | None = None,
) -> list[Callable[[], RoboCasaEnv]]:
    """Build n_envs factory callables for a dataset."""

    def _make_env(episode_index: int, **kwargs) -> RoboCasaEnv:
        local_kwargs = dict(kwargs)
        
        # Extract ep_meta for this worker from ep_metas list if provided
        if ep_metas is not None:
            if len(ep_metas) <= episode_index:
                raise ValueError(
                    f"ep_metas list has {len(ep_metas)} elements, but episode_index {episode_index} "
                    f"requires at least {episode_index + 1} elements."
                )
            ep_meta = ep_metas[episode_index].copy()  # Make a copy to avoid mutations
        else:
            # No ep_metas provided: use defaults
            ep_meta = None

        # Extract seed from ep_meta if present, otherwise use episode_index as default
        seed = local_kwargs.pop("seed", episode_index)
        
        # Remove ep_meta from local_kwargs if present (shouldn't be there, but just in case)
        local_kwargs.pop("ep_meta", None)
        
        return RoboCasaEnv(
            task=task_name,
            camera_name=camera_names,
            seed=seed,
            ep_meta=ep_meta,
            **local_kwargs,
        )

    fns: list[Callable[[], RoboCasaEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns

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

# ---- Main API ----------------------------------------------------------------
def create_robocasa_envs(
    task_name: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "",
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized RoboCasa environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - For RoboCasa, we use a single suite_name "robocasa" and task_id 0.
    Args:
        task_name: Name of the task
        n_envs: Number of environments to create
        gym_kwargs: Additional arguments to pass to RoboCasaEnv. Can include 'ep_metas' (list of dicts)
            to provide different ep_meta for each worker. Each ep_meta dict can include a 'seed' key
            to set a specific seed for that worker.
        camera_name: Camera name(s) to use for observations, overrides gym_kwargs['camera_name'] if provided
        env_cls: Callable that wraps a list of environment factory callables (for vectorization)
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    gym_kwargs_camera_name = gym_kwargs.pop("camera_name", None)
    camera_name = camera_name if camera_name != "" else gym_kwargs_camera_name
    parsed_camera_names = _parse_camera_names(camera_name)
    
    # Extract ep_metas from gym_kwargs (similar to how camera_name is handled)
    # This prevents it from being passed to the main env class
    ep_metas = gym_kwargs.pop("ep_metas", None)
    if ep_metas is not None:
        if not isinstance(ep_metas, (list, tuple)):
            raise TypeError(f"ep_metas must be a list or tuple, got {type(ep_metas).__name__}")
        if len(ep_metas) < n_envs:
            raise ValueError(
                f"ep_metas list has {len(ep_metas)} elements, but n_envs={n_envs} requires at least {n_envs} elements."
            )

    suite_name = "robocasa"
    task_id = 0
    
    print(f"Creating RoboCasa envs | task={task_name} | n_envs(per task)={n_envs}")
    
    out: dict[str, dict[int, Any]] = defaultdict(dict)
    
    fns = _make_env_fns(
        task_name=task_name,
        n_envs=n_envs,
        camera_names=parsed_camera_names,
        gym_kwargs=gym_kwargs,
        ep_metas=ep_metas,
    )
    out[suite_name][task_id] = env_cls(fns)
    print(f"Built vec env | suite={suite_name} | task_id={task_id} | n_envs={n_envs}")

    # return plain dicts for predictability
    return {suite: dict(task_map) for suite, task_map in out.items()}