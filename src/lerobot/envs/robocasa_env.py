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

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import robosuite
from gymnasium import spaces
from robosuite.controllers import load_part_controller_config

from lerobot.utils.constants import HF_LEROBOT_HOME


@dataclass
class EnvArgs:
    """Environment arguments for creating robosuite environments."""

    env_name: str = ""
    robots: str | list[str] = "PandaOmron"
    controller: str | None = "OSC_POSE"
    has_renderer: bool = False
    renderer: str = "mjviewer"
    control_freq: int = 20
    use_object_obs: bool = True
    use_camera_obs: bool = True
    camera_names: list = field(default_factory=lambda: ["robot0_agentview_center", "robot0_eye_in_hand"])
    camera_heights: int = 128
    camera_widths: int = 128
    camera_depths: bool = False
    # camera-segmentations is not accepted in robocasa
    # camera_segmentations: str = "instance"
    seed: int = 0
    controller_configs: dict = field(default_factory=dict)
    layout_ids: list = field(default_factory=list)
    style_ids: list = field(default_factory=list)
    translucent_robot: bool = False
    reward_shaping: bool = False
    has_offscreen_renderer: bool = False
    ignore_done: bool = False
    # ep_meta is not supported directly in the __init__ method of RoboCasaEnv, but it can be set later using set_ep_meta
    # ep_meta: defaultdict = field(default_factory=defaultdict)

    def __post_init__(self):
        if list(self.controller_configs.keys()) == [] and robosuite.__version__ > "1.4.0":
            self.controller_configs = load_part_controller_config(
                default_controller=self.controller,
            )

    def env_dict(self):
        exclude_keys = ["controller"]
        return {k: v for k, v in self.__dict__.items() if k not in exclude_keys}


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


def _parse_env_args_from_hdf5(dataset_path: str) -> dict[str, Any]:
    """Extract environment arguments from dataset."""
    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"]) if "data" in f else json.loads(f.attrs["env_args"])
        if isinstance(env_args, str):
            env_args = json.loads(env_args)  # double leads to dict type
    return env_args


def _parse_env_meta_from_hdf5(dataset_path: str, episode_index: int = 0) -> dict[str, Any]:
    """Extract environment metadata from dataset."""
    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        data = f.get("data", f)
        keys = list(data.keys())
        env_meta = data[keys[episode_index]].attrs["ep_meta"]
        env_meta = json.loads(env_meta)
        assert isinstance(env_meta, dict), f"Expected dict type but got {type(env_meta)}"
    return env_meta


def _parse_env_meta_from_repo_id(repo_id: str, episode_index: int = 0) -> dict[str, Any]:
    """Extract environment metadata from dataset."""
    dataset_path = HF_LEROBOT_HOME / repo_id
    with open(dataset_path / "meta" / "episodes" / "ep_metas.json") as f:
        env_metas = json.load(f)
    return env_metas[episode_index]


def get_robocasa_dummy_action(env):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    active_robot = env.robots[0]
    if env.action_dim == 12:
        assert len(env.robots) == 1, "Only one robot is supported in this function"
        assert env.robots[0].name == "PandaOmron", "Only PandaOmron is supported in this function"
        arms = ["right"]
        zero_action_dict = {}
        for arm in arms:
            # controller has absolute actions, so we need to set the initial action to be the current position
            zero_action = np.zeros(7)
            if active_robot.part_controllers[arm].input_type == "absolute":
                raise NotImplementedError("Dummy actions assume relative actions")
            zero_action_dict[f"{arm}"] = zero_action[: zero_action.shape[0] - 1]
            zero_action_dict[f"{arm}_gripper"] = zero_action[zero_action.shape[0] - 1 :]
        zero_action_dict["base_mode"] = -1
        zero_action_dict["base"] = np.zeros(3)
        zero_action = active_robot.create_action_vector(zero_action_dict)
    else:
        # For single-arm robots, try to get the arm name
        raise NotImplementedError("If you're using another robot, need to update this function")
    return zero_action


# Default constants
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


class RoboCasaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        # dataset_path: str,
        task_name: str,
        camera_name: str | Sequence[str] = "robot0_agentview_center,robot0_eye_in_hand",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        max_episode_steps: int | None = None,
        ep_meta: dict | None = None,
        seed: int = 0,
        return_raw_obs: bool = False,
        **env_kwargs,
    ):
        """
        Initialize RoboCasa environment from HDF5 dataset.

        Args:
            task_name: Name of the task
            camera_name: Camera name(s) to use for observations
            obs_type: Observation type ('pixels' or 'pixels_agent_pos')
            render_mode: Render mode
            observation_width: Width of observation images
            observation_height: Height of observation images
            camera_name_mapping: Mapping from raw camera names to output names
            num_steps_wait: Number of steps to wait after reset for stability
            max_episode_steps: Maximum number of steps per episode
            return_raw_obs: Whether to return raw observations
            **env_kwargs: Additional arguments to pass to environment creation
        """
        super().__init__()
        self.task_name = task_name
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.num_steps_wait = num_steps_wait
        self.max_episode_steps = max_episode_steps or DEFAULT_MAX_EPISODE_STEPS_BY_TASK.get(
            task_name, DEFAULT_MAX_EPISODE_STEPS
        )
        self._max_episode_steps = (
            self.max_episode_steps
        )  # Required by gymnasium for env.call("_max_episode_steps")
        self.return_raw_obs = return_raw_obs
        self._step_count = 0

        # Parse camera names
        self.camera_name = _parse_camera_names(camera_name)

        # Map raw camera names to "image" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        if camera_name_mapping is None:
            camera_name_mapping = {
                "robot0_agentview_center_image": "robot0_agentview_center",
                "robot0_eye_in_hand_image": "robot0_eye_in_hand",
            }
        self.camera_name_mapping = camera_name_mapping

        # Load environment arguments from dataset
        env_args = EnvArgs(
            env_name=task_name,
            robots="PandaOmron",
            controller="OSC_POSE",
            has_renderer=(render_mode == "human"),
            has_offscreen_renderer=(render_mode == "rgb_array"),
            use_object_obs=True,
            use_camera_obs=(render_mode == "rgb_array"),
            camera_names=self.camera_name,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            camera_depths=False,
            seed=seed,
            style_ids=ep_meta.get("style_ids", [-1]) if ep_meta is not None else [-1],
            layout_ids=ep_meta.get("layout_ids", [-1]) if ep_meta is not None else [-1],
        )

        # Create environment (following make_env_from_args pattern)
        env_dict = env_args.env_dict()
        self._env = robosuite.make(**env_dict)
        if ep_meta is not None:
            self._env.set_ep_meta(ep_meta)
        self._env_args = env_args

        # Set up observation space
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

        # Set up action space
        action_dim = self._env.action_dim
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(action_dim,), dtype=np.float32
        )

        # Store task info
        self.task = self.task_name
        self.task_description = None

    def render(self):
        """Render the environment."""
        raw_obs = self._env._get_observations()
        image = self._format_raw_obs(raw_obs)["pixels"]["robot0_agentview_center"]
        return image

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Format raw observations from RoboCasa into the expected format."""
        if self.return_raw_obs:
            return raw_obs
        images = {}
        for camera_name in self.camera_name:
            # RoboCasa uses camera_name + "_image" suffix
            image_key = f"{camera_name}_image"
            if image_key in raw_obs:
                image = raw_obs[image_key]
                # Images are already in correct format, no rotation needed
                images[self.camera_name_mapping.get(camera_name, camera_name)] = image[::-1]
            else:
                # Fallback: try without _image suffix
                raise ValueError(
                    f"Camera name {camera_name} not found in raw observations:\n{raw_obs.keys()}"
                )

        # Extract agent position (end-effector pose + gripper)
        if "robot0_eef_pos" in raw_obs and "robot0_eef_quat" in raw_obs:
            state = np.concatenate(
                (
                    raw_obs["robot0_joint_pos_cos"],
                    raw_obs["robot0_joint_pos_sin"],
                    raw_obs["robot0_gripper_qpos"],
                )
            )
            agent_pos = state
        else:
            # Fallback: use zeros if not available
            agent_pos = np.zeros(OBS_STATE_DIM)

        if self.obs_type == "pixels":
            return {"pixels": images.copy()}
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": images.copy(),
                "agent_pos": agent_pos,
            }
        else:
            raise NotImplementedError(
                f"The observation type '{self.obs_type}' is not supported in RoboCasaEnv."
            )

    def reset(self, seed: int | None = None, ep_meta: dict | None = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed)
        self._step_count = 0

        if ep_meta is not None:
            self._env.set_ep_meta(ep_meta)

        # Reset environment
        raw_obs = self._env.reset()
        self.task_description = self._env.get_ep_meta().get("lang", None)

        # After reset, objects may be unstable. Step the simulator with a no-op action
        # for a few frames so everything settles.
        zero_action = get_robocasa_dummy_action(self._env)
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(zero_action)

        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step the environment."""
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )

        self._step_count += 1
        raw_obs, reward, done, info = self._env.step(action)

        # Check for success if available
        is_success = False
        is_success = self._env._check_success()

        terminated = done or is_success
        truncated = self._step_count >= self.max_episode_steps

        info.update(
            {
                "task": self.task,
                "done": done,
                "is_success": is_success,
            }
        )

        observation = self._format_raw_obs(raw_obs)

        if terminated or truncated:
            info["final_info"] = {
                "task": self.task,
                "done": bool(done),
                "is_success": bool(is_success),
            }

        return observation, reward, terminated, truncated, info

    def close(self):
        """Close the environment."""
        self._env.close()


def _make_env_fns(
    *,
    task_name: str,
    n_envs: int,
    camera_names: list[str],
    gym_kwargs: Mapping[str, Any],
) -> list[Callable[[], RoboCasaEnv]]:
    """Build n_envs factory callables for a dataset."""

    def _make_env(seed: int, **kwargs) -> RoboCasaEnv:
        local_kwargs = dict(kwargs)
        return RoboCasaEnv(
            task_name=task_name,
            camera_name=camera_names,
            seed=seed,
            **local_kwargs,
        )

    fns: list[Callable[[], RoboCasaEnv]] = []
    for seed in range(n_envs):
        fns.append(partial(_make_env, seed=seed, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------
def create_robocasa_envs(
    task_name: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "",
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> RoboCasaEnv | Any:
    """
    Create vectorized RoboCasa environments from an HDF5 dataset.

    Args:
        task_name: Name of the task
        n_envs: Number of environments to create
        gym_kwargs: Additional arguments to pass to RoboCasaEnv
        camera_name: Camera name(s) to use for observations, overrides gym_kwargs['camera_name'] if provided
        env_cls: Callable that wraps a list of environment factory callables (for vectorization)

    Returns:
        If env_cls is provided, returns vectorized environment. Otherwise returns a single RoboCasaEnv.
    """
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    gym_kwargs_camera_name = gym_kwargs.pop("camera_name", None)
    camera_name = camera_name if camera_name != "" else gym_kwargs_camera_name
    parsed_camera_names = _parse_camera_names(camera_name)

    if env_cls is None:
        # Return a single environment
        return RoboCasaEnv(
            task_name=task_name,
            camera_name=parsed_camera_names,
            **gym_kwargs,
        )
    else:
        # Return vectorized environment
        if not callable(env_cls):
            raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")

        fns = _make_env_fns(
            task_name=task_name,
            n_envs=n_envs,
            camera_names=parsed_camera_names,
            gym_kwargs=gym_kwargs,
        )
        vec_env = env_cls(fns)
        print(f"Built vec env | n_envs={n_envs}")
        return vec_env
