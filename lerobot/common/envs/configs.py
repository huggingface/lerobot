# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import draccus

from lerobot.common.constants import ACTION, OBS_ENV, OBS_IMAGE, OBS_IMAGES, OBS_ROBOT
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.configs.types import FeatureType, PolicyFeature


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "environment_state": OBS_ENV,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(type=FeatureType.VISUAL, shape=(384, 384, 3))
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("xarm")
@dataclass
class XarmEnv(EnvConfig):
    task: str = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@dataclass
class VideoRecordConfig:
    """Configuration for video recording in ManiSkill environments."""

    enabled: bool = False
    record_dir: str = "videos"
    trajectory_name: str = "trajectory"


@dataclass
class WrapperConfig:
    """Configuration for environment wrappers."""

    joint_masking_action_space: list[bool] | None = None


@dataclass
class EEActionSpaceConfig:
    """Configuration parameters for end-effector action space."""

    x_step_size: float
    y_step_size: float
    z_step_size: float
    bounds: Dict[str, Any]  # Contains 'min' and 'max' keys with position bounds
    use_gamepad: bool = False


@dataclass
class EnvWrapperConfig:
    """Configuration for environment wrappers."""

    display_cameras: bool = False
    use_relative_joint_positions: bool = True
    add_joint_velocity_to_observation: bool = False
    add_current_to_observation: bool = False
    add_ee_pose_to_observation: bool = False
    crop_params_dict: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    resize_size: Optional[Tuple[int, int]] = None
    control_time_s: float = 20.0
    fixed_reset_joint_positions: Optional[Any] = None
    reset_time_s: float = 5.0
    joint_masking_action_space: Optional[Any] = None
    ee_action_space_params: Optional[EEActionSpaceConfig] = None
    use_gripper: bool = False
    gripper_quantization_threshold: float | None = 0.8
    gripper_penalty: float = 0.0
    gripper_penalty_in_reward: bool = False
    open_gripper_on_reset: bool = False


@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    robot: Optional[RobotConfig] = None
    wrapper: Optional[EnvWrapperConfig] = None
    fps: int = 10
    name: str = "real_robot"
    mode: str = None  # Either "record", "replay", None
    repo_id: Optional[str] = None
    dataset_root: Optional[str] = None
    task: str = ""
    num_episodes: int = 10  # only for record mode
    episode: int = 0
    device: str = "cuda"
    push_to_hub: bool = True
    pretrained_policy_name_or_path: Optional[str] = None
    reward_classifier_pretrained_path: Optional[str] = None

    def gym_kwargs(self) -> dict:
        return {}


@EnvConfig.register_subclass("maniskill_push")
@dataclass
class ManiskillEnvConfig(EnvConfig):
    """Configuration for the ManiSkill environment."""

    name: str = "maniskill/pushcube"
    task: str = "PushCube-v1"
    image_size: int = 64
    control_mode: str = "pd_ee_delta_pose"
    state_dim: int = 25
    action_dim: int = 7
    fps: int = 200
    episode_length: int = 50
    obs_type: str = "rgb"
    render_mode: str = "rgb_array"
    render_size: int = 64
    device: str = "cuda"
    robot: str = "so100"  # This is a hack to make the robot config work
    video_record: VideoRecordConfig = field(default_factory=VideoRecordConfig)
    wrapper: WrapperConfig = field(default_factory=WrapperConfig)
    mock_gripper: bool = False
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(25,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.image": OBS_IMAGE,
            "observation.state": OBS_ROBOT,
        }
    )
    reward_classifier_pretrained_path: Optional[str] = None

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
            "control_mode": self.control_mode,
            "sensor_configs": {"width": self.image_size, "height": self.image_size},
            "num_envs": 1,
        }
