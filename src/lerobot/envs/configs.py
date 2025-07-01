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
from typing import Any, Optional

import draccus

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.robots import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    @abc.abstractmethod
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
            "agent_pos": OBS_STATE,
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
            "agent_pos": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
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
            "agent_pos": OBS_STATE,
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
class EnvTransformConfig:
    """Configuration for environment wrappers."""

    # ee_action_space_params: EEActionSpaceConfig = field(default_factory=EEActionSpaceConfig)
    control_mode: str = "gamepad"
    display_cameras: bool = False
    add_joint_velocity_to_observation: bool = False
    add_current_to_observation: bool = False
    add_ee_pose_to_observation: bool = False
    crop_params_dict: Optional[dict[str, tuple[int, int, int, int]]] = None
    resize_size: Optional[tuple[int, int]] = None
    control_time_s: float = 20.0
    fixed_reset_joint_positions: Optional[Any] = None
    reset_time_s: float = 5.0
    use_gripper: bool = True
    gripper_quantization_threshold: float | None = 0.8
    gripper_penalty: float = 0.0
    gripper_penalty_in_reward: bool = False


@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    robot: Optional[RobotConfig] = None
    teleop: Optional[TeleoperatorConfig] = None
    wrapper: Optional[EnvTransformConfig] = None
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
    # For the reward classifier, to record more positive examples after a success
    number_of_steps_after_success: int = 0

    def gym_kwargs(self) -> dict:
        return {}


@EnvConfig.register_subclass("hil")
@dataclass
class HILEnvConfig(EnvConfig):
    """Configuration for the HIL environment."""

    type: str = "hil"
    name: str = "PandaPickCube"
    task: str = "PandaPickCubeKeyboard-v0"
    use_viewer: bool = True
    gripper_penalty: float = 0.0
    use_gamepad: bool = True
    state_dim: int = 18
    action_dim: int = 4
    fps: int = 100
    episode_length: int = 100
    video_record: VideoRecordConfig = field(default_factory=VideoRecordConfig)
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(18,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.image": OBS_IMAGE,
            "observation.state": OBS_STATE,
        }
    )
    ################# args from hilserlrobotenv
    reward_classifier_pretrained_path: Optional[str] = None
    robot_config: Optional[RobotConfig] = None
    teleop_config: Optional[TeleoperatorConfig] = None
    wrapper: Optional[EnvTransformConfig] = None
    mode: str = None  # Either "record", "replay", None
    repo_id: Optional[str] = None
    dataset_root: Optional[str] = None
    num_episodes: int = 10  # only for record mode
    episode: int = 0
    device: str = "cuda"
    push_to_hub: bool = True
    pretrained_policy_name_or_path: Optional[str] = None
    # For the reward classifier, to record more positive examples after a success
    number_of_steps_after_success: int = 0
    ############################

    @property
    def gym_kwargs(self) -> dict:
        return {
            "use_viewer": self.use_viewer,
            "use_gamepad": self.use_gamepad,
            "gripper_penalty": self.gripper_penalty,
        }
