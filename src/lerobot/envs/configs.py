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
from dataclasses import dataclass, field, fields
from typing import Any

import draccus

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.robots import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import (
    ACTION,
    LIBERO_KEY_EEF_MAT,
    LIBERO_KEY_EEF_POS,
    LIBERO_KEY_EEF_QUAT,
    LIBERO_KEY_GRIPPER_QPOS,
    LIBERO_KEY_GRIPPER_QVEL,
    LIBERO_KEY_JOINTS_POS,
    LIBERO_KEY_JOINTS_VEL,
    LIBERO_KEY_PIXELS_AGENTVIEW,
    LIBERO_KEY_PIXELS_EYE_IN_HAND,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
)


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    max_parallel_tasks: int = 1
    disable_env_checker: bool = True

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    def package_name(self) -> str:
        """Package name to import if environment not found in gym registry"""
        return f"gym_{self.type}"

    @property
    def gym_id(self) -> str:
        """ID string used in gym.make() to instantiate the environment"""
        return f"{self.package_name}/{self.task}"

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@dataclass
class HubEnvConfig(EnvConfig):
    """Base class for environments that delegate creation to a hub-hosted make_env.

    Hub environments download and execute remote code from the HF Hub.
    The hub_path points to a repository containing an env.py with a make_env function.
    """

    hub_path: str | None = None  # required: e.g., "username/repo" or "username/repo@branch:file.py"

    @property
    def gym_kwargs(self) -> dict:
        # Not used for hub environments - the hub's make_env handles everything
        return {}


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str | None = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    observation_height: int = 480
    observation_width: int = 640
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )

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
    task: str | None = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    observation_height: int = 384
    observation_width: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
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


@dataclass
class ImagePreprocessingConfig:
    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None


@dataclass
class RewardClassifierConfig:
    """Configuration for reward classification."""

    pretrained_path: str | None = None
    success_threshold: float = 0.5
    success_reward: float = 1.0


@dataclass
class InverseKinematicsConfig:
    """Configuration for inverse kinematics processing."""

    urdf_path: str | None = None
    target_frame_name: str | None = None
    end_effector_bounds: dict[str, list[float]] | None = None
    end_effector_step_sizes: dict[str, float] | None = None


@dataclass
class ObservationConfig:
    """Configuration for observation processing."""

    add_joint_velocity_to_observation: bool = False
    add_current_to_observation: bool = False
    display_cameras: bool = False


@dataclass
class GripperConfig:
    """Configuration for gripper control and penalties."""

    use_gripper: bool = True
    gripper_penalty: float = 0.0


@dataclass
class ResetConfig:
    """Configuration for environment reset behavior."""

    fixed_reset_joint_positions: Any | None = None
    reset_time_s: float = 5.0
    control_time_s: float = 20.0
    terminate_on_success: bool = True


@dataclass
class HILSerlProcessorConfig:
    """Configuration for environment processing pipeline."""

    control_mode: str = "gamepad"
    observation: ObservationConfig | None = None
    image_preprocessing: ImagePreprocessingConfig | None = None
    gripper: GripperConfig | None = None
    reset: ResetConfig | None = None
    inverse_kinematics: InverseKinematicsConfig | None = None
    reward_classifier: RewardClassifierConfig | None = None
    max_gripper_pos: float | None = 100.0


@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    processor: HILSerlProcessorConfig = field(default_factory=HILSerlProcessorConfig)

    name: str = "real_robot"

    @property
    def gym_kwargs(self) -> dict:
        return {}


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_10"  # can also choose libero_spatial, libero_object, etc.
    task_ids: list[int] | None = None
    fps: int = 30
    episode_length: int | None = None
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    init_states: bool = True
    camera_name_mapping: dict[str, str] | None = None
    observation_height: int = 360
    observation_width: int = 360
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            LIBERO_KEY_EEF_POS: f"{OBS_STATE}.eef_pos",
            LIBERO_KEY_EEF_QUAT: f"{OBS_STATE}.eef_quat",
            LIBERO_KEY_EEF_MAT: f"{OBS_STATE}.eef_mat",
            LIBERO_KEY_GRIPPER_QPOS: f"{OBS_STATE}.gripper_qpos",
            LIBERO_KEY_GRIPPER_QVEL: f"{OBS_STATE}.gripper_qvel",
            LIBERO_KEY_JOINTS_POS: f"{OBS_STATE}.joint_pos",
            LIBERO_KEY_JOINTS_VEL: f"{OBS_STATE}.joint_vel",
            LIBERO_KEY_PIXELS_AGENTVIEW: f"{OBS_IMAGES}.image",
            LIBERO_KEY_PIXELS_EYE_IN_HAND: f"{OBS_IMAGES}.image2",
        }
    )
    control_mode: str = "relative"  # or "absolute"

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features[LIBERO_KEY_PIXELS_AGENTVIEW] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features[LIBERO_KEY_PIXELS_EYE_IN_HAND] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features[LIBERO_KEY_PIXELS_AGENTVIEW] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features[LIBERO_KEY_PIXELS_EYE_IN_HAND] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features[LIBERO_KEY_EEF_POS] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(3,),
            )
            self.features[LIBERO_KEY_EEF_QUAT] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(4,),
            )
            self.features[LIBERO_KEY_EEF_MAT] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(3, 3),
            )
            self.features[LIBERO_KEY_GRIPPER_QPOS] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(2,),
            )
            self.features[LIBERO_KEY_GRIPPER_QVEL] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(2,),
            )
            self.features[LIBERO_KEY_JOINTS_POS] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(7,),
            )
            self.features[LIBERO_KEY_JOINTS_VEL] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(7,),
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {"obs_type": self.obs_type, "render_mode": self.render_mode}
        if self.task_ids is not None:
            kwargs["task_ids"] = self.task_ids
        return kwargs


@EnvConfig.register_subclass("metaworld")
@dataclass
class MetaworldEnv(EnvConfig):
    task: str = "metaworld-push-v2"  # add all tasks
    fps: int = 80
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    multitask_eval: bool = True
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "top": f"{OBS_IMAGE}",
            "pixels/top": f"{OBS_IMAGE}",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 480, 3))

        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 480, 3))

        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
        }


@EnvConfig.register_subclass("isaaclab_arena")
@dataclass
class IsaaclabArenaEnv(HubEnvConfig):
    hub_path: str = "nvidia/isaaclab-arena-envs"
    episode_length: int = 300
    num_envs: int = 1
    embodiment: str | None = "gr1_pink"
    object: str | None = "power_drill"
    mimic: bool = False
    teleop_device: str | None = None
    seed: int | None = 42
    device: str | None = "cuda:0"
    disable_fabric: bool = False
    enable_cameras: bool = False
    headless: bool = False
    enable_pinocchio: bool = True
    environment: str | None = "gr1_microwave"
    task: str | None = "Reach out to the microwave and open it."
    state_dim: int = 54
    action_dim: int = 36
    camera_height: int = 512
    camera_width: int = 512
    video: bool = False
    video_length: int = 100
    video_interval: int = 200
    # Comma-separated keys, e.g., "robot_joint_pos,left_eef_pos"
    state_keys: str = "robot_joint_pos"
    # Comma-separated keys, e.g., "robot_pov_cam_rgb,front_cam_rgb"
    # Set to None or "" for environments without cameras
    camera_keys: str | None = None
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    kwargs: dict | None = None

    def __post_init__(self):
        if self.kwargs:
            # dynamically convert kwargs to fields in the dataclass
            # NOTE! the new fields will not bee seen by the dataclass repr
            field_names = {f.name for f in fields(self)}
            for key, value in self.kwargs.items():
                if key not in field_names and key != "kwargs":
                    setattr(self, key, value)
            self.kwargs = None

        # Set action feature
        self.features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        self.features_map[ACTION] = ACTION

        # Set state feature
        self.features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,))
        self.features_map[OBS_STATE] = OBS_STATE

        # Add camera features for each camera key
        if self.enable_cameras and self.camera_keys:
            for cam_key in self.camera_keys.split(","):
                cam_key = cam_key.strip()
                if cam_key:
                    self.features[cam_key] = PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(self.camera_height, self.camera_width, 3),
                    )
                    self.features_map[cam_key] = f"{OBS_IMAGES}.{cam_key}"

    @property
    def gym_kwargs(self) -> dict:
        return {}
