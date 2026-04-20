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

from __future__ import annotations

import abc
import importlib
from dataclasses import dataclass, field, fields
from typing import Any

import draccus
import gymnasium as gym
from gymnasium.envs.registration import registry as gym_registry

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.processor import IsaaclabArenaProcessorStep, LiberoProcessorStep, PolicyProcessorPipeline
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


def _make_vec_env_cls(use_async: bool, n_envs: int):
    """Return the right VectorEnv constructor."""
    if use_async and n_envs > 1:
        return gym.vector.AsyncVectorEnv
    return gym.vector.SyncVectorEnv


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

    def create_envs(
        self,
        n_envs: int,
        use_async_envs: bool = False,
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """Create {suite: {task_id: VectorEnv}}.

        Default: single-task env via gym.make(). Multi-task benchmarks override.
        AsyncVectorEnv is the default for n_envs > 1; auto-downgraded to Sync for n_envs=1.
        """
        env_cls = gym.vector.AsyncVectorEnv if (use_async_envs and n_envs > 1) else gym.vector.SyncVectorEnv

        if self.gym_id not in gym_registry:
            print(f"gym id '{self.gym_id}' not found, attempting to import '{self.package_name}'...")
            try:
                importlib.import_module(self.package_name)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Package '{self.package_name}' required for env '{self.type}' not found. "
                    f"Please install it or check PYTHONPATH."
                ) from e

            if self.gym_id not in gym_registry:
                raise gym.error.NameNotFound(
                    f"Environment '{self.gym_id}' not registered even after importing '{self.package_name}'."
                )

        def _make_one():
            return gym.make(self.gym_id, disable_env_checker=self.disable_env_checker, **self.gym_kwargs)

        extra_kwargs: dict = {}
        if env_cls is gym.vector.AsyncVectorEnv:
            extra_kwargs["context"] = "forkserver"
        try:
            from gymnasium.vector import AutoresetMode

            vec = env_cls(
                [_make_one for _ in range(n_envs)], autoreset_mode=AutoresetMode.SAME_STEP, **extra_kwargs
            )
        except ImportError:
            vec = env_cls([_make_one for _ in range(n_envs)], **extra_kwargs)
        return {self.type: {0: vec}}

    def get_env_processors(self):
        """Return (preprocessor, postprocessor) for this env. Default: identity."""
        return PolicyProcessorPipeline(steps=[]), PolicyProcessorPipeline(steps=[])


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
    add_ee_pose_to_observation: bool = False
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

        if self.camera_name_mapping is not None:
            mapped_agentview = self.camera_name_mapping.get("agentview_image", "image")
            mapped_eye_in_hand = self.camera_name_mapping.get("robot0_eye_in_hand_image", "image2")
            self.features_map[LIBERO_KEY_PIXELS_AGENTVIEW] = f"{OBS_IMAGES}.{mapped_agentview}"
            self.features_map[LIBERO_KEY_PIXELS_EYE_IN_HAND] = f"{OBS_IMAGES}.{mapped_eye_in_hand}"

    @property
    def gym_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
        }
        if self.task_ids is not None:
            kwargs["task_ids"] = self.task_ids
        return kwargs

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        from .libero import create_libero_envs

        if self.task is None:
            raise ValueError("LiberoEnv requires a task to be specified")
        env_cls = _make_vec_env_cls(use_async_envs, n_envs)
        return create_libero_envs(
            task=self.task,
            n_envs=n_envs,
            camera_name=self.camera_name,
            init_states=self.init_states,
            gym_kwargs=self.gym_kwargs,
            env_cls=env_cls,
            control_mode=self.control_mode,
            episode_length=self.episode_length,
            camera_name_mapping=self.camera_name_mapping,
        )

    def get_env_processors(self):
        return (
            PolicyProcessorPipeline(steps=[LiberoProcessorStep()]),
            PolicyProcessorPipeline(steps=[]),
        )


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

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        from .metaworld import create_metaworld_envs

        if self.task is None:
            raise ValueError("MetaWorld requires a task to be specified")
        env_cls = _make_vec_env_cls(use_async_envs, n_envs)
        return create_metaworld_envs(
            task=self.task,
            n_envs=n_envs,
            gym_kwargs=self.gym_kwargs,
            env_cls=env_cls,
        )


@EnvConfig.register_subclass("robocasa")
@dataclass
class RoboCasaEnv(EnvConfig):
    task: str = "CloseFridge"
    fps: int = 20
    episode_length: int = 1000
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "robot0_agentview_left,robot0_eye_in_hand,robot0_agentview_right"
    observation_height: int = 256
    observation_width: int = 256
    visualization_height: int = 512
    visualization_width: int = 512
    split: str | None = None
    # Object-mesh registries to sample from. Upstream default is
    # ("objaverse", "lightwheel"), but objaverse is ~30GB and the CI image
    # only ships the lightwheel pack. Override to include objaverse once
    # you've run `python -m robocasa.scripts.download_kitchen_assets
    # --type objaverse` locally.
    obj_registries: list[str] = field(default_factory=lambda: ["lightwheel"])
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))}
    )
    features_map: dict[str, str] = field(default_factory=lambda: {ACTION: ACTION, "agent_pos": OBS_STATE})

    def __post_init__(self):
        if self.obs_type not in ("pixels", "pixels_agent_pos"):
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

        # Preserve raw RoboCasa camera names end-to-end (e.g.
        # `observation.images.robot0_agentview_left`). This matches the
        # naming convention used by the RoboCasa datasets on the Hub, so
        # trained policies don't need a `--rename_map` at eval time.
        cams = [c.strip() for c in self.camera_name.split(",") if c.strip()]
        for cam in cams:
            self.features[f"pixels/{cam}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.observation_height, self.observation_width, 3),
            )
            self.features_map[f"pixels/{cam}"] = f"{OBS_IMAGES}.{cam}"

        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
            "visualization_height": self.visualization_height,
            "visualization_width": self.visualization_width,
        }
        if self.split is not None:
            kwargs["split"] = self.split
        return kwargs

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        from .robocasa import create_robocasa_envs

        if self.task is None:
            raise ValueError("RoboCasaEnv requires a task to be specified")
        env_cls = _make_vec_env_cls(use_async_envs, n_envs)
        return create_robocasa_envs(
            task=self.task,
            n_envs=n_envs,
            camera_name=self.camera_name,
            gym_kwargs=self.gym_kwargs,
            env_cls=env_cls,
            episode_length=self.episode_length,
            obj_registries=tuple(self.obj_registries),
        )


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

    def get_env_processors(self):
        state_keys = tuple(k.strip() for k in (self.state_keys or "").split(",") if k.strip())
        camera_keys = tuple(k.strip() for k in (self.camera_keys or "").split(",") if k.strip())
        if not state_keys and not camera_keys:
            raise ValueError("At least one of state_keys or camera_keys must be specified.")
        return (
            PolicyProcessorPipeline(
                steps=[IsaaclabArenaProcessorStep(state_keys=state_keys, camera_keys=camera_keys)]
            ),
            PolicyProcessorPipeline(steps=[]),
        )
