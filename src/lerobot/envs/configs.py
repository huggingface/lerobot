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
import logging
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any

import draccus
import gymnasium as gym

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.processor import IsaaclabArenaProcessorStep, PolicyProcessorPipeline
from lerobot.robots import RobotConfig
from lerobot.sims.backend import BackendConfig
from lerobot.sims.conventions import resolve, served_differences
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)

# Per-tick bound for a served environment; opening and resetting it get the same.
SERVED_TIMEOUT_S = 120.0


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    # Filled from the environment's description when it is created: its canonical observation names.
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    max_parallel_tasks: int = 1
    disable_env_checker: bool = True
    # Where `lerobot-env-server` serves this environment; unset, it runs in this process.
    endpoint: str | None = None

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

    def sim_settings(self) -> dict[str, Any]:
        """The simulator this config describes, as a server YAML's ``sim`` section.

        Settings left out (fps, control, semantics, component names) follow the simulator's conventions.
        By default, the environment Gymnasium registers as ``gym_id``, built with ``gym_kwargs`` and
        commanded with deltas; override it to declare other conventions.
        """
        return {
            "type": "gym",
            "task": self.task,
            "fps": self.fps,
            "control": "delta",
            "semantics": f"{self.type}-delta-v1",
            "kwargs": {"gym_id": self.gym_id, "gym_kwargs": self.gym_kwargs},
        }

    def backend_config(self) -> BackendConfig:
        """The resolved simulator settings, as the env server and the in-process client run them."""
        return BackendConfig(**resolve(self.sim_settings()))

    def create_envs(
        self,
        n_envs: int,
        use_async_envs: bool = False,
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """Create {suite: {task_id: BatchedEnv}} of ``n_envs`` worlds each, in-process or served.

        ``use_async_envs`` runs each world in its own process, in-process or on the server, so a batch
        steps in parallel.
        """
        from lerobot.env_server.client import EnvClient
        from lerobot.env_server.in_process import InProcessEnvClient

        from .batched import BatchedEnv

        settings = self.sim_settings()
        endpoint = self.endpoint
        if endpoint is not None:
            if self.max_parallel_tasks != 1:
                raise ValueError("A served environment runs one task at a time")
            probe = EnvClient(endpoint, self.type, SERVED_TIMEOUT_S)
            try:
                descriptor = probe.describe()
            finally:
                probe.close()
            check_served(descriptor.config, settings, endpoint)

            def client() -> EnvClient | InProcessEnvClient:
                return EnvClient(endpoint, self.type, SERVED_TIMEOUT_S)
        else:
            sim = self.backend_config()
            local = InProcessEnvClient(sim)
            try:
                descriptor = local.describe()
            finally:
                local.close()

            def client() -> EnvClient | InProcessEnvClient:
                return InProcessEnvClient(sim, descriptor=descriptor)

        self.fps = int(descriptor.fps)
        self.features = {
            f.name: PolicyFeature(
                type=FeatureType.VISUAL if f.kind == "rgb" else FeatureType.STATE, shape=f.shape
            )
            for f in descriptor.features
        }
        self.features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=descriptor.action_feature.shape)
        # Canonical keys reach the policy unchanged; --rename_map aliases them like any env.
        self.features_map = {name: name for name in self.features}
        self.descriptor = descriptor
        groups = self.task.split(",") if self.task else list(descriptor.tasks)
        result = {}
        for group in groups:
            if group not in descriptor.tasks:
                raise ValueError(f"Unknown task {group!r}; this environment has {sorted(descriptor.tasks)}")
            result[group] = {
                int(i): BatchedEnv(client(), n_envs, group, int(i), descriptor, use_async_envs and n_envs > 1)
                for i in descriptor.tasks[group]
            }
        return result

    def get_env_processors(self):
        """Return (preprocessor, postprocessor) for this env. Default: identity."""
        return PolicyProcessorPipeline(steps=[]), PolicyProcessorPipeline(steps=[])


def check_served(served: dict[str, Any], expected: dict[str, Any], endpoint: str) -> None:
    """Refuse a server that runs another environment than the one expected; warn on other options."""
    errors, warnings = served_differences(served, expected)
    if errors:
        raise ValueError(
            f"The env server at {endpoint} runs another environment than this command describes "
            f"({'; '.join(errors)}). Pass the same --env options as the server's configuration."
        )
    for warning in warnings:
        logger.warning("The env server at %s differs in %s", endpoint, warning)


@EnvConfig.register_subclass("envhub")
@dataclass
class HubEnvConfig(EnvConfig):
    """An EnvHub repository: its ``env.py`` defines the environment and ``make_env``.

    Running it executes the repository's code; ``trust_remote_code`` is that consent.
    """

    hub_path: str | None = None  # required: e.g., "username/repo" or "username/repo@branch:file.py"
    trust_remote_code: bool = False

    @property
    def gym_kwargs(self) -> dict:
        # Not used for hub environments - the hub's make_env handles everything
        return {}

    def sim_settings(self) -> dict[str, Any]:
        if not self.hub_path:
            raise ValueError("--env.hub_path is required for an EnvHub environment")
        # The suites come from the repository; --env.task selects among them.
        return {
            "type": "envhub",
            "task": "",
            "kwargs": {"hub_path": self.hub_path, "trust_remote_code": self.trust_remote_code},
        }


def _gym_settings(config: EnvConfig, state_names: tuple[str, ...]) -> dict[str, Any]:
    """An environment ``gym.make`` builds, commanded with absolute positions of its observed state."""
    return {
        "type": "gym",
        "task": config.task,
        "fps": config.fps,
        "control": "position",
        "semantics": f"{config.type}-position-v1",
        "state_names": state_names,
        "action_names": state_names,
        "kwargs": {"gym_id": config.gym_id, "gym_kwargs": config.gym_kwargs},
    }


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str | None = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }

    def sim_settings(self) -> dict[str, Any]:
        return _gym_settings(self, tuple(f"joint_{i}.pos" for i in range(14)))


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

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }

    def sim_settings(self) -> dict[str, Any]:
        return _gym_settings(self, ("agent.x", "agent.y"))


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

    fixed_reset_joint_positions: list[float] | None = None
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

    def sim_settings(self) -> dict[str, Any]:
        raise NotImplementedError("gym_manipulator wraps a robot for RL; it does not describe a simulator")


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_10"  # can also choose libero_spatial, libero_object, etc.
    task_ids: list[int] | None = None
    fps: int = 20  # Must match robosuite's default control_freq (20 Hz)
    episode_length: int | None = None
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    init_states: bool = True
    hard_reset: bool = True
    # Raw camera name -> observation name; default agentview_image -> image, robot0_eye_in_hand_image -> image2.
    camera_name_mapping: dict[str, str] | None = None
    observation_height: int = 360
    observation_width: int = 360
    is_libero_plus: bool = False
    control_mode: str = "relative"  # or "absolute"

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if not self.hard_reset and not self.init_states:
            raise ValueError("hard_reset=False requires init_states=True")
        if self.obs_type not in ("pixels", "pixels_agent_pos"):
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
            "control_freq": self.fps,
            "hard_reset": self.hard_reset,
        }
        if self.task_ids is not None:
            kwargs["task_ids"] = self.task_ids
        return kwargs

    def sim_settings(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "task": self.task,
            "fps": self.fps,
            "kwargs": {
                "gym_kwargs": self.gym_kwargs,
                "camera_name": self.camera_name,
                "init_states": self.init_states,
                "control_mode": self.control_mode,
                "episode_length": self.episode_length,
                "camera_name_mapping": self.camera_name_mapping,
                "is_libero_plus": self.is_libero_plus,
            },
        }


@EnvConfig.register_subclass("metaworld")
@dataclass
class MetaworldEnv(EnvConfig):
    task: str = "metaworld-push-v3"  # add all tasks
    fps: int = 80
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"

    def __post_init__(self):
        if self.obs_type not in ("pixels", "pixels_agent_pos"):
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
        }

    def sim_settings(self) -> dict[str, Any]:
        return {"type": "metaworld", "task": self.task, "kwargs": {"gym_kwargs": self.gym_kwargs}}


@EnvConfig.register_subclass("robocasa")
@dataclass
class RoboCasaEnv(EnvConfig):
    task: str = "CloseFridge"
    fps: int = 20
    episode_length: int | None = None
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    # Raw RoboCasa camera names are the observation names (`observation.images.robot0_agentview_left`), as
    # in the RoboCasa datasets on the Hub.
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

    def __post_init__(self):
        if self.obs_type not in ("pixels", "pixels_agent_pos"):
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

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

    def sim_settings(self) -> dict[str, Any]:
        return {
            "type": "robocasa",
            "task": self.task,
            "kwargs": {
                "gym_kwargs": self.gym_kwargs,
                "camera_name": self.camera_name,
                "episode_length": self.episode_length,
                "obj_registries": list(self.obj_registries),
            },
        }


@EnvConfig.register_subclass("vlabench")
@dataclass
class VLABenchEnv(EnvConfig):
    task: str = "select_fruit"
    fps: int = 10
    episode_length: int = 500
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    render_resolution: tuple[int, int] = (480, 480)
    robot: str = "franka"
    action_mode: str = "eef"

    def __post_init__(self):
        if self.obs_type not in ("pixels", "pixels_agent_pos"):
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "render_resolution": list(self.render_resolution),
            "robot": self.robot,
            "max_episode_steps": self.episode_length,
            "action_mode": self.action_mode,
        }

    def sim_settings(self) -> dict[str, Any]:
        return {"type": "vlabench", "task": self.task, "kwargs": {"gym_kwargs": self.gym_kwargs}}


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

    def sim_settings(self) -> dict[str, Any]:
        raise NotImplementedError(f"--env.type={self.type} runs through its repository's make_env only")

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        from .factory import make_hub_envs

        assert self.hub_path is not None
        return make_hub_envs(self.hub_path, n_envs, use_async_envs, self.trust_remote_code, cfg=self)

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


class G1EndEffector(StrEnum):
    """What the G1's arms carry, by hardware name.

    Shared with UnitreeG1Config, which owns the robot-level flag.
    """

    DUMMY = "dummy"  # bare wrists
    DEX1 = "dex1"  # parallel grippers
    DEX3 = "dex3"  # three-finger hands

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(f"`end_effector` is expected to be in {list(cls)}, but {value} is provided.")


@EnvConfig.register_subclass("unitree_g1_mujoco")
@dataclass
class UnitreeG1MujocoEnv(HubEnvConfig):
    """Config for the MuJoCo simulation of the Unitree G1.

    The end effector selects the MuJoCo model, and with it the finger actuators and the
    wrist cameras that exist: "dummy" for bare wrists, "dex1" for the parallel grippers,
    "dex3" for the three-finger hands.

    The sim renders either its cameras or its window, never both: the offscreen contexts
    are bound to the thread that creates them, which is not the one driving the viewer.
    `onscreen` is therefore resolved from `publish_images` unless it is set explicitly.
    """

    hub_path: str = "lerobot/unitree-g1-mujoco"
    end_effector: G1EndEffector = G1EndEffector.DEX1
    publish_images: bool = True
    camera_port: int = 5555
    onscreen: bool | None = None

    def __post_init__(self) -> None:
        self.end_effector = G1EndEffector(self.end_effector)
        if self.onscreen is None:
            self.onscreen = not self.publish_images
        elif self.onscreen and self.publish_images:
            raise ValueError(
                "The G1 sim cannot publish camera images and open its viewer in the same "
                "process. Set publish_images=False to watch the window, or onscreen=False "
                "to take the image stream."
            )

    def sim_settings(self) -> dict[str, Any]:
        raise NotImplementedError(f"--env.type={self.type} runs through its repository's make_env only")

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        from .factory import make_hub_envs

        assert self.hub_path is not None
        return make_hub_envs(self.hub_path, n_envs, use_async_envs, self.trust_remote_code, cfg=self)


@EnvConfig.register_subclass("libero_plus")
@dataclass
class LiberoPlusEnv(LiberoEnv):
    """Config for LIBERO-plus robustness benchmark evaluation.

    LIBERO-plus extends LIBERO with 7 perturbation dimensions (camera viewpoints,
    object layouts, robot initial states, language instructions, lighting, background
    textures, sensor noise) producing ~10k task variants.

    The gym interface is identical to LIBERO so this class reuses ``LiberoEnv``
    entirely — only the registered name and default task suite differ.

    Install: see the libero_plus_native stage of docker/Dockerfile.sim — LIBERO-plus ships
    as a namespace package from a git fork and must be cloned + PYTHONPATH'd
    rather than installed as a pyproject extra.

    See Also:
        https://github.com/sylvestf/LIBERO-plus
    """

    task: str = "libero_spatial"
    is_libero_plus: bool = True


@EnvConfig.register_subclass("robocerebra")
@dataclass
class RoboCerebraEnv(LiberoEnv):
    """RoboCerebra long-horizon tasks on LIBERO scenes, with its datasets' camera names and resolution."""

    task: str = "libero_10"
    observation_height: int = 256
    observation_width: int = 256
    camera_name_mapping: dict[str, str] | None = field(
        default_factory=lambda: {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}
    )


@EnvConfig.register_subclass("robotwin")
@dataclass
class RoboTwinEnvConfig(EnvConfig):
    """Configuration for RoboTwin 2.0 benchmark environments.

    RoboTwin 2.0 is a dual-arm manipulation benchmark with 50 tasks built on the
    SAPIEN simulator. The robot is an Aloha-AgileX bimanual platform with 14 DOF
    (7 per arm). All three cameras are enabled by default.

    See: https://robotwin-platform.github.io
    Dataset: https://huggingface.co/datasets/lerobot/robotwin_unified
    """

    task: str = "beat_block_hammer"  # single task or comma-separated list
    fps: int = 25
    episode_length: int = 1200
    # Available cameras from RoboTwin's aloha-agilex embodiment: head_camera
    # (torso-mounted) + left_camera / right_camera (wrists).
    camera_names: str = "head_camera,left_camera,right_camera"
    # Match the D435 dims in task_config/demo_clean.yml (_camera_config.yml).
    observation_height: int = 240
    observation_width: int = 320
    # "joint": 14-d joint-space control. "ee": 16-d end-effector-pose deltas executed via CuRobo IK
    # (for world-model policies like LingBot-VA that predict per-arm xyz+quaternion+gripper poses).
    action_mode: str = "joint"

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def sim_settings(self) -> dict[str, Any]:
        return {
            "type": "robotwin",
            "task": self.task,
            "kwargs": {
                "camera_names": [c.strip() for c in self.camera_names.split(",") if c.strip()],
                "observation_height": self.observation_height,
                "observation_width": self.observation_width,
                "episode_length": self.episode_length,
                "action_mode": self.action_mode,
            },
        }


@EnvConfig.register_subclass("robomme")
@dataclass
class RoboMMEEnv(EnvConfig):
    """RoboMME memory-augmented manipulation benchmark (ManiSkill/SAPIEN).

    16 tasks across 4 suites: Counting, Permanence, Reference, Imitation.
    Dataset: lerobot/robomme (LeRobot v3.0, 1,600 episodes).
    Benchmark: https://github.com/RoboMME/robomme_benchmark

    Requires the `robomme` git package installed separately (Linux only);
    see the robomme_native stage of docker/Dockerfile.sim for the canonical install.
    """

    task: str = "PickXtimes"
    fps: int = 10
    episode_length: int = 300
    action_space: str = "joint_angle"  # or "ee_pose" (7-D)
    dataset_split: str = "test"  # "train" | "val" | "test"
    task_ids: list[int] | None = None
    front_camera_name: str = "camera1"
    wrist_camera_name: str = "camera2"

    def __post_init__(self):
        if self.front_camera_name == self.wrist_camera_name:
            raise ValueError("RoboMME front and wrist camera names must be distinct.")

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def sim_settings(self) -> dict[str, Any]:
        return {
            "type": "robomme",
            "task": self.task,
            "kwargs": {
                "action_space_type": self.action_space,
                "dataset": self.dataset_split,
                "episode_length": self.episode_length,
                "task_ids": self.task_ids,
                "front_camera_name": self.front_camera_name,
                "wrist_camera_name": self.wrist_camera_name,
            },
        }
