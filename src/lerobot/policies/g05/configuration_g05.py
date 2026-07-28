# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Configuration for the OpenGalaxea G0.5 policy adapter."""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE

G05_SOURCE_REVISION = "b34966f387dd2ae0f003143b81494afd9213e613"
G05_HUB_REVISION = "e312be81e90c56a55bcb26b57429bd39a335b449"

G05_CAMERA_PROFILES: dict[str, tuple[str, ...]] = {
    "libero": (
        "observation.images.image",
        "observation.images.wrist_image",
    ),
    "robotwin": (
        "observation.images.head_camera",
        "observation.images.left_camera",
        "observation.images.right_camera",
    ),
    "so100": (
        "observation.images.exterior",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    ),
    "galaxea_r1lite": (
        "observation.images.head_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb",
    ),
    "galaxea_r1pro": (
        "observation.images.head_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb",
    ),
    "atomic_4": (
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_right",
    ),
}

G05_CAMERA_SIZE_PROFILES: dict[str, dict[str, tuple[int, int]]] = {
    "libero": dict.fromkeys(G05_CAMERA_PROFILES["libero"], (224, 224)),
    "robotwin": dict.fromkeys(G05_CAMERA_PROFILES["robotwin"], (256, 256)),
    "so100": dict.fromkeys(G05_CAMERA_PROFILES["so100"], (256, 256)),
    "galaxea_r1lite": dict.fromkeys(G05_CAMERA_PROFILES["galaxea_r1lite"], (256, 256)),
    "galaxea_r1pro": dict.fromkeys(G05_CAMERA_PROFILES["galaxea_r1pro"], (256, 256)),
    "atomic_4": dict.fromkeys(G05_CAMERA_PROFILES["atomic_4"], (256, 256)),
}


def make_g05_prompt_template(num_images: int, *, predict_cot: bool, flow_only: bool) -> str:
    """Reproduce the selected author SamplesBuilder template exactly."""

    images = "".join(f"<image{index}_image_!>" for index in range(num_images))
    prefix = (
        f"<chat_user_prefix>{images}<bos>"
        "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> "
        "State: <proprio_proprio_!>;"
        "<chat_user_suffix><chat_assistant_prefix>"
    )
    if predict_cot:
        action = "Action: <EOV><eos>" if flow_only else "Action: <EOV><action_action>|<eos>"
        return f"{prefix}<prompt_text_!>\n<EOC><atomic_task_text>|{action}"
    if flow_only:
        # BaseActionSamplesBuilderFMOnly intentionally has no chat wrapper.
        return (
            f"{images}<bos>Embodiment: <embodiment_text_!>; "
            "Task: <command_text_!_200> State: <proprio_proprio_!>;\n"
            "Action: <EOV><EOC><eos>"
        )
    return f"{prefix}Action: <EOV><EOC><action_action>|<eos>"


# Raw dimensions are inserted in these exact policy slots. The G0.5 shared layout is:
# left_control[9] | left_gripper[1] | right_control[9] | right_gripper[1] | lower_body[7].
# LIBERO uses only the right EEF delta and right gripper. atomic_4 is a single-arm mobile
# manipulator and therefore has a deliberately separate state/action map.
G05_EMBODIMENT_MAPPINGS: dict[str, dict[str, tuple[int, ...]]] = {
    "libero": {
        "state": (10, 11, 12, 13, 14, 15, 19),
        "action": (10, 11, 12, 13, 14, 15, 19),
    },
    "robotwin": {
        "state": (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19),
        "action": (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19),
    },
    "so100": {
        "state": (10, 11, 12, 13, 14, 15),
        "action": (10, 11, 12, 13, 14, 15),
    },
    "galaxea_r1lite": {
        "state": (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19),
        "action": (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19),
    },
    "galaxea_r1pro": {
        "state": (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 19),
        "action": (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 19),
    },
    "atomic_4": {
        # EEF relative xyz+quat -> right_control[0:7], base xyz+quat -> lower_body[0:7],
        # the two parallel-jaw qpos values -> the two one-dimensional gripper slots.
        "state": (10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 9, 19),
        # EEF delta xyz+rpy -> right_control[0:6], gripper -> right_gripper,
        # base motion[4] -> lower_body[0:4], control mode -> lower_body[4].
        "action": (10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24),
    },
}

G05_POLICY_PARTS: dict[int, dict[str, int]] = {
    20: {
        "left_control": 9,
        "left_gripper": 1,
        "right_control": 9,
        "right_gripper": 1,
    },
    27: {
        "left_control": 9,
        "left_gripper": 1,
        "right_control": 9,
        "right_gripper": 1,
        "lower_body": 7,
    },
}

_PROFILE_DEFAULTS = {
    "g05-base": ("z_score_tail_mixed", 27, 32),
    "g05-libero": ("q01_q99", 20, 32),
    "g05-robotwin20": ("q01_q99", 20, 32),
    "g05-so101": ("q01_q99", 20, 32),
}


@PreTrainedConfig.register_subclass("g05")
@dataclass
class G05Config(PreTrainedConfig):
    """LeRobot-side, checkpoint-auditable configuration for G0.5.

    ``author_model_config`` comes from the packaged checkpoint's resolved Hydra
    config. It is intentionally checkpoint state rather than a collection of
    guessed LeRobot defaults.
    """

    checkpoint_profile: str = "g05-base"
    embodiment: str = "libero"
    action_head: str = "actioncodec"  # actioncodec (AR) or flow (continuous)
    runtime_system: str = "system1"  # system1 actions, or unified system2 CoT+actions
    predict_cot: bool = False
    discrete_action: bool = True
    continuous_action: bool = False
    return_continuous_action: bool = False
    model_weights_to_bf16: bool = True

    policy_action_dim: int = 20
    policy_state_dim: int = 20
    raw_action_dim: int = 7
    raw_state_dim: int = 7
    chunk_size: int = 16
    n_action_steps: int = 16
    normalization_mode: str = "checkpoint"
    normalization_clip: tuple[float, float] | None = None
    use_relative_actions: bool = False
    relative_exclude_joints: tuple[str, ...] = ()
    action_feature_names: tuple[str, ...] = ()
    use_stepwise_action_norm: bool = False
    gripper_indices: tuple[int, ...] = (6,)
    libero_gripper_binarize: bool = False
    camera_order: tuple[str, ...] = field(default_factory=lambda: G05_CAMERA_PROFILES["libero"])
    camera_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    optional_camera_keys: tuple[str, ...] = ()
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    num_input_images: int = 0
    num_prompt_images: int = 0

    author_source_revision: str = G05_SOURCE_REVISION
    source_checkpoint_revision: str = G05_HUB_REVISION
    author_model_config: dict[str, Any] = field(default_factory=dict)
    processor_metadata: dict[str, Any] = field(default_factory=dict)
    action_codec_metadata: dict[str, Any] = field(default_factory=dict)
    prompt_template: str = ""

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )
    optimizer_lr: float = 8e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    optimizer_backbone_lr_multiplier: float = 1.0
    optimizer_vision_lr_multiplier: float = 1.0
    optimizer_apply_decay_on_norm_and_bias: bool = False
    scheduler_warmup_steps: int = 500

    def __post_init__(self) -> None:
        super().__post_init__()
        self.camera_order = tuple(self.camera_order)
        self.camera_sizes = {key: tuple(size) for key, size in (self.camera_sizes or {}).items()}
        self.optional_camera_keys = tuple(self.optional_camera_keys)
        if self.normalization_clip is not None:
            self.normalization_clip = tuple(self.normalization_clip)
            if len(self.normalization_clip) != 2 or self.normalization_clip[0] >= self.normalization_clip[1]:
                raise ValueError("normalization_clip must be an increasing (minimum, maximum) pair.")
        self.relative_exclude_joints = tuple(self.relative_exclude_joints)
        self.action_feature_names = tuple(self.action_feature_names)
        if not self.camera_sizes and self.embodiment in G05_CAMERA_SIZE_PROFILES:
            self.camera_sizes = G05_CAMERA_SIZE_PROFILES[self.embodiment].copy()
        if self.num_input_images == 0:
            self.num_input_images = len(self.camera_order) * self.n_obs_steps
        if self.num_prompt_images == 0:
            self.num_prompt_images = len(self.camera_order)
        if not self.prompt_template:
            samples_builder = self.processor_metadata.get("samples_builder") or {}
            if isinstance(samples_builder, dict):
                samples_builder_target = str(samples_builder.get("_target_", ""))
            else:
                samples_builder_target = str(samples_builder)
            self.prompt_template = make_g05_prompt_template(
                self.num_prompt_images,
                predict_cot=self.predict_cot,
                flow_only=samples_builder_target.endswith("FMOnly"),
            )
        if self.checkpoint_profile not in _PROFILE_DEFAULTS and self.checkpoint_profile != "custom":
            raise ValueError(
                f"Unknown G0.5 checkpoint_profile={self.checkpoint_profile!r}; "
                f"expected one of {sorted(_PROFILE_DEFAULTS)} or 'custom'."
            )
        if self.action_head not in {"actioncodec", "flow"}:
            raise ValueError("action_head must be 'actioncodec' or 'flow'.")
        if self.runtime_system not in {"system1", "system2"}:
            raise ValueError("runtime_system must be 'system1' or 'system2'.")
        if self.runtime_system == "system2" and not self.predict_cot:
            raise ValueError("G0.5 System 2 requires predict_cot=True in the packaged checkpoint.")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be between 1 and chunk_size.")
        if self.action_head == "actioncodec" and not self.discrete_action:
            raise ValueError("The ActionCodec runtime requires discrete_action=True.")
        if self.action_head == "flow" and not self.continuous_action:
            raise ValueError("The flow runtime requires continuous_action=True.")
        if self.action_head == "flow" and not self.return_continuous_action:
            raise ValueError("The flow runtime requires return_continuous_action=True.")
        if self.policy_action_dim not in G05_POLICY_PARTS:
            raise ValueError(
                f"No named G0.5 shared action layout for policy_action_dim={self.policy_action_dim}."
            )
        if not (self.discrete_action or self.continuous_action):
            raise ValueError("At least one G0.5 action path must be enabled.")
        if self.embodiment not in G05_EMBODIMENT_MAPPINGS:
            raise ValueError(f"No named G0.5 embodiment mapping for {self.embodiment!r}.")
        if self.embodiment == "atomic_4":
            if self.policy_action_dim < 27 or self.policy_state_dim < 27:
                raise ValueError(
                    "atomic_4 includes mobile-base/control-mode semantics and requires the 27D "
                    "G0.5 shared layout; a 20D LIBERO checkpoint is incompatible."
                )
            if self.raw_action_dim != 12 or self.raw_state_dim != 16:
                raise ValueError("atomic_4 requires raw_state_dim=16 and raw_action_dim=12.")
        mapping = G05_EMBODIMENT_MAPPINGS.get(self.embodiment)
        if mapping is not None:
            if len(mapping["state"]) != self.raw_state_dim:
                raise ValueError("raw_state_dim does not match the selected embodiment mapping.")
            if len(mapping["action"]) != self.raw_action_dim:
                raise ValueError("raw_action_dim does not match the selected embodiment mapping.")
            if max(mapping["state"]) >= self.policy_state_dim:
                raise ValueError("Selected state mapping exceeds policy_state_dim.")
            if max(mapping["action"]) >= self.policy_action_dim:
                raise ValueError("Selected action mapping exceeds policy_action_dim.")
        if self.normalization_mode not in {
            "checkpoint",
            "q01_q99",
            "z_score",
            "z_score_tail_mixed",
            "identity",
        }:
            raise ValueError(
                "normalization_mode must be checkpoint, q01_q99, z_score, z_score_tail_mixed, or identity."
            )
        if self.checkpoint_profile == "g05-libero":
            if self.chunk_size != 32 or self.normalization_mode != "q01_q99":
                raise ValueError("g05-libero requires a 32-step chunk and q01/q99 normalization.")
            if self.action_head != "flow":
                raise ValueError("The released g05-libero config enables only the continuous flow path.")
            if not self.libero_gripper_binarize:
                raise ValueError("g05-libero requires the official binary gripper command transform.")
        expected_cameras = G05_CAMERA_PROFILES.get(self.embodiment)
        if expected_cameras is not None and tuple(self.camera_order) != expected_cameras:
            raise ValueError(
                f"{self.embodiment} camera_order must be {expected_cameras}, got {self.camera_order}."
            )
        if set(self.camera_sizes) != set(self.camera_order):
            raise ValueError("camera_sizes must contain exactly the ordered checkpoint camera keys.")
        if not set(self.optional_camera_keys) <= set(self.camera_order):
            raise ValueError("optional_camera_keys must be a subset of camera_order.")
        if self.num_input_images != len(self.camera_order) * self.n_obs_steps:
            raise ValueError(
                "num_input_images must equal len(camera_order) * n_obs_steps for the selected checkpoint."
            )
        if self.num_prompt_images != len(self.camera_order):
            raise ValueError("num_prompt_images must equal len(camera_order).")
        if any(len(size) != 2 or min(size) <= 0 for size in self.camera_sizes.values()):
            raise ValueError("Every G0.5 camera size must be a positive (height, width) pair.")
        if len(self.image_mean) != 3 or len(self.image_std) != 3 or min(self.image_std) <= 0:
            raise ValueError("G0.5 image_mean/image_std must be three channels with positive std.")

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}
        state = self.input_features.get(OBS_STATE)
        if state is not None and state.shape[-1] != self.raw_state_dim:
            raise ValueError(
                f"G0.5 {self.embodiment} expects {self.raw_state_dim} raw state dimensions, "
                f"got {state.shape[-1]}."
            )
        action = self.output_features.get(ACTION)
        if action is not None and action.shape[-1] != self.raw_action_dim:
            raise ValueError(
                f"G0.5 {self.embodiment} expects {self.raw_action_dim} raw action dimensions, "
                f"got {action.shape[-1]}."
            )
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.raw_state_dim,)
            )
        for key in self.camera_order:
            if key not in self.input_features:
                height, width = self.camera_sizes[key]
                self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(self.raw_action_dim,)
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(-(self.n_obs_steps - 1), 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
