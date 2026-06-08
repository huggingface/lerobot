#!/usr/bin/env python

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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

COSMOS3_LEFT_IMAGE = f"{OBS_IMAGES}.over_shoulder_left_camera"
COSMOS3_RIGHT_IMAGE = f"{OBS_IMAGES}.over_shoulder_right_camera"
COSMOS3_WRIST_IMAGE = f"{OBS_IMAGES}.wrist_cam"

COSMOS3_DROID_DOMAIN_ID = 8
COSMOS3_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. "
    "The bottom row contains two horizontally concatenated third-person perspective views of the scene from "
    "opposite sides, with the robot visible."
)


@PreTrainedConfig.register_subclass("cosmos3")
@dataclass
class Cosmos3Config(PreTrainedConfig):
    """Configuration for LeRobot-format Cosmos3 policy checkpoints."""

    # Converted LeRobot checkpoints store component configs here and load all
    # model weights through PreTrainedPolicy.from_pretrained(model.safetensors).
    text_processor_name_or_path: str | None = None
    transformer_config: dict[str, Any] | None = None
    vae_config: dict[str, Any] | None = None
    scheduler_config: dict[str, Any] | None = None

    # Public model-level controls.
    freeze_vae: bool = True
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"
    local_files_only: bool = True

    # RoboLab/DROID policy contract.
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32
    raw_action_dim: int = 8
    max_action_dim: int = 64
    # Upper bound on the proprioceptive state width accepted by the processor.
    # Wider state raises; narrower state is zero-padded. Defaults to raw_action_dim.
    max_state_dim: int | None = None
    joint_position_dim: int = 7
    gripper_position_dim: int = 1
    use_state: bool = True
    history_length: int = 1
    action_space: str = "joint_pos"
    invert_gripper: bool = True

    # Cosmos3 action generation settings matching the RoboLab policy server defaults.
    domain_name: str = "droid_lerobot"
    domain_id: int = COSMOS3_DROID_DOMAIN_ID
    eos_token_id: int = 151645
    start_of_generation_token_id: int = 151652
    mode: str = "policy"
    viewpoint: str = "concat_view"
    additional_view_description: str = COSMOS3_CONCAT_VIEW_DESCRIPTION
    conditioning_fps: float = 15.0
    resolution_tier: int = 480
    guidance_scale: float = 3.0
    num_inference_steps: int = 4
    shift: float = 5.0
    seed: int = 0
    deterministic_seed: bool = False
    generate_video: bool = False
    output_type: str = "latent"
    train_time_video_distribution: str = "waver"
    video_loss_weight: float = 10.0
    action_loss_weight: float = 10.0
    normalize_loss_by_active: bool = False

    # Multi-view image composition. `image_keys[0]` is rendered as the primary
    # (top) view and the remaining views tile the bottom row. The contract is
    # `num_views` views: fewer observed views are zero-padded up to `num_views`.
    # RoboLab's three DROID cameras are the default, but any ordered list works.
    image_keys: list[str] = field(
        default_factory=lambda: [COSMOS3_WRIST_IMAGE, COSMOS3_LEFT_IMAGE, COSMOS3_RIGHT_IMAGE]
    )
    num_views: int = 3
    image_height: int = 360
    image_width: int = 640
    composed_image_height: int = 540
    composed_image_width: int = 640
    prompt_key: str = "task"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Conservative training defaults until task-specific recipes are added.
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"Invalid dtype: {self.dtype!r}")
        if self.mode != "policy":
            raise ValueError("Cosmos3Config currently supports only action mode='policy'.")
        if self.action_space != "joint_pos":
            raise ValueError("Cosmos3Config currently supports only action_space='joint_pos'.")
        if self.history_length < int(self.use_state):
            raise ValueError("history_length must be at least 1 when use_state=True.")
        if self.raw_action_dim != self.joint_position_dim + self.gripper_position_dim:
            raise ValueError("raw_action_dim must equal joint_position_dim + gripper_position_dim.")
        if self.max_state_dim is None:
            self.max_state_dim = self.raw_action_dim
        if self.max_state_dim > self.raw_action_dim:
            raise ValueError(
                f"max_state_dim ({self.max_state_dim}) cannot exceed raw_action_dim ({self.raw_action_dim})."
            )
        if self.num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {self.num_views}.")
        if len(self.image_keys) > self.num_views:
            raise ValueError(
                f"Got {len(self.image_keys)} image_keys but num_views={self.num_views}; "
                "raise num_views or shorten image_keys."
            )

    @property
    def transformer_backbone_config(self) -> dict[str, Any]:
        if self.transformer_config is None:
            raise ValueError(
                "Cosmos3Config.transformer_config is required. "
                "Load a converted LeRobot Cosmos3 checkpoint or provide the serialized transformer config."
            )
        config = dict(self.transformer_config)
        config.setdefault("action_dim", self.max_action_dim)
        config.setdefault("action_gen", True)
        return config

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}

        default_image_shape = (3, self.image_height, self.image_width)
        for image_key in self.image_keys:
            self.input_features.setdefault(
                image_key,
                PolicyFeature(type=FeatureType.VISUAL, shape=default_image_shape),
            )

        self.input_features.setdefault(
            OBS_STATE,
            PolicyFeature(type=FeatureType.STATE, shape=(self.raw_action_dim,)),
        )
        self.output_features.setdefault(
            ACTION,
            PolicyFeature(type=FeatureType.ACTION, shape=(self.raw_action_dim,)),
        )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size + 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
