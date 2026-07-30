#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import Qwen2_5_VLConfig


@PreTrainedConfig.register_subclass("wall_oss_05")
@dataclass
class WallOSS05Config(PreTrainedConfig):
    """LeRobot configuration for the Wall-OSS-0.5 release.

    The 0.5 checkpoint is deliberately a separate policy type from ``wall_x``.
    Its 26D action/state contract, 1024-wide action expert, quantile-range
    normalizers, and x-prediction flow objective are not compatible with the
    older ``wall-oss-flow`` defaults.
    """

    pretrained_name_or_path: str = "lerobot/wall-oss-0.5"
    vlm_config: dict[str, Any] | None = None
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32
    max_action_dim: int = 26
    max_state_dim: int = 26
    num_inference_timesteps: int = 10
    action_branch: str = "flow"

    # LeRobot image key -> model camera slot. Insertion order is prompt order.
    camera_key_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.face_view": "face_view",
            "observation.images.right_wrist_view": "right_wrist_view",
        }
    )

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    optimizer_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 200_000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_action_dim != 26 or self.max_state_dim != 26:
            raise ValueError("Wall-OSS-0.5 has a fixed 26D action/state contract.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size.")
        if self.action_branch != "flow":
            raise ValueError(
                "The public checkpoint exposes only continuous flow deployment. "
                "The AR normalizer is preserved in the checkpoint, but AR inference is not exposed."
            )
        if self.num_inference_timesteps != 10:
            raise ValueError("The checkpoint uses exactly 10 Euler flow steps.")

    @property
    def vlm_backbone_config(self) -> Qwen2_5_VLConfig:
        require_package("transformers", extra="wall_oss_05")
        if self.vlm_config is None:
            raise ValueError("WallOSS05Config.vlm_config must be loaded from a pretrained checkpoint.")
        return Qwen2_5_VLConfig.from_dict(deepcopy(self.vlm_config))

    def validate_features(self) -> None:
        image_features = {key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL}
        missing_cameras = sorted(set(self.camera_key_mapping) - image_features)
        if missing_cameras:
            raise ValueError(
                f"Configured Wall-OSS-0.5 cameras are absent from input_features: {missing_cameras}"
            )
        if OBS_STATE not in self.input_features:
            raise ValueError("Wall-OSS-0.5 requires observation.state.")
        if ACTION not in self.output_features:
            raise ValueError("Wall-OSS-0.5 requires an action output feature.")

        state_dim = self.input_features[OBS_STATE].shape[-1]
        action_dim = self.output_features[ACTION].shape[-1]
        if state_dim != self.max_state_dim or action_dim != self.max_action_dim:
            raise ValueError(
                "Wall-OSS-0.5 currently requires canonical 26D state and action features; "
                f"got {state_dim}D state and {action_dim}D action."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
