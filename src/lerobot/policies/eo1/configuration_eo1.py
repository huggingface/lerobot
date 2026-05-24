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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )
else:
    Qwen2_5_VLConfig = None
    Qwen2_5_VLTextConfig = None
    Qwen2_5_VLVisionConfig = None


@PreTrainedConfig.register_subclass("eo1")
@dataclass
class EO1Config(PreTrainedConfig):
    """Configuration for native EO1 policy integration in LeRobot."""

    vlm_base: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_config: dict | None = None

    # Vision processor settings.
    image_min_pixels: int | None = 64 * 28 * 28
    image_max_pixels: int | None = 128 * 28 * 28
    use_fast_processor: bool = False

    # Execution and action horizon.
    n_obs_steps: int = 1
    chunk_size: int = 8
    n_action_steps: int = 8

    # State/action padding to match EO1 flow head dimensionality.
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching sampling.
    num_denoise_steps: int = 10
    num_action_layers: int = 2
    action_act: str = "linear"
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    supervise_padding_action_dims: bool = True
    supervise_padding_actions: bool = True

    # Policy-level dtype request for the Qwen backbone.
    # - "auto": follow the backbone config/checkpoint default dtype. For Qwen2.5-VL this resolves to bf16.
    #           The EO1 flow-matching head still keeps its own parameters in fp32.
    # - "bfloat16": force the backbone to initialize/load in bf16 regardless of the saved config default.
    # - "float32": force the backbone to initialize/load in fp32 for maximum numerical conservatism.
    dtype: str = "auto"  # Options: "auto", "bfloat16", "float32"
    force_fp32_autocast: bool = True

    # Optional attention backend request passed through to the Qwen backbone.
    # Common values: None, "eager", "sdpa", "flash_attention_2".
    attn_implementation: str | None = None

    # Training settings.
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Optimizer settings aligned with EO1/experiments/2_libero/train.sh and EO1 TrainPipelineConfig defaults.
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.1
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings aligned with EO1 train.sh: cosine schedule with warmup_ratio=0.03.
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 900  # 0.03 * 30_000 long-run steps
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        # Populate the serialized backbone config only when the caller did not provide one.
        if self.vlm_config is None:
            require_package("transformers", extra="eo1")
            self.vlm_config = Qwen2_5_VLConfig.from_pretrained(self.vlm_base).to_dict()

    @property
    def vlm_backbone_config(self) -> Qwen2_5_VLConfig:
        require_package("transformers", extra="eo1")
        config_dict = deepcopy(self.vlm_config)
        if self.attn_implementation is not None:
            config_dict["attn_implementation"] = self.attn_implementation
        return Qwen2_5_VLConfig(**config_dict)

    @property
    def text_config(self) -> Qwen2_5_VLTextConfig:
        return self.vlm_backbone_config.text_config

    @property
    def vision_config(self) -> Qwen2_5_VLVisionConfig:
        return self.vlm_backbone_config.vision_config

    def validate_features(self) -> None:
        """Validate and set up EO1 input and output features."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "EO1 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature

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
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
