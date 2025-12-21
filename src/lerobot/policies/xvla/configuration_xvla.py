#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import XVLAAdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

# Conditional import for type checking and lazy loading
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from .configuration_florence2 import Florence2Config
else:
    Florence2Config = None


@PreTrainedConfig.register_subclass("xvla")
@dataclass
class XVLAConfig(PreTrainedConfig):
    """
    Configuration class for the XVLA (Extended Vision-Language-Action) policy so it can
    plug into the LeRobot training stack.

    The config mirrors the knobs exposed in the original XVLA repository but also
    declares the input/output feature contract required by LeRobot.
    """

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Florence2 backbone and tokenizer configuration
    florence_config: dict[str, Any] = field(default_factory=dict)
    tokenizer_name: str = "facebook/bart-large"
    tokenizer_max_length: int = 64
    tokenizer_padding_side: str = "right"
    pad_language_to: str = "max_length"

    # Transformer head
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_domains: int = 30
    len_soft_prompts: int = 32
    dim_time: int = 32
    max_len_seq: int = 512
    use_hetero_proj: bool = False

    # Action & proprioception
    action_mode: str = "ee6d"
    num_denoising_steps: int = 10
    use_proprio: bool = True
    max_state_dim: int = 32
    max_action_dim: int = 20  # Maximum action dimension for padding (used by "auto" action mode)
    domain_feature_key: str | None = None

    # Vision preprocessing
    resize_imgs_with_padding: tuple[int, int] | None = None
    num_image_views: int | None = None
    empty_cameras: int = 0

    # Freezing options for VLM components
    # By default, VLM encoders are frozen and only policy transformer + soft prompts train
    freeze_vision_encoder: bool = False  # Freeze VLM vision encoder weights
    freeze_language_encoder: bool = False  # Freeze VLM language encoder weights
    train_policy_transformer: bool = True  # Allow policy transformer to train
    train_soft_prompts: bool = True  # Allow soft prompts to train

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.99)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 10.0
    # Soft-prompt LR settings (for optional warm-up)
    optimizer_soft_prompt_lr_scale: float = 1.0  # Scale factor for soft-prompt LR
    optimizer_soft_prompt_warmup_lr_scale: float | None = None  # Start scale for warmup (e.g., 0.01)

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.chunk_size <= 0:
            raise ValueError("`chunk_size` must be strictly positive.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be <= `chunk_size` ({self.chunk_size})."
            )
        if self.num_image_views is not None and self.num_image_views <= 0:
            raise ValueError("`num_image_views` must be > 0 when specified.")
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        self._florence_config_obj: Florence2Config | None = None

    def get_florence_config(self) -> Florence2Config:
        """
        Build (and cache) the Florence2 transformer config that should back the VLM.
        """
        if self._florence_config_obj is None:
            config_dict = dict(self.florence_config)
            if "vision_config" not in config_dict or config_dict["vision_config"] is None:
                raise ValueError("vision_config is required")

            if "text_config" not in config_dict or config_dict["text_config"] is None:
                raise ValueError("text_config is required")
            self._florence_config_obj = Florence2Config(**config_dict)
        return self._florence_config_obj

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("XVLA requires at least one visual feature in the inputs.")
        if self.use_proprio and self.robot_state_feature is None:
            raise ValueError("`use_proprio=True` requires a proprioceptive state feature.")
        if self.num_image_views is None:
            self.num_image_views = len(self.image_features) + self.empty_cameras
        else:
            self.num_image_views = max(self.num_image_views, len(self.image_features) + self.empty_cameras)

        if self.empty_cameras > 0:
            height, width = (480, 640)
            if self.resize_imgs_with_padding is not None:
                height, width = self.resize_imgs_with_padding
            for idx in range(self.empty_cameras):
                key = f"{OBS_IMAGES}.empty_camera_{idx}"
                if key not in self.input_features:
                    self.input_features[key] = PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, height, width),
                    )

    def get_optimizer_preset(self) -> XVLAAdamWConfig:
        """Return the XVLA-specific optimizer with differential learning rates.

        This optimizer applies:
        - 1/10 LR for VLM parameters (stable optimization)
        - Full LR for transformer/action head
        - Configurable LR for soft-prompts (with optional warm-up)
        """
        return XVLAAdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            soft_prompt_lr_scale=self.optimizer_soft_prompt_lr_scale,
            soft_prompt_warmup_lr_scale=self.optimizer_soft_prompt_warmup_lr_scale,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None
