#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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

import logging
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("multi_task_dit")
@dataclass
class MultiTaskDiTConfig(PreTrainedConfig):
    """Configuration for the Multi-Task Diffusion Transformer (DiT) policy.

    A transformer-based policy that supports both diffusion and flow matching objectives
    for multi-task robot learning with text and vision conditioning.
    """

    n_obs_steps: int = 2  # Number of observation steps for temporal context
    horizon: int = 32  # Number of action steps to predict
    n_action_steps: int = 24  # Actions executed per policy call (~0.8s at 30Hz)

    # Objective Selection
    objective: str = "diffusion"  # "diffusion" or "flow_matching"

    # --- Diffusion-specific (used when objective="diffusion") ---
    noise_scheduler_type: str = "DDPM"  # "DDPM" or "DDIM"
    num_train_timesteps: int = 100  # Number of diffusion timesteps
    beta_schedule: str = "squaredcos_cap_v2"  # Noise schedule type
    beta_start: float = 0.0001  # Starting noise level
    beta_end: float = 0.02  # Ending noise level
    prediction_type: str = "epsilon"  # "epsilon" (predict noise) or "sample" (predict clean)
    clip_sample: bool = True  # Clip samples during denoising
    clip_sample_range: float = 1.0  # Clipping range [-x, x]
    num_inference_steps: int | None = None  # Denoising steps at inference (defaults to num_train_timesteps)

    # --- Flow Matching-specific (used when objective="flow_matching") ---
    sigma_min: float = 0.0  # Minimum noise in flow interpolation path
    num_integration_steps: int = 100  # ODE integration steps at inference
    integration_method: str = "euler"  # ODE solver: "euler" or "rk4"
    timestep_sampling_strategy: str = "beta"  # "uniform" or "beta"

    timestep_sampling_s: float = 0.999  # (beta only) Max timestep threshold
    timestep_sampling_alpha: float = 1.5  # (beta only) Beta distribution alpha
    timestep_sampling_beta: float = 1.0  # (beta only) Beta distribution beta

    # Transformer Architecture
    hidden_dim: int = 512  # Transformer hidden dimension
    num_layers: int = 6  # Number of transformer layers
    num_heads: int = 8  # Number of attention heads
    dropout: float = 0.1  # Dropout rate
    use_positional_encoding: bool = False  # Use absolute positional encoding
    timestep_embed_dim: int = 256  # Timestep embedding dimension
    use_rope: bool = True  # Use Rotary Position Embedding
    rope_base: float = 10000.0  # RoPE base frequency

    # Vision Encoder (CLIP)
    vision_encoder_name: str = "openai/clip-vit-base-patch16"  # HuggingFace CLIP model
    use_separate_rgb_encoder_per_camera: bool = False  # Separate encoder per camera view
    vision_encoder_lr_multiplier: float = 0.1  # LR multiplier for vision encoder
    image_resize_shape: tuple[int, int] | None = None  # Resize images before crop
    image_crop_shape: tuple[int, int] | None = (224, 224)  # Crop shape (CLIP default)
    image_crop_is_random: bool = True  # Random crop during training, center at inference

    # Text Encoder (CLIP)
    text_encoder_name: str = "openai/clip-vit-base-patch16"  # HuggingFace CLIP model
    tokenizer_max_length: int = 77  # Max length for tokenized text (CLIP default is 77)
    tokenizer_padding: str = "max_length"  # Padding strategy: "max_length" or "longest"
    tokenizer_padding_side: str = "right"  # Padding side: "left" or "right"
    tokenizer_truncation: bool = True  # Whether to truncate sequences longer than max_length

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Training/Optimizer
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 0
    do_mask_loss_for_padding: bool = False

    # Auto-calculated
    drop_n_last_frames: int | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1

        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        # Objective validation
        if self.objective not in ["diffusion", "flow_matching"]:
            raise ValueError(f"objective must be 'diffusion' or 'flow_matching', got '{self.objective}'")

        # Transformer validation
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0")

        # Vision encoder validation
        if "clip" not in self.vision_encoder_name.lower():
            raise ValueError(
                f"vision_encoder_name must be a CLIP model (contain 'clip'), got '{self.vision_encoder_name}'"
            )
        if (
            self.image_resize_shape
            and self.image_crop_shape
            and (
                self.image_crop_shape[0] > self.image_resize_shape[0]
                or self.image_crop_shape[1] > self.image_resize_shape[1]
            )
        ):
            logging.warning(
                "image_crop_shape %s must be <= image_resize_shape %s; disabling cropping.",
                self.image_crop_shape,
                self.image_resize_shape,
            )
            self.image_crop_shape = None

        # Text encoder validation
        if "clip" not in self.text_encoder_name.lower():
            raise ValueError(
                f"text_encoder_name must be a CLIP model (contain 'clip'), got '{self.text_encoder_name}'"
            )

        # Objective-specific validation
        if self.objective == "diffusion":
            if self.noise_scheduler_type not in ["DDPM", "DDIM"]:
                raise ValueError(
                    f"noise_scheduler_type must be 'DDPM' or 'DDIM', got {self.noise_scheduler_type}"
                )
            if self.prediction_type not in ["epsilon", "sample"]:
                raise ValueError(f"prediction_type must be 'epsilon' or 'sample', got {self.prediction_type}")
            if self.num_train_timesteps <= 0:
                raise ValueError(f"num_train_timesteps must be positive, got {self.num_train_timesteps}")
            if not (0.0 <= self.beta_start <= self.beta_end <= 1.0):
                raise ValueError(f"Invalid beta values: {self.beta_start}, {self.beta_end}")

        elif self.objective == "flow_matching":
            if not (0.0 <= self.sigma_min <= 1.0):
                raise ValueError(f"sigma_min must be in [0, 1], got {self.sigma_min}")
            if self.num_integration_steps <= 0:
                raise ValueError(f"num_integration_steps must be positive, got {self.num_integration_steps}")
            if self.integration_method not in ["euler", "rk4"]:
                raise ValueError(
                    f"integration_method must be 'euler' or 'rk4', got {self.integration_method}"
                )
            if self.timestep_sampling_strategy not in ["uniform", "beta"]:
                raise ValueError("timestep_sampling_strategy must be 'uniform' or 'beta'")
            if self.timestep_sampling_strategy == "beta":
                if not (0.0 < self.timestep_sampling_s <= 1.0):
                    raise ValueError(f"timestep_sampling_s must be in (0, 1], got {self.timestep_sampling_s}")
                if self.timestep_sampling_alpha <= 0:
                    raise ValueError("timestep_sampling_alpha must be positive")
                if self.timestep_sampling_beta <= 0:
                    raise ValueError("timestep_sampling_beta must be positive")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        """Validate that required input features are present and properly configured."""
        # If the configured crop doesn't fit, disable cropping instead of erroring.
        # Note: if image_resize_shape is set, cropping is applied *after* resizing.
        if self.image_crop_shape is not None:
            for key, image_ft in self.image_features.items():
                # image_ft.shape is (C, H, W)
                effective_h, effective_w = (
                    self.image_resize_shape
                    if self.image_resize_shape is not None
                    else (image_ft.shape[1], image_ft.shape[2])
                )
                if self.image_crop_shape[0] > effective_h or self.image_crop_shape[1] > effective_w:
                    logging.warning(
                        "image_crop_shape %s doesn't fit within effective image shape (%s, %s) for '%s'; disabling cropping.",
                        self.image_crop_shape,
                        effective_h,
                        effective_w,
                        key,
                    )
                    self.image_crop_shape = None
                    break

        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_ft.shape:
                    raise ValueError(
                        f"Image '{key}' shape {image_ft.shape} != '{first_key}' shape {first_ft.shape}"
                    )

    @property
    def is_diffusion(self) -> bool:
        return self.objective == "diffusion"

    @property
    def is_flow_matching(self) -> bool:
        return self.objective == "flow_matching"

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
