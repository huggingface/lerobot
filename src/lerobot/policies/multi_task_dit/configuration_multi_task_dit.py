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

from dataclasses import dataclass, field

import draccus

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@dataclass
class ObjectiveConfig(draccus.ChoiceRegistry):
    """Base configuration for model objectives (diffusion, flow matching, etc.)."""

    pass


@ObjectiveConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(ObjectiveConfig):
    """Configuration for standard diffusion model training and inference.

    These parameters control the noise scheduling and denoising process for
    standard DDPM/DDIM diffusion models.
    """

    objective_name: str = field(default="diffusion", init=False)

    # Noise scheduler configuration - controls diffusion process
    noise_scheduler_type: str = "DDPM"  # "DDPM" or "DDIM"
    num_train_timesteps: int = 100  # 100 noise levels for fine-grained control
    beta_schedule: str = "squaredcos_cap_v2"  # Cosine schedule prevents extreme noise
    beta_start: float = 0.0001  # Small initial noise level
    beta_end: float = 0.02  # Moderate final noise level
    prediction_type: str = "epsilon"  # Predict noise (works better than direct prediction)
    clip_sample: bool = True  # Prevent extreme action values
    clip_sample_range: float = 1.0  # Clip to [-1, 1] range

    # Inference configuration
    num_inference_steps: int | None = None  # Default to num_train_timesteps

    def __post_init__(self):
        """Validate diffusion-specific parameters."""
        if self.noise_scheduler_type not in ["DDPM", "DDIM"]:
            raise ValueError(
                f"noise_scheduler_type must be 'DDPM' or 'DDIM', got {self.noise_scheduler_type}"
            )

        if self.prediction_type not in ["epsilon", "sample"]:
            raise ValueError(f"prediction_type must be 'epsilon' or 'sample', got {self.prediction_type}")

        if self.num_train_timesteps <= 0:
            raise ValueError(f"num_train_timesteps must be positive, got {self.num_train_timesteps}")

        if not (0.0 <= self.beta_start <= self.beta_end <= 1.0):
            raise ValueError(
                "beta values must satisfy 0 <= beta_start <= beta_end <= 1, "
                f"got {self.beta_start}, {self.beta_end}"
            )


@dataclass
class TimestepSamplingConfig(draccus.ChoiceRegistry):
    """Base configuration for timestep sampling strategies during training."""

    pass


@TimestepSamplingConfig.register_subclass("uniform")
@dataclass
class UniformTimestepSamplingConfig(TimestepSamplingConfig):
    """Uniform timestep sampling from [0, 1]."""

    strategy_name: str = field(default="uniform", init=False)


@TimestepSamplingConfig.register_subclass("beta")
@dataclass
class BetaTimestepSamplingConfig(TimestepSamplingConfig):
    """Beta distribution timestep sampling.

    Samples from Beta distribution emphasizing low timesteps (high noise).

    This was inspired on the work from Physical Intelligence PI-0 model,
    where they suggested the beta distribution for sampling timesteps
    during training improved sample quality.
    """

    strategy_name: str = field(default="beta", init=False)

    s: float = 0.999  # Max timestep threshold for beta sampling
    alpha: float = 1.5  # Beta distribution alpha parameter
    beta: float = 1.0  # Beta distribution beta parameter

    def __post_init__(self):
        if not (0.0 < self.s <= 1.0):
            raise ValueError(f"s must be in (0, 1], got {self.s}")

        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")


@ObjectiveConfig.register_subclass("flow_matching")
@dataclass
class FlowMatchingConfig(ObjectiveConfig):
    """Configuration for flow matching training and inference.

    These parameters control the velocity field learning and ODE integration
    process for flow matching models.
    """

    objective_name: str = field(default="flow_matching", init=False)

    # Flow path construction
    sigma_min: float = 0.0  # Minimum noise level in flow interpolation path

    # ODE integration for inference
    num_integration_steps: int = (
        100  # Number of ODE integration steps (increased from 50 for smoother trajectories)
    )
    integration_method: str = "euler"  # ODE solver: "euler" or "rk4"

    # Timestep sampling strategy for training
    # Beta distribution found to be the most effective in practice, so it is the default
    timestep_sampling: TimestepSamplingConfig = field(default_factory=BetaTimestepSamplingConfig)

    def __post_init__(self):
        if not (0.0 <= self.sigma_min <= 1.0):
            raise ValueError(f"sigma_min must be in [0, 1], got {self.sigma_min}")

        if self.num_integration_steps <= 0:
            raise ValueError(f"num_integration_steps must be positive, got {self.num_integration_steps}")

        if self.integration_method not in ["euler", "rk4"]:
            raise ValueError(f"integration_method must be 'euler' or 'rk4', got {self.integration_method}")


@dataclass
class TransformerConfig:
    """Configuration for Transformer-based prediction model.

    These parameters control the transformer architecture used for noise/velocity
    prediction in diffusion and flow matching models.
    """

    # Transformer architecture parameters
    hidden_dim: int = 512  # Hidden dimension of transformer
    num_layers: int = 6  # Number of transformer layers
    num_heads: int = 8  # Number of attention heads
    dropout: float = 0.1  # Dropout rate
    use_positional_encoding: bool = False  # Whether to use absolute positional encoding
    diffusion_step_embed_dim: int = 256  # Timestep embedding size

    # RoPE (Rotary Position Embedding) configuration
    use_rope: bool = True  # Whether to use Rotary Position Embedding in attention (baseline is True)
    rope_base: float = 10000.0  # Base frequency for RoPE computation

    def __post_init__(self):
        """Validate Transformer-specific parameters."""
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

        if self.diffusion_step_embed_dim <= 0:
            raise ValueError("diffusion_step_embed_dim must be positive")


@dataclass
class VisionEncoderConfig:
    """Configuration for CLIP vision encoder.

    Uses CLIPVisionModel from transformers library.
    CLS token usage is handled automatically.
    CLIP's internal preprocessing (resize to 224x224) can be overridden
    by setting resize_shape and crop_shape.

    All image preprocessing is centralized here:
    1. Resize (optional) - resize images to target resolution
    2. Crop (optional) - crop after resize, must be smaller than resize_shape
    3. Random crop - whether to use random cropping during training

    Any CLIP model from transformers can be used. Examples:
    - openai/clip-vit-base-patch16 (default, 768 dims)
    - openai/clip-vit-large-patch14 (1024 dims)
    - laion/CLIP-ViT-B-32-xlaai256 (alternative CLIP model)
    """

    model_name: str = "openai/clip-vit-base-patch16"
    use_separate_encoder_per_camera: bool = False

    # Learning rate multiplier for vision encoder parameters
    # Vision encoder learning rate = optimizer_lr * lr_multiplier
    lr_multiplier: float = 0.1

    # Image preprocessing (centralized)
    resize_shape: tuple[int, int] | None = None
    crop_shape: tuple[int, int] | None = (224, 224)  # default input size for CLIP
    crop_is_random: bool = True

    def __post_init__(self):
        # Validate that model name contains "clip" to ensure correct encoder type
        if "clip" not in self.model_name.lower():
            raise ValueError(
                f"model_name must be a CLIP model from transformers (contain 'clip'), got '{self.model_name}'"
            )

        if (
            self.resize_shape
            and self.crop_shape
            and (self.crop_shape[0] > self.resize_shape[0] or self.crop_shape[1] > self.resize_shape[1])
        ):
            raise ValueError(
                f"crop_shape {self.crop_shape} must be smaller than or equal to "
                f"resize_shape {self.resize_shape}. Got crop={self.crop_shape}, resize={self.resize_shape}"
            )


@dataclass
class TextEncoderConfig:
    """Configuration for CLIP text encoder.

    Uses CLIP's text encoder to embed task descriptions, which are then
    used to condition the policy. The text embeddings are processed by
    a learnable projection layer before being concatenated into the
    conditioning vector.

    Any HuggingFace CLIP model can be used. Examples:
    - openai/clip-vit-base-patch16 (default)
    - openai/clip-vit-large-patch14
    """

    model: str = "openai/clip-vit-base-patch16"

    def __post_init__(self):
        # Validate that model name contains "clip" to ensure correct encoder type
        if "clip" not in self.model.lower():
            raise ValueError(f"CLIP text encoder requires a CLIP model (contain 'clip'). Got '{self.model}'")


@dataclass
class ObservationEncoderConfig:
    """Top-level configuration for observation encoding.

    This config combines:
    - Vision encoding (required): CLIP vision encoder from transformers
    """

    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)


@PreTrainedConfig.register_subclass("multi_task_dit")
@dataclass
class MultiTaskDiTConfig(PreTrainedConfig):
    """
    Configuration class for the Multi-Task Diffusion Transformer (DiT) policy.
    """

    # Temporal structure - controls how the policy processes time and predicts actions
    n_obs_steps: int = 2  # num observations for temporal context (..., t-1, t)
    horizon: int = 100  # predicted action steps into the future
    n_action_steps: int = 24  # actions per policy call (receding horizon) -- ~0.8s is a good place to start

    # Normalization strategy - critical for diffusion model performance
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,  # Standard ImageNet normalization for vision
            "STATE": NormalizationMode.MIN_MAX,  # [-1,1] range for proper diffusion clipping
            "ACTION": NormalizationMode.MIN_MAX,  # [-1,1] range required for diffusion process
        }
    )

    drop_n_last_frames: int | None = None  # Auto-calculated: horizon - n_action_steps - n_obs_steps + 1
    observation_encoder: ObservationEncoderConfig = field(default_factory=ObservationEncoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    objective: ObjectiveConfig = field(default_factory=DiffusionConfig)
    do_mask_loss_for_padding: bool = False  #  same logic as is implemented in LeRobot DP implementation

    # training optimizer and scheduler hyperparameters
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0  # No weight decay is suggested to be optimal
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 0  # No warmup found to be optimal

    def __post_init__(self):
        super().__post_init__()

        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1

    def get_optimizer_preset(self) -> AdamConfig:
        """Return Adam optimizer configuration optimized for diffusion training.

        Note: Vision encoder learning rate is set separately via get_optim_params.
        """
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        """Return learning rate scheduler configuration."""
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        """Validate that required input features are present and properly configured."""
        # Robot state is always present via self.robot_state_feature, so we don't need to enforce images/env_state
        # This allows for testing and simple state-only policies

        # Validate crop shape fits within image dimensions
        crop_shape = self.observation_encoder.vision.crop_shape
        if crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if crop_shape[0] > image_ft.shape[1] or crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Ensure all images have same shape (current limitation)
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def model_objective(self) -> str:
        return self.objective.objective_name

    @property
    def is_diffusion(self) -> bool:
        return isinstance(self.objective, DiffusionConfig)

    @property
    def is_flow_matching(self) -> bool:
        return isinstance(self.objective, FlowMatchingConfig)

    def get_objective_config(self) -> DiffusionConfig | FlowMatchingConfig:
        """Get the objective-specific configuration with proper typing."""
        return self.objective

    @property
    def observation_delta_indices(self) -> list:
        """Delta indices for stacking observations. Provides temporal context."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        """Delta indices for action horizon prediction."""
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        """Indices for reward deltas (not used in diffusion policy)."""
        return None
