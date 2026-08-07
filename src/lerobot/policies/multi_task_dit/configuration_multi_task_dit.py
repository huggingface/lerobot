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

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("multi_task_dit")
@dataclass
class MultiTaskDiTConfig(PreTrainedConfig):
    """Configuration for the Multi-Task Diffusion Transformer (DiT) policy.

    A transformer-based policy that supports both diffusion and flow matching objectives
    for multi-task robot learning with text and vision conditioning.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 2):
            Number of observation timesteps used for temporal context.
        input_features (`dict[str, PolicyFeature]`, *optional*):
            Input feature specification, keyed by feature name. Left empty to infer from the dataset.
        output_features (`dict[str, PolicyFeature]`, *optional*):
            Output feature specification, keyed by feature name. Left empty to infer from the dataset.
        device (`str`, *optional*):
            Torch device to run the policy on, e.g. `"cuda"` or `"cpu"`. Auto-selected when unset or
            unavailable.
        use_amp (`bool`, *optional*, defaults to `False`):
            Whether to use Automatic Mixed Precision for training and evaluation.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether this policy is trained with PEFT adapters.
        push_to_hub (`bool`, *optional*, defaults to `True`):
            Whether to push the trained policy to the Hugging Face Hub.
        repo_id (`str`, *optional*):
            Hub repository id to push the policy to.
        private (`bool`, *optional*):
            Whether the pushed Hub repository is private.
        tags (`list[str]`, *optional*):
            Tags to attach to the policy on the Hub.
        license (`str`, *optional*):
            License identifier for the policy on the Hub.
        pretrained_path (`Path`, *optional*):
            Repo id or local directory of pretrained weights saved with `save_pretrained`. Left unset to
            initialize from scratch.
        pretrained_revision (`str`, *optional*):
            Hub revision to pin when loading `pretrained_path`.
        horizon (`int`, *optional*, defaults to 32):
            Number of action steps predicted per policy call.
        n_action_steps (`int`, *optional*, defaults to 24):
            Number of actions from a predicted chunk that are actually executed before re-querying the
            policy, roughly 0.8s of actions at 30Hz.
        objective (`str`, *optional*, defaults to `"diffusion"`):
            Action-generation objective, either `"diffusion"` or `"flow_matching"`.
        noise_scheduler_type (`str`, *optional*, defaults to `"DDPM"`):
            Diffusion noise scheduler, either `"DDPM"` or `"DDIM"`. Used when `objective="diffusion"`.
        num_train_timesteps (`int`, *optional*, defaults to 100):
            Number of diffusion timesteps used during training. Used when `objective="diffusion"`.
        beta_schedule (`str`, *optional*, defaults to `"squaredcos_cap_v2"`):
            Noise schedule type for the diffusion scheduler. Used when `objective="diffusion"`.
        beta_start (`float`, *optional*, defaults to 0.0001):
            Starting noise level of the diffusion schedule. Used when `objective="diffusion"`.
        beta_end (`float`, *optional*, defaults to 0.02):
            Ending noise level of the diffusion schedule. Used when `objective="diffusion"`.
        prediction_type (`str`, *optional*, defaults to `"epsilon"`):
            What the diffusion model predicts: `"epsilon"` for the noise, or `"sample"` for the clean
            action. Used when `objective="diffusion"`.
        clip_sample (`bool`, *optional*, defaults to `True`):
            Whether to clip samples to `clip_sample_range` during denoising. Used when
            `objective="diffusion"`.
        clip_sample_range (`float`, *optional*, defaults to 1.0):
            Clipping range `[-x, x]` applied when `clip_sample` is `True`.
        num_inference_steps (`int`, *optional*):
            Number of denoising steps at inference. Defaults to `num_train_timesteps` when left unset.
            Used when `objective="diffusion"`.
        sigma_min (`float`, *optional*, defaults to 0.0):
            Minimum noise level in the flow-matching interpolation path. Used when
            `objective="flow_matching"`.
        num_integration_steps (`int`, *optional*, defaults to 100):
            Number of ODE integration steps at inference. Used when `objective="flow_matching"`.
        integration_method (`str`, *optional*, defaults to `"euler"`):
            ODE solver for flow-matching sampling, either `"euler"` or `"rk4"`.
        timestep_sampling_strategy (`str`, *optional*, defaults to `"beta"`):
            How training timesteps are sampled for flow matching, either `"uniform"` or `"beta"`.
        timestep_sampling_s (`float`, *optional*, defaults to 0.999):
            Maximum timestep threshold, used only when `timestep_sampling_strategy="beta"`.
        timestep_sampling_alpha (`float`, *optional*, defaults to 1.5):
            Alpha parameter of the Beta distribution, used only when `timestep_sampling_strategy="beta"`.
        timestep_sampling_beta (`float`, *optional*, defaults to 1.0):
            Beta parameter of the Beta distribution, used only when `timestep_sampling_strategy="beta"`.
        hidden_dim (`int`, *optional*, defaults to 512):
            Transformer hidden dimension.
        num_layers (`int`, *optional*, defaults to 6):
            Number of transformer layers.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads. Must divide `hidden_dim`.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate applied inside the transformer.
        use_positional_encoding (`bool`, *optional*, defaults to `False`):
            Whether to add a learned absolute positional encoding to the action sequence.
        timestep_embed_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the diffusion/flow-matching timestep embedding.
        use_rope (`bool`, *optional*, defaults to `True`):
            Whether to use Rotary Position Embedding in self-attention instead of standard multi-head
            attention.
        rope_base (`float`, *optional*, defaults to 10000.0):
            Base frequency for Rotary Position Embedding. Used when `use_rope` is `True`.
        vision_encoder_name (`str`, *optional*, defaults to `"openai/clip-vit-base-patch16"`):
            Hugging Face Hub id of the CLIP vision model used to encode camera images. Must be a CLIP
            model.
        use_separate_rgb_encoder_per_camera (`bool`, *optional*, defaults to `False`):
            Whether to instantiate one vision encoder per camera view instead of sharing a single one.
        vision_encoder_lr_multiplier (`float`, *optional*, defaults to 0.1):
            Learning-rate multiplier applied to the vision encoder's parameter group.
        image_resize_shape (`tuple[int, int]`, *optional*):
            Size images are resized to before cropping. `None` skips resizing.
        image_crop_shape (`tuple[int, int]`, *optional*, defaults to `(224, 224)`):
            Crop shape applied after resizing. Disabled automatically when it does not fit within the
            (resized) image.
        image_crop_is_random (`bool`, *optional*, defaults to `True`):
            Whether to crop randomly during training. Inference always uses a center crop.
        text_encoder_name (`str`, *optional*, defaults to `"openai/clip-vit-base-patch16"`):
            Hugging Face Hub id of the CLIP text model used to encode the language instruction. Must be a
            CLIP model.
        tokenizer_max_length (`int`, *optional*, defaults to 77):
            Maximum length for tokenized text.
        tokenizer_padding (`str`, *optional*, defaults to `"max_length"`):
            Tokenizer padding strategy, either `"max_length"` or `"longest"`.
        tokenizer_padding_side (`str`, *optional*, defaults to `"right"`):
            Tokenizer padding side, either `"left"` or `"right"`.
        tokenizer_truncation (`bool`, *optional*, defaults to `True`):
            Whether to truncate sequences longer than `tokenizer_max_length`.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps each `FeatureType` to the `NormalizationMode` used to normalize/unnormalize it.
        optimizer_lr (`float`, *optional*, defaults to 2e-05):
            Learning rate used to build the default `AdamConfig` optimizer preset.
        optimizer_betas (`tuple`, *optional*, defaults to `(0.95, 0.999)`):
            Adam beta coefficients for the default optimizer preset.
        optimizer_eps (`float`, *optional*, defaults to 1e-08):
            Adam epsilon for the default optimizer preset.
        optimizer_weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay for the default optimizer preset.
        scheduler_name (`str`, *optional*, defaults to `"cosine"`):
            Name of the learning-rate scheduler preset.
        scheduler_warmup_steps (`int`, *optional*, defaults to 0):
            Number of warmup steps for the learning-rate scheduler preset.
        do_mask_loss_for_padding (`bool`, *optional*, defaults to `False`):
            Whether to exclude padded action timesteps, marked by `action_is_pad`, from the loss.
        drop_n_last_frames (`int`, *optional*):
            Number of trailing frames dropped per episode when building training windows.
            Auto-computed from `horizon`, `n_action_steps`, and `n_obs_steps` in `__post_init__` when left
            unset.
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
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates the DiT backbone and diffusion/flow-matching schedule configuration."""
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
        """See [`~configs.PreTrainedConfig.get_optimizer_preset`]."""
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        """See [`~configs.PreTrainedConfig.get_scheduler_preset`]."""
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
        """`True` if `objective` is `"diffusion"`."""
        return self.objective == "diffusion"

    @property
    def is_flow_matching(self) -> bool:
        """`True` if `objective` is `"flow_matching"`."""
        return self.objective == "flow_matching"

    @property
    def observation_delta_indices(self) -> list:
        """See [`~configs.PreTrainedConfig.observation_delta_indices`]."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        """See [`~configs.PreTrainedConfig.action_delta_indices`]."""
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        """See [`~configs.PreTrainedConfig.reward_delta_indices`]."""
        return None
