#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # VLM MLP FP8 settings (NVIDIA Transformer Engine). Only the te_layernorm_mlp backend is
    # implemented: each VLM language-model layer's post_attention_layernorm (plain RMSNorm) is
    # fused with its gate/up/down projections into one te.LayerNormMLP FP8 kernel. The action
    # expert (adaRMS) stays bf16. Defaults keep FP8 disabled → identical to the bf16 path.
    vlm_mlp_fp8_enable: bool = False
    vlm_mlp_fp8_format: str = "hybrid"  # Options: "hybrid" (E4M3 fwd/E5M2 bwd), "e4m3"
    vlm_mlp_fp8_amax_history_len: int = 16
    vlm_mlp_fp8_amax_compute_algo: str = "max"  # Options: "max", "most_recent"
    vlm_mlp_fp8_margin: int = 0
    vlm_mlp_fp8_strict_shape_check: bool = True
    vlm_mlp_fp8_log_once: bool = True
    # FP8 scaling recipe selector.
    # "delayed_scaling"      — per-tensor, 16-step amax history (default, HYBRID format).
    # "float8_block_scaling" — block-wise: 1D for activations/grads, 2D for weights (TE-managed,
    #                          works on Hopper and Blackwell). Defaults to power-of-2 scales.
    vlm_mlp_fp8_recipe_kind: str = "delayed_scaling"
    # Float8BlockScaling tuning knobs (used only when vlm_mlp_fp8_recipe_kind="float8_block_scaling").
    # Defaults match TE's defaults — leaving these alone reproduces the original recipe construction.
    vlm_mlp_fp8_blockscale_use_f32_scales: bool = False  # False = E8M0 power-of-2 scales; True = FP32
    vlm_mlp_fp8_blockscale_x_dim: int = 1  # Activation block scaling dim: 1=row-wise (1D), 2=tile (2D)
    vlm_mlp_fp8_blockscale_w_dim: int = 2  # Weight block scaling dim: default 2 (tile); 1 for 1D
    vlm_mlp_fp8_blockscale_grad_dim: int = 1  # Gradient block scaling dim: default 1 (row-wise)
    # When False, te.autocast is skipped during inference (model.eval()) even if FP8 is enabled,
    # so the forward runs in bf16 from the stored master weights. Training always quantizes.
    vlm_mlp_quant_inference: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        # VLM MLP FP8 validations
        if self.vlm_mlp_fp8_format not in ["hybrid", "e4m3"]:
            raise ValueError(f"Invalid vlm_mlp_fp8_format: {self.vlm_mlp_fp8_format}")

        if self.vlm_mlp_fp8_amax_compute_algo not in ["max", "most_recent"]:
            raise ValueError(
                f"Invalid vlm_mlp_fp8_amax_compute_algo: {self.vlm_mlp_fp8_amax_compute_algo}"
            )

        if self.vlm_mlp_fp8_amax_history_len <= 0:
            raise ValueError(
                f"vlm_mlp_fp8_amax_history_len must be > 0, got {self.vlm_mlp_fp8_amax_history_len}"
            )

        if self.vlm_mlp_fp8_margin < 0:
            raise ValueError(f"vlm_mlp_fp8_margin must be >= 0, got {self.vlm_mlp_fp8_margin}")

        if self.vlm_mlp_fp8_recipe_kind not in ["delayed_scaling", "float8_block_scaling"]:
            raise ValueError(
                f"Invalid vlm_mlp_fp8_recipe_kind: {self.vlm_mlp_fp8_recipe_kind}. "
                f"Must be one of: 'delayed_scaling', 'float8_block_scaling'."
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
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
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
