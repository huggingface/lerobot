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

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("flow_matching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    """Configuration for a conditional Flow Matching action policy.

    The policy learns a velocity field on straight paths between Gaussian noise
    and normalized action chunks. A transformer predicts the velocity field,
    conditioned on any combination of robot state, environment state, and RGB
    observations declared in ``input_features``.

    Args:
        n_obs_steps: Number of observation steps used for temporal context.
        horizon: Number of action steps modeled by the velocity field.
        n_action_steps: Number of generated actions executed before replanning.
        vision_backbone: Torchvision ResNet used to encode RGB observations.
        pretrained_backbone_weights: Optional torchvision weights identifier.
        use_group_norm: Replace ResNet BatchNorm with GroupNorm. This supports
            small training batches but is incompatible with pretrained weights.
        text_encoder_name: Optional Hugging Face CLIP text encoder. When set,
            task descriptions are tokenized by the processor and condition the
            velocity field. This is required for multi-task benchmarks such as
            LIBERO, where the same scene can have different goals.
        freeze_text_encoder: Freeze the pretrained text encoder while keeping
            its output projection trainable.
        hidden_dim: Transformer and observation projection width.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        feed_forward_dim: Transformer feed-forward width.
        dropout: Transformer dropout probability.
        num_inference_steps: Number of Euler integration steps.
        conditioning_dropout_prob: Probability of replacing the conditioning
            vector with zero during training for classifier-free guidance.
        guidance_scale: Classifier-free guidance scale used during inference.
            ``1.0`` disables the extra unconditional network evaluation.
        do_mask_loss_for_padding: Exclude copy-padded action steps from loss.
    """

    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Observation encoder.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True

    # Optional language task conditioning.
    text_encoder_name: str | None = None
    freeze_text_encoder: bool = True
    tokenizer_max_length: int = 77
    tokenizer_padding: str = "max_length"
    tokenizer_padding_side: str = "right"
    tokenizer_truncation: bool = True

    # Velocity-field transformer.
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    feed_forward_dim: int = 1024
    dropout: float = 0.1

    # Flow Matching.
    num_inference_steps: int = 10
    conditioning_dropout_prob: float = 0.1
    guidance_scale: float = 1.0
    do_mask_loss_for_padding: bool = True

    # Optimization.
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.n_obs_steps < 1:
            raise ValueError(f"`n_obs_steps` must be positive, got {self.n_obs_steps}.")
        if self.horizon < 1:
            raise ValueError(f"`horizon` must be positive, got {self.horizon}.")
        if self.n_action_steps < 1:
            raise ValueError(f"`n_action_steps` must be positive, got {self.n_action_steps}.")
        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError(
                "`n_action_steps` must fit between the current observation and the end of the action "
                f"horizon. Got {self.n_action_steps=}, {self.horizon=}, and {self.n_obs_steps=}."
            )
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be a torchvision ResNet variant, got {self.vision_backbone!r}."
            )
        if self.use_group_norm and self.pretrained_backbone_weights is not None:
            raise ValueError(
                "`use_group_norm=True` is incompatible with pretrained BatchNorm weights. "
                "Set `pretrained_backbone_weights=None` or disable GroupNorm."
            )
        if self.text_encoder_name is not None and "clip" not in self.text_encoder_name.lower():
            raise ValueError(
                "`text_encoder_name` must identify a CLIP model so its tokenizer and text encoder are "
                f"compatible, got {self.text_encoder_name!r}."
            )
        if self.tokenizer_max_length < 1:
            raise ValueError(f"`tokenizer_max_length` must be positive, got {self.tokenizer_max_length}.")
        if self.tokenizer_padding not in {"longest", "max_length"}:
            raise ValueError(
                f"`tokenizer_padding` must be 'longest' or 'max_length', got {self.tokenizer_padding!r}."
            )
        if self.tokenizer_padding_side not in {"left", "right"}:
            raise ValueError(
                f"`tokenizer_padding_side` must be 'left' or 'right', got {self.tokenizer_padding_side!r}."
            )
        if self.hidden_dim < 4 or self.hidden_dim % 2 != 0:
            raise ValueError(f"`hidden_dim` must be an even integer of at least 4, got {self.hidden_dim}.")
        if self.num_layers < 1:
            raise ValueError(f"`num_layers` must be positive, got {self.num_layers}.")
        if self.num_heads < 1 or self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"`num_heads` must divide `hidden_dim`, got {self.num_heads=} and {self.hidden_dim=}."
            )
        if self.feed_forward_dim < 1:
            raise ValueError(f"`feed_forward_dim` must be positive, got {self.feed_forward_dim}.")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"`dropout` must be in [0, 1], got {self.dropout}.")
        if self.num_inference_steps < 1:
            raise ValueError(f"`num_inference_steps` must be positive, got {self.num_inference_steps}.")
        if not 0.0 <= self.conditioning_dropout_prob <= 1.0:
            raise ValueError(
                f"`conditioning_dropout_prob` must be in [0, 1], got {self.conditioning_dropout_prob}."
            )
        if self.guidance_scale < 0.0:
            raise ValueError(f"`guidance_scale` must be non-negative, got {self.guidance_scale}.")

    def validate_features(self) -> None:
        if not self.input_features:
            raise ValueError("FlowMatchingPolicy requires at least one input feature.")
        if self.robot_state_feature is None and self.env_state_feature is None and not self.image_features:
            raise ValueError(
                "FlowMatchingPolicy requires robot state, environment state, or at least one RGB image."
            )
        if self.action_feature is None:
            raise ValueError("FlowMatchingPolicy requires an `action` output feature.")

        first_image = None
        for key, image_feature in self.image_features.items():
            if len(image_feature.shape) != 3 or image_feature.shape[0] != 3:
                raise ValueError(
                    f"Visual feature `{key}` must use channel-first RGB shape (3, H, W), "
                    f"got {image_feature.shape}."
                )
            if first_image is None:
                first_image = (key, image_feature)
            elif image_feature.shape != first_image[1].shape:
                raise ValueError(
                    f"Visual feature `{key}` has shape {image_feature.shape}, but "
                    f"`{first_image[0]}` has shape {first_image[1].shape}. All cameras must match."
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
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
