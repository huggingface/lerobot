#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.act_simple_with_awm_head.planning import PlanningConfig


@PreTrainedConfig.register_subclass("act_simple_with_awm_head")
@dataclass
class ACTSimpleWithAWMHeadConfig(PreTrainedConfig):
    """Configuration for ACT Simple + World Model Head policy.

    Combines the act_simple encoder-decoder (non-autoregressive, continuous actions, L1 loss)
    with the world model decoder from AWM. The action decoder is identical to act_simple;
    the world model decoder takes continuous action embeddings (no tokenization) and predicts
    future encoder representations via cosine similarity loss.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation.
        vision_backbone: Name of the torchvision resnet backbone for encoding images.
        pretrained_backbone_weights: Pretrained weights to initialize the backbone.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final stride with dilation.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads in multi-head attention.
        dim_feedforward: The feed-forward expansion dimension.
        feedforward_activation: Activation function for feed-forward layers.
        n_encoder_layers: Number of transformer encoder layers.
        n_decoder_layers: Number of transformer decoder layers (action decoder).
        dropout: Dropout rate for transformer layers.
        wm_loss_weight: Weight on world model loss relative to action prediction loss.
        wm_warmup_steps: Number of steps to linearly ramp wm_loss_weight from 0 to target.
        detach_encoder_from_wm: Detach encoder outputs before WM cross-attention.
        n_wm_decoder_layers: Number of layers in the world model decoder.
        use_ema_target: Use an EMA copy of the encoder to compute z_target.
        ema_momentum: EMA decay coefficient.
        ema_momentum_end: Final EMA momentum after annealing.
        ema_anneal_steps: Steps over which to anneal EMA momentum.
        decoder_loss_weight: Weight on image reconstruction loss.
        n_image_viz_pairs: Number of GT/decoded image pairs to log.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 10

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture — Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4

    # Training and loss computation.
    dropout: float = 0.1

    # World model head.
    wm_loss_weight: float = 0.2
    wm_warmup_steps: int = 0
    detach_encoder_from_wm: bool = False
    n_wm_decoder_layers: int = 4
    use_ema_target: bool = False
    ema_momentum: float = 0.996
    ema_momentum_end: float = 0.999
    ema_anneal_steps: int = 50_000
    normalize_wm_representations: bool = False  # L2-normalize z_pred and z_target to unit sphere before WM loss and image decoding
    decoder_loss_weight: float = 0.1
    n_image_viz_pairs: int = 12

    # Training preset.
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 0.0
    optimizer_lr_backbone: float = 1e-5
    optimizer_grad_clip_norm: float = 10.0

    # LR schedule.
    use_lr_schedule: bool = False
    scheduler_warmup_steps: int = 5000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 0.0

    # Test-time planning.
    use_planning: bool = False
    planning: PlanningConfig = field(default_factory=PlanningConfig)

    # Deprecated — kept for checkpoint compatibility with older configs.
    image_resize: int | None = None
    wm_visual_pool: bool = False
    wm_pool_size: int = 9
    log_wm_action_sensitivity: bool = False

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig | None:
        if not self.use_lr_schedule:
            return None
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0, self.chunk_size]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
