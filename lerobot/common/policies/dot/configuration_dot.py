#!/usr/bin/env python

# Copyright 2025 Ilia Larchenko and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import CosineAnnealingSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("dot")
@dataclass
class DOTConfig(PreTrainedConfig):
    """Configuration class for the Decision Transformer (DOT) policy.

    DOT is a transformer-based policy for sequential decision making that predicts future actions based on
    a history of past observations and actions. This configuration enables fine-grained
    control over the model’s temporal horizon, input normalization, architectural parameters, and
    augmentation strategies.

    Defaults are configured for general robot manipulation tasks like Push-T and ALOHA insert/transfer.

    The parameters you will most likely need to modify are those related to temporal structure and
    normalization:
        - `train_horizon` and `inference_horizon`
        - `lookback_obs_steps` and `lookback_aug`
        - `alpha` and `train_alpha`
        - `normalization_mapping`

    Notes on the temporal design:
        - `train_horizon`: Length of action sequence the model is trained on. Must be ≥ `inference_horizon`.
        - `inference_horizon`: How far into the future the model predicts during inference (in environment steps).
            A good rule of thumb is 2×FPS (e.g., 30–50 for 15–25 FPS environments).
        - `alpha` / `train_alpha`: Control exponential decay of loss weights for inference and training.
            These should be tuned such that all predicted steps contribute meaningful signal.

    Notes on the inputs:
        - Observations can come from:
            - Images (e.g., keys starting with `"observation.images"`)
            - Proprioceptive state (`"observation.state"`)
            - Environment state (`"observation.environment_state"`)
        - At least one of image or environment state inputs must be provided.
        - The "action" key is required as an output.

    Args:
        n_obs_steps: Number of past steps passed to the model, including the current step.
        train_horizon: Number of future steps the model is trained to predict.
        inference_horizon: Number of future steps predicted during inference.
        lookback_obs_steps: Number of past steps to include for temporal context.
        lookback_aug: Number of steps into the far past from which to randomly sample for augmentation.
        normalization_mapping: Dictionary specifying normalization mode for each input/output group.
        override_dataset_stats: If True, replaces the dataset's stats with manually defined `new_dataset_stats`.
        new_dataset_stats: Optional manual min/max overrides used if `override_dataset_stats=True`.
        vision_backbone: Name of the ResNet variant used for image encoding (e.g., "resnet18").
        pretrained_backbone_weights: Optional pretrained weights (e.g., "ResNet18_Weights.IMAGENET1K_V1").
        pre_norm: Whether to apply pre-norm in transformer layers.
        lora_rank: If > 0, applies LoRA adapters of the given rank to transformer layers.
        merge_lora: Whether to merge LoRA weights at inference time.
        dim_model: Dimension of the transformer hidden state.
        n_heads: Number of attention heads.
        dim_feedforward: Dimension of the feedforward MLP inside the transformer.
        n_decoder_layers: Number of transformer decoder layers.
        rescale_shape: Resize shape for input images (e.g., (96, 96)).
        crop_scale: Image crop scale for augmentation.
        state_noise: Magnitude of additive uniform noise for state inputs.
        noise_decay: Decay factor applied to `crop_scale` and `state_noise` during training.
        dropout: Dropout rate used in transformer layers.
        alpha: Decay factor for inference loss weighting.
        train_alpha: Decay factor for training loss weighting.
        predict_every_n: Predict actions every `n` frames instead of every frame.
        return_every_n: Return every `n`-th predicted action during inference.
        optimizer_lr: Initial learning rate.
        optimizer_min_lr: Minimum learning rate for cosine scheduler.
        optimizer_lr_cycle_steps: Total steps in one learning rate cycle.
        optimizer_weight_decay: L2 weight decay for optimizer.

    Raises:
        ValueError: If the temporal settings are inconsistent (e.g., `train_horizon < inference_horizon`,
                    or `predict_every_n` > allowed bounds).
    """

    # Input / output structure.
    n_obs_steps: int = 3
    train_horizon: int = 20
    inference_horizon: int = 20
    lookback_obs_steps: int = 10
    lookback_aug: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Align with the new config system
    override_dataset_stats: bool = False
    new_dataset_stats: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            "action": {"max": [512.0] * 2, "min": [0.0] * 2},
            "observation.environment_state": {"max": [512.0] * 16, "min": [0.0] * 16},
            "observation.state": {"max": [512.0] * 2, "min": [0.0] * 2},
        }
    )

    # Architecture.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    pre_norm: bool = True
    lora_rank: int = 20
    merge_lora: bool = False

    dim_model: int = 128
    n_heads: int = 8
    dim_feedforward: int = 512
    n_decoder_layers: int = 8
    rescale_shape: tuple[int, int] = (96, 96)

    # Augmentation.
    crop_scale: float = 0.8
    state_noise: float = 0.01
    noise_decay: float = 0.999995

    # Training and loss computation.
    dropout: float = 0.1

    # Weighting and inference.
    alpha: float = 0.75
    train_alpha: float = 0.9
    predict_every_n: int = 1
    return_every_n: int = 1

    # Training preset
    optimizer_lr: float = 1.0e-4
    optimizer_min_lr: float = 1.0e-4
    optimizer_lr_cycle_steps: int = 300000
    optimizer_weight_decay: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if self.predict_every_n > self.inference_horizon:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be less than or equal to horizon ({self.inference_horizon})."
            )
        if self.return_every_n > self.inference_horizon:
            raise ValueError(
                f"return_every_n ({self.return_every_n}) must be less than or equal to horizon ({self.inference_horizon})."
            )
        if self.predict_every_n > self.inference_horizon // self.return_every_n:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be less than or equal to horizon //  return_every_n({self.inference_horizon // self.return_every_n})."
            )
        if self.train_horizon < self.inference_horizon:
            raise ValueError(
                f"train_horizon ({self.train_horizon}) must be greater than or equal to horizon ({self.inference_horizon})."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return CosineAnnealingSchedulerConfig(
            min_lr=self.optimizer_min_lr, T_max=self.optimizer_lr_cycle_steps
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        far_past_obs = list(
            range(
                -self.lookback_aug - self.lookback_obs_steps, self.lookback_aug + 1 - self.lookback_obs_steps
            )
        )
        recent_obs = list(range(2 - self.n_obs_steps, 1))

        return far_past_obs + recent_obs

    @property
    def action_delta_indices(self) -> list:
        far_past_actions = list(
            range(
                -self.lookback_aug - self.lookback_obs_steps, self.lookback_aug + 1 - self.lookback_obs_steps
            )
        )
        recent_actions = list(range(2 - self.n_obs_steps, self.train_horizon))

        return far_past_actions + recent_actions

    @property
    def reward_delta_indices(self) -> None:
        return None
