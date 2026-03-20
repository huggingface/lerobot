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


@PreTrainedConfig.register_subclass("awm")
@dataclass
class AWMConfig(PreTrainedConfig):
    """Configuration class for the AWM (Autoregressive Action-World-Model) policy.

    Nearly identical to ACTSimpleConfig but uses an autoregressive decoder with teacher-forcing during
    training and step-by-step generation at inference. Optionally reduces the encoder-to-decoder
    cross-attention dimension via a 2-layer MLP projection.

    The parameters you will most likely need to change are the ones which depend on the environment /
    sensors. Those are: `input_shapes` and `output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image" is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple
          camera views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the
            policy. This should be no greater than the chunk size.
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality (e.g.
            "observation.state"), and the value specifies the normalization mode to apply.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to
            the original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a
            dilated convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the
            feed-forward layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        dropout: Dropout to use in the transformer layers.
        cross_attn_dim: Dimension to project encoder outputs to before using them as keys/values in the
            decoder's cross-attention. `None` (the default) resolves to `dim_model` (no compression).
        action_token_vocab_size: Number of uniform bins per action dimension used to tokenise actions into
            discrete tokens.  The joint vocabulary size is ``action_token_vocab_size ** action_dim``, so
            keep this small for high-dimensional action spaces (e.g. V=64 is fine for D≤3).
        action_ranges: Per-dimension ``[lo, hi]`` bounds for uniform tokenisation.  ``None`` (the default)
            resolves to ``[[-1.0, 1.0]] * action_dim`` at model construction time.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Architecture.
    # Vision backbone.
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
    # Cross-attention dimension reduction (None → dim_model, i.e. no compression).
    cross_attn_dim: int | None = None
    # Discrete action tokenisation.
    action_token_vocab_size: int = 128
    action_ranges: list[list[float]] | None = None  # None → [[-1.0, 1.0]] * action_dim

    # Training and loss computation.
    dropout: float = 0.1

    # World model
    wm_loss_weight: float = 0.2       # Weight on world model loss relative to action prediction loss
    wm_warmup_steps: int = 0          # Number of steps to linearly ramp wm_loss_weight from 0 to target
    detach_encoder_from_wm: bool = False  # Detach encoder outputs before WM cross-attention
    normalize_wm_representations: bool = False  # L2-normalize z_pred and z_target to unit sphere before WM loss and image decoding.
    use_normalized_mse_wm_loss: bool = False  # Replace cosine WM loss with normalized-MSE + variance regularization
    wm_variance_loss_weight: float = 0.1  # Weight on VICReg-style variance regularization when normalized MSE loss is enabled
    use_ema_target: bool = False      # Use an EMA copy of the encoder to compute z_target
    ema_momentum: float = 0.996       # EMA decay coefficient (higher = slower target evolution)
    ema_momentum_end: float = 0.999   # Final EMA momentum after annealing
    ema_anneal_steps: int = 50_000    # Steps over which to anneal EMA momentum
    n_wm_decoder_layers: int = 4      # Number of layers in the world model decoder
    decoder_loss_weight: float = 0.1  # Weight on image reconstruction loss (detached from main model)
    n_image_viz_pairs: int = 12       # Number of GT/decoded image pairs to log at each log step

    # Training preset
    optimizer_lr: float = 2e-4
    optimizer_weight_decay: float = 0.0
    optimizer_lr_backbone: float = 1e-5
    optimizer_grad_clip_norm: float = 1.0

    # LR schedule: cosine decay with linear warmup.
    # Set use_lr_schedule=True to enable; constant lr is used by default.
    use_lr_schedule: bool = False
    scheduler_warmup_steps: int = 5000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
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
        if self.cross_attn_dim is None:
            self.cross_attn_dim = self.dim_model

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
        return [0, self.chunk_size]  # Load obs at t (idx=0) and t+H (idx=1) for world model

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
