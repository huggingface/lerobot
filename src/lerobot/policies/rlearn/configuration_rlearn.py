#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


@PreTrainedConfig.register_subclass("rlearn")
@dataclass
class RLearNConfig(PreTrainedConfig):
    """Configuration for a video-language conditioned reward model (RLearN).

    Inputs:
      - Visual frames (one or multiple cameras). Optionally a short sequence.
      - A language instruction/goal string.

    Output:
      - Per-timestep reward logits or a single-step reward logit.

    Notes:
      - This is the initial architecture. It uses frozen vision/text encoders
        (e.g. SigLIP2) and trains a lightweight temporal aggregator + head.
    """

    # Encoders
    model_name: str = "google/siglip2-large-patch16-256"
    freeze_backbones: bool = True

    # Temporal aggregator
    dim_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    pre_norm: bool = True
    use_first_frame_positional_bias: bool = True
    frame_dropout_p: float = 0.0
    stride: int = 1

    # Sequence length, amount of past frames including current one to use in the temporal model
    max_seq_len: int = 16

    # Head
    use_tanh_head: bool = False  # when True, bound outputs in [-1, 1]

    # Training
    learning_rate: float = 5e-5  # Reduced for stability
    weight_decay: float = 0.01
    loss_type: str = "composite"  # Always use composite loss with spatial awareness
    ranking_margin: float = 0.1

    # Composite loss weights (with spatial awareness and ReWiND reversibility)
    lambda_prog: float = 1.0  # Progress regression weight
    lambda_spatial_nce: float = 0.5  # Spatial-aware InfoNCE weight
    lambda_rewind: float = 0.4  # ReWiND reversible ranking weight

    # Loss hyperparameters
    nce_temperature: float = 0.07  # Temperature for InfoNCE
    zscore_eps: float = 1e-5  # Epsilon for z-score normalization
    min_rank_gap: int = 1  # Minimum gap for temporal ranking pairs
    num_ranking_pairs: int = 64  # Number of (far, near) pairs to sample for ReWiND
    last_k_for_nce: int = 3  # Use last k frames for InfoNCE
    mismatch_lang_prob: float = 0.2  # Probability of language mismatch augmentation

    # Value-based pairwise loss hyperparameters (for value_pairwise mode)
    lambda_dir: float = 1.0  # Intra-trajectory directional ranking
    lambda_text: float = 0.5  # Inter-instruction contrastive ranking
    lambda_flat: float = 0.25  # Flatness under mismatch
    dir_margin: float = 0.2  # Margin for directional ranking
    text_margin: float = 0.2  # Margin for text contrastive ranking
    flat_epsilon: float = 0.05  # Epsilon band for flatness loss
    num_pairs_per_loss: int = 64  # Number of pairs to sample per loss term
    use_hard_negatives: bool = True  # Whether to generate hard negative instructions

    # Normalization presets
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            # Language is tokenized at the encoder level; no numeric normalization here.
        }
    )

    def validate_features(self) -> None:
        # Require at least one image feature. Language is recommended but optional (can be blank).
        if not self.image_features:
            raise ValueError(
                "You must provide at least one image feature for RLearN (e.g. 'observation.image')."
            )

    @property
    def observation_delta_indices(self) -> list | None:
        # Use temporal sequences: past frames from -(max_seq_len-1) to current (0)
        # This gives us max_seq_len frames total, e.g. [-15, -14, ..., -1, 0] for max_seq_len=16
        return list(range(1 - self.max_seq_len, 1))

    @property
    def action_delta_indices(self) -> list | None:
        # Not an action chunking policy.
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        # By default we supervise every provided timestep equally.
        return None

    def get_optimizer_preset(self):  # type: ignore[override]
        from lerobot.optim.optimizers import AdamWConfig

        return AdamWConfig(lr=self.learning_rate, weight_decay=self.weight_decay)

    def get_scheduler_preset(self):  # type: ignore[override]
        # No scheduler by default.
        return None
