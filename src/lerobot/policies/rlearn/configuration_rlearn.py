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
    model_name: str = "google/siglip2-base-patch16-256"
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
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # ReWiND-specific parameters
    use_video_rewind: bool = True  # Enable video rewinding augmentation
    rewind_prob: float = 0.5  # Probability of applying rewind to each batch
    use_mismatch_loss: bool = True  # Enable mismatched language-video loss

    # Loss hyperparameters (simplified for ReWiND)
    # The main loss is just MSE between predicted and target progress

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
        # This gives us max_seq_len frames total, e.g. [-3, -2, -1, 0] for max_seq_len=4
        # The dataset will handle padding/repeating frames for episodes shorter than this
        return list(range(1 - self.max_seq_len, 1))

    @property
    def action_delta_indices(self) -> list | None:
        # Not an action chunking policy.
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        # ReWiND generates progress labels on-the-fly, doesn't need reward data
        return None

    def get_optimizer_preset(self):  # type: ignore[override]
        from lerobot.optim.optimizers import AdamWConfig

        return AdamWConfig(lr=self.learning_rate, weight_decay=self.weight_decay)

    def get_scheduler_preset(self):  # type: ignore[override]
        # No scheduler by default.
        return None
