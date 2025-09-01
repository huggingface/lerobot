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
      - This follows the ReWiND paper architecture. It uses frozen vision/text encoders
        (DINOv3 for vision, SigLIP2 for language) and trains a
        lightweight temporal aggregator + head.
    """

    # Encoders - Use SigLIP2 for both vision and text (shared checkpoint)
    vision_model_name: str = "google/siglip2-base-patch16-224"
    text_model_name: str = "google/siglip2-base-patch16-224"
    freeze_backbones: bool = True

    # Sequence length, amount of past frames including current one to use in the temporal model
    max_seq_len: int = 16
    # Temporal sampling stride (2 = skip every other frame for wider temporal coverage)
    temporal_sampling_stride: int = 3 # Open x mostly has fps 10, and rewind has seq len 16, ours is 30fps so 30/10 = 3 stride lenght to have same timeframe!

    # Model dimensions and transformer
    dim_model: int = 512
    num_layers: int = 4
    num_heads: int = 8
    ff_mult: int = 4  # Feed-forward multiplier, hidden = dim_model * ff_mult
    dropout: float = 0.10
    num_register_tokens: int = 4

    # --- reward head options ---
    use_categorical_rewards: bool = False      # classification over bins
    num_reward_bins: int = 25
    reward_min_value: float = 0.0              # for HL-Gauss range
    reward_max_value: float = 1.0
    use_hl_gauss_loss: bool = True             # if False -> plain regression
    hl_gauss_num_bins: int = 25                # histogram resolution

    # Inference-time subsampling and regularization
    inference_stride: int = 1 # inference_stride is an extra, second downsampling applied in forward after window sampling/rewind. Keep it at 1 to disable extra skipping
    frame_dropout_p: float = 0.10

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    head_lr_multiplier: float = 5.0
    logit_eps: float = 1e-4
    
    # Performance optimizations
    use_amp: bool = False
    compile_model: bool = True

    # ReWiND augmentation
    rewind_prob: float = 0.8
    rewind_last3_prob: float = 0.3
    mismatch_prob: float = 0.2
    
    # Normalization presets
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
        }
    )

    # Required path to episodes.jsonl for episode boundaries
    episodes_jsonl_path: str | None = "meta/episodes.jsonl"

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
