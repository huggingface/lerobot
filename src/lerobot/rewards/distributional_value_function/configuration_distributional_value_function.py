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

"""Configuration for RECAP's distributional value function.

Paper: "π*0.6: a VLA That Learns From Experience" (Physical Intelligence, 2025)
       https://pi.website/blog/pistar06

Implements the distributional value function V^{pi_ref}(o_t, l) from Section IV-A.
Architecture: the paper uses a 670M-parameter Gemma 3 VLM (the actor is 4B Gemma 3).
We match that scale on PaliGemma (PI05's Gemma 2B backbone) by truncating to 6 Gemma
LM layers and 13 SigLIP vision layers (~670M params), with a [CLS] token and linear
head predicting a categorical distribution over B=201 discrete value bins in [-1, 0].

Training: cross-entropy on HL-Gauss soft targets (or Dirac delta projection),
with optional one-hot targets for terminal states; MC returns normalized per task.
Weights initialized from a pre-trained PI05 actor checkpoint.
"""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@RewardModelConfig.register_subclass("distributional_value_function")
@dataclass
class DistributionalVFConfig(RewardModelConfig):
    """Configuration for RECAP's distributional value function.

    The value function predicts V^{pi_ref}(o_t, l) as a distribution over B discrete
    bins spanning [value_support_min, value_support_max]. It is trained with cross-entropy
    on HL-Gauss soft targets or Dirac delta projection, derived from Monte Carlo returns
    (Eq. 1 in the paper).

    Architecture: the paper value function is a 670M Gemma 3 VLM; the actor is 4B Gemma 3.
    We use truncated PaliGemma (``num_hidden_layers=6``, ``num_vision_layers=13``) to reach
    about 670M params and initialize from the PI05 actor checkpoint.
    """

    # Backbone
    paligemma_variant: str = "gemma_2b"
    num_hidden_layers: int = 6
    num_vision_layers: int = 13

    # Distributional head
    num_value_bins: int = 201
    value_support_min: float = -1.0
    value_support_max: float = 0.0
    hl_gauss_sigma_ratio: float = 5.0

    # Target distribution method: "hl_gauss" (default, soft) or "dirac_delta" (C51, hard)
    target_method: str = "hl_gauss"

    # Whether to use one-hot targets for terminal states (exact return, no smoothing).
    # When False, terminal states use the same target method as non-terminal states.
    use_one_hot_terminal: bool = True

    # Image
    image_resolution: tuple[int, int] = (224, 224)

    # Tokenizer
    tokenizer_max_length: int = 64

    # Init from actor (required for first training: provides SigLIP vision tower + Gemma embeddings).
    # Pass a PI05 checkpoint path or Hub repo_id here.
    # After training, load the value function with RewardModel.from_pretrained() instead.
    init_from_actor_path: str = ""

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
        }
    )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=3e-4,
            weight_decay=1e-4,
            grad_clip_norm=1.0,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=500,
            num_decay_steps=50000,
        )

    def validate_features(self) -> None:
        if not self.input_features:
            return
        has_image = any(ft.type == FeatureType.VISUAL for ft in self.input_features.values())
        if not has_image:
            raise ValueError("DistributionalVFConfig requires at least one VISUAL input feature.")
