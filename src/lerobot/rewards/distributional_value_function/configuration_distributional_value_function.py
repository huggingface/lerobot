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
Architecture source of truth: "π0.6 Model Card", Section 2 (Model Design)
       https://website.pi-asset.com/pi06star/PI06_model_card.pdf

Distributional value function V^{pi_ref}(o_t, l) (Section IV-A).

Architecture (~670M params):
    Vision:  SigLIP2-so400m — 27 layers, 1152-dim, 1024 patches/image at 448px
    LM:      Gemma3-270M   — 18 layers, 640-dim
    Proj:    2x2 pool → RMSNorm → Linear(1152, 640), 256 soft tokens/image
    Readout: one-way learned value query → 2-layer MLP → 201 bins

Inputs:  multi-camera images (3 x 256 soft tokens) + ``"Task: {task}."`` prompt
Targets: MC returns in [-1, 0], cross-entropy on Dirac delta (default) or HL-Gauss
Init:    SigLIP2 + Gemma3 from pretrained HF checkpoints; head normal_(std=0.02)
"""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@RewardModelConfig.register_subclass("distributional_value_function")
@dataclass
class DistributionalVFConfig(RewardModelConfig):
    """Configuration for RECAP's distributional value function.

    Predicts V^{pi_ref}(o_t, l) as a categorical distribution over B=201 bins in [-1, 0].
    Trained with cross-entropy on Dirac delta (C51, default) or HL-Gauss soft targets,
    with optional one-hot targets for terminal states.

    Architecture: adapted from the native Gemma3 multimodal VLM design and
    scaled to π0.6's ~670M value backbone:
    448px SigLIP2-so400m images are pooled from 1024 patches to 256 soft
    tokens, RMS-normalized, projected into Gemma3-270M, and followed by a
    one-way learned value-query token. Image tokens attend bidirectionally;
    text and the value query remain causal.
    """

    # Backbone pretrained paths
    siglip_path: str = "google/siglip2-so400m-patch14-384"
    gemma3_path: str = "google/gemma-3-270m"
    # Optional standard Gemma3ForConditionalGeneration checkpoint produced by
    # standalone VLM alignment. When set, it supplies vision, connector, and LM.
    vlm_pretrained_path: str | None = None

    # Distributional head
    num_value_bins: int = 201
    value_support_min: float = -1.0
    value_support_max: float = 0.0
    hl_gauss_sigma_ratio: float = 5.0

    # Target distribution method: "dirac_delta" (paper-faithful C51) or "hl_gauss" (soft)
    target_method: str = "dirac_delta"

    # Whether to use one-hot targets for terminal states (exact return, no smoothing).
    use_one_hot_terminal: bool = True

    # Image
    image_resolution: tuple[int, int] = (448, 448)
    num_image_tokens: int = 256

    # Tokenizer (uses Gemma3's tokenizer)
    tokenizer_max_length: int = 200

    # Training controls
    value_dropout: float = 0.0
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False
    stop_gradient_to_vlm: bool = False
    vision_encoder_lr_multiplier: float = 0.5

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
        }
    )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=5e-5,
            weight_decay=1e-10,
            grad_clip_norm=1.0,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=500,
            num_decay_steps=40000,
            peak_lr=5e-5,
            decay_lr=5e-5,
        )

    def validate_features(self) -> None:
        if not self.input_features:
            return
        has_image = any(ft.type == FeatureType.VISUAL for ft in self.input_features.values())
        if not has_image:
            raise ValueError("DistributionalVFConfig requires at least one VISUAL input feature.")
