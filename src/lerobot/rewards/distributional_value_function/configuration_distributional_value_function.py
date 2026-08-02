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
    # Stop Regressing (Farebrother et al., 2024) default: spreads most
    # probability mass across approximately six neighboring bins.
    hl_gauss_sigma_ratio: float = 0.75

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
    value_dropout: float = 0.1
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False
    stop_gradient_to_vlm: bool = False
    optimizer_vision_lr: float = 1e-6
    optimizer_language_model_lr: float = 1e-5
    optimizer_multimodal_projector_lr: float = 5e-5
    optimizer_value_query_lr: float = 1e-4
    optimizer_value_head_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-2
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 40000
    scheduler_decay_lr: float = 1e-6
    # Deprecated compatibility field. Component-specific learning rates above
    # now control optimization directly.
    vision_encoder_lr_multiplier: float = 0.5

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        learning_rates = {
            "optimizer_vision_lr": self.optimizer_vision_lr,
            "optimizer_language_model_lr": self.optimizer_language_model_lr,
            "optimizer_multimodal_projector_lr": self.optimizer_multimodal_projector_lr,
            "optimizer_value_query_lr": self.optimizer_value_query_lr,
            "optimizer_value_head_lr": self.optimizer_value_head_lr,
        }
        for name, learning_rate in learning_rates.items():
            if learning_rate <= 0:
                raise ValueError(f"{name} must be > 0, got {learning_rate}")
        if self.optimizer_weight_decay < 0:
            raise ValueError(f"optimizer_weight_decay must be >= 0, got {self.optimizer_weight_decay}")
        if not 0 <= self.value_dropout <= 1:
            raise ValueError(f"value_dropout must be in [0,1], got {self.value_dropout}")
        if self.scheduler_warmup_steps < 0 or self.scheduler_decay_steps < 1:
            raise ValueError("scheduler_warmup_steps must be >= 0 and scheduler_decay_steps must be >= 1")
        if self.scheduler_decay_lr < 0:
            raise ValueError(f"scheduler_decay_lr must be >= 0, got {self.scheduler_decay_lr}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_value_head_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=1.0,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=max(
                self.optimizer_vision_lr,
                self.optimizer_language_model_lr,
                self.optimizer_multimodal_projector_lr,
                self.optimizer_value_query_lr,
                self.optimizer_value_head_lr,
            ),
            decay_lr=self.scheduler_decay_lr,
        )

    def validate_features(self) -> None:
        if not self.input_features:
            return
        has_image = any(ft.type == FeatureType.VISUAL for ft in self.input_features.values())
        if not has_image:
            raise ValueError("DistributionalVFConfig requires at least one VISUAL input feature.")
