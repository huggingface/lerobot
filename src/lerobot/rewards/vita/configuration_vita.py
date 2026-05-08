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

import math
from dataclasses import dataclass, field

from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@LRSchedulerConfig.register_subclass("vita_paper_cosine")
@dataclass
class VitaPaperCosineSchedulerConfig(LRSchedulerConfig):
    """Cosine scheduler with warmup ratio fixed to paper setting."""

    num_warmup_steps: int | None = None
    peak_lr: float = 1e-4
    decay_lr: float = 1e-5
    warmup_ratio: float = 0.1

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        warmup_steps = max(1, int(self.warmup_ratio * num_training_steps))
        decay_steps = max(1, num_training_steps)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return max(1e-8, float(current_step + 1) / float(warmup_steps))

            step = min(current_step, decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
            alpha = self.decay_lr / self.peak_lr
            return (1 - alpha) * cosine_decay + alpha

        return LambdaLR(optimizer, lr_lambda, -1)


@RewardModelConfig.register_subclass("vita")
@dataclass
class VitaConfig(RewardModelConfig):
    """Configuration for VITA reward modeling."""

    backbone_type: str = "openclip"
    openclip_model_name: str = "ViT-B-32"
    openclip_pretrained: str = "openai"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    freeze_backbone: bool = True
    raw_image_key: str = "observation.images.top"
    raw_text_key: str = "task"
    image_feature_key: str = "image_features"
    text_feature_key: str = "text_features"
    adaptation_lr: float = 0.1
    adaptation_dim: int = 64
    image_feature_dim: int = 512
    text_feature_dim: int = 512
    reward_hidden_dim: int = 128
    device: str | None = None
    meta_enabled: bool = True
    inner_steps: int = 1
    inner_lr: float = 0.1
    outer_loss_weight: float = 1.0
    self_supervised_loss_weight: float = 0.5
    first_order: bool = False
    support_len: int = 1
    query_len: int = 1
    target_reward_key: str = "next.reward"
    sampling_strategy: str = "dissimilarity"
    sampling_window_size: int = 8
    sampling_num_windows: int = 8
    sampling_stride: int = 1
    force_temporal_progress_targets: bool = True
    scheduler_decay_lr: float = 1e-5

    input_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "image_features": PolicyFeature(type=FeatureType.VISUAL, shape=(512,)),
            "text_features": PolicyFeature(type=FeatureType.LANGUAGE, shape=(512,)),
        }
    )
    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {"reward": PolicyFeature(type=FeatureType.REWARD, shape=(1,))}
    )
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "LANGUAGE": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    @property
    def latent_dim(self) -> int:
        return self.image_feature_dim + self.text_feature_dim

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=1e-4, weight_decay=1e-4)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return VitaPaperCosineSchedulerConfig(
            peak_lr=1e-4,
            decay_lr=self.scheduler_decay_lr,
            warmup_ratio=0.1,
        )

    def validate_features(self) -> None:
        if self.support_len < 1 or self.query_len < 1:
            raise ValueError("support_len and query_len must be >= 1.")
        if self.inner_steps < 1:
            raise ValueError("inner_steps must be >= 1.")
        if self.backbone_type not in {"identity", "clip", "openclip"}:
            raise ValueError(
                f"Unsupported backbone_type '{self.backbone_type}'. Expected one of: identity, clip, openclip."
            )
        if self.sampling_strategy not in {"contiguous", "dissimilarity"}:
            raise ValueError(
                f"Unsupported sampling_strategy '{self.sampling_strategy}'. "
                "Expected one of: contiguous, dissimilarity."
            )
        if self.sampling_window_size < 1 or self.sampling_num_windows < 1 or self.sampling_stride < 1:
            raise ValueError("sampling_window_size, sampling_num_windows, and sampling_stride must be >= 1.")
        if self.backbone_type == "identity":
            if self.image_feature_key not in self.input_features:
                raise ValueError(f"Missing expected image feature key '{self.image_feature_key}' in input_features.")
            if self.text_feature_key not in self.input_features:
                raise ValueError(f"Missing expected text feature key '{self.text_feature_key}' in input_features.")
        else:
            if not self.raw_image_key:
                raise ValueError("raw_image_key must be set for clip/openclip backbone mode.")
            if not self.raw_text_key:
                raise ValueError("raw_text_key must be set for clip/openclip backbone mode.")
