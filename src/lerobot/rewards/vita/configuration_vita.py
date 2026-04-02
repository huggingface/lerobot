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

from dataclasses import dataclass, field

from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig


@RewardModelConfig.register_subclass("vita")
@dataclass
class VitaConfig(RewardModelConfig):
    """Configuration for VITA reward modeling."""

    image_feature_key: str = "image_features"
    text_feature_key: str = "text_features"
    adaptation_lr: float = 1e-2
    adaptation_dim: int = 128
    image_feature_dim: int = 128
    text_feature_dim: int = 128
    reward_hidden_dim: int = 128
    device: str | None = None
    meta_enabled: bool = True
    inner_steps: int = 1
    inner_lr: float = 1e-2
    outer_loss_weight: float = 1.0
    first_order: bool = True
    support_len: int = 1
    query_len: int = 1
    target_reward_key: str = "reward"

    input_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "image_features": PolicyFeature(type=FeatureType.VISUAL, shape=(128,)),
            "text_features": PolicyFeature(type=FeatureType.LANGUAGE, shape=(128,)),
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
        return AdamWConfig(lr=1e-4, weight_decay=1e-2)

    def validate_features(self) -> None:
        if self.support_len < 1 or self.query_len < 1:
            raise ValueError("support_len and query_len must be >= 1.")
        if self.inner_steps < 1:
            raise ValueError("inner_steps must be >= 1.")
        if self.image_feature_key not in self.input_features:
            raise ValueError(f"Missing expected image feature key '{self.image_feature_key}' in input_features.")
        if self.text_feature_key not in self.input_features:
            raise ValueError(f"Missing expected text feature key '{self.text_feature_key}' in input_features.")
