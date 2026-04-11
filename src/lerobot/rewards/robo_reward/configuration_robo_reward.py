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

"""
RoboReward: General-Purpose Vision-Language Reward Models for Robotics.
Paper: https://arxiv.org/abs/2601.00675
Models: teetone/RoboReward-4B, teetone/RoboReward-8B
"""

from dataclasses import dataclass, field

from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig


@RewardModelConfig.register_subclass("robo_reward")
@dataclass
class RoboRewardConfig(RewardModelConfig):
    """Configuration for the RoboReward vision-language reward model.

    RoboReward is a fine-tuned Qwen3-VL model that assigns discrete progress scores
    (1–5) from robot rollout videos and task descriptions. Scores are mapped to
    normalized float rewards in [0, 1].

    Args:
        model_name: HuggingFace Hub model path. Use "teetone/RoboReward-4B" for
            a lighter-weight variant.
        max_new_tokens: Maximum tokens to generate for the score output. 16 is
            sufficient to capture "ANSWER: 5".
        score_to_reward: Maps discrete scores 1–5 to float rewards. Default maps
            linearly from 0.0 (no success) to 1.0 (perfect).
        task_key: Batch key containing the task language instruction string(s).
        image_key: Batch key for the primary camera image(s). Accepts tensors of
            shape (B, C, H, W) for single-frame or (B, T, C, H, W) for video.
    """

    model_name: str = "teetone/RoboReward-8B"
    device: str = "cpu"
    max_new_tokens: int = 16
    score_to_reward: dict[int, float] = field(
        default_factory=lambda: {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}
    )
    task_key: str = "observation.language_instruction"
    image_key: str = "observation.images.top"

    input_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.language_instruction": PolicyFeature(type=FeatureType.LANGUAGE, shape=(1,)),
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

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=1e-5, weight_decay=1e-2)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        if self.image_key not in self.input_features:
            raise ValueError(
                f"image_key '{self.image_key}' not found in input_features. "
                f"Available keys: {list(self.input_features.keys())}"
            )
        if not self.score_to_reward:
            raise ValueError("score_to_reward mapping cannot be empty.")
