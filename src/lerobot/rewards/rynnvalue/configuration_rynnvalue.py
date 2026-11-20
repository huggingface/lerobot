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
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.rewards import RewardModelConfig
from lerobot.utils.constants import OBS_IMAGES

RYNNVALUE_FEATURE_PREFIX = "observation.rynnvalue."
# LeRobot checkpoints keep RynnValue's tokenizer and image processor in this folder.
RYNNVALUE_PROCESSOR_SUBFOLDER = "processor"


@RewardModelConfig.register_subclass("rynnvalue")
@dataclass
class RynnValueConfig(RewardModelConfig):
    """Configuration for the inference-only RynnValue temporal-distance model."""

    pretrained_path: str | None = "lerobot/RynnValue-4B"
    torch_dtype: str = "bfloat16"
    # Native architecture config, embedded on save so a LeRobot checkpoint rebuilds
    # the model from its own `config.json` and `model.safetensors`.
    model_config: dict[str, Any] | None = None

    image_key: str = OBS_IMAGES + ".top"
    task_key: str = "task"
    default_task: str | None = None
    max_frames: int | None = 8
    max_image_side: int | None = 640
    robot_description: str | None = None
    camera_description: str | None = None
    use_meta: bool | None = None

    # Offline scoring (`lerobot-score`): run RynnValue on about this many frames per second of
    # video and interpolate the frames in between.
    inference_fps: float = 1.0

    license: str | None = "apache-2.0"
    tags: list[str] | None = field(
        default_factory=lambda: [
            "reward-model",
            "value-model",
            "vision-language",
            "qwen3-vl",
            "temporal-distance",
        ]
    )
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pretrained_path is None:
            raise ValueError(
                "RynnValue loads from a LeRobot checkpoint: set pretrained_path, such as lerobot/RynnValue-4B."
            )
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {self.max_frames}")
        if self.max_image_side is not None and self.max_image_side < 1:
            raise ValueError(f"max_image_side must be >= 1, got {self.max_image_side}")
        if self.inference_fps <= 0:
            raise ValueError(f"inference_fps must be > 0, got {self.inference_fps}")
        if self.image_key not in self.input_features:
            self.input_features[self.image_key] = PolicyFeature(shape=(3, 224, 224), type=FeatureType.VISUAL)
        self.output_features.setdefault(
            "remaining_time_s", PolicyFeature(shape=(1,), type=FeatureType.REWARD)
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.image_key not in self.input_features:
            raise ValueError(f"RynnValue requires image input feature {self.image_key!r}")
