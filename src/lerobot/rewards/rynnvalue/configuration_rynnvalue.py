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


@RewardModelConfig.register_subclass("rynnvalue")
@dataclass
class RynnValueConfig(RewardModelConfig):
    """Configuration for the inference-only RynnValue temporal-distance model."""

    model_id: str = "Alibaba-DAMO-Academy/RynnValue-4B"
    model_revision: str | None = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "pred_slot_isolated_eager"
    # Embedded by the conversion script so LeRobot checkpoints can construct
    # the architecture without consulting the original repository.
    model_config: dict[str, Any] | None = None

    image_key: str = OBS_IMAGES + ".top"
    task_key: str = "task"
    default_task: str | None = None
    max_frames: int | None = 8
    robot_description: str | None = None
    camera_description: str | None = None
    use_meta: bool | None = None

    # Load-only compatibility for checkpoints created by the unpublished
    # pre-capability integration. Native inference always returns remaining
    # time; consumers decide whether to derive potential or progress.
    reward_output: str | None = None

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
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {self.max_frames}")
        if self.reward_output not in {None, "potential", "remaining_time"}:
            raise ValueError(
                "reward_output is a load-only compatibility field and must be "
                f"None, 'potential', or 'remaining_time', got {self.reward_output!r}"
            )
        if self.attn_implementation != "pred_slot_isolated_eager":
            raise ValueError("RynnValue checkpoints require attn_implementation='pred_slot_isolated_eager'")
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
