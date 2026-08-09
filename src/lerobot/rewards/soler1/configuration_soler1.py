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

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.rewards import RewardModelConfig
from lerobot.utils.constants import OBS_IMAGES


@RewardModelConfig.register_subclass("sole-r1")
@dataclass
class SOLER1Config(RewardModelConfig):
    """Configuration for inference with SOLE-R1.

    SOLE-R1 predicts task progress using the first, previous, and current
    observations in a trajectory. It supports an external camera, a wrist
    camera, or both.

    Args:
        model_name: Hugging Face Hub identifier or local SOLE-R1 checkpoint.
        torch_dtype: Torch dtype passed to the Transformers model loader.
        attn_implementation: Optional Transformers attention implementation.
        external_image_key: Optional external-camera observation key.
        wrist_image_key: Optional wrist-camera observation key. At least one
            camera key must be configured.
        task_key: Key containing the task description.
        default_task: Task used when ``task_key`` is absent.
        max_new_tokens: Maximum number of generated tokens.
        temperature: Sampling temperature. Zero enables greedy decoding.
        top_p: Nucleus-sampling probability.
        top_k: Top-k sampling parameter.
        max_input_length: Maximum tokenized prompt length.
        min_progress: Minimum accepted progress percentage.
        max_progress: Maximum accepted progress percentage.
        reward_scale: Scale applied to predicted percentages.
        fallback_to_previous: Reuse the previous prediction when parsing fails.
        reward_output: ``"progress"`` or ``"success"``.
        success_threshold: Threshold applied to scaled final progress when
            ``reward_output="success"``.
        downsample_to: Maximum number of uniformly spaced frames processed
            per trajectory. ``None`` processes every frame.
    """

    # A saved LeRobot wrapper contains only this configuration. The underlying
    # SOLE-R1 weights remain referenced by model_name.
    pretrained_path: str | None = None

    model_name: str = "Philip-MIT/SOLE-R1-8B"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None

    external_image_key: str | None = OBS_IMAGES + ".top"
    wrist_image_key: str | None = None
    task_key: str = "task"
    default_task: str | None = "Complete the task."
    downsample_to: int | None = 10

    max_new_tokens: int = 600
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    max_input_length: int = 16384

    min_progress: float = -100.0
    max_progress: float = 100.0
    reward_scale: float = 0.01
    fallback_to_previous: bool = True
    reward_output: str = "progress"
    success_threshold: float = 0.80

    license: str | None = "mit"
    tags: list[str] | None = field(
        default_factory=lambda: [
            "reward-model",
            "vision-language",
            "qwen3-vl",
            "robotics",
            "reasoning",
            "inference",
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

        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.max_input_length < 1:
            raise ValueError(f"max_input_length must be >= 1, got {self.max_input_length}")
        if self.min_progress >= self.max_progress:
            raise ValueError("min_progress must be smaller than max_progress")
        if self.reward_scale <= 0:
            raise ValueError(f"reward_scale must be > 0, got {self.reward_scale}")
        if self.reward_output not in {"progress", "success"}:
            raise ValueError(f"reward_output must be 'progress' or 'success', got {self.reward_output!r}")
        if self.external_image_key is None and self.wrist_image_key is None:
            raise ValueError("SOLE-R1 requires at least one of external_image_key or wrist_image_key")
        if self.downsample_to is not None and self.downsample_to < 1:
            raise ValueError(f"downsample_to must be >= 1 or None, got {self.downsample_to}")

        if self.external_image_key is not None:
            self.input_features.setdefault(
                self.external_image_key,
                PolicyFeature(
                    shape=(224, 224, 3),
                    type=FeatureType.VISUAL,
                ),
            )

        if self.wrist_image_key is not None:
            self.input_features.setdefault(
                self.wrist_image_key,
                PolicyFeature(
                    shape=(224, 224, 3),
                    type=FeatureType.VISUAL,
                ),
            )

        self.output_features.setdefault(
            "progress",
            PolicyFeature(
                shape=(1,),
                type=FeatureType.REWARD,
            ),
        )
        self.output_features.setdefault(
            "success",
            PolicyFeature(
                shape=(1,),
                type=FeatureType.REWARD,
            ),
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int] | None:
        return None

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None

    def validate_features(self) -> None:
        if self.external_image_key is not None and self.external_image_key not in self.input_features:
            raise ValueError(f"SOLE-R1 requires external image feature {self.external_image_key!r}")

        if self.wrist_image_key is not None and self.wrist_image_key not in self.input_features:
            raise ValueError(f"SOLE-R1 requires wrist image feature {self.wrist_image_key!r}")
