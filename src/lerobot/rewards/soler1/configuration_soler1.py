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

    SOLE-R1 consumes already-selected trajectories. Temporal sampling belongs
    to the caller or unified reward pipeline. Camera tensors use canonical
    ``(B,T,C,H,W)`` layout.

    ``temperature=1.0`` matches the public online server. Use ``0.0`` for
    greedy, reproducible offline inference.

    ``input_features`` is intentionally left empty by default. ``PolicyFeature``
    requires a fixed shape, while SOLE-R1 accepts caller-provided raw images
    with arbitrary positive spatial dimensions before 384 x 384 letterboxing.
    """

    pretrained_path: str | None = None

    model_name: str = "Philip-MIT/SOLE-R1-8B"
    torch_dtype: str = "auto"
    attn_implementation: str | None = None

    external_image_key: str | None = OBS_IMAGES + ".top"
    wrist_image_key: str | None = None
    task_key: str = "task"
    default_task: str | None = None

    max_new_tokens: int = 200
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    max_input_length: int = 16384

    smart_resize_factor: int = 28
    min_pixels: int = 3136
    max_pixels: int = 12845056

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

        if self.external_image_key is None and self.wrist_image_key is None:
            raise ValueError("SOLE-R1 requires at least one of external_image_key or wrist_image_key")
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
        if self.smart_resize_factor < 1:
            raise ValueError(f"smart_resize_factor must be >= 1, got {self.smart_resize_factor}")
        if self.min_pixels < 1 or self.max_pixels < self.min_pixels:
            raise ValueError("min_pixels and max_pixels must satisfy 1 <= min_pixels <= max_pixels")
        if self.min_progress >= self.max_progress:
            raise ValueError("min_progress must be smaller than max_progress")
        if self.reward_scale <= 0:
            raise ValueError(f"reward_scale must be > 0, got {self.reward_scale}")
        if self.reward_output not in {"progress", "success"}:
            raise ValueError(f"reward_output must be 'progress' or 'success', got {self.reward_output!r}")

        self.output_features.setdefault(
            "reward",
            PolicyFeature(shape=(1,), type=FeatureType.REWARD),
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
        if (
            self.input_features
            and self.external_image_key is not None
            and self.external_image_key not in self.input_features
        ):
            raise ValueError(f"SOLE-R1 requires external image feature {self.external_image_key!r}")

        if (
            self.input_features
            and self.wrist_image_key is not None
            and self.wrist_image_key not in self.input_features
        ):
            raise ValueError(f"SOLE-R1 requires wrist image feature {self.wrist_image_key!r}")
