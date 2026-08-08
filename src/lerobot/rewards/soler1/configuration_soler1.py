# Copyright 2026 Philip Schroeder and The HuggingFace Inc. team. All rights reserved.
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

SYSTEM_PROMPT = (
    "You are an expert roboticist with the goal of predicting task progress percentages "
    "given frames from a video of a robot attempting to complete a task. "
    "You first think, in the form of an internal monologue, before providing your final answer. "
    "Your reasoning process MUST BE enclosed within <think> </think> tags and should include "
    "detailed reasoning. "
    "Your final answer MUST BE enclosed within <answer> </answer> tags and should be an integer "
    "(positive or negative) representing current task progress percentage. "
    "Example output format: "
    "<think>[detailed reasoning process]</think><answer>[current task progress]%</answer>"
)

DUAL_VIEW_QUESTION_TEMPLATE = (
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
    "The views on the top are from an external camera. The views on the bottom are from the "
    "robot's wrist camera. "
    "The views from the very first timestep are shown to the left. The views from the previous "
    "timestep are shown in the middle. The views from the current timestep are shown to the right. "
    "The task description is: {task_description}. "
    "The task progress for the very first timestep is 0%. The task progress for the previous "
    "timestep is {previous_progress}%. Predict the task progress for the current timestep."
)

EXTERNAL_VIEW_QUESTION_TEMPLATE = (
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
    "The views from the very first timestep are shown to the left. The views from the previous "
    "timestep are shown in the middle. The views from the current timestep are shown to the right. "
    "The task description is: {task_description}. "
    "The task progress for the very first timestep is 0%. The task progress for the previous "
    "timestep is {previous_progress}%. Predict the task progress for the current timestep."
)

DUAL_VIEW_FROM_ZERO_QUESTION_TEMPLATE = (
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
    "The views on the top are from an external camera. The views on the bottom are from the "
    "robot's wrist camera. "
    "The views from the very first timestep are shown to the left. The views from the current "
    "timestep are shown to the right. "
    "The task description is: {task_description}. "
    "The task progress for the very first timestep is 0%. Predict the task progress for the "
    "current timestep."
)

EXTERNAL_VIEW_FROM_ZERO_QUESTION_TEMPLATE = (
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
    "The views from the very first timestep are shown to the left. The views from the current "
    "timestep are shown to the right. "
    "The task description is: {task_description}. "
    "The task progress for the very first timestep is 0%. Predict the task progress for the "
    "current timestep."
)


@RewardModelConfig.register_subclass("sole-r1")
@dataclass
class SOLER1Config(RewardModelConfig):
    """Configuration for inference with SOLE-R1.

    SOLE-R1 predicts dense task progress from the first, previous, and current
    observations in an episode. The preprocessor maintains the visual episode
    context, while the reward model maintains the previously predicted progress.

    The first observation after reset returns zero without invoking the VLM.
    Subsequent observations produce a composite image and a progress prediction.

    Args:
        model_name: Hugging Face Hub identifier or local path for SOLE-R1.
        torch_dtype: Torch dtype passed to the Transformers model loader.
        attn_implementation: Optional Transformers attention implementation.
        external_image_key: Observation key containing the external-camera image.
        wrist_image_key: Optional observation key containing the wrist-camera image.
        task_key: Complementary-data key containing the task description.
        default_task: Task used when ``task_key`` is absent.
        from_zero: Predict every timestep relative to the first frame rather than
            including the previous predicted progress.
        max_new_tokens: Maximum number of generated reasoning/answer tokens.
        temperature: Sampling temperature. Zero enables greedy decoding.
        top_p: Nucleus-sampling probability.
        top_k: Top-k sampling parameter.
        max_input_length: Maximum tokenized prompt length.
        min_progress: Minimum accepted percentage prediction.
        max_progress: Maximum accepted percentage prediction.
        reward_scale: Scale applied to percentages. The default maps percentages
            to progress values in approximately ``[-1, 1]``.
        fallback_to_previous: Reuse the previous prediction when output parsing fails.
    """

    # A LeRobot wrapper checkpoint contains only this configuration. The SOLE-R1
    # weights are referenced separately by model_name.
    pretrained_path: str | None = None

    model_name: str = "Philip-MIT/SOLE-R1-8B"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None

    external_image_key: str = OBS_IMAGES + ".top"
    wrist_image_key: str | None = None
    task_key: str = "task"
    default_task: str | None = "Complete the task."

    from_zero: bool = False

    max_new_tokens: int = 600
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    max_input_length: int = 16384

    min_progress: float = -100.0
    max_progress: float = 100.0
    reward_scale: float = 0.01
    fallback_to_previous: bool = True

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

        self.input_features.setdefault(
            self.external_image_key,
            PolicyFeature(shape=(3, 224, 224), type=FeatureType.VISUAL),
        )
        if self.wrist_image_key is not None:
            self.input_features.setdefault(
                self.wrist_image_key,
                PolicyFeature(shape=(3, 224, 224), type=FeatureType.VISUAL),
            )

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
        if self.external_image_key not in self.input_features:
            raise ValueError(f"SOLE-R1 requires external image feature {self.external_image_key!r}")
        if self.wrist_image_key is not None and self.wrist_image_key not in self.input_features:
            raise ValueError(f"SOLE-R1 requires wrist image feature {self.wrist_image_key!r}")
