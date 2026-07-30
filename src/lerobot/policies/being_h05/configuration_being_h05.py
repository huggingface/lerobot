#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION

ROBOCASA_CAMERA_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]


@PreTrainedConfig.register_subclass("being_h05")
@dataclass
class BeingH05Config(PreTrainedConfig):
    """LeRobot configuration for BeingBeyond Being-H0.5.

    ``author_config`` is the untouched Hugging Face ``config.json`` payload. Keeping it
    nested prevents LeRobot defaults from silently changing the checkpoint architecture.
    """

    author_model_id: str = "BeingBeyond/Being-H05-2B"
    author_config: dict = field(default_factory=dict)
    tokenizer_name: str = "BeingBeyond/Being-H05-2B"
    tokenizer_revision: str | None = None
    embodiment: str = "robocasa_human"
    embodiment_id: int = 31
    image_keys: list[str] = field(default_factory=lambda: list(ROBOCASA_CAMERA_KEYS))
    image_size: int = 224
    unified_state_dim: int = 200
    unified_action_dim: int = 200
    chunk_size: int = 16
    n_action_steps: int = 8
    num_inference_steps: int = 4
    prompt_template: str = (
        "According to the instruction '{task_description}', what's the micro-step actions "
        "in the next {k} steps?"
    )
    recipe_path: str | None = None
    action_loss_weight: float = 1.0
    text_loss_weight: float = 1.0
    metadata: dict = field(default_factory=dict)
    optimizer_lr: float = 2e-5
    optimizer_weight_decay: float = 0.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 2e-6
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.unified_state_dim != 200 or self.unified_action_dim != 200:
            raise ValueError("Being-H0.5 checkpoints require the semantic 200D state/action spaces.")
        if self.chunk_size < self.n_action_steps:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        author_chunk_size = self.author_config.get("action_chunk_length")
        if author_chunk_size is not None and self.chunk_size != author_chunk_size:
            raise ValueError(
                f"chunk_size={self.chunk_size} does not match checkpoint action_chunk_length={author_chunk_size}."
            )
        if self.image_size != 224:
            raise ValueError("Released Being-H0.5 checkpoints require 224px images.")
        if self.action_loss_weight < 0 or self.text_loss_weight < 0:
            raise ValueError("Being-H0.5 loss weights must be non-negative.")
        if self.action_loss_weight == 0 and self.text_loss_weight == 0:
            raise ValueError("At least one Being-H0.5 training loss must be enabled.")

    def validate_features(self) -> None:
        visual = [key for key, value in self.input_features.items() if value.type == FeatureType.VISUAL]
        missing = [key for key in self.image_keys if key not in visual]
        if missing:
            raise ValueError(f"Being-H0.5 is missing configured camera features: {missing}")
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(12,))

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def should_rebuild_pretrained_processors(self) -> bool:
        return self.recipe_path is not None or any(
            self.normalization_mapping.get(feature, NormalizationMode.IDENTITY) != NormalizationMode.IDENTITY
            for feature in ("STATE", "ACTION")
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
