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
from pathlib import Path

from lerobot.configs import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
    PreTrainedConfig,
    recipe as recipe_module,
)
from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION

ROBOCASA_CAMERA_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]


def _being_h05_default_recipe() -> TrainingRecipe:
    """Being-H0.5's subtask wording.

    The upstream release ships no subtask-generation prompt (its only official
    template is the micro-step action instruction in ``prompt_template``), so this
    wording is LeRobot's; the recipe pins it so training and inference share it.
    """
    return TrainingRecipe(
        messages=[
            MessageTurn(
                role="user",
                content=(
                    "The robot's task is: ${task}\n"
                    "What is the next concise, executable subtask? Answer with only the subtask."
                ),
                stream="high_level",
            ),
            MessageTurn(
                role="assistant",
                content="${subtask}",
                stream="high_level",
                target=True,
                if_present="subtask",
            ),
        ]
    )


def _load_recipe(path_str: str) -> TrainingRecipe:
    """Load a recipe YAML, resolving relative paths against src/lerobot/configs/recipes/."""
    path = Path(path_str)
    if not path.is_absolute() and not path.exists():
        candidate = Path(recipe_module.__file__).resolve().parent / path
        if candidate.exists():
            path = candidate
    return TrainingRecipe.from_yaml(path)


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
    # Setting `recipe_path` opts training into recipe-rendered language supervision
    # and loads the YAML into `recipe` (relative paths resolve against
    # src/lerobot/configs/recipes/).
    recipe_path: str | None = None
    # Being-H0.5's language contract; see `_being_h05_default_recipe`.
    recipe: TrainingRecipe | dict | None = field(default_factory=lambda: _being_h05_default_recipe())
    action_loss_weight: float = 1.0
    text_loss_weight: float = 0.1
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
        if self.recipe_path is not None:
            try:
                self.recipe = _load_recipe(self.recipe_path)
            except FileNotFoundError:
                if self.recipe is None:
                    raise
                # A reloaded checkpoint already carries its recipe inline; a stale
                # path only matters on the training machine that set it.
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

    @property
    def rebuild_pretrained_processors(self) -> bool:
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
