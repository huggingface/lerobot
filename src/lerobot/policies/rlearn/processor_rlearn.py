#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any

from lerobot.configs.types import PolicyFeature
from lerobot.constants import OBS_LANGUAGE
from lerobot.policies.rlearn.configuration_rlearn import RLearNConfig
from lerobot.processor import (
    DeviceProcessor,
    NormalizerProcessor,
    RenameProcessor,
    RobotProcessor,
    ToBatchProcessor,
    TokenizerProcessor,
    UnnormalizerProcessor,
)
from lerobot.processor.pipeline import (
    ComplementaryDataProcessor,
    EnvTransition,
    ProcessorStepRegistry,
    TransitionKey,
)


def make_rlearn_processor(
    config: RLearNConfig, dataset_stats: dict[str, dict[str, Any]] | None = None
) -> tuple[RobotProcessor, RobotProcessor]:
    """Build pre/post processors for RLearN.

    Responsibilities moved out of the model:
      - Normalize inputs (images) using dataset stats
      - Ensure batching
      - Map complementary_data.task to observation.language when available
      - Tokenize language into observation.language.tokens / attention_mask
      - Move to/from device
    """

    input_steps = [
        # No renaming by default, but keep for future extensibility
        RenameProcessor(rename_map={}),
        NormalizerProcessor(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        ToBatchProcessor(),
        RLearnLanguageFromTaskProcessor(),
        # Use the text model name for tokenizer to keep vocab aligned with text tower
        TokenizerProcessor(
            tokenizer_name=config.text_model_name,
            max_length=128,
            padding="max_length",
            truncation=True,
            padding_side="right",
        ),
        DeviceProcessor(device=config.device),
    ]

    output_steps = [
        DeviceProcessor(device="cpu"),
        UnnormalizerProcessor(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]

    return RobotProcessor(steps=input_steps, name="robot_preprocessor"), RobotProcessor(
        steps=output_steps, name="robot_postprocessor"
    )


@dataclass
@ProcessorStepRegistry.register(name="rlearn_language_from_task")
class RLearnLanguageFromTaskProcessor(ComplementaryDataProcessor):
    """Copy complementary_data['task'] into observation['observation.language'] if present.

    This ensures the model can consume a raw language string when tokenization is not used,
    while TokenizerProcessor can still create tokenized fields.
    """

    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:  # type: ignore[override]
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if not complementary_data or self.task_key not in complementary_data:
            return transition

        task = complementary_data.get(self.task_key)
        if task is None:
            return transition

        # Normalize to list[str]
        if isinstance(task, str):
            task_list = [task]
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            task_list = task
        else:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION) or {}
        # Do not overwrite if user already provided observation.language
        if OBS_LANGUAGE not in observation:
            observation[OBS_LANGUAGE] = task_list
            transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:  # noqa: D401
        # Adds nothing to features; only mirrors complementary_data.task into observation
        return features

    def get_config(self) -> dict[str, Any]:
        return {"task_key": self.task_key}
