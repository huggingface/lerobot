#!/usr/bin/env python

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

from dataclasses import asdict, dataclass
from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.configs.recipe import TrainingRecipe
from lerobot.datasets.language_render import render_message_turns
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_PREFIX

from .converters import create_transition
from .pipeline import ProcessorStep, ProcessorStepRegistry


def text_generation_request_to_transition(request: dict[str, Any]) -> EnvTransition:
    """Convert a generation request without dropping policy-specific batch values."""
    if not isinstance(request, dict):
        raise TypeError(f"Text generation request must be a dict, got {type(request).__name__}.")
    observation = {key: value for key, value in request.items() if key.startswith(OBS_PREFIX)}
    complementary_data = {key: value for key, value in request.items() if key not in observation}
    return create_transition(
        observation=observation or None,
        complementary_data=complementary_data,
    )


def text_generation_transition_to_request(transition: EnvTransition) -> dict[str, Any]:
    """Flatten a rendered request while preserving the original batch objects."""
    request = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
    observation = transition.get(TransitionKey.OBSERVATION)
    if isinstance(observation, dict):
        request.update(observation)
    return request


@dataclass
@ProcessorStepRegistry.register(name="render_generation_prompt_processor")
class RenderGenerationPromptStep(ProcessorStep):
    """Render the semantic messages for one public text-generation request.

    ``raw`` never consults the recipe and preserves caller text as one user
    payload. ``subtask`` renders the recipe prefix immediately before the
    assistant target that supervises ``${subtask}``, binding caller text to
    ``${task}``. The step only prepares messages; model-native formatting,
    tokenization, decoding, runtime state, and action dispatch remain outside.
    """

    recipe: TrainingRecipe | None = None

    def __post_init__(self) -> None:
        if isinstance(self.recipe, dict):
            self.recipe = TrainingRecipe.from_dict(self.recipe)

    def get_config(self) -> dict[str, Any]:
        return {"recipe": asdict(self.recipe) if self.recipe is not None else None}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        text = complementary_data.get("text")
        template = complementary_data.get("text_template", "raw")

        if not isinstance(text, str):
            raise TypeError("Text generation requires caller-owned `text` to be a string.")

        if template == "raw":
            messages = [{"role": "user", "content": text}]
        elif template == "subtask":
            if self.recipe is None:
                raise ValueError(
                    "Subtask generation requires a checkpoint recipe with an assistant target "
                    "that supervises `${subtask}`."
                )
            # Existing batch values may satisfy optional/required prefix bindings,
            # while the public text argument always owns the high-level task.
            bindings = dict(complementary_data)
            bindings["task"] = text
            turns = self.recipe.prompt_turns("subtask")
            messages = render_message_turns(turns, bindings)["messages"]
        else:
            raise ValueError(f"Unsupported text template: {template!r}. Expected one of: 'raw', 'subtask'.")

        new_transition = transition.copy()
        new_complementary_data = dict(complementary_data)
        new_complementary_data["messages"] = messages
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
