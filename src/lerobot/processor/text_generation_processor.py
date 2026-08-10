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
from typing import Any, Literal

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.configs.recipe import TrainingRecipe, render_message_turns
from lerobot.lerobot_types import EnvTransition, TransitionKey

from .pipeline import ProcessorStep, ProcessorStepRegistry

TextPromptKind = Literal["query", "vqa", "subtask"]
TEXT_KIND = "text_kind"
TEXT = "text"


@dataclass
@ProcessorStepRegistry.register(name="render_generation_prompt_processor")
class RenderGenerationPromptStep(ProcessorStep):
    """Render the semantic messages for one public text-generation request.

    Runtime adds ``text_kind`` and ``text`` to complementary data before the
    ordinary policy input pipeline runs. ``query`` and ``vqa`` preserve the text
    as one user payload. ``subtask`` renders the recipe prefix immediately before
    the assistant target supervising ``${subtask}``, binding text to ``${task}``.
    Inputs without ``text_kind`` pass through unchanged, so ordinary action
    preprocessing is unaffected.
    """

    recipe: TrainingRecipe | None = None

    def __post_init__(self) -> None:
        if isinstance(self.recipe, dict):
            self.recipe = TrainingRecipe.from_dict(self.recipe)

    def get_config(self) -> dict[str, Any]:
        return {"recipe": asdict(self.recipe) if self.recipe is not None else None}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        kind = complementary_data.get(TEXT_KIND)
        if kind is None:
            return transition

        text = complementary_data.get(TEXT)

        if not isinstance(text, str):
            raise TypeError(f"Text generation requires complementary data {TEXT!r} to be a string.")

        if kind in ("query", "vqa"):
            messages = [{"role": "user", "content": text}]
        elif kind == "subtask":
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
            raise ValueError(f"Unsupported text kind: {kind!r}. Expected one of: 'query', 'vqa', 'subtask'.")

        new_transition = transition.copy()
        new_complementary_data = dict(complementary_data)
        new_complementary_data.pop(TEXT_KIND)
        new_complementary_data.pop(TEXT)
        new_complementary_data["messages"] = messages
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
