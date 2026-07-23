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

from dataclasses import dataclass
from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.configs.recipe import TrainingRecipe
from lerobot.datasets.language import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT
from lerobot.datasets.language_render import render_sample
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.utils import unwrap_scalar

from .pipeline import ProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="render_messages_processor")
class RenderMessagesStep(ProcessorStep):
    """Processor step that turns raw language columns into rendered chat messages.

    Reads ``language_persistent`` and ``language_events`` from the transition's
    complementary data, renders them through ``recipe`` at the sample timestamp,
    and replaces the raw columns with the resulting ``messages`` /
    ``message_streams`` / ``target_message_indices`` keys.
    """

    recipe: TrainingRecipe
    dataset_ctx: Any | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition | None:
        """Render messages for a single transition; return ``None`` to drop it."""
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        persistent = complementary_data.get(LANGUAGE_PERSISTENT) or []
        events = complementary_data.get(LANGUAGE_EVENTS) or []

        if not persistent and not events:
            return transition

        timestamp = complementary_data.get("timestamp")
        if timestamp is None:
            raise KeyError("RenderMessagesStep requires sample timestamp in complementary data.")

        sample_idx = complementary_data.get("index", 0)
        rendered = render_sample(
            recipe=self.recipe,
            persistent=persistent,
            events=events,
            t=unwrap_scalar(timestamp),
            sample_idx=int(unwrap_scalar(sample_idx)),
            task=complementary_data.get("task"),
            dataset_ctx=self.dataset_ctx,
        )
        if rendered is None:
            return None

        new_transition = transition.copy()
        new_complementary_data = dict(complementary_data)
        new_complementary_data.pop(LANGUAGE_PERSISTENT, None)
        new_complementary_data.pop(LANGUAGE_EVENTS, None)
        new_complementary_data.update(rendered)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass features through unchanged; rendering only touches complementary data."""
        return features
