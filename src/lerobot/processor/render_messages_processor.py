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

        if _is_batched_language(persistent) or _is_batched_language(events):
            return self._call_batch(transition, complementary_data, persistent, events)

        timestamp = complementary_data.get("timestamp")
        if timestamp is None:
            raise KeyError("RenderMessagesStep requires sample timestamp in complementary data.")

        sample_idx = complementary_data.get("index", 0)
        rendered = render_sample(
            recipe=self.recipe,
            persistent=persistent,
            events=events,
            t=_scalar(timestamp),
            sample_idx=int(_scalar(sample_idx)),
            task=complementary_data.get("task"),
            dataset_ctx=self.dataset_ctx,
        )
        if rendered is None:
            rendered = _fallback_low_level_render(complementary_data.get("task"))
            if rendered is None:
                return None

        new_transition = transition.copy()
        new_complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        new_complementary_data.pop(LANGUAGE_PERSISTENT, None)
        new_complementary_data.pop(LANGUAGE_EVENTS, None)
        new_complementary_data.update(rendered)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return new_transition

    def _call_batch(
        self,
        transition: EnvTransition,
        complementary_data: dict[str, Any],
        persistent_batch: list,
        events_batch: list,
    ) -> EnvTransition | None:
        timestamp = complementary_data.get("timestamp")
        if timestamp is None:
            raise KeyError("RenderMessagesStep requires sample timestamp in complementary data.")

        batch_size = max(len(persistent_batch), len(events_batch))
        messages: list[list[dict[str, Any]]] = []
        message_streams: list[list[str | None]] = []
        target_message_indices: list[list[int]] = []
        keep_indices: list[int] = []

        for i in range(batch_size):
            rendered = render_sample(
                recipe=self.recipe,
                persistent=persistent_batch[i] if i < len(persistent_batch) else [],
                events=events_batch[i] if i < len(events_batch) else [],
                t=_batch_value(timestamp, i),
                sample_idx=int(_batch_value(complementary_data.get("index", 0), i)),
                task=_batch_value(complementary_data.get("task"), i),
                dataset_ctx=self.dataset_ctx,
            )
            if rendered is None:
                rendered = _fallback_low_level_render(_batch_value(complementary_data.get("task"), i))
                if rendered is None:
                    continue
            keep_indices.append(i)
            messages.append(rendered["messages"])
            message_streams.append(rendered["message_streams"])
            target_message_indices.append(rendered["target_message_indices"])

        if not messages:
            return None

        new_transition = (
            _select_batch_indices(transition, keep_indices)
            if len(keep_indices) != batch_size
            else transition.copy()
        )
        new_complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        new_complementary_data.pop(LANGUAGE_PERSISTENT, None)
        new_complementary_data.pop(LANGUAGE_EVENTS, None)
        new_complementary_data["messages"] = messages
        new_complementary_data["message_streams"] = message_streams
        new_complementary_data["target_message_indices"] = target_message_indices
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass features through unchanged; rendering only touches complementary data."""
        return features


def _scalar(value: Any) -> float | int:
    """Unwrap a tensor/array/single-element list into a Python scalar."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected a scalar, got list of length {len(value)}: {value!r}")
        return _scalar(value[0])
    return value


def _is_batched_language(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], list)


def _batch_value(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return value[index]
    if hasattr(value, "ndim") and getattr(value, "ndim") > 0:
        return _scalar(value[index])
    return _scalar(value)


def _select_batch_indices(transition: EnvTransition, indices: list[int]) -> EnvTransition:
    selected = transition.copy()
    for key in (TransitionKey.OBSERVATION, TransitionKey.COMPLEMENTARY_DATA):
        data = selected.get(key)
        if isinstance(data, dict):
            selected[key] = {k: _select_value(v, indices) for k, v in data.items()}
    action = selected.get(TransitionKey.ACTION)
    if action is not None:
        selected[TransitionKey.ACTION] = _select_value(action, indices)
    return selected


def _select_value(value: Any, indices: list[int]) -> Any:
    if isinstance(value, list) and len(value) >= len(indices):
        return [value[i] for i in indices]
    if hasattr(value, "index_select") and hasattr(value, "new_tensor") and getattr(value, "ndim", 0) > 0:
        return value.index_select(0, value.new_tensor(indices).long())
    return value


def _fallback_low_level_render(task: Any) -> dict[str, Any] | None:
    """Keep action-only samples trainable when no recipe branch matches."""
    if hasattr(task, "item"):
        task = task.item()
    if not isinstance(task, str) or not task:
        return None
    return {
        "messages": [{"role": "user", "content": task}],
        "message_streams": ["low_level"],
        "target_message_indices": [],
    }
