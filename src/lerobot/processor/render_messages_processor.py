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

import numpy as np

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.configs.recipe import TrainingRecipe
from lerobot.datasets.language import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT
from lerobot.datasets.language_render import render_sample
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.utils import unwrap_scalar

from .pipeline import ProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="render_messages_processor")
class RenderMessagesStep(ProcessorStep):
    """Turn raw language columns into recipe-defined messages and supervision.

    Reads ``language_persistent`` and ``language_events`` from complementary
    data, renders them at each sample timestamp, and replaces the raw columns
    with ``messages``, ``message_streams``, and ``target_message_indices``.
    Batched inputs are filtered to samples with applicable supervision; samples
    without language annotations use their task string as low-level supervision
    when one is available.
    """

    recipe: TrainingRecipe
    dataset_ctx: Any | None = None

    def __post_init__(self) -> None:
        if isinstance(self.recipe, dict):
            self.recipe = TrainingRecipe.from_dict(self.recipe)

    def get_config(self) -> dict[str, Any]:
        return {"recipe": asdict(self.recipe)}

    def __call__(self, transition: EnvTransition) -> EnvTransition | None:
        """Render messages, preserving unannotated samples and dropping unmatched annotated ones."""
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        persistent = complementary_data.get(LANGUAGE_PERSISTENT) or []
        events = complementary_data.get(LANGUAGE_EVENTS) or []

        if not persistent and not events:
            # A dataset without language annotations remains usable: render its
            # task as low-level supervision, or pass it through when no task exists.
            rendered = _fallback_low_level_render(complementary_data.get("task"))
            if rendered is None:
                return transition
            new_transition = transition.copy()
            new_complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
            new_complementary_data.update(rendered)
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
            return new_transition

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
            t=unwrap_scalar(timestamp),
            sample_idx=int(unwrap_scalar(sample_idx)),
            task=complementary_data.get("task"),
            dataset_ctx=self.dataset_ctx,
        )
        if rendered is None:
            # Language is present but this sparse frame has no applicable recipe
            # branch. Keep it only when task-level action supervision is possible.
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
        """Render a language batch.

        Non-empty persistent and event batches must have the same size. Either
        list may be empty when that language column is absent from the batch.
        """
        timestamp = complementary_data.get("timestamp")
        if timestamp is None:
            raise KeyError("RenderMessagesStep requires sample timestamp in complementary data.")

        non_empty_batch_sizes = {len(batch) for batch in (persistent_batch, events_batch) if batch}
        if len(non_empty_batch_sizes) > 1:
            raise ValueError(
                "Batched language columns must have equal lengths when both are non-empty, "
                f"got persistent={len(persistent_batch)} and events={len(events_batch)}."
            )
        batch_size = next(iter(non_empty_batch_sizes), 0)
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
            _select_batch_indices(transition, keep_indices, batch_size)
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


def _is_batched_language(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], list)


def _batch_value(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return value[index]
    if hasattr(value, "ndim") and value.ndim > 0:
        return unwrap_scalar(value[index])
    return unwrap_scalar(value)


def _select_batch_indices(transition: EnvTransition, indices: list[int], batch_size: int) -> EnvTransition:
    selected = transition.copy()
    for key in (TransitionKey.OBSERVATION, TransitionKey.COMPLEMENTARY_DATA):
        data = selected.get(key)
        if isinstance(data, dict):
            selected[key] = {
                name: _select_value(value, indices, batch_size, f"{key}.{name}")
                for name, value in data.items()
            }
    action = selected.get(TransitionKey.ACTION)
    if action is not None:
        selected[TransitionKey.ACTION] = _select_value(action, indices, batch_size, str(TransitionKey.ACTION))
    return selected


def _select_value(value: Any, indices: list[int], batch_size: int, path: str) -> Any:
    if isinstance(value, dict):
        return {key: _select_value(item, indices, batch_size, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        if len(value) != batch_size:
            raise ValueError(
                f"Cannot filter batched field {path!r}: expected {batch_size} values, got {len(value)}."
            )
        return [value[i] for i in indices]
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return value[indices]
    if hasattr(value, "index_select") and hasattr(value, "new_tensor") and getattr(value, "ndim", 0) > 0:
        return value.index_select(0, value.new_tensor(indices).long())
    return value


def _fallback_low_level_render(task: Any) -> dict[str, Any] | None:
    """Keep action-only samples trainable when no recipe branch matches."""
    if hasattr(task, "item"):
        task = task.item()
    if isinstance(task, list):
        if not task:
            return None
        messages = []
        message_streams = []
        target_message_indices = []
        missing_indices = []
        for index, t in enumerate(task):
            rendered = _fallback_low_level_render(t)
            if rendered is None:
                missing_indices.append(index)
                continue
            messages.append(rendered["messages"])
            message_streams.append(rendered["message_streams"])
            target_message_indices.append(rendered["target_message_indices"])
        if missing_indices:
            if len(missing_indices) == len(task):
                return None
            raise ValueError(
                "Batched low-level fallback requires a non-empty task for every sample; "
                f"missing task at indices {missing_indices}."
            )
        return {
            "messages": messages,
            "message_streams": message_streams,
            "target_message_indices": target_message_indices,
        }
    if not isinstance(task, str) or not task:
        return None
    return {
        "messages": [{"role": "user", "content": task}],
        "message_streams": ["low_level"],
        "target_message_indices": [],
    }
