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

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..manifest import ProcessorSpec
from ..normalize import Normalizer

OBS_STATE = "observation.state"
ACTION = "action"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"

__all__ = ["ExportProcessorPipeline", "build_processor_pipeline"]


class _ExportProcessor(Protocol):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]: ...

    def reset(self) -> None: ...


class ExportProcessorPipeline:
    def __init__(self, processors: list[_ExportProcessor] | None = None) -> None:
        self._processors = processors or []

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = data
        for processor in self._processors:
            result = processor(result)
        return result

    def reset(self) -> None:
        for processor in self._processors:
            processor.reset()


class _NormalizeProcessor:
    def __init__(self, normalizer: Normalizer) -> None:
        self._normalizer = normalizer

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._normalizer.normalize_inputs(data)

    def reset(self) -> None:
        return None


class _DenormalizeProcessor:
    def __init__(self, normalizer: Normalizer, features: list[str]) -> None:
        self._normalizer = normalizer
        self._features = features

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in data.items() if key != "task"}
        for feature in self._features:
            if feature in result:
                result[feature] = self._normalizer.denormalize_outputs(result[feature], key=feature)
        return result

    def reset(self) -> None:
        return None


class _RelativeActionsProcessor:
    def __init__(
        self,
        *,
        enabled: bool,
        exclude_joints: list[str] | None = None,
        action_names: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.exclude_joints = exclude_joints or []
        self.action_names = action_names
        self.last_state: np.ndarray | None = None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = data.get(OBS_STATE)
        if state is not None:
            self.last_state = np.asarray(state)

        if not self.enabled or ACTION not in data or state is None:
            return data

        result = dict(data)
        result[ACTION] = _to_relative_actions(np.asarray(data[ACTION]), np.asarray(state), self._build_mask)
        return result

    def _build_mask(self, action_dim: int) -> np.ndarray:
        if not self.exclude_joints or self.action_names is None:
            return np.ones((action_dim,), dtype=np.float32)

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        mask: list[float] = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(0.0 if is_excluded else 1.0)
        if len(mask) < action_dim:
            mask.extend([1.0] * (action_dim - len(mask)))
        return np.asarray(mask, dtype=np.float32)

    def reset(self) -> None:
        self.last_state = None


class _AbsoluteActionsProcessor:
    def __init__(self, *, enabled: bool, relative_processor: _RelativeActionsProcessor | None) -> None:
        self.enabled = enabled
        self._relative_processor = relative_processor

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return data
        if self._relative_processor is None:
            raise RuntimeError("absolute_actions requires a paired relative_actions preprocessor")
        if self._relative_processor.last_state is None:
            raise RuntimeError("absolute_actions requires cached state from relative_actions")
        if ACTION not in data:
            return data

        result = dict(data)
        action = np.asarray(data[ACTION])
        state = self._relative_processor.last_state
        result[ACTION] = _to_absolute_actions(action, state, self._relative_processor._build_mask)
        return result

    def reset(self) -> None:
        return None


class _PI05PrepareStateProcessor:
    def __init__(self, *, max_state_dim: int) -> None:
        self._max_state_dim = max_state_dim

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if OBS_STATE not in data:
            return data
        state = np.asarray(data[OBS_STATE])
        if state.shape[-1] >= self._max_state_dim:
            return data
        result = dict(data)
        padding = np.zeros((*state.shape[:-1], self._max_state_dim - state.shape[-1]), dtype=state.dtype)
        result[OBS_STATE] = np.concatenate([state, padding], axis=-1)
        return result

    def reset(self) -> None:
        return None


class _TokenizeProcessor:
    def __init__(
        self,
        *,
        tokenizer_path: Path,
        max_length: int,
        padding_side: str = "right",
        padding: str = "max_length",
        truncation: bool = True,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers is required for exported tokenize processor") from e

        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        self._tokenizer.padding_side = padding_side
        self._max_length = max_length
        self._padding = padding
        self._truncation = truncation

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "task" not in data:
            return data
        task = data["task"]
        if isinstance(task, str):
            task = [task]
        tokenized = self._tokenizer(
            task,
            max_length=self._max_length,
            truncation=self._truncation,
            padding=self._padding,
            return_tensors="np",
        )
        result = dict(data)
        result.pop("task", None)
        result[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"].astype(np.int64)
        result[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].astype(np.bool_)
        return result

    def reset(self) -> None:
        return None


def build_processor_pipeline(
    specs: list[ProcessorSpec] | None,
    *,
    package_path: Path,
    normalizer: Normalizer | None = None,
    relative_processor: _RelativeActionsProcessor | None = None,
) -> tuple[ExportProcessorPipeline, _RelativeActionsProcessor | None]:
    processors: list[_ExportProcessor] = []
    current_relative = relative_processor

    for spec in specs or []:
        if spec.type == "normalize":
            if normalizer is None:
                raise ValueError("normalize processor declared but normalization stats could not be loaded")
            processors.append(_NormalizeProcessor(normalizer))
        elif spec.type == "denormalize":
            if normalizer is None:
                raise ValueError("denormalize processor declared but normalization stats could not be loaded")
            processors.append(_DenormalizeProcessor(normalizer, spec.features or [ACTION]))
        elif spec.type == "relative_actions":
            current_relative = _RelativeActionsProcessor(
                enabled=bool(spec.extra_params.get("enabled", False)),
                exclude_joints=list(spec.extra_params.get("exclude_joints", [])),
                action_names=_optional_str_list(spec.extra_params.get("action_names")),
            )
            processors.append(current_relative)
        elif spec.type == "absolute_actions":
            processors.append(
                _AbsoluteActionsProcessor(
                    enabled=bool(spec.extra_params.get("enabled", False)),
                    relative_processor=current_relative,
                )
            )
        elif spec.type == "pi05_prepare_state":
            processors.append(
                _PI05PrepareStateProcessor(max_state_dim=int(spec.extra_params["max_state_dim"]))
            )
        elif spec.type == "tokenize":
            if spec.artifact is None:
                raise ValueError("tokenize processor requires a tokenizer artifact")
            processors.append(
                _TokenizeProcessor(
                    tokenizer_path=package_path / spec.artifact,
                    max_length=int(spec.extra_params["max_length"]),
                    padding_side=str(spec.extra_params.get("padding_side", "right")),
                    padding=str(spec.extra_params.get("padding", "max_length")),
                    truncation=bool(spec.extra_params.get("truncation", True)),
                )
            )
        else:
            raise ValueError(f"Unknown export processor type: {spec.type!r}")

    return ExportProcessorPipeline(processors), current_relative


def _optional_str_list(value: object) -> list[str] | None:
    if value is None:
        return None
    return [str(item) for item in value]


def _to_relative_actions(
    action: np.ndarray,
    state: np.ndarray,
    build_mask: Callable[[int], np.ndarray],
) -> np.ndarray:
    mask = build_mask(action.shape[-1]).astype(action.dtype)
    state_offset = state[..., : mask.shape[0]] * mask
    if action.ndim == 3:
        state_offset = np.expand_dims(state_offset, axis=-2)
    result = action.copy()
    result[..., : mask.shape[0]] -= state_offset
    return result


def _to_absolute_actions(
    action: np.ndarray,
    state: np.ndarray,
    build_mask: Callable[[int], np.ndarray],
) -> np.ndarray:
    mask = build_mask(action.shape[-1]).astype(action.dtype)
    state_offset = state[..., : mask.shape[0]] * mask
    if action.ndim == 3:
        state_offset = np.expand_dims(state_offset, axis=-2)
    result = action.copy()
    result[..., : mask.shape[0]] += state_offset
    return result
