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

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.converters import from_tensor_to_numpy, to_tensor
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import ACTION, OBS_STATE


def _serialize_value(value: Any) -> Any:
    value = from_tensor_to_numpy(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer | np.floating):
        return value.item()
    return value


def _serialize_stats(stats: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if not stats:
        return {}

    serialized: dict[str, dict[str, Any]] = {}
    for key, key_stats in stats.items():
        serialized[key] = {stat_name: _serialize_value(stat_value) for stat_name, stat_value in key_stats.items()}
    return serialized


def _validate_index_map(index_map: list[int], raw_dim: int, target_dim: int, name: str) -> None:
    if len(index_map) != raw_dim:
        if name == "state_index_map":
            raise ValueError("state_index_map の長さが raw_state_dim と一致しません")
        raise ValueError("action_index_map の長さが raw_action_dim と一致しません")

    if any(index < 0 or index >= target_dim for index in index_map):
        raise ValueError(f"{name} に 0〜{target_dim - 1} の範囲外の値があります")

    if len(set(index_map)) != len(index_map):
        raise ValueError(f"{name} に重複 index があります")


def _build_projection_matrix(
    target_dim: int,
    raw_dim: int,
    projection_init: str,
    projection_seed: int,
) -> torch.Tensor:
    if projection_init != "random_orthonormal_columns":
        raise ValueError(f"projection_init が不正です: {projection_init}")

    if target_dim < raw_dim:
        raise ValueError("projection 行列の形状が不正です")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(projection_seed)
    random_matrix = torch.randn(target_dim, raw_dim, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
    if q.shape != (target_dim, raw_dim):
        raise ValueError("projection 行列の形状が不正です")
    return q.contiguous()


@ProcessorStepRegistry.register(name="toyota_hsr/state_action_32_adapter")
@dataclass
class StateAction32AdapterProcessorStep(ProcessorStep):
    enabled: bool = True
    mode: str = "index_embedding"  # "index_embedding" | "linear_projection"

    target_state_dim: int = 32
    target_action_dim: int = 32
    raw_state_dim: int = 8
    raw_action_dim: int = 11

    state_index_map: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 6, 11, 12])
    action_index_map: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15])

    projection_init: str = "random_orthonormal_columns"
    projection_seed: int = 0

    apply_mean_std_normalization: bool = True
    dataset_stats: dict[str, dict[str, Any]] | None = None
    eps: float = 1e-8

    gripper_enabled: bool = False
    gripper_raw_index_in_state: int = 5
    gripper_raw_index_in_action: int = 5
    gripper_conversion_type: str = "hsr_open_close_to_pi0_angular"
    gripper_raw_open_value: float = 1.0
    gripper_raw_closed_value: float = 0.0
    gripper_pi0_open_value: float = 1.0
    gripper_pi0_closed_value: float = -1.0

    _state_projection: torch.Tensor | None = field(default=None, init=False, repr=False)
    _action_projection: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"index_embedding", "linear_projection"}:
            raise ValueError(f"mode が不正です: {self.mode}")

        if self.mode == "index_embedding":
            _validate_index_map(
                index_map=self.state_index_map,
                raw_dim=self.raw_state_dim,
                target_dim=self.target_state_dim,
                name="state_index_map",
            )
            _validate_index_map(
                index_map=self.action_index_map,
                raw_dim=self.raw_action_dim,
                target_dim=self.target_action_dim,
                name="action_index_map",
            )
        else:
            self._state_projection = _build_projection_matrix(
                target_dim=self.target_state_dim,
                raw_dim=self.raw_state_dim,
                projection_init=self.projection_init,
                projection_seed=self.projection_seed,
            )
            self._action_projection = _build_projection_matrix(
                target_dim=self.target_action_dim,
                raw_dim=self.raw_action_dim,
                projection_init=self.projection_init,
                projection_seed=self.projection_seed + 1,
            )

        if self.gripper_enabled:
            if self.gripper_raw_index_in_state < 0 or self.gripper_raw_index_in_state >= self.raw_state_dim:
                raise ValueError("gripper_raw_index_in_state が範囲外です")
            if self.gripper_raw_index_in_action < 0 or self.gripper_raw_index_in_action >= self.raw_action_dim:
                raise ValueError("gripper_raw_index_in_action が範囲外です")

    @property
    def state_projection_matrix(self) -> torch.Tensor | None:
        return self._state_projection

    @property
    def action_projection_matrix(self) -> torch.Tensor | None:
        return self._action_projection

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "target_state_dim": self.target_state_dim,
            "target_action_dim": self.target_action_dim,
            "raw_state_dim": self.raw_state_dim,
            "raw_action_dim": self.raw_action_dim,
            "state_index_map": self.state_index_map,
            "action_index_map": self.action_index_map,
            "projection_init": self.projection_init,
            "projection_seed": self.projection_seed,
            "apply_mean_std_normalization": self.apply_mean_std_normalization,
            "dataset_stats": _serialize_stats(self.dataset_stats),
            "eps": self.eps,
            "gripper_enabled": self.gripper_enabled,
            "gripper_raw_index_in_state": self.gripper_raw_index_in_state,
            "gripper_raw_index_in_action": self.gripper_raw_index_in_action,
            "gripper_conversion_type": self.gripper_conversion_type,
            "gripper_raw_open_value": self.gripper_raw_open_value,
            "gripper_raw_closed_value": self.gripper_raw_closed_value,
            "gripper_pi0_open_value": self.gripper_pi0_open_value,
            "gripper_pi0_closed_value": self.gripper_pi0_closed_value,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        if self._state_projection is not None:
            state["state_projection"] = self._state_projection.cpu()
        if self._action_projection is not None:
            state["action_projection"] = self._action_projection.cpu()
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if "state_projection" in state:
            self._state_projection = state["state_projection"].to(dtype=torch.float32)
        if "action_projection" in state:
            self._action_projection = state["action_projection"].to(dtype=torch.float32)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        new_transition = dict(transition)
        observation = dict(new_transition.get(TransitionKey.OBSERVATION) or {})

        if OBS_STATE not in observation:
            raise ValueError("observation.state が存在しません")

        state = to_tensor(observation[OBS_STATE], dtype=torch.float32)
        if state.shape[-1] != self.raw_state_dim:
            raise ValueError(
                f"observation.state の最終次元が raw_state_dim と一致しません: "
                f"{state.shape[-1]} != {self.raw_state_dim}"
            )

        state = self._apply_gripper_conversion(
            state,
            raw_index=self.gripper_raw_index_in_state,
            enabled=self.gripper_enabled,
        )
        state = self._maybe_normalize(state, OBS_STATE)
        state = self._to_target_dim(
            tensor=state,
            raw_dim=self.raw_state_dim,
            target_dim=self.target_state_dim,
            index_map=self.state_index_map,
            projection=self._state_projection,
        )

        observation[OBS_STATE] = state
        new_transition[TransitionKey.OBSERVATION] = observation

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            action_tensor = to_tensor(action, dtype=torch.float32)
            if action_tensor.shape[-1] != self.raw_action_dim:
                raise ValueError(
                    f"action の最終次元が raw_action_dim と一致しません: "
                    f"{action_tensor.shape[-1]} != {self.raw_action_dim}"
                )

            action_tensor = self._apply_gripper_conversion(
                action_tensor,
                raw_index=self.gripper_raw_index_in_action,
                enabled=self.gripper_enabled,
            )
            action_tensor = self._maybe_normalize(action_tensor, ACTION)
            action_tensor = self._to_target_dim(
                tensor=action_tensor,
                raw_dim=self.raw_action_dim,
                target_dim=self.target_action_dim,
                index_map=self.action_index_map,
                projection=self._action_projection,
            )
            new_transition[TransitionKey.ACTION] = action_tensor

        return new_transition

    def _maybe_normalize(self, tensor: torch.Tensor, feature_key: str) -> torch.Tensor:
        if not self.apply_mean_std_normalization:
            return tensor

        stats = (self.dataset_stats or {}).get(feature_key)
        if not stats or "mean" not in stats or "std" not in stats:
            return tensor

        mean = to_tensor(stats["mean"], dtype=tensor.dtype, device=tensor.device)
        std = to_tensor(stats["std"], dtype=tensor.dtype, device=tensor.device)

        if mean.shape[-1] != tensor.shape[-1] or std.shape[-1] != tensor.shape[-1]:
            raise ValueError(f"{feature_key} の統計 shape が入力 shape と一致しません")

        std = torch.clamp(std, min=self.eps)
        return (tensor - mean) / std

    def _to_target_dim(
        self,
        tensor: torch.Tensor,
        raw_dim: int,
        target_dim: int,
        index_map: list[int],
        projection: torch.Tensor | None,
    ) -> torch.Tensor:
        if tensor.shape[-1] != raw_dim:
            raise ValueError(f"入力の最終次元が raw_dim と一致しません: {tensor.shape[-1]} != {raw_dim}")

        if self.mode == "index_embedding":
            output = torch.zeros(*tensor.shape[:-1], target_dim, dtype=tensor.dtype, device=tensor.device)
            for raw_index, target_index in enumerate(index_map):
                output[..., target_index] = tensor[..., raw_index]
            return output

        if projection is None or projection.shape != (target_dim, raw_dim):
            raise ValueError("projection 行列の形状が不正です")

        projection = projection.to(dtype=tensor.dtype, device=tensor.device)
        return tensor @ projection.transpose(0, 1)

    def _apply_gripper_conversion(
        self,
        tensor: torch.Tensor,
        *,
        raw_index: int,
        enabled: bool,
    ) -> torch.Tensor:
        if not enabled:
            return tensor

        if self.gripper_conversion_type != "hsr_open_close_to_pi0_angular":
            raise ValueError(f"未対応の gripper_conversion_type: {self.gripper_conversion_type}")

        raw_span = self.gripper_raw_open_value - self.gripper_raw_closed_value
        if abs(raw_span) < self.eps:
            raise ValueError("gripper_raw_open_value と gripper_raw_closed_value が同じです")

        converted = tensor.clone()
        gripper_value = converted[..., raw_index]

        normalized = (gripper_value - self.gripper_raw_closed_value) / raw_span
        projected = (
            normalized * (self.gripper_pi0_open_value - self.gripper_pi0_closed_value)
            + self.gripper_pi0_closed_value
        )

        converted[..., raw_index] = projected
        return converted

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {
            PipelineFeatureType.OBSERVATION: dict(features.get(PipelineFeatureType.OBSERVATION, {})),
            PipelineFeatureType.ACTION: dict(features.get(PipelineFeatureType.ACTION, {})),
        }

        obs_features = new_features[PipelineFeatureType.OBSERVATION]
        if OBS_STATE in obs_features:
            obs_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.target_state_dim,))

        action_features = new_features[PipelineFeatureType.ACTION]
        for key, feature in action_features.items():
            if feature.type == FeatureType.ACTION:
                action_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=(self.target_action_dim,))

        return new_features
