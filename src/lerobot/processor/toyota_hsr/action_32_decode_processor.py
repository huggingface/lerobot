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

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.converters import to_tensor
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry

from .state_action_32_adapter_processor import _build_projection_matrix, _validate_index_map


@ProcessorStepRegistry.register(name="toyota_hsr/action_32_decode")
@dataclass
class Action32DecodeProcessorStep(ProcessorStep):
    enabled: bool = True
    mode: str = "index_embedding"  # "index_embedding" | "linear_projection"

    target_action_dim: int = 32
    raw_action_dim: int = 11
    action_index_map: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15])

    projection_init: str = "random_orthonormal_columns"
    projection_seed: int = 0

    gripper_enabled: bool = False
    gripper_raw_index_in_action: int = 5
    gripper_conversion_type: str = "hsr_open_close_to_pi0_angular"
    gripper_raw_open_value: float = 1.0
    gripper_raw_closed_value: float = 0.0
    gripper_pi0_open_value: float = 1.0
    gripper_pi0_closed_value: float = -1.0

    eps: float = 1e-8

    _action_projection: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"index_embedding", "linear_projection"}:
            raise ValueError(f"mode が不正です: {self.mode}")

        if self.mode == "index_embedding":
            _validate_index_map(
                index_map=self.action_index_map,
                raw_dim=self.raw_action_dim,
                target_dim=self.target_action_dim,
                name="action_index_map",
            )
        else:
            self._action_projection = _build_projection_matrix(
                target_dim=self.target_action_dim,
                raw_dim=self.raw_action_dim,
                projection_init=self.projection_init,
                projection_seed=self.projection_seed + 1,
            )

        if self.gripper_enabled:
            if self.gripper_raw_index_in_action < 0 or self.gripper_raw_index_in_action >= self.raw_action_dim:
                raise ValueError("gripper_raw_index_in_action が範囲外です")

    @property
    def action_projection_matrix(self) -> torch.Tensor | None:
        return self._action_projection

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "target_action_dim": self.target_action_dim,
            "raw_action_dim": self.raw_action_dim,
            "action_index_map": self.action_index_map,
            "projection_init": self.projection_init,
            "projection_seed": self.projection_seed,
            "gripper_enabled": self.gripper_enabled,
            "gripper_raw_index_in_action": self.gripper_raw_index_in_action,
            "gripper_conversion_type": self.gripper_conversion_type,
            "gripper_raw_open_value": self.gripper_raw_open_value,
            "gripper_raw_closed_value": self.gripper_raw_closed_value,
            "gripper_pi0_open_value": self.gripper_pi0_open_value,
            "gripper_pi0_closed_value": self.gripper_pi0_closed_value,
            "eps": self.eps,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        if self._action_projection is not None:
            state["action_projection"] = self._action_projection.cpu()
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if "action_projection" in state:
            self._action_projection = state["action_projection"].to(dtype=torch.float32)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        new_transition = dict(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        action_tensor = to_tensor(action, dtype=torch.float32)
        if action_tensor.shape[-1] != self.target_action_dim:
            raise ValueError(
                f"action の最終次元が target_action_dim と一致しません: "
                f"{action_tensor.shape[-1]} != {self.target_action_dim}"
            )

        decoded_action = self._decode_action(action_tensor)
        decoded_action = self._apply_inverse_gripper_conversion(decoded_action)
        new_transition[TransitionKey.ACTION] = decoded_action
        return new_transition

    def _decode_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.mode == "index_embedding":
            slices = [action[..., index] for index in self.action_index_map]
            return torch.stack(slices, dim=-1)

        if self._action_projection is None or self._action_projection.shape != (
            self.target_action_dim,
            self.raw_action_dim,
        ):
            raise ValueError("projection 行列の形状が不正です")

        projection = self._action_projection.to(dtype=action.dtype, device=action.device)
        # For orthonormal columns, this is the left-inverse of the encoding projection.
        return action @ projection

    def _apply_inverse_gripper_conversion(self, action: torch.Tensor) -> torch.Tensor:
        if not self.gripper_enabled:
            return action

        if self.gripper_conversion_type != "hsr_open_close_to_pi0_angular":
            raise ValueError(f"未対応の gripper_conversion_type: {self.gripper_conversion_type}")

        projected_span = self.gripper_pi0_open_value - self.gripper_pi0_closed_value
        if abs(projected_span) < self.eps:
            raise ValueError("gripper_pi0_open_value と gripper_pi0_closed_value が同じです")

        decoded = action.clone()
        projected_value = decoded[..., self.gripper_raw_index_in_action]

        normalized = (projected_value - self.gripper_pi0_closed_value) / projected_span
        raw_value = (
            normalized * (self.gripper_raw_open_value - self.gripper_raw_closed_value)
            + self.gripper_raw_closed_value
        )

        decoded[..., self.gripper_raw_index_in_action] = raw_value
        return decoded

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {
            PipelineFeatureType.OBSERVATION: dict(features.get(PipelineFeatureType.OBSERVATION, {})),
            PipelineFeatureType.ACTION: dict(features.get(PipelineFeatureType.ACTION, {})),
        }

        action_features = new_features[PipelineFeatureType.ACTION]
        for key, feature in action_features.items():
            if feature.type == FeatureType.ACTION:
                action_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=(self.raw_action_dim,))

        return new_features
