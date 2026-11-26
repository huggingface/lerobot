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
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry

_QUANTILE = re.compile(r"q(\d\d)")


def _to_float64(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=np.float64)


@dataclass
@ProcessorStepRegistry.register(name="joint_affine_processor")
class JointAffineProcessorStep(ProcessorStep):
    """Map joint values from a robot's units and frame to a policy's: `scales * x + offsets`.

    LeRobot robots report joints in degrees and grippers from 0 to 100. A policy trained on other units (radians,
    grippers from 0 to 1) or another calibration keeps them with this step: it maps the leading `len(scales)`
    entries of the state and of the action. `inverse=True` maps predicted actions back to the robot.

    Args:
        scales (`list[float]`, *optional*):
            Factor of each mapped entry, non-zero. A negative factor flips the joint. Empty disables the step.
        offsets (`list[float]`, *optional*):
            Offset of each mapped entry, in the policy's frame. `None` uses zeros.
        inverse (`bool`, *optional*, defaults to `False`):
            Map from the policy's frame to the robot's: `(x - offsets) / scales`.
    """

    scales: list[float] = field(default_factory=list)
    offsets: list[float] | None = None
    inverse: bool = False

    def __post_init__(self) -> None:
        if self.offsets is not None and len(self.offsets) != len(self.scales):
            raise ValueError(
                f"offsets must match scales in length, got {len(self.offsets)} and {len(self.scales)}."
            )
        if 0.0 in self.scales:
            raise ValueError("scales must be non-zero so the mapping can be inverted.")

    def _coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Factor and offset of the mapping in the step's direction."""
        scales = np.asarray(self.scales, dtype=np.float64)
        offsets = (
            np.zeros_like(scales) if self.offsets is None else np.asarray(self.offsets, dtype=np.float64)
        )
        if self.inverse:
            return 1.0 / scales, -offsets / scales
        return scales, offsets

    def _map(self, values: torch.Tensor | np.ndarray) -> torch.Tensor:
        values = torch.as_tensor(values)
        if not values.is_floating_point():
            values = values.float()
        factors, offsets = self._coefficients()
        mapped = values.clone()
        n = len(factors)
        mapped[..., :n] = values[..., :n] * values.new_tensor(factors) + values.new_tensor(offsets)
        return mapped

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.scales:
            return transition
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is not None and OBS_STATE in observation:
            transition[TransitionKey.OBSERVATION] = {
                **observation,
                OBS_STATE: self._map(observation[OBS_STATE]),
            }
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor | np.ndarray):
            transition[TransitionKey.ACTION] = self._map(action)
        return transition

    def transform_stats(self, stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Map the state and action stats the way the step maps their values.

        A normalizer placed after the step needs the stats in the step's output frame, while a dataset's stats are
        in the robot's. `mean`, `std`, `min`, `max` and the quantiles (`q01`, ...) are mapped; a negative factor
        swaps the low and high ones (`min` and `max`, `q01` and `q99`). Other entries, such as `count`, are kept.

        Args:
            stats (`dict[str, dict[str, Any]]`):
                Stats per feature key, then per statistic.

        Returns:
            `dict[str, dict[str, Any]]`: The mapped stats.
        """
        if not self.scales:
            return stats
        factors, offsets = self._coefficients()
        n = len(factors)
        flipped = factors < 0
        mapped_stats = dict(stats)
        for key in (OBS_STATE, ACTION):
            if key not in stats:
                continue
            feature_stats = dict(stats[key])
            for name, value in stats[key].items():
                quantile = _QUANTILE.fullmatch(name)
                if name not in {"mean", "std", "min", "max"} and not quantile:
                    continue
                mapped = _to_float64(value)
                if name == "std":
                    mapped[..., :n] *= np.abs(factors)
                else:
                    partner = (
                        f"q{100 - int(quantile[1]):02d}"
                        if quantile
                        else {"min": "max", "max": "min"}.get(name, name)
                    )
                    partner_values = _to_float64(stats[key].get(partner, value))
                    mapped[..., :n] = (
                        np.where(flipped, partner_values[..., :n], mapped[..., :n]) * factors + offsets
                    )
                feature_stats[name] = mapped
            mapped_stats[key] = feature_stats
        return mapped_stats

    def get_config(self) -> dict[str, Any]:
        return {"scales": self.scales, "offsets": self.offsets, "inverse": self.inverse}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
