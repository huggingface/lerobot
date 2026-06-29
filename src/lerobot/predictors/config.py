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

"""Overhead cube-predictor configuration.

``PredictorConfig`` bundles the wiring (enable flag, camera key) and the cube
detector parameters. It is embedded by the RTC inference engine; when enabled,
the predictor advances the cube on the configured camera by the inference
latency and the engine feeds the time-advanced frame to the policy. Disabled by
default so existing behavior is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cube_predictor import CubePredictor


@dataclass
class CubePredictorConfig:
    """Red-cube HSV-mask detector parameters."""

    hue_tolerance_deg: float = 20.0
    saturation_min: float = 0.45
    value_min: float = 0.25
    min_area_ratio: float = 0.001

    def __post_init__(self):
        if not 0 <= self.hue_tolerance_deg <= 180:
            raise ValueError(f"hue_tolerance_deg must be between 0 and 180, got {self.hue_tolerance_deg}")
        if not 0 <= self.saturation_min <= 1:
            raise ValueError(f"saturation_min must be between 0 and 1, got {self.saturation_min}")
        if not 0 <= self.value_min <= 1:
            raise ValueError(f"value_min must be between 0 and 1, got {self.value_min}")
        if not 0 < self.min_area_ratio <= 1:
            raise ValueError(f"min_area_ratio must be in (0, 1], got {self.min_area_ratio}")

    def make(self) -> CubePredictor:
        return CubePredictor(
            hue_tolerance_deg=self.hue_tolerance_deg,
            saturation_min=self.saturation_min,
            value_min=self.value_min,
            min_area_ratio=self.min_area_ratio,
        )


@dataclass
class PredictorConfig:
    """Overhead cube-position predictor for time-advanced observations.

    When ``enabled``, the RTC engine runs the cube predictor on ``camera`` and
    shifts the cube forward by the inference latency before feeding the frame to
    the policy. Disabled by default -> behaviour is unchanged.
    """

    enabled: bool = False
    camera: str = "overall"
    cube: CubePredictorConfig = field(default_factory=CubePredictorConfig)

    def __post_init__(self):
        if self.enabled and not self.camera:
            raise ValueError("camera must be set when the predictor is enabled")

    def make(self) -> CubePredictor:
        """Instantiate the cube predictor callable described by this config."""
        return self.cube.make()
