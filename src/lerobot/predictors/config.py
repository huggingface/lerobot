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

``PredictorConfig`` bundles the wiring (enable flag, camera key, time-advance
``mode``) and the cube detector parameters. It is embedded by the RTC inference
engine; when enabled, the predictor advances the cube on the configured camera by
the inference latency and the engine feeds the time-advanced observation to the
policy. Disabled by default so existing behavior is unchanged.

Time-advance ``mode`` selects how (and where) the cube is advanced:

- ``"image_shift"`` (default): analytic colour-tracked velocity; edit RGB pixels
  with :func:`shift.shift_cube_in_frame`; the policy re-encodes the edited frame.
- ``"latent_warp"``: analytic colour-tracked velocity; rigidly translate the cube
  on the policy's vision patch-token grid (:mod:`latent_warp`). No pixel edit / re-encode.
- ``"latent_flow"``: *dense optical flow* (:mod:`optical_flow`) -> per-patch velocity;
  advance each patch token along its own flow (:func:`latent_warp.warp_token_grid_by_flow`).
  No colour heuristic, no rigid-motion assumption.

The two latent modes require a policy that exposes a latent-warp hook (currently SmolVLA).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .cube_predictor import CubePredictor

PredictorMode = Literal["image_shift", "latent_warp", "latent_flow"]


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
    advances the cube forward by the inference latency before feeding it to the
    policy. ``mode`` selects how the time-advance is realised (pixel edit vs.
    latent patch-token warp). Disabled by default -> behaviour is unchanged.
    """

    enabled: bool = False
    camera: str = "overall"
    mode: PredictorMode = "image_shift"
    # latent_warp only: a patch is treated as cube when its fractional cube
    # coverage exceeds this threshold (0.0 -> any overlap counts).
    latent_mask_threshold: float = 0.0
    # latent_flow only: dense optical-flow backend and the patch-unit flow
    # magnitude below which a patch is held static (0.0 -> warp every patch).
    flow_algorithm: str = "dis"
    flow_motion_threshold: float = 0.0
    cube: CubePredictorConfig = field(default_factory=CubePredictorConfig)

    def __post_init__(self):
        if self.enabled and not self.camera:
            raise ValueError("camera must be set when the predictor is enabled")
        if self.mode not in ("image_shift", "latent_warp", "latent_flow"):
            raise ValueError(
                f"mode must be 'image_shift', 'latent_warp', or 'latent_flow', got {self.mode!r}"
            )
        if not 0 <= self.latent_mask_threshold < 1:
            raise ValueError(f"latent_mask_threshold must be in [0, 1), got {self.latent_mask_threshold}")
        if self.flow_algorithm not in ("dis", "farneback"):
            raise ValueError(f"flow_algorithm must be 'dis' or 'farneback', got {self.flow_algorithm!r}")
        if self.flow_motion_threshold < 0:
            raise ValueError(f"flow_motion_threshold must be >= 0, got {self.flow_motion_threshold}")

    def make(self) -> CubePredictor:
        """Instantiate the cube predictor callable described by this config."""
        return self.cube.make()

    def make_flow(self):
        """Instantiate the dense optical-flow estimator for ``latent_flow`` mode."""
        from .optical_flow import DenseFlowEstimator

        return DenseFlowEstimator(algorithm=self.flow_algorithm)
