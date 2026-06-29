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

"""Predictor output type for overhead cube-position prediction.

A *predictor* inspects a single overhead camera frame and estimates the target
cube's image-plane position and velocity, so the RTC engine can advance the cube
forward by the inference latency (the "PE gap") and feed a time-advanced
observation to the policy. Predictors are pure and transport-agnostic: the same
code backs both the async-inference client and the RTC rollout engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictorOutput:
    """Structured predictor output for overhead cube-position prediction.

    ``velocity_px_s`` is an image-plane velocity *vector* ``(vx, vy)`` (pixels per
    second), so a downstream consumer can shift the cube along its travel
    direction by ``velocity_px_s * lead_s``. It is ``None`` until two frames have
    been observed (no velocity estimate yet).
    """

    target_visible: bool | None = None
    center_px: tuple[float, float] | None = None
    velocity_px_s: tuple[float, float] | None = None
    reason: str = ""
