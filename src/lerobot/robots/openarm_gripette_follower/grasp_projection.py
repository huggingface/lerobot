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

"""Minimal (strategy, closure) <-> joint-angle map for OpenArm Gripette eval.

TODO / WARNING
--------------
This is a **deliberate fork** of the encode/decode surface from
``grabette_dataset_tools.grasp_projection`` in
https://github.com/pollen-robotics/grabette-policy
(``libs/grabette-dataset-tools/grabette_dataset_tools/grasp_projection.py``).

We do **not** depend on ``grabette-dataset-tools`` here: that package pulls the
whole training/dataset toolbox (and a lerobot pin) into the robot extra. Only
``GraspProjection.encode`` / ``decode`` and the reachable-travel constants are
needed for the follower processors and rollout.

KEEP IN SYNC with the upstream module. If the map or the measured limits
change there, update this file in the same change or grasps will silently
disagree between dataset conversion and eval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Measured travel on hardware (rgripette-v2, 2026-07-30). Must match
# grabette_dataset_tools.grasp_projection.REACHABLE_{PROXIMAL,DISTAL}.
REACHABLE_PROXIMAL = math.radians(93.5)
REACHABLE_DISTAL = math.radians(102.0)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass(frozen=True)
class GraspProjection:
    """Invertible map between joint angles and (strategy, closure)."""

    lim_prox: float = REACHABLE_PROXIMAL
    lim_dist: float = REACHABLE_DISTAL

    def __post_init__(self) -> None:
        if not (self.lim_prox > 0 and self.lim_dist > 0):
            raise ValueError(f"joint limits must be positive, got {self.lim_prox}, {self.lim_dist}")

    def decode(self, s: float, c: float) -> tuple[float, float]:
        """(strategy, closure) -> (proximal, distal) angles in radians."""
        return _clamp01(c) * self.lim_prox, _clamp01(s) * self.lim_dist

    def encode(self, prox: float, dist: float) -> tuple[float, float]:
        """(proximal, distal) angles in radians -> (strategy, closure)."""
        u = _clamp01(prox / self.lim_prox)
        v = _clamp01(dist / self.lim_dist)
        return v, u
