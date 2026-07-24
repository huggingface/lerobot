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

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class JointActionMapping:
    source_zero: float
    source_for_target_min: float | None
    source_for_target_max: float | None
    target_min: float
    target_zero: float
    target_max: float

    def __post_init__(self) -> None:
        values = (self.source_zero, self.target_min, self.target_zero, self.target_max)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Action mapping values must be finite")
        if not self.target_min <= self.target_zero <= self.target_max:
            raise ValueError("Action mapping target_zero must be inside target limits")
        if self.source_for_target_min is None and self.source_for_target_max is None:
            raise ValueError("At least one action mapping source endpoint is required")

        for endpoint in (self.source_for_target_min, self.source_for_target_max):
            if endpoint is not None and not math.isfinite(endpoint):
                raise ValueError("Action mapping source endpoints must be finite")
            if endpoint == self.source_zero:
                raise ValueError("Action mapping source endpoints must differ from source_zero")

        if self.source_for_target_min is not None and self.source_for_target_max is not None:
            min_delta = self.source_for_target_min - self.source_zero
            max_delta = self.source_for_target_max - self.source_zero
            if min_delta * max_delta >= 0:
                raise ValueError("Action mapping source endpoints must be on opposite sides of source_zero")

    def map(self, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Action value must be finite")

        min_endpoint = self.source_for_target_min
        max_endpoint = self.source_for_target_max
        if min_endpoint is None:
            source_endpoint, target_endpoint = max_endpoint, self.target_max
        elif max_endpoint is None:
            source_endpoint, target_endpoint = min_endpoint, self.target_min
        else:
            max_side_sign = math.copysign(1.0, max_endpoint - self.source_zero)
            if (value - self.source_zero) * max_side_sign >= 0:
                source_endpoint, target_endpoint = max_endpoint, self.target_max
            else:
                source_endpoint, target_endpoint = min_endpoint, self.target_min

        scale = (target_endpoint - self.target_zero) / (source_endpoint - self.source_zero)
        mapped = self.target_zero + (value - self.source_zero) * scale
        return min(self.target_max, max(self.target_min, mapped))


@dataclass(frozen=True)
class ActionMappingProfile:
    joints: dict[str, JointActionMapping]

    def __post_init__(self) -> None:
        if not self.joints:
            raise ValueError("Action mapping must contain at least one joint")

    def map_action(self, action: dict[str, float]) -> dict[str, float]:
        expected = {f"{joint}.pos" for joint in self.joints}
        if set(action) != expected:
            raise ValueError(f"Action keys must be exactly {sorted(expected)}")
        return {key: self.joints[key.removesuffix(".pos")].map(float(value)) for key, value in action.items()}
