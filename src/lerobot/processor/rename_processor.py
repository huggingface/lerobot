#!/usr/bin/env python

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
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.types import PolicyFeature
from lerobot.processor.pipeline import (
    ObservationProcessor,
    ProcessorStepRegistry,
)


@dataclass
@ProcessorStepRegistry.register(name="rename_processor")
class RenameProcessor(ObservationProcessor):
    """Rename processor that renames keys in the observation."""

    rename_map: dict[str, str] = field(default_factory=dict)

    def observation(self, observation):
        processed_obs = {}
        for key, value in observation.items():
            if key in self.rename_map:
                processed_obs[self.rename_map[key]] = value
            else:
                processed_obs[key] = value

        return processed_obs

    def get_config(self) -> dict[str, Any]:
        return {"rename_map": self.rename_map}

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        """Transforms:
        - Each key in the observation that appears in `rename_map` is renamed to its value.
        - Keys not in `rename_map` remain unchanged.
        """
        return {self.rename_map.get(k, k): v for k, v in features.items()}
