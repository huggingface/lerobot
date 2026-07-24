#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Utility functions for tactile sensors."""

from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .configs import TactileSensorConfig
from .tactile import TactileSensor


def make_tactile_sensors_from_configs(
    tactile_configs: dict[str, TactileSensorConfig],
) -> dict[str, TactileSensor]:
    """Create tactile sensor instances from configuration dictionary.

    Args:
        tactile_configs: Dictionary mapping sensor names to their configurations.

    Returns:
        Dictionary mapping sensor names to instantiated TactileSensor objects.

    Example:
        ```python
        configs = {
            "left_hand": Tac3DConfig(udp_port=9988),
            "right_hand": Tac3DConfig(udp_port=9989),
        }
        sensors = make_tactile_sensors_from_configs(configs)
        ```
    """
    sensors: dict[str, TactileSensor] = {}

    for key, cfg in tactile_configs.items():
        if cfg.type == "simulated":
            from .simulated import SimulatedTactile

            sensors[key] = SimulatedTactile(cfg)

        elif cfg.type == "tac3d":
            from .tac3d import Tac3DTactile

            sensors[key] = Tac3DTactile(cfg)

        else:
            try:
                sensors[key] = cast(TactileSensor, make_device_from_device_class(cfg))
            except Exception as e:
                raise ValueError(f"Error creating tactile sensor {key} with config {cfg}: {e}") from e

    return sensors
