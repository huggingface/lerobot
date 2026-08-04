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

"""Configuration for simulated tactile sensors."""

from dataclasses import dataclass

from ..configs import TactileSensorConfig

__all__ = ["SimulatedTactileConfig"]


@TactileSensorConfig.register_subclass("simulated")
@dataclass
class SimulatedTactileConfig(TactileSensorConfig):
    """Configuration for simulated tactile sensors.

    Used for testing and development without physical hardware.

    Attributes:
        noise_std: Standard deviation of simulated Gaussian noise. Defaults to 0.01.
        seed: Random seed for reproducible simulation. Defaults to 42.
        warmup_frames: Number of frames to discard on connect. Defaults to 5.
        simulate_delay: Whether to simulate realistic frame timing. Defaults to False.

    Example:
        ```python
        config = SimulatedTactileConfig(num_points=400, fps=30)
        with SimulatedTactile(config) as sensor:
            data = sensor.read()  # (400, 6) array
        ```
    """

    noise_std: float = 0.01
    seed: int = 42
    warmup_frames: int = 5
    simulate_delay: bool = False
