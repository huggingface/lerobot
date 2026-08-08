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

"""Configuration for the LeKiwi leader teleoperator.

NOTE: document fields with comments rather than a Google-style ``Attributes:`` block. draccus
parses these docstrings line by line to build ``--help``, and reads ``name: description`` lines
inside a docstring as if they were field definitions.
"""

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig
from ..so_leader import SOLeaderConfig


def default_teleop_keys() -> dict[str, str]:
    return {
        # Movement
        "forward": "w",
        "backward": "s",
        "left": "a",
        "right": "d",
        "rotate_left": "z",
        "rotate_right": "x",
        # Speed control
        "speed_up": "r",
        "speed_down": "f",
        # quit teleop
        "quit": "q",
    }


def default_speed_levels() -> list[dict[str, float]]:
    return [
        {"xy": 0.1, "theta": 30.0},  # slow
        {"xy": 0.2, "theta": 60.0},  # medium
        {"xy": 0.3, "theta": 90.0},  # fast
    ]


@TeleoperatorConfig.register_subclass("lekiwi_leader")
@dataclass
class LeKiwiLeaderConfig(TeleoperatorConfig):
    """LeKiwi is driven by two devices at once: an arm leader and a keyboard for the base."""

    # The leader arm that drives LeKiwi's follower arm.
    arm_config: SOLeaderConfig
    # Which key triggers each base movement / speed action.
    teleop_keys: dict[str, str] = field(default_factory=default_teleop_keys)
    # Selectable (xy m/s, theta deg/s) pairs for the base, cycled with the speed keys.
    speed_levels: list[dict[str, float]] = field(default_factory=default_speed_levels)
