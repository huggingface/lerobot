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

"""Configuration of a Metal arm used as a leader."""

from dataclasses import dataclass
from typing import Literal

from lerobot.robots.metal_follower import MetalFollowerConfig

from ..config import TeleoperatorConfig


@dataclass
class MetalLeaderConfig(MetalFollowerConfig):
    """Settings of one Metal leader arm: the follower's settings, floating and with a free gripper."""

    idle_mode: Literal["hold", "float"] = "float"
    gripper_kp: float = 0.0
    gripper_kd: float = 0.1


@TeleoperatorConfig.register_subclass("metal_leader")
@dataclass
class MetalLeaderTeleopConfig(TeleoperatorConfig, MetalLeaderConfig):
    """A Metal arm used as a leader: the follower's settings, floating, with a free gripper.

    It never receives an action, so it stays in float mode and you move it by hand. See
    `MetalFollowerRobotConfig` for the options; only these defaults differ.

    Args:
        idle_mode (`Literal["hold", "float"]`, *optional*, defaults to `"float"`):
            Float mode: no stiffness, gravity compensated.
        gripper_kp (`float`, *optional*, defaults to 0.0):
            No position gain, so the gripper moves freely.
        gripper_kd (`float`, *optional*, defaults to 0.1):
            Light gripper damping.
    """

    def __post_init__(self) -> None:
        MetalLeaderConfig.__post_init__(self)
