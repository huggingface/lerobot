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
from typing import TypeAlias

from lerobot.cameras import CameraConfig

from ..config import RobotConfig, parse_max_relative_target_cli


@dataclass
class SOFollowerConfig:
    """Base configuration class for SO Follower robots."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    #
    # Note: this is declared as a plain string for CLI compatibility with Draccus/argparse.
    # Accepted formats:
    # - scalar: "5.0"
    # - json dict: '{"shoulder_pan": 5.0, "elbow_flex": 3.0}'
    max_relative_target: str = ""

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    def __post_init__(self) -> None:
        # Let parent classes run their post-init (if any)
        post_init = getattr(super(), "__post_init__", None)
        if callable(post_init):
            post_init()

        self.max_relative_target = parse_max_relative_target_cli(self.max_relative_target)


SO100FollowerConfig: TypeAlias = SOFollowerRobotConfig
SO101FollowerConfig: TypeAlias = SOFollowerRobotConfig
