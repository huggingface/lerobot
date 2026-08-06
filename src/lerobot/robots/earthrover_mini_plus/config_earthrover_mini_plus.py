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
"""Configuration for EarthRover Mini Plus robot."""

from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("earthrover_mini_plus")
@dataclass
class EarthRoverMiniPlusConfig(RobotConfig):
    """Configuration for the EarthRover Mini Plus rover.

    This robot is driven over the cloud through the Frodobots SDK's HTTP API rather than a local bus, so
    there is no serial port and no LeRobot calibration file. Camera frames come from SDK HTTP endpoints.

    Args:
        sdk_url (`str`, *optional*, defaults to `"http://localhost:8000"`):
            Base URL of the Frodobots SDK server. Commands and camera frames both go through it.
        id (`str`, *optional*):
            Identifier for this particular rover.
        calibration_dir (`Path`, *optional*):
            Unused: the rover exposes no calibration.
    """

    sdk_url: str = "http://localhost:8000"
