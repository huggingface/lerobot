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
    """Configuration for EarthRover Mini Plus robot using Frodobots SDK.

    This robot uses cloud-based control via the Frodobots SDK HTTP API.
    Camera frames are accessed directly through SDK HTTP endpoints.

    Attributes:
        sdk_url: URL of the Frodobots SDK server (default: http://localhost:8000)
    """

    sdk_url: str = "http://localhost:8000"
