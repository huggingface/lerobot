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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("almond_axol_vr")
@dataclass
class AlmondAxolVRConfig(TeleoperatorConfig):
    """Configuration for the Almond Axol VR teleoperator.

    The teleoperator runs a WebSocket server the VR headset connects to (open
    https://axol.almond.bot in the headset browser and enter the robot
    machine's address) and an inverse-kinematics subprocess that turns the
    tracked hand poses into joint-position actions for the ``almond_axol``
    robot. Session parameters beyond these fields (rest poses, smoothing, IK
    weights, TLS certificates) use the almond-axol SDK defaults.

    Attributes:
        port: Port of the VR WebSocket (wss) server the headset connects to.
        has_gripper: Whether the robot has grippers. ``False`` (the gripperless
            SKU) drops the gripper keys from the emitted actions so they match
            the robot's action features.
    """

    port: int = 8000
    has_gripper: bool = True
