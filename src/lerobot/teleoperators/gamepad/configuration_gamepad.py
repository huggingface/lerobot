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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gamepad")
@dataclass
class GamepadTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True

    # When True, add a 4th continuous DOF (yaw, rotation about the tool Z axis) to the
    # action vector. Right Stick X drives delta_yaw. Defaults False so existing robots
    # (RC10, etc.) and existing recorded datasets are unaffected.
    use_yaw: bool = False

    # Per-axis sign flips applied in GamepadTeleop.get_action(). Use these when the robot's
    # base-frame x/y/z axes don't match the operator's intuitive forward/back/left/right/up/down
    # at the workstation (e.g. UR10e mounted facing the operator versus RC10's frame). Defaults
    # are False so existing robots (RC10, etc.) are unaffected.
    invert_delta_x: bool = False
    invert_delta_y: bool = False
    invert_delta_z: bool = False
    invert_delta_yaw: bool = False

    # Symmetric deadzone applied to each stick axis BEFORE sign-flip. Any |delta| <= deadzone
    # is forced to 0.0; values above are linearly rescaled so 1.0 still maps to full deflection
    # (no jump at the deadzone edge). Catches resting-stick drift AND the spring-back overshoot
    # that occurs when an analog stick momentarily reads past-center on release — without this,
    # robots that latch deltas into a target (UR10e) accumulate the spring-back as a real
    # backward move. Default is small enough to be invisible on healthy sticks.
    # deadzone: float = 0.1
