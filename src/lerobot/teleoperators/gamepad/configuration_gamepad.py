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

    # If True, read the right-stick horizontal axis (axis 3) and emit a
    # ``delta_yaw`` field in addition to delta_x / delta_y / delta_z. Enables
    # yaw control for envs that accept a 5D action (dx, dy, dz, dyaw, gripper).
    # Right Stick X drives delta_yaw. Defaults False so existing robots
    # (RC10, etc.) and existing recorded datasets are unaffected.
    use_yaw: bool = False

    # SDL/pygame button index that advances to the next SARM recording stage
    # (consumed by StageAnnotatorProcessorStep). Default 0: Cross on DualSense
    # via SDL2, which the existing gamepad update() loop leaves unhandled.
    # Remap per-controller via env JSON if 0 collides on your pad.
    stage_advance_button: int = 0

    # When True, emit gripper as a continuous width target in [0, 1]
    # (close → 0.0, open → 1.0, stay → hold last). Pairs with the env-side
    # `record_gripper_width` flag to record continuous gripper datasets.
    record_gripper_width: bool = False

    # Per-axis sign flips applied in GamepadTeleop.get_action(). Use these when the robot's
    # base-frame x/y/z axes don't match the operator's intuitive forward/back/left/right/up/down
    # at the workstation (e.g. UR10e mounted facing the operator versus RC10's frame). Defaults
    # are False so existing robots (RC10, etc.) are unaffected.
    invert_delta_x: bool = False
    invert_delta_y: bool = False
    invert_delta_z: bool = False
    invert_delta_yaw: bool = False
