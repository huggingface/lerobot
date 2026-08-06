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
    """Configuration for the gamepad teleoperator.

    Args:
        use_gripper (`bool`, *optional*, defaults to `True`):
            Whether to include a `gripper` entry in the produced actions.
        hidapi_fallback (`bool`, *optional*, defaults to `False`):
            Read the gamepad through `hidapi` instead of `pygame`. Set this on macOS if `pygame` does not
            reliably detect input from your controller.
        id (`str`, *optional*):
            Identifier for this particular unit, used to tell apart several teleoperators of the same
            type. It also names the calibration file, so keep it stable for a given piece of hardware.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to a per-teleoperator directory under
            the LeRobot calibration home.
    """

    use_gripper: bool = True
    # Use hidapi instead of pygame for controllers that pygame cannot detect reliably.
    hidapi_fallback: bool = False
