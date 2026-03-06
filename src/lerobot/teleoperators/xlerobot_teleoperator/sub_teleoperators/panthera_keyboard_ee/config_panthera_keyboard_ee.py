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

from ....config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("panthera_keyboard_ee")
@dataclass
class PantheraKeyboardEETeleopConfig(TeleoperatorConfig):
    """Keyboard teleop config for Panthera EE polar/cartesian commands.

    Defaults are chosen to avoid collisions with KeyboardRoverTeleop
    (W/A/S/D/Q/E/X for base driving).
    """

    key_radial_out: str = "t"
    key_radial_in: str = "g"
    key_orbit_ccw: str = "f"
    key_orbit_cw: str = "h"
    key_up: str = "r"
    key_down: str = "v"

    key_pitch_pos: str = "i"
    key_pitch_neg: str = "k"
    key_yaw_pos: str = "j"
    key_yaw_neg: str = "l"
    key_roll_pos: str = "u"
    key_roll_neg: str = "o"

    key_gripper_close: str = "z"
    key_gripper_open: str = "c"
