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

from .config import TeleoperatorConfig
from .teleoperator import Teleoperator


def make_teleoperator_from_config(config: TeleoperatorConfig) -> Teleoperator:
    if config.type == "keyboard":
        from .keyboard import KeyboardTeleop

        return KeyboardTeleop(config)
    elif config.type == "koch_leader":
        from .koch_leader import KochLeader

        return KochLeader(config)
    elif config.type == "so100_leader":
        from .so100_leader import SO100Leader

        return SO100Leader(config)
    elif config.type == "so101_leader":
        from .so101_leader import SO101Leader

        return SO101Leader(config)
    elif config.type == "stretch3":
        from .stretch3_gamepad import Stretch3GamePad

        return Stretch3GamePad(config)
    elif config.type == "widowx":
        from .widowx import WidowX

        return WidowX(config)
    elif config.type == "mock_teleop":
        from tests.mocks.mock_teleop import MockTeleop

        return MockTeleop(config)
    elif config.type == "gamepad":
        from .gamepad.teleop_gamepad import GamepadTeleop

        return GamepadTeleop(config)
    elif config.type == "keyboard_ee":
        from .keyboard.teleop_keyboard import KeyboardEndEffectorTeleop

        return KeyboardEndEffectorTeleop(config)
    elif config.type == "homunculus_glove":
        from .homunculus import HomunculusGlove

        return HomunculusGlove(config)
    elif config.type == "homunculus_arm":
        from .homunculus import HomunculusArm

        return HomunculusArm(config)
    elif config.type == "bi_so100_leader":
        from .bi_so100_leader import BiSO100Leader

        return BiSO100Leader(config)
    elif config.type == "reachy2_teleoperator":
        from .reachy2_teleoperator import Reachy2Teleoperator

        return Reachy2Teleoperator(config)
    else:
        raise ValueError(config.type)
