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

from enum import Enum
from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .config import TeleoperatorConfig
from .teleoperator import Teleoperator


class TeleopEvents(Enum):
    """Shared constants for teleoperator events across teleoperators."""

    SUCCESS = "success"
    FAILURE = "failure"
    RERECORD_EPISODE = "rerecord_episode"
    IS_INTERVENTION = "is_intervention"
    TERMINATE_EPISODE = "terminate_episode"


def make_teleoperator_from_config(config: TeleoperatorConfig) -> Teleoperator:
    # TODO(Steven): Consider just using the make_device_from_device_class for all types
    if config.type == "keyboard":
        from .keyboard import KeyboardTeleop

        return KeyboardTeleop(config)
    elif config.type == "koch_leader":
        from .koch_leader import KochLeader

        return KochLeader(config)
    elif config.type == "omx_leader":
        from .omx_leader import OmxLeader

        return OmxLeader(config)
    elif config.type == "so100_leader":
        from .so_leader import SO100Leader

        return SO100Leader(config)
    elif config.type == "so101_leader":
        from .so_leader import SO101Leader

        return SO101Leader(config)
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
    elif config.type == "bi_so_leader":
        from .bi_so_leader import BiSOLeader

        return BiSOLeader(config)
    elif config.type == "reachy2_teleoperator":
        from .reachy2_teleoperator import Reachy2Teleoperator

        return Reachy2Teleoperator(config)
    else:
        try:
            return cast(Teleoperator, make_device_from_device_class(config))
        except Exception as e:
            raise ValueError(f"Error creating robot with config {config}: {e}") from e
