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

import math
from typing import Tuple

from .config import TeleoperatorConfig
from .teleoperator import Teleoperator


def rtz_to_xyz_delta(
    delta_r: float, 
    delta_theta: float, 
    delta_z: float,
    current_r: float,
    current_theta: float,
    current_z: float
) -> Tuple[float, float, float, float, float, float]:
    """
    Convert RTZ polar coordinate deltas to XYZ cartesian coordinate deltas.
    
    Args:
        delta_r: Radial delta (distance from origin)
        delta_theta: Angular delta (rotation around Z-axis, in radians)
        delta_z: Vertical delta (same as Z in cartesian)
        current_r: Current radial position
        current_theta: Current angular position (in radians)
        current_z: Current vertical position
        
    Returns:
        Tuple of (delta_x, delta_y, delta_z, new_r, new_theta, new_z) where:
        - delta_x, delta_y, delta_z are the cartesian coordinate deltas
        - new_r, new_theta, new_z are the updated polar positions
    """
    # Update current polar position
    new_r = current_r + delta_r
    new_theta = current_theta + delta_theta
    new_z = current_z + delta_z
    
    # Convert to cartesian coordinates
    # x = r * cos(theta)
    # y = r * sin(theta)
    # z = z (same in both systems)
    
    # Calculate current cartesian position
    current_x = new_r * math.cos(new_theta)
    current_y = new_r * math.sin(new_theta)
    current_z_cart = new_z
    
    # Calculate previous cartesian position
    prev_x = current_r * math.cos(current_theta)
    prev_y = current_r * math.sin(current_theta)
    prev_z = current_z
    
    # Calculate deltas
    delta_x = current_x - prev_x
    delta_y = current_y - prev_y
    delta_z_cart = current_z_cart - prev_z
    
    return delta_x, delta_y, delta_z_cart, new_r, new_theta, new_z


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
    else:
        raise ValueError(config.type)
