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

from ....config import RobotConfig


@RobotConfig.register_subclass("lekiwi_base")
@dataclass
class LeKiwiBaseConfig(RobotConfig):
    port: str = "/dev/ttyACM0"

    disable_torque_on_disconnect: bool = True

    # Geometry and kinematics parameters of the omni-wheel base
    wheel_radius_m: float = 0.05
    base_radius_m: float = 0.125
    wheel_axis_angles_deg: tuple[float, float, float] = (240.0, 0.0, 120.0)
    # Motor IDs in order: (left, back, right).
    base_motor_ids: tuple[int, int, int] = (2, 1, 3)
    max_wheel_raw: int = 3000

    # Whether to perform the UART handshake when connecting to the bus.
    handshake_on_connect: bool = True
