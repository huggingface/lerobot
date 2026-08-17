#!/usr/bin/env python

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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("reachy2_teleoperator")
@dataclass
class Reachy2TeleoperatorConfig(TeleoperatorConfig):
    """Configuration for reading teleoperation actions from a Reachy 2.

    Reachy 2 can act as its own teleoperator: instead of a leader arm, another Reachy 2 (or the same one in
    a different mode) reports its joint positions over the network as the action. There is no LeRobot
    calibration file; Reachy 2 manages its own calibration.

    Which joints are reported is selected by the `with_*` flags: turning a part off removes its joints
    entirely. At least one part must stay enabled.

    Args:
        ip_address (`str`, *optional*, defaults to `"localhost"`):
            Address of the Reachy 2 robot to read actions from.
        use_present_position (`bool`, *optional*, defaults to `False`):
            Whether to report each joint's present position as the action. If `False`, the joint's goal
            position is reported instead.
        with_mobile_base (`bool`, *optional*, defaults to `True`):
            Whether to include the mobile base's velocity in actions.
        with_l_arm (`bool`, *optional*, defaults to `True`):
            Whether to include the left arm's joints.
        with_r_arm (`bool`, *optional*, defaults to `True`):
            Whether to include the right arm's joints.
        with_neck (`bool`, *optional*, defaults to `True`):
            Whether to include the neck's joints.
        with_antennas (`bool`, *optional*, defaults to `True`):
            Whether to include the antennas' joints.
        id (`str`, *optional*):
            Identifier for this particular teleoperator.
        calibration_dir (`Path`, *optional*):
            Unused: Reachy 2 manages its own calibration.
    """

    # IP address of the Reachy 2 robot used as teleoperator
    ip_address: str | None = "localhost"

    # Whether to use the present position of the joints as actions
    # if False, the goal position of the joints will be used
    use_present_position: bool = False

    # Which parts of the robot to use
    with_mobile_base: bool = True
    with_l_arm: bool = True
    with_r_arm: bool = True
    with_neck: bool = True
    with_antennas: bool = True

    def __post_init__(self):
        """Validate that at least one robot part is enabled.

        Raises:
            ValueError: If every robot part is disabled, which would leave no joints to report.
        """
        if not (
            self.with_mobile_base
            or self.with_l_arm
            or self.with_r_arm
            or self.with_neck
            or self.with_antennas
        ):
            raise ValueError(
                "No Reachy2Teleoperator part used.\n"
                "At least one part of the robot must be set to True "
                "(with_mobile_base, with_l_arm, with_r_arm, with_neck, with_antennas)"
            )
