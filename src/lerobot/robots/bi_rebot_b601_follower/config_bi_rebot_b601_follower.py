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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..rebot_b601_follower import RebotB601FollowerConfig


@RobotConfig.register_subclass("bi_rebot_b601_follower")
@dataclass
class BiRebotB601FollowerConfig(RobotConfig):
    """Configuration class for the bimanual reBot B601 follower robot.

    Both arms must use the same motor family. Mixed DM/RS pairs are intentionally
    rejected until there is a concrete hardware use case and validation matrix.
    """

    left_arm_config: RebotB601FollowerConfig
    right_arm_config: RebotB601FollowerConfig

    # Top-level cameras not attached to a specific side. Keys are kept as-is in
    # observations (no `left_`/`right_` prefix). Per-arm cameras (declared on
    # `{left,right}_arm_config.cameras`) are prefixed.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        left = self.left_arm_config
        right = self.right_arm_config
        if left.motor_family is not right.motor_family:
            raise ValueError("Mixed DM/RS bimanual reBot configurations are not supported.")

        same_channel = left.can_adapter == right.can_adapter and left.port == right.port
        if same_channel:
            left_send_ids = {send_id for send_id, _ in left.motor_can_ids.values()}
            right_send_ids = {send_id for send_id, _ in right.motor_can_ids.values()}
            overlap = sorted(left_send_ids & right_send_ids)
            if overlap:
                rendered = ", ".join(f"0x{can_id:X}" for can_id in overlap)
                raise ValueError(
                    f"Bimanual reBot arms on the same CAN channel have overlapping send IDs: {rendered}."
                )
