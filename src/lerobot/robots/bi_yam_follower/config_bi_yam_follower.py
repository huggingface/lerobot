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

"""Bimanual YAM follower configuration."""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..yam_follower.config_yam_follower import YamFollowerConfig, motor_feature_names


@RobotConfig.register_subclass("bi_yam_follower")
@dataclass(kw_only=True)
class BiYamFollowerConfig(RobotConfig):
    """Configuration of two YAM follower arms on separate CAN interfaces.

    Each arm keeps its own gripper calibration file, named after `id` with a `_left` or
    `_right` suffix, so calibrate with the same `id` used for control.
    """

    id: str | None = "bi_yam_follower"

    left_arm_config: YamFollowerConfig
    right_arm_config: YamFollowerConfig

    # Top-level cameras not attached to a specific side. Keys are kept as-is in
    # observations (no `left_`/`right_` prefix). Per-arm cameras (declared on
    # `{left,right}_arm_config.cameras`) are prefixed.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.left_arm_config.port == self.right_arm_config.port:
            raise ValueError("YAM arms use duplicate motor IDs and require distinct CAN interfaces")
        if self.left_arm_config.read_only != self.right_arm_config.read_only:
            # A powered arm next to a read-only one could never be commanded.
            raise ValueError("Set the same read_only on both YAM arms")
        motor_features = {
            f"{side}_{name}" for side in ("left", "right") for name in motor_feature_names(True)
        }
        if set(self.cameras) & motor_features:
            raise ValueError("Camera names collide with YAM motor features")
