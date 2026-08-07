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
from ..metal_follower import MetalFollowerConfigBase


@RobotConfig.register_subclass("bi_metal_follower")
@dataclass
class BiMetalFollowerConfig(RobotConfig):
    """Configuration class for the bimanual Metal arm follower robot.

    Each arm must sit on its own physically independent CAN bus (e.g. "can0" /
    "can1"): both arms ship with the same motor CAN IDs, so sharing a bus is impossible
    unless one arm's `motor_can_ids` is reflashed and overridden.
    """

    # Typed as the unregistered base (not the registered MetalFollowerConfig): a field typed
    # as a RobotConfig choice subclass would make the draccus CLI parser tree self-referential
    # (bi -> arm -> choice registry -> bi -> ...) and recurse forever.
    left_arm_config: MetalFollowerConfigBase
    right_arm_config: MetalFollowerConfigBase

    # Top-level cameras not attached to a specific side (e.g. a top view). Keys are
    # kept as-is in observations (no `left_`/`right_` prefix). Per-arm cameras
    # (declared on `{left,right}_arm_config.cameras`) are prefixed.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
