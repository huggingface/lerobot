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

"""Bimanual Metal follower configuration."""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..metal_follower.config_metal_follower import MetalFollowerConfig


@RobotConfig.register_subclass("bi_metal_follower")
@dataclass(kw_only=True)
class BiMetalFollowerConfig(RobotConfig):
    """Configuration of two Metal follower arms on separate CAN interfaces.

    Metal needs no calibration file: the joint zeros are stored in the motors.

    Args:
        id (`str | None`, *optional*, defaults to `"bi_metal_follower"`):
            Robot identifier; each arm gets it with a `_left` or `_right` suffix.
        calibration_dir (`pathlib.Path | None`, *optional*):
            Unused: Metal has no calibration file.
        left_arm_config (`MetalFollowerConfig`):
            Settings of the left arm.
        right_arm_config (`MetalFollowerConfig`):
            Settings of the right arm.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras not attached to an arm. Their keys aren't prefixed, unlike those of per-arm cameras.
    """

    id: str | None = "bi_metal_follower"

    left_arm_config: MetalFollowerConfig
    right_arm_config: MetalFollowerConfig
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.left_arm_config.port == self.right_arm_config.port:
            raise ValueError("Metal arms use the same motor IDs and require distinct CAN interfaces")
