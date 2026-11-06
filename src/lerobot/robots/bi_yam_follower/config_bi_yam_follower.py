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
from ..yam_follower.config_yam_follower import YamFollowerConfig


@RobotConfig.register_subclass("bi_yam_follower")
@dataclass(kw_only=True)
class BiYamFollowerConfig(RobotConfig):
    """Configuration of two YAM follower arms on separate CAN interfaces.

    Args:
        id (`str | None`, *optional*, defaults to `"bi_yam_follower"`):
            Robot identifier. Each arm saves its gripper calibration under `id` with a `_left` or `_right`
            suffix, so calibrate with the same `id` used for control.
        calibration_dir (`pathlib.Path | None`, *optional*):
            Directory of the gripper calibration files.
        left_arm_config (`YamFollowerConfig`):
            Settings of the left arm.
        right_arm_config (`YamFollowerConfig`):
            Settings of the right arm.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras not attached to an arm. Their keys aren't prefixed, unlike those of per-arm cameras.
    """

    id: str | None = "bi_yam_follower"

    left_arm_config: YamFollowerConfig
    right_arm_config: YamFollowerConfig
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.left_arm_config.port == self.right_arm_config.port:
            raise ValueError("YAM arms use duplicate motor IDs and require distinct CAN interfaces")
