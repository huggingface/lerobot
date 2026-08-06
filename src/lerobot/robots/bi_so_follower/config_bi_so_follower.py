#!/usr/bin/env python

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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..so_follower import SOFollowerConfig


@RobotConfig.register_subclass("bi_so_follower")
@dataclass
class BiSOFollowerConfig(RobotConfig):
    """Configuration for a bimanual pair of SO follower arms.

    The two arms are configured independently, then driven as one robot: observation and action keys from
    each arm are prefixed with `left_` and `right_`.

    Calibration is per arm, taken from each arm config's own `id` and `calibration_dir`.

    Args:
        left_arm_config (`SOFollowerConfig`):
            Configuration for the left arm, including its own `port`.
        right_arm_config (`SOFollowerConfig`):
            Configuration for the right arm, including its own `port`.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras not attached to either arm, such as an overhead view. These keys appear in
            observations unchanged, whereas cameras declared on an arm config are prefixed with that
            arm's side.
        id (`str`, *optional*):
            Identifier for the pair as a whole.
        calibration_dir (`Path`, *optional*):
            Unused at this level; each arm calibrates through its own config.
    """

    left_arm_config: SOFollowerConfig
    right_arm_config: SOFollowerConfig

    # Top-level cameras not attached to a specific side. Keys are kept as-is in
    # observations (no `left_`/`right_` prefix). Per-arm cameras (declared on
    # `{left,right}_arm_config.cameras`) are prefixed.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
