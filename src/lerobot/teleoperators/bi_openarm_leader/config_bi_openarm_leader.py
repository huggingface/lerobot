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

from dataclasses import dataclass

from ..config import TeleoperatorConfig
from ..openarm_leader import OpenArmLeaderConfigBase


@TeleoperatorConfig.register_subclass("bi_openarm_leader")
@dataclass
class BiOpenArmLeaderConfig(TeleoperatorConfig):
    """Configuration for a bimanual pair of OpenArm leader arms.

    The two arms are configured independently, then driven as one teleoperator: action keys from each arm
    are prefixed with `left_` and `right_`.

    Calibration is per arm, taken from each arm config's own `id` and `calibration_dir`.

    Args:
        left_arm_config (`OpenArmLeaderConfigBase`):
            Configuration for the left arm, including its own `port` and `motor_config`.
        right_arm_config (`OpenArmLeaderConfigBase`):
            Configuration for the right arm, including its own `port` and `motor_config`.
        id (`str`, *optional*):
            Identifier for the pair as a whole.
        calibration_dir (`Path`, *optional*):
            Unused at this level; each arm calibrates through its own config.
    """

    left_arm_config: OpenArmLeaderConfigBase
    right_arm_config: OpenArmLeaderConfigBase
