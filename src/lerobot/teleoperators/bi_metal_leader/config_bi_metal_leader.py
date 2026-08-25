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
from ..metal_leader import MetalLeaderConfigBase


@TeleoperatorConfig.register_subclass("bi_metal_leader")
@dataclass
class BiMetalLeaderConfig(TeleoperatorConfig):
    """Configuration class for the bimanual Metal arm leader teleoperator.

    Each arm must sit on its own physically independent CAN bus (e.g. "can0" / "can1"): both arms
    ship with the same motor CAN IDs, so sharing a bus is impossible unless one arm's
    `motor_can_ids` is reflashed and overridden. Each arm also runs its own gravity-compensation
    thread, which is another reason not to share: two threads on one bus would double an already
    substantial frame rate.
    """

    # Typed as the unregistered base (not the registered MetalLeaderConfig): a field typed as a
    # TeleoperatorConfig choice subclass would make the draccus CLI parser tree self-referential
    # (bi -> arm -> choice registry -> bi -> ...) and recurse forever.
    left_arm_config: MetalLeaderConfigBase
    right_arm_config: MetalLeaderConfigBase
