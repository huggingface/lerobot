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

from lerobot.so101_7dof import DEFAULT_MOTOR_IDS, validate_motor_ids

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_7dof_leader")
@dataclass
class SO1017DoFLeaderConfig(TeleoperatorConfig):
    port: str
    motor_ids: dict[str, int] = field(default_factory=lambda: DEFAULT_MOTOR_IDS.copy())

    def __post_init__(self) -> None:
        validate_motor_ids(self.motor_ids)
