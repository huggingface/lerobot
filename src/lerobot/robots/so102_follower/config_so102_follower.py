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
from lerobot.so102 import DEFAULT_MOTOR_IDS, validate_motor_ids

from ..config import RobotConfig


@RobotConfig.register_subclass("so102_follower")
@dataclass
class SO102FollowerConfig(RobotConfig):
    port: str
    disable_torque_on_disconnect: bool = True
    # SO-101의 6개 모터 구성 대신 wrist_yaw가 포함된 7개 ID를 검증한다.
    motor_ids: dict[str, int] = field(default_factory=lambda: DEFAULT_MOTOR_IDS.copy())
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_motor_ids(self.motor_ids)
