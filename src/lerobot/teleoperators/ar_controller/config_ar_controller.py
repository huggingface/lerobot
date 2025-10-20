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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("ar_controller")
@dataclass
class ARControllerConfig(TeleoperatorConfig):
    bi_controller: bool = False
    use_degrees: bool = True
    base_max_speed_mps: float = 1.0
    base_yaw_speed_deg: float = 30.0
    thumbstick_deadzone: float = 0.15
    mount_pan_key: str = "mount_pan.pos"
    mount_tilt_key: str = "mount_tilt.pos"
    mount_pan_speed_deg: float = 45.0
    mount_tilt_speed_deg: float = 45.0
    mount_pan_limits: tuple[float, float] = (-90.0, 90.0)
    mount_tilt_limits: tuple[float, float] = (-45.0, 45.0)
