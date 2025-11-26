#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import Optional

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xlerobot_vr")
@dataclass
class XLerobotVRTeleopConfig(TeleoperatorConfig):

    
    # VR sysytem setting
    vr_enabled: bool = True
    vr_connection_timeout: float = 10.0 
    vr_data_timeout: float = 5.0  

    kp : float = 1.0  # Proportional gain for arm control

    xlevr_path: Optional[str] = "/home/jetson/XLeRobot/XLeVR" # need to be modified
