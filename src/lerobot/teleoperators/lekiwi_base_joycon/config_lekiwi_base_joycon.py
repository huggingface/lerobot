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
from typing import Optional, Tuple

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("lekiwi_base_joycon")
@dataclass
class LeKiwiBaseJoyconConfig(TeleoperatorConfig):
    """Configuration for the JoyCon-based LeKiwi base teleoperator."""

    which: str = "left"  # "left" or "right"
    max_speed_mps: float = 1.0
    deadzone: float = 0.20
    yaw_speed_deg: float = 30.0
    normalize_diagonal: bool = True
    invert_x: bool = False
    invert_y: bool = False
    discovery_timeout: float = 6.0
    discovery_poll_interval: float = 0.4
    serial_hint: Optional[str] = None
    yaw_buttons_left: Tuple[str, str] = ("left", "right")
    yaw_buttons_right: Tuple[str, str] = ("y", "a")
