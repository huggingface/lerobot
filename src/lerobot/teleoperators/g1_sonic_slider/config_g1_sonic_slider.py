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


@TeleoperatorConfig.register_subclass("g1_sonic_slider")
@dataclass
class G1SonicSliderTeleopConfig(TeleoperatorConfig):
    """Pygame sliders for 29-DOF G1 poses (SONIC encoder mode 0 reference)."""

    window_width: int = 780
    window_height: int = 720
    slider_width: int = 200
    row_height: int = 22
    scroll_step: int = 40
    foot_panel_width: int = 248
    use_leg_ik: bool = True
    foot_xyz_margin: tuple[float, float, float] = (0.22, 0.18, 0.18)
    """Per-axis slider half-range (m) around standing foot FK position in pelvis frame."""
