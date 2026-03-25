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
from pathlib import Path

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("bi_so_follower_simulated")
@dataclass
class BiSOFollowerSimulatedConfig(RobotConfig):
    """Configuration for the bimanual simulated SO follower backed by MuJoCo."""

    sim_root: Path | None = None
    bridge_path: Path | None = None
    xml_path: Path | None = None
    bridge_factory_name: str = "make_task2_bimanual_buses"

    robot_dofs: int = 6
    render_size: tuple[int, int] | None = None
    camera_names: tuple[str, ...] = field(default_factory=tuple)

    realtime: bool = True
    slowmo: float = 1.0
    launch_viewer: bool = False

    max_relative_target: float | dict[str, float] | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.robot_dofs != 6:
            raise ValueError(f"`robot_dofs` must be 6 for the SO-arm Task2 bridge, got {self.robot_dofs}.")

        if self.camera_names and self.render_size is None:
            raise ValueError("`render_size` is required when `camera_names` is not empty.")

        if self.render_size is not None:
            if len(self.render_size) != 2 or any(x <= 0 for x in self.render_size):
                raise ValueError(f"Invalid `render_size`: {self.render_size}.")

