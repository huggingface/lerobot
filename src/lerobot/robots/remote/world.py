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

"""Optional world capability for simulated robots."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class EpisodeStatus:
    success: bool
    terminated: bool
    truncated: bool
    step: int
    reward: float


@runtime_checkable
class World(Protocol):
    def reset_world(self, task: str | None = None, seed: int | None = None) -> None: ...
    @property
    def task_description(self) -> str: ...
    @property
    def episode_status(self) -> EpisodeStatus: ...
    @property
    def sim_time(self) -> float: ...
    def render(self) -> np.ndarray: ...


def get_world(robot: object) -> World | None:
    """Discover optional episode/world support without identifying a simulator driver."""
    world = getattr(robot, "world", None)
    return world if isinstance(world, World) else None
