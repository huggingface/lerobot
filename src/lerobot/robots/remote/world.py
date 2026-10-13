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


@dataclass(frozen=True)
class EpisodeStatus:
    success: bool
    terminated: bool
    truncated: bool
    step: int
    reward: float


@runtime_checkable
class World(Protocol):
    """What rollout strategies need from a simulated robot: episode reset, task and status."""

    def reset_world(self, seed: int | None = None) -> None: ...
    @property
    def task_description(self) -> str: ...
    @property
    def episode_status(self) -> EpisodeStatus: ...


def get_world(robot: object) -> World | None:
    """Discover optional episode/world support without identifying a simulator driver."""
    return robot if isinstance(robot, World) else None
