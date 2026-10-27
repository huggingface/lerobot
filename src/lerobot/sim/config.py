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

"""The simulator an environment runs, as its integration resolves it."""

import importlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.sim.integrations import simulator_required


@dataclass
class SimConfig:
    """The simulator an environment runs: its integration, task, options and control conventions.

    ``semantics`` defaults to ``<type>-<control>-v1``. ``robot_type`` is recorded as a sim robot's dataset
    ``robot_type`` (the hardware robot's name when the body has one, e.g. ``so_follower``); left empty, a sim
    robot records ``sim``.
    """

    type: str
    task: str
    fps: float = 30
    control_mode: str = "position"
    semantics: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    state_names: tuple[str, ...] = ()
    action_names: tuple[str, ...] = ()
    robot_type: str = ""

    def __post_init__(self) -> None:
        """Normalize component order and reject invalid controller declarations."""
        self.state_names = tuple(self.state_names)
        self.action_names = tuple(self.action_names)
        if (
            self.control_mode not in {"position", "delta", "eef_pose"}
            or not np.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("A simulator requires a known controller and a positive FPS")
        if not self.semantics:
            self.semantics = f"{self.type}-{self.control_mode.replace('_', '-')}-v1"

    @classmethod
    def resolve(cls, settings: dict[str, Any]) -> "SimConfig":
        """Build from ``settings``, filling those left out or null from the integration's ``conventions``.

        Imports the integration, so it runs where the simulator is installed.
        """
        with simulator_required(settings["type"]):
            integration = importlib.import_module(f"lerobot.sim.integrations.{settings['type']}")
            conventions = getattr(integration, "conventions", None)
            declared = conventions(settings.get("kwargs") or {}) if conventions else {}
        return cls(**{**declared, **{key: value for key, value in settings.items() if value is not None}})
