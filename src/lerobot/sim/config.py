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

"""Config for the simulator an env runs."""

import importlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from lerobot.sim.integrations import simulator_required
from lerobot.sim.types import CONTROL_MODES

# What a simulator runs with when neither the config nor its integration's `conventions` set a value.
_DEFAULTS: dict[str, Any] = {
    "fps": 30,
    "control_mode": "position",
    "semantics": "",
    "state_names": (),
    "action_names": (),
    "robot_type": "",
}


@dataclass
class SimConfig:
    """The simulator an env runs: integration, task, options and control settings.

    A server YAML's `sim` section is one, usually with only `type`, `task` and `kwargs`. Settings left unset
    (`None`) come from the integration's `conventions` when `resolve` fills them in; the env runs a resolved one.

    Args:
        type (`str`):
            Integration, a module of `lerobot.sim.integrations`, e.g. `libero`.
        task (`str`):
            Task to run, as the integration names it, e.g. `libero_spatial`. Some integrations take a
            comma-separated list.
        fps (`float`, *optional*):
            Control rate. Resolves to `30` when the integration doesn't declare one.
        control_mode (`str`, *optional*):
            What an action commands: `position`, `delta` or `eef_pose`. Resolves to `position`.
        semantics (`str`, *optional*):
            Name of the observation and action conventions. Resolves to `<type>-<control_mode>-v1`.
        kwargs (`dict[str, Any]`, *optional*):
            Options passed to the integration's `env_fns`.
        state_names (`tuple[str, ...]`, *optional*):
            Component names of `observation.state`. Left empty, they are `state_<i>`.
        action_names (`tuple[str, ...]`, *optional*):
            Component names of the action. Left empty, they are `command_<i>`.
        robot_type (`str`, *optional*):
            Body a sim robot records in its dataset. Left empty, it records `sim_robot`.
    """

    type: str = ""
    task: str = ""
    fps: float | None = None
    control_mode: str | None = None
    semantics: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    state_names: tuple[str, ...] | None = None
    action_names: tuple[str, ...] | None = None
    robot_type: str | None = None

    def __post_init__(self) -> None:
        """Check the `control_mode` and `fps` that are set."""
        if self.state_names is not None:
            self.state_names = tuple(self.state_names)
        if self.action_names is not None:
            self.action_names = tuple(self.action_names)
        if self.control_mode is not None and self.control_mode not in CONTROL_MODES:
            raise ValueError(f"Unknown control_mode {self.control_mode!r}; expected one of {CONTROL_MODES}")
        if self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError(f"A simulator needs a positive fps, got {self.fps}")

    @property
    def resolved(self) -> bool:
        """Whether every setting is filled in."""
        return all(getattr(self, key) is not None for key in _DEFAULTS)

    @classmethod
    def resolve(cls, settings: "dict[str, Any] | SimConfig") -> "SimConfig":
        """Build a config from `settings`, taking unset values from the integration's `conventions`."""
        if isinstance(settings, SimConfig):
            settings = asdict(settings)
        with simulator_required(settings["type"]):
            integration = importlib.import_module(f"lerobot.sim.integrations.{settings['type']}")
            conventions = getattr(integration, "conventions", None)
            declared = conventions(settings.get("kwargs") or {}) if conventions else {}
        given = {key: value for key, value in settings.items() if value is not None}
        values = {**_DEFAULTS, **declared, **given}
        if not values["semantics"]:
            values["semantics"] = f"{values['type']}-{values['control_mode'].replace('_', '-')}-v1"
        return cls(**values)
