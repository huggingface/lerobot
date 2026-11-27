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
from dataclasses import dataclass, field
from typing import Any

from lerobot.sim.integrations import simulator_required
from lerobot.sim.types import CONTROL_MODES
from lerobot.utils.constants import ACTION, OBS_PREFIX


@dataclass
class SimConfig:
    """The simulator an env runs: integration, task, options and control settings.

    Args:
        type (`str`):
            Integration, a module of `lerobot.sim.integrations`, e.g. `libero`.
        task (`str`):
            Task to run, as the integration names it, e.g. `libero_spatial`. Some integrations take a
            comma-separated list.
        fps (`float`, *optional*, defaults to `30`):
            Control rate.
        control_mode (`str`, *optional*, defaults to `"position"`):
            What an action commands: `position`, `delta` or `eef_pose`.
        semantics (`str`, *optional*):
            Name of the observation and action conventions. Defaults to `<type>-<control_mode>-v1`.
        kwargs (`dict[str, Any]`, *optional*):
            Options passed to the integration's `env_fns`.
        state_names (`tuple[str, ...]`, *optional*):
            Component names of `observation.state`. Left empty, they are `state_<i>`.
        action_names (`tuple[str, ...]`, *optional*):
            Component names of the action. Left empty, they are `command_<i>`.
        robot_type (`str`, *optional*, defaults to `""`):
            Body a sim robot records in its dataset. Left empty, it records `sim_robot`.
        features (`dict[str, tuple[int, ...]]`, *optional*):
            Shape of each observation, with LeRobot keys, and of the `action`. Given, the env is described
            without building its simulator, and the simulator is checked against them when it is built.
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
    features: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check `control_mode`, `fps` and `features`, and fill in the default `semantics`."""
        self.state_names = tuple(self.state_names)
        self.action_names = tuple(self.action_names)
        self.features = {name: tuple(shape) for name, shape in self.features.items()}
        if self.features and ACTION not in self.features:
            raise ValueError(f"features must give the shape of {ACTION!r}, got {sorted(self.features)}")
        unknown = [name for name in self.features if name != ACTION and not name.startswith(OBS_PREFIX)]
        if unknown:
            raise ValueError(f"features are observations ({OBS_PREFIX}*) or {ACTION!r}, got {unknown}")
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(f"Unknown control_mode {self.control_mode!r}; expected one of {CONTROL_MODES}")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"A simulator needs a positive fps, got {self.fps}")
        if not self.semantics:
            self.semantics = f"{self.type}-{self.control_mode.replace('_', '-')}-v1"

    @classmethod
    def resolve(cls, settings: dict[str, Any], suite: str) -> "SimConfig":
        """Build `suite`'s config from `settings`, taking missing values from the integration's `sim_settings`."""
        if not suite:
            raise ValueError("Settings are resolved for one suite; name it")
        declared: dict[str, Any] = {}
        with simulator_required(settings["type"]):
            integration = importlib.import_module(f"lerobot.sim.integrations.{settings['type']}")
            sim_settings = getattr(integration, "sim_settings", None)
            if sim_settings is not None:
                declared = sim_settings(suite, **(settings.get("kwargs") or {}))
        return cls(**{**declared, **{key: value for key, value in settings.items() if value is not None}})


@dataclass
class SimSettings:
    """`SimConfig`'s fields as a config file gives them; `SimConfig.resolve` fills in the unset (`None`) ones.

    They come from the integration's `sim_settings`, so a file only names the simulator, its task and its options.
    """

    type: str = ""
    task: str = ""
    fps: float | None = None
    control_mode: str | None = None
    semantics: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    state_names: list[str] | None = None
    action_names: list[str] | None = None
    robot_type: str | None = None
    features: dict[str, list[int]] | None = None
