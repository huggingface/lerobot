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

import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from draccus.choice_types import ChoiceType

from lerobot.envs.configs import EnvConfig
from lerobot.robots.config import RobotConfig


def _embeds_robot(config_class: type) -> bool:
    return any(
        isinstance(arg, type) and issubclass(arg, RobotConfig)
        for hint in typing.get_type_hints(config_class).values()
        for arg in (hint, *typing.get_args(hint))
    )


# A dataclass, so that draccus routes `_SimulatedEnvChoices | None` through its choice handling.
@dataclass
class _SimulatedEnvChoices(ChoiceType):
    """Exclude robot-wrapping envs to avoid recursive RobotConfig/EnvConfig choices in draccus."""

    @classmethod
    def get_known_choices(cls) -> dict[str, Any]:
        return {name: c for name, c in EnvConfig.get_known_choices().items() if not _embeds_robot(c)}

    @classmethod
    def get_choice_class(cls, name: str) -> Any:
        return cls.get_known_choices()[name]

    @classmethod
    def get_choice_name(cls, subcls: type) -> str:
        return EnvConfig.get_choice_name(subcls)

    @classmethod
    def default_choice_name(cls) -> str | None:
        return None


if TYPE_CHECKING:
    from lerobot.envs.configs import EnvConfig as SimulatedEnvConfig
else:
    SimulatedEnvConfig = _SimulatedEnvChoices


@RobotConfig.register_subclass("sim")
@dataclass
class SimRobotConfig(RobotConfig):
    """One simulated robot in a task environment, running locally or on a server.

    ``env`` names the environment (``--robot.env.type=libero``); its ``task`` selects the suite and
    ``task_id`` the task in it. With ``--robot.env.endpoint`` set, the world runs behind
    ``lerobot-env-server``, which must serve that environment, as for a batched env.
    No hardware calibration is required.
    """

    env: SimulatedEnvConfig | None = None
    task_id: int = 0
    # Seed the initial reset; subsequent unseeded resets continue the simulator RNG.
    seed: int | None = None
    # Remote only: bounds step RPCs. A local native call cannot be preempted by a timeout.
    timeout_s: float = 5.0
    # Opening a session or resetting an episode may construct a native simulator.
    setup_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.env is None:
            raise ValueError("A sim robot needs --robot.env.type")
