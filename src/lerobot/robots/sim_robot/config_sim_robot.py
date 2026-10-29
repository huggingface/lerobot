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
from typing import TYPE_CHECKING, Any, get_args, get_type_hints

from draccus.choice_types import ChoiceType

from lerobot.envs.configs import EnvConfig

from ..config import RobotConfig


def _embeds_robot(config_class: type) -> bool:
    return any(
        isinstance(arg, type) and issubclass(arg, RobotConfig)
        for hint in get_type_hints(config_class).values()
        for arg in (hint, *get_args(hint))
    )


class _SimulatedEnvChoices(ChoiceType):
    """The environments a sim robot can run: every ``EnvConfig`` except those built around a robot.

    It narrows the ``--robot.env.type`` choices. An env that embeds a robot config (``gym_manipulator``)
    would make the robot and env registries contain each other, and draccus expands every choice of a
    config tree. It is not an ``EnvConfig`` subclass, which draccus would not treat as a choice type.
    """

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
    type SimulatedEnvConfig = EnvConfig
else:
    SimulatedEnvConfig = _SimulatedEnvChoices


@RobotConfig.register_subclass("sim")
@dataclass
class SimRobotConfig(RobotConfig):
    """A simulated environment driven as a robot. It needs no calibration."""

    # The environment, e.g. --robot.env.type=libero; its `task` selects the suite. With
    # --robot.env.server.endpoint set, it runs in lerobot-env-server instead of in this process.
    env: SimulatedEnvConfig
    # Index of the task within the suite.
    task_id: int = 0
