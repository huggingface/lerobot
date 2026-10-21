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

CLOCKS = ("realtime", "lockstep")


def _embeds_robot(config_class: type) -> bool:
    return any(
        isinstance(arg, type) and issubclass(arg, RobotConfig)
        for hint in typing.get_type_hints(config_class).values()
        for arg in (hint, *typing.get_args(hint))
    )


# A dataclass, so that draccus routes `_SimulatedEnvChoices | None` through its choice handling.
@dataclass
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
    SimulatedEnvConfig = EnvConfig
else:
    SimulatedEnvConfig = _SimulatedEnvChoices


@RobotConfig.register_subclass("sim")
@dataclass
class SimRobotConfig(RobotConfig):
    """One simulated world driven as a robot, in this process or served.

    ``env`` names the environment (``--robot.env.type=libero``); its ``task`` selects the suite and
    ``task_id`` the task in it. With ``--robot.env.endpoint`` set, the world runs behind
    ``lerobot-env-server``, which must serve that environment, as for a batched env.
    A simulated world needs no calibration: ``calibration_dir`` is unused.
    """

    env: SimulatedEnvConfig | None = None
    task_id: int = 0
    # realtime: the world advances at its control rate, also while the policy computes, as hardware does.
    # lockstep: each sent action advances the world exactly one step.
    clock: str = "realtime"
    # Per-tick snapshot/apply/hold bound: a dead simulator must not stall the control loop.
    timeout_s: float = 5.0
    # Session open and world reset may build or rebuild native simulators.
    setup_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.clock not in CLOCKS:
            raise ValueError(f"--robot.clock must be one of {CLOCKS}, got {self.clock!r}")
        if self.env is None:
            raise ValueError("A sim robot needs --robot.env.type")


def check_teleop_clock(robot: RobotConfig | None, teleop: object | None) -> None:
    """A person drives a world that moves on its own: refuse a teleoperator for a lockstep sim robot."""
    if teleop is not None and isinstance(robot, SimRobotConfig) and robot.clock == "lockstep":
        raise ValueError(
            "--teleop needs --robot.clock=realtime: a lockstep world waits for each action instead of "
            "moving with the person driving it"
        )
