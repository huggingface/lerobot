import abc
from dataclasses import dataclass

import draccus


@dataclass
class MotorsBusConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@MotorsBusConfig.register_subclass("dynamixel")
@dataclass
class DynamixelMotorsBusConfig:
    port: str
    motors: dict[str, tuple[int, str]]


@MotorsBusConfig.register_subclass("feetech")
@dataclass
class FeetechMotorsBusConfig:
    port: str
    motors: dict[str, tuple[int, str]]
