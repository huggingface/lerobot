from abc import abstractmethod, ABC
from enum import Enum
from typing import Literal

import acton_ai
import numpy as np
import numpy.typing as npt


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


class MyArmBaseClass(ABC):

    def __init__(
            self,
            port: str,
            motors: dict[str, tuple[int, str]],
            extra_model_control_table: dict[str, list[tuple]] | None = None,
            extra_model_resolution: dict[str, int] | None = None,
            mock=False,
    ):
        self.port = port
        self.motors = motors
        self.mock = mock

    @abstractmethod
    def connect(self) -> None:
        pass

    def set_up_presets(self) -> None:
        """Optional. Override this method to do further set up on a device."""




class MyArmLeader(MyArmBaseClass):
    _handle: acton_ai.MyArmC
    """Instantiated in `connect`"""

    def connect(self) -> None:
        self._handle = acton_ai.find_myarm_controller()

    def read(self, cmd: Literal["Present_Position"]) -> npt.NDArray[np.float64]:
        match cmd:
            case "Present_Position":
                raise NotImplementedError()
            case _:
                raise ValueError(f"Unsupported {cmd=}")


class MyArmFollower(MyArmBaseClass):
    _handle: acton_ai.HelpfulMyArmM
    """Instantiated in `connect`"""

    def connect(self) -> None:
        self._handle = acton_ai.find_myarm_motor()

    def write(self, cmd: Literal["Goal_Position"],
              data: npt.NDArray[np.float64]) -> None:
        """This should mirror 'read' in the order of the data"""
        # TODO: Implement
        match cmd:
            case "Goal_Position":
                raise NotImplementedError()
            case "Torque_Enable":
                self._handle.bring_up_motors()
            case _:
                raise ValueError(f"Unsupported {cmd=}")
