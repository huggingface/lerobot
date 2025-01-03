from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal

import acton_ai
import numpy as np
import numpy.typing as npt


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


class MyArmBaseClass(ABC):
    _handle: acton_ai.HelpfulMyArmC | acton_ai.HelpfulMyArmM

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

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    def set_up_presets(self) -> None:
        """Optional. Override this method to do further set up on a device."""
        return

    def set_calibration(self, _: Any) -> None:
        """Calibration is not implemented on LeRobot. Use the MyArm interface."""
        return


class MyArmLeader(MyArmBaseClass):
    _handle: acton_ai.HelpfulMyArmC
    """Instantiated in `connect`"""

    def connect(self) -> None:
        self._handle = acton_ai.find_myarm_controller()

    def disconnect(self) -> None:
        """There is no need to disconnect from the MyArmC"""
        pass

    def read(self, cmd: Literal["Present_Position"]) -> npt.NDArray[np.float32]:
        """This should mirror 'write' in the order of the data"""
        match cmd:
            case "Present_Position":
                joint_angles = self._handle.get_joint_angles_in_mover_space()
                return np.array(joint_angles, dtype=np.float32)
            case _:
                raise ValueError(f"Unsupported {cmd=}")

    def write(self, cmd: Literal["Torque_Enable"], data: npt.NDArray[np.float32]) -> None:
        """Nothing needs doing here"""
        match cmd:
            case "Torque_Enable":
                # Doesn't make sense, but it does get requested to enable torque
                pass
            case _:
                raise ValueError(f"Unexpected write to follower arm {cmd=}")


class MyArmFollower(MyArmBaseClass):
    _handle: acton_ai.HelpfulMyArmM
    """Instantiated in `connect`"""

    def connect(self) -> None:
        self._handle = acton_ai.find_myarm_motor()

    def disconnect(self) -> None:
        self._handle.set_robot_power_off()

    def read(self, cmd: Literal["Present_Position"]) -> npt.NDArray[np.float32]:
        """This should mirror 'write' in the order of the data"""
        match cmd:
            case "Present_Position":
                joint_angles = self._handle.get_joints_angle()
                return np.array(joint_angles, dtype=np.float32)
            case _:
                raise ValueError(f"Unsupported {cmd=}")

    def write(
        self, cmd: Literal["Goal_Position", "Torque_Enable"], data: npt.NDArray[np.float32] | int
    ) -> None:
        """This should mirror 'read' in the order of the data"""
        # TODO: Implement
        match cmd:
            case "Goal_Position":
                self._handle.set_joints_from_controller_angles(data.tolist(), speed=20)
            case "Torque_Enable":
                if data == TorqueMode.ENABLED.value:
                    self._handle.bring_up_motors()
                    self._handle.prompt_user_to_bring_motors_into_bounds()
                elif data == TorqueMode.DISABLED.value:
                    self._handle.set_servos_enabled(False)
            case _:
                raise ValueError(f"Unsupported {cmd=}")
