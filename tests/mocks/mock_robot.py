import random
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from lerobot.common.cameras import CameraConfig, make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.robots import Robot, RobotConfig


@RobotConfig.register_subclass("mock_robot")
@dataclass
class MockRobotConfig(RobotConfig):
    n_motors: int = 3
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    random_values: bool = True
    static_values: list[float] | None = None
    calibrated: bool = True

    def __post_init__(self):
        if self.n_motors < 1:
            raise ValueError(self.n_motors)

        if self.random_values and self.static_values is not None:
            raise ValueError("Choose either random values or static values")

        if self.static_values is not None and len(self.static_values) != self.n_motors:
            raise ValueError("Specify the same number of static values as motors")

        if len(self.cameras) > 0:
            raise NotImplementedError  # TODO with the cameras refactor


class MockRobot(Robot):
    """Mock Robot to be used for testing."""

    config_class = MockRobotConfig
    name = "mock_robot"

    def __init__(self, config: MockRobotConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = config.calibrated
        self.motors = [f"motor_{i + 1}" for i in range(config.n_motors)]
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._is_connected = True
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_calibrated = True

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.random_values:
            return {f"{motor}.pos": random.uniform(-100, 100) for motor in self.motors}
        else:
            return {
                f"{motor}.pos": val for motor, val in zip(self.motors, self.config.static_values, strict=True)
            }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        return action

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_connected = False
