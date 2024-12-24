from typing import Protocol

from lerobot.common.robot_devices.motors.configs import MotorsBusConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus


class MotorsBus(Protocol):
    def motor_names(self): ...
    def set_calibration(self): ...
    def apply_calibration(self): ...
    def revert_calibration(self): ...
    def read(self): ...
    def write(self): ...


def build_motors_buses(motors_bus_configs: dict[str, MotorsBusConfig]):
    motors_buses = {}
    for key, cfg in motors_bus_configs.items():
        if cfg.type == "dynamixel":
            motors_buses[key] = DynamixelMotorsBus(cfg)
        elif cfg.type == "feetech":
            motors_buses[key] = FeetechMotorsBus(cfg)
        else:
            raise ValueError(f"{cfg.type} type is not found.")
    return motors_buses
