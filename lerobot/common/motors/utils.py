from .configs import MotorsBusConfig
from .motors_bus import MotorsBus


def make_motors_buses_from_configs(motors_bus_configs: dict[str, MotorsBusConfig]) -> list[MotorsBus]:
    motors_buses = {}

    for key, cfg in motors_bus_configs.items():
        if cfg.type == "dynamixel":
            from .dynamixel import DynamixelMotorsBus

            motors_buses[key] = DynamixelMotorsBus(cfg)

        elif cfg.type == "feetech":
            from lerobot.common.motors.feetech.feetech import FeetechMotorsBus

            motors_buses[key] = FeetechMotorsBus(cfg)

        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return motors_buses


def make_motors_bus(motor_type: str, **kwargs) -> MotorsBus:
    if motor_type == "dynamixel":
        from .configs import DynamixelMotorsBusConfig
        from .dynamixel import DynamixelMotorsBus

        config = DynamixelMotorsBusConfig(**kwargs)
        return DynamixelMotorsBus(config)

    elif motor_type == "feetech":
        from feetech import FeetechMotorsBus

        from .configs import FeetechMotorsBusConfig

        config = FeetechMotorsBusConfig(**kwargs)
        return FeetechMotorsBus(config)

    else:
        raise ValueError(f"The motor type '{motor_type}' is not valid.")
