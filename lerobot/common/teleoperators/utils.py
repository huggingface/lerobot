from .config import TeleoperatorConfig
from .teleoperator import Teleoperator


def make_teleoperator_from_config(config: TeleoperatorConfig) -> Teleoperator:
    if config.type == "keyboard":
        from .keyboard import KeyboardTeleop

        return KeyboardTeleop(config)
    elif config.type == "koch_leader":
        from .koch_leader import KochLeader

        return KochLeader(config)
    elif config.type == "so100_leader":
        from .so100_leader import SO100Leader

        return SO100Leader(config)
    elif config.type == "so101_leader":
        from .so101_leader import SO101Leader

        return SO101Leader(config)
    elif config.type == "stretch3":
        from .stretch3_gamepad import Stretch3GamePad

        return Stretch3GamePad(config)
    elif config.type == "widowx":
        from .widowx import WidowX

        return WidowX(config)
    elif config.type == "mock_teleop":
        from tests.mocks.mock_teleop import MockTeleop

        return MockTeleop(config)
    else:
        raise ValueError(config.type)
