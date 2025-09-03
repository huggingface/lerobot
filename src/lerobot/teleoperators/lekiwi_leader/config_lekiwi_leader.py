from dataclasses import dataclass, field
from ..config import TeleoperatorConfig
from ..keyboard import KeyboardTeleopConfig
from ..so100_leader import SO100LeaderConfig
from ..so101_leader import SO101LeaderConfig

@TeleoperatorConfig.register_subclass('lekiwi_leader')
@dataclass
class LekiwiLeaderConfig(TeleoperatorConfig):

    port: str #for Leader arm,
    use_degrees: bool = False # for SO101
    # Define three speed levels and a current index

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )