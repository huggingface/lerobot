from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rudder_pedal")
@dataclass
class RudderPedalConfig(TeleoperatorConfig):
    """
    Configuration for Logitech/Saitek rudder pedals as a differential drive
    wheel teleoperator.

    Axis mapping (Saitek Pro Flight Rudder Pedals):
      Axis 0: Right toe brake (-1=released, +1=fully pressed) → forward
      Axis 1: Left toe brake  (-1=released, +1=fully pressed) → reverse
      Axis 2: Rudder bar      ( 0=center, -1=full left, +1=full right) → turn

    Control scheme:
      Right brake → forward throttle
      Left brake  → reverse throttle
      Rudder      → differential steering
    """
    max_speed: float = 500.0   # raw velocity units sent to STS3250 (0-2000 practical range)
    deadzone: float = 0.05     # ignore inputs below this to prevent drift
    joystick_index: int = 0    # pygame joystick index if multiple devices connected
