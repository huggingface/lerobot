"""Offline STS3215 gripper feedback parsing.

Input values are decoded register values returned by RustyPot.
This module never connects to hardware or sends motor commands.
"""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SO101GripperFeedback:
    load_value: int
    load_magnitude: int
    load_direction: str

    voltage_raw: int
    voltage_v: float

    temperature_raw: int
    temperature_c: int

    current_raw: int
    status_raw: int


def _validate_integer(name: str, value: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")

    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, got {value}"
        )

    return value


def parse_sts3215_feedback(
    raw: Mapping[str, int],
) -> SO101GripperFeedback:
    """Parse RustyPot-decoded STS3215 register values.

    Expected keys:
        present_load
        present_voltage
        present_temperature
        present_current
        status

    RustyPot already decodes the STS3215 sign-magnitude encoding.
    Therefore present_load is signed and must not be decoded again.
    """

    required = {
        "present_load",
        "present_voltage",
        "present_temperature",
        "present_current",
        "status",
    }

    missing = required - raw.keys()

    if missing:
        raise ValueError(
            f"Missing feedback registers: {sorted(missing)}"
        )

    load = _validate_integer(
        "present_load", raw["present_load"], -1023, 1023
    )

    voltage = _validate_integer(
        "present_voltage", raw["present_voltage"], 0, 255
    )

    temperature = _validate_integer(
        "present_temperature", raw["present_temperature"], 0, 255
    )

    current = _validate_integer(
        "present_current", raw["present_current"], 0, 65535
    )

    status = _validate_integer(
        "status", raw["status"], 0, 255
    )

    magnitude = abs(load)

    if load == 0:
        direction = "zero"
    elif load < 0:
        direction = "negative"
    else:
        direction = "positive"

    return SO101GripperFeedback(
        load_value=load,
        load_magnitude=magnitude,
        load_direction=direction,
        voltage_raw=voltage,
        voltage_v=voltage * 0.1,
        temperature_raw=temperature,
        temperature_c=temperature,
        current_raw=current,
        status_raw=status,
    )
