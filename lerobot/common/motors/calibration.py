from dataclasses import dataclass


@dataclass
class MotorCalibration:
    name: str
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int
