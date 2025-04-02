import time
import numpy as np

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, CalibrationMode

@staticmethod
def degps_to_raw(degps: float) -> int:
    steps_per_deg = 4096.0 / 360.0
    speed_in_steps = abs(degps) * steps_per_deg
    speed_int = int(round(speed_in_steps))
    if speed_int > 0x7FFF:
        speed_int = 0x7FFF
    if degps < 0:
        return speed_int | 0x8000
    else:
        return speed_int & 0x7FFF

@staticmethod
def raw_to_degps(raw_speed: int) -> float:
    steps_per_deg = 4096.0 / 360.0
    magnitude = raw_speed & 0x7FFF
    degps = magnitude / steps_per_deg
    if raw_speed & 0x8000:
        degps = -degps
    return degps

def main():
    # Instantiate the bus for a single motor on port /dev/ttyACM0.
    arm_bus = FeetechMotorsBus(
        port="/dev/ttyACM0",
        motors={"wrist_pitch": [1, "scs0009"]},
        protocol_version=1,
        group_sync_read=False,  # using individual read calls
    )
    arm_bus.connect()
    # Read the current raw motor position.
    # Note that "Present_Position" is in the raw units.
    current_raw = arm_bus.read("Present_Position", ["wrist_pitch"])[0]
    print("Current raw position:", current_raw)
    arm_bus.write("Goal_Position", 1000)
    arm_bus.disconnect()
    exit()

if __name__ == "__main__":
    main()
