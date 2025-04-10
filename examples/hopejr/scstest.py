import time
import serial
import numpy as np
import matplotlib.pyplot as plt

# Import the motor bus (adjust the import path as needed)
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

def main():

    bus = FeetechMotorsBus(
        port="/dev/ttyACM0",
        motors={
            "leader": [1, "scs0009"],
            "follower": [2, "scs0009"]
        },
        protocol_version=1,
        group_sync_read=False
    )
    bus.connect()
    print(bus.read("Present_Position", "leader"))
    bus.write("Torque_Enable", 0, ["leader"])
    bus.write("Torque_Enable", 1, ["follower"])
    for i in range(10000000):
        time.sleep(0.01)
        pos = bus.read("Present_Position", "leader")
        if pos[0] > 1 and pos[0] < 1022:
            bus.write("Goal_Position", pos, ["follower"])

if __name__ == "__main__":
    main()
