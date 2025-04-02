import time
import numpy as np
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

def main():
    # Instantiate the bus for a single motor on port /dev/ttyACM0.
    arm_bus = FeetechMotorsBus(
        port="/dev/ttyACM0",
        motors={"wrist_pitch": [1, "scs0009"]},
        protocol_version=1,
        group_sync_read=False,  # using individual read calls
    )
    arm_bus.connect()
    
    # Configure continuous rotation mode.
    arm_bus.write("Min_Angle_Limit", 0)
    arm_bus.write("Max_Angle_Limit", 1024)
    
    # For model "scs0009", the raw reading runs from 0 to ~1022.
    resolution_max = 1022  # use 1022 as the effective maximum raw value
    
    # Read initial raw motor position.
    prev_raw = arm_bus.read("Present_Position", ["wrist_pitch"])[0]
    print("Initial raw position:", prev_raw)
    
    # Command continuous rotation.
    arm_bus.write("Goal_Position", 1024)
    
    # Initialize loop counter.
    loops_count = 0
    target_effective = 1780
    tolerance = 50  # stop when effective position is within ±50 of target
    
    while True:
        current_raw = arm_bus.read("Present_Position", ["wrist_pitch"])[0]
        
        # Detect wrap-around: if the previous reading was near the top (>= 1020)
        # and current reading is low (< 100), count that as one full loop.
        if prev_raw >= 1020 and current_raw < 100:
            loops_count += 1
            print(f"Wrap detected! loops_count increased to {loops_count}")
        
        # Compute the effective position.
        effective_position = loops_count * resolution_max + current_raw
        print(f"Raw position: {current_raw} | loops_count: {loops_count} | Effective position: {effective_position}")
        
        # Check if effective position is within tolerance of the target.
        if abs(effective_position - target_effective) <= tolerance:
            # Command motor to stop by setting the current raw position as goal.
            arm_bus.write("Goal_Position", current_raw)
            print(f"Target reached (effective position: {effective_position}). Stopping motor at raw position {current_raw}.")
            break
        
        prev_raw = current_raw
        time.sleep(0.01)  # 10 ms delay

    time.sleep(1)
    arm_bus.disconnect()

if __name__ == "__main__":
    main()
