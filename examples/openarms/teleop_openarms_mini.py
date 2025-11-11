"""
OpenArms Mini Teleoperation Example

This script demonstrates teleoperation of an OpenArms follower robot using 
an OpenArms Mini leader (Feetech-based) with dual arms (16 motors total).

The OpenArms Mini has:
- Right arm: 8 motors (joint_1 to joint_7 + gripper)
- Left arm: 8 motors (joint_1 to joint_7 + gripper)
"""

import time

from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.teleoperators.openarms_mini.openarms_mini import OpenArmsMini
from lerobot.teleoperators.openarms_mini.config_openarms_mini import OpenArmsMiniConfig


# Configure the OpenArms follower (Damiao motors on CAN bus)
follower_config = OpenArmsFollowerConfig(
    port_left="can2",   # CAN interface for follower left arm
    port_right="can3",  # CAN interface for follower right arm
    can_interface="socketcan",  # Linux SocketCAN
    id="openarms_follower",
    disable_torque_on_disconnect=True,
    max_relative_target=10.0,  # Safety limit (degrees per step)
)

# Configure the OpenArms Mini leader (Feetech motors on serial)
leader_config = OpenArmsMiniConfig(
    port_right="/dev/ttyUSB0",  # Serial port for right arm
    port_left="/dev/ttyUSB1",   # Serial port for left arm
    id="openarms_mini",
    use_degrees=True,
)

print("OpenArms Mini → OpenArms Follower Teleoperation")

# Initialize devices
follower = OpenArmsFollower(follower_config)
leader = OpenArmsMini(leader_config)

# Connect and calibrate follower
print("Note: If you have existing calibration, just press ENTER to use it.")
follower.connect(calibrate=True)

# Connect and calibrate leader
print("Note: The leader arms will have torque disabled for manual control.")
leader.connect(calibrate=True)

print("\nPress ENTER to start teleoperation...")
input()

print("Press Ctrl+C to stop.\n")

# All joints for both arms (16 motors total)
all_joints = [
    # Right arm
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_joint_7",
    "right_gripper",
    # Left arm
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_joint_7",
    "left_gripper",
]

# Performance monitoring
loop_times = []
start_time = time.perf_counter()
last_print_time = start_time

try:
    while True:
        loop_start = time.perf_counter()
        
        # Get action from leader (OpenArms Mini)
        leader_action = leader.get_action()
        
        # Filter to only position data for all joints (both arms)
        joint_action = {}
        for joint in all_joints:
            pos_key = f"{joint}.pos"
            if pos_key in leader_action:
                joint_action[pos_key] = leader_action[pos_key]
        
        # Send action to follower (OpenArms)
        if joint_action:
            follower.send_action(joint_action)
        
        # Measure loop time
        loop_end = time.perf_counter()
        loop_time = loop_end - loop_start
        loop_times.append(loop_time)
        
        # Print stats every 2 seconds
        if loop_end - last_print_time >= 2.0:
            if loop_times:
                avg_time = sum(loop_times) / len(loop_times)
                current_hz = 1.0 / avg_time if avg_time > 0 else 0
                min_time = min(loop_times)
                max_time = max(loop_times)
                max_hz = 1.0 / min_time if min_time > 0 else 0
                min_hz = 1.0 / max_time if max_time > 0 else 0
                
                print(f"[Hz Stats] Avg: {current_hz:.1f} Hz | "
                      f"Range: {min_hz:.1f}-{max_hz:.1f} Hz | "
                      f"Avg loop time: {avg_time*1000:.1f} ms")
                
                # Reset for next measurement window
                loop_times = []
                last_print_time = loop_end
            
except KeyboardInterrupt:
    print("\n\nStopping teleoperation...")
finally:
    # Disconnect devices
    print("Disconnecting devices...")
    try:
        follower.disconnect()
    except Exception as e:
        print(f"Error disconnecting follower: {e}")
    
    try:
        leader.disconnect()
    except Exception as e:
        print(f"Error disconnecting leader: {e}")
    
    print("Done!")

