"""
OpenArms Mini Teleoperation Example

This script demonstrates teleoperation of an OpenArms follower robot using 
an OpenArms Mini leader (Feetech-based) with dual arms (16 motors total).

The OpenArms Mini has:
- Right arm: 8 motors (joint_1 to joint_7 + gripper)
- Left arm: 8 motors (joint_1 to joint_7 + gripper)

Note on gripper normalization:
- OpenArms Mini gripper: 0-100 scale (0=closed, 100=open)
- OpenArms follower gripper: degrees (0=closed, -65=open)
- This script automatically converts between the two ranges
"""

import time
import os
import sys

from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.teleoperators.openarms_mini.openarms_mini import OpenArmsMini
from lerobot.teleoperators.openarms_mini.config_openarms_mini import OpenArmsMiniConfig
from lerobot.utils.robot_utils import busy_wait

# Target control frequency
TARGET_FPS = 30

# Configure the OpenArms follower (Damiao motors on CAN bus)
follower_config = OpenArmsFollowerConfig(
    port_left="can0",   # CAN interface for follower left arm
    port_right="can1",  # CAN interface for follower right arm
    can_interface="socketcan",  # Linux SocketCAN
    id="openarms_follower",
    disable_torque_on_disconnect=True,
    max_relative_target=10.0,  # Safety limit (degrees per step)
)

# Configure the OpenArms Mini leader (Feetech motors on serial)
leader_config = OpenArmsMiniConfig(
    port_right="/dev/ttyACM0",  # Serial port for right arm
    port_left="/dev/ttyACM1",   # Serial port for left arm
    id="openarms_minis",
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
avg_loop_time = 0.0
min_loop_time = float('inf')
max_loop_time = 0.0
stats_update_interval = 1.0  # Update stats every 1 second
last_stats_update = time.perf_counter()

JOINT_DIRECTION = {
    # invert direction
    "right_joint_1": -1,
    "right_joint_2": -1,
    "right_joint_3": -1,
    "right_joint_4": -1,
    "right_joint_5": -1,
    "left_joint_1": -1,
    "left_joint_3": -1,
    "left_joint_4": -1,
    "left_joint_5": -1,
    "left_joint_6": -1,
    "left_joint_7": -1,
}

SWAPPED_JOINTS = {
    "right_joint_6": "right_joint_7",
    "right_joint_7": "right_joint_6",
    "left_joint_6": "left_joint_7",
    "left_joint_7": "left_joint_6",
}

try:
    while True:
        loop_start = time.perf_counter()

        # Get actions and observations
        leader_action = leader.get_action()
        follower_obs = follower.get_observation()

        joint_action = {}
        for joint in all_joints:
            leader_key = f"{joint}.pos"

            # Determine which follower joint this leader joint controls
            follower_joint = SWAPPED_JOINTS.get(joint, joint)
            follower_key = f"{follower_joint}.pos"
            if "left" in follower_key:
                 continue

            # Get leader position (default 0 if missing)
            pos = leader_action.get(leader_key, 0.0)

            # Convert gripper values: Mini uses 0-100, OpenArms uses 0 to -65 degrees
            if "gripper" in joint:
                # Map 0-100 (Mini) to 0 to -65 (OpenArms)
                # 0 (closed) -> 0°, 100 (open) -> -65°
                pos = (pos / 100.0) * -65.0
            else:
                # Apply direction reversal if specified (non-gripper joints only)
                pos *= JOINT_DIRECTION.get(joint, 1)

            # Store in action dict for follower
            joint_action[follower_key] = pos

        #follower.send_action(joint_action)

        # Loop timing
        loop_end = time.perf_counter()
        loop_time = loop_end - loop_start
        loop_times.append(loop_time)

        # Update stats periodically
        current_time = time.perf_counter()
        if current_time - last_stats_update >= stats_update_interval:
            if loop_times:
                avg_loop_time = sum(loop_times) / len(loop_times)
                min_loop_time = min(loop_times)
                max_loop_time = max(loop_times)
                loop_times = []
                last_stats_update = current_time

        # Display everything
        sys.stdout.write("\033[H\033[J")  # Clear screen
        
        # Show timing stats at the top
        if avg_loop_time > 0:
            avg_hz = 1.0 / avg_loop_time
            min_hz = 1.0 / max_loop_time if max_loop_time > 0 else 0
            max_hz = 1.0 / min_loop_time if min_loop_time > 0 and min_loop_time < float('inf') else 0
            print(f"[Performance] Target: {TARGET_FPS} Hz | Avg: {avg_hz:.1f} Hz | Range: {min_hz:.1f}-{max_hz:.1f} Hz | Loop: {avg_loop_time*1000:.1f} ms\n")
        else:
            print(f"[Performance] Target: {TARGET_FPS} Hz | Measuring...\n")

        # Show joint positions
        print(f"{'Joint':<20} {'Leader':>15} {'Follower':>15}")
        print(f"{'':20} {'(0-100/deg)':>15} {'(deg)':>15}")
        print("-" * 52)

        for joint in all_joints:
            leader_key = f"{joint}.pos"
            follower_joint = SWAPPED_JOINTS.get(joint, joint)
            follower_key = f"{follower_joint}.pos"

            leader_pos = leader_action.get(leader_key, 0.0)
            follower_pos = follower_obs.get(follower_key, 0.0)

            print(f"{joint:<20} {leader_pos:>15.2f} {follower_pos:>15.2f}")

        # Smart sleep to maintain target FPS
        dt_s = time.perf_counter() - loop_start
        busy_wait(max(0, 1.0 / TARGET_FPS - dt_s))
            
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

