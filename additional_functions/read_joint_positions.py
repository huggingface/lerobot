#!/usr/bin/env python3
"""
G1 Robot Joint Position Reader

Displays real-time joint positions, velocities, and accelerations for all 35 motors.

Usage:
    python read_joint_positions.py <network_interface>
    Example: python read_joint_positions.py en7
"""

import sys
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


# G1 Joint Names (35 motors)
JOINT_NAMES = {
    # Legs (0-11)
    0: "left_hip_pitch",
    1: "left_hip_roll", 
    2: "left_hip_yaw",
    3: "left_knee",
    4: "left_ankle_pitch",
    5: "left_ankle_roll",
    6: "right_hip_pitch",
    7: "right_hip_roll",
    8: "right_hip_yaw", 
    9: "right_knee",
    10: "right_ankle_pitch",
    11: "right_ankle_roll",
    
    # Waist (12-14)
    12: "waist_yaw",
    13: "waist_roll",
    14: "waist_pitch",
    
    # Head (15-16)
    15: "head_yaw",
    16: "head_pitch",
    
    # Left Arm (17-23)
    17: "left_shoulder_pitch",
    18: "left_shoulder_roll",
    19: "left_shoulder_yaw",
    20: "left_elbow_pitch",
    21: "left_elbow_roll",
    22: "left_wrist_yaw",
    23: "left_wrist_pitch",
    
    # Right Arm (24-30)
    24: "right_shoulder_pitch",
    25: "right_shoulder_roll",
    26: "right_shoulder_yaw",
    27: "right_elbow_pitch",
    28: "right_elbow_roll",
    29: "right_wrist_yaw",
    30: "right_wrist_pitch",
    
    # Hands (31-34) - if applicable
    31: "left_hand",
    32: "right_hand",
    33: "reserved_33",
    34: "reserved_34",
}


class JointReader:
    def __init__(self):
        self.latest_state = None
        self.update_count = 0
        
    def state_handler(self, msg: LowState_):
        """Handle low-level state updates"""
        self.latest_state = msg
        self.update_count += 1
    
    def print_positions(self, mode='all'):
        """
        Print joint positions
        mode: 'all', 'legs', 'arms', 'compact'
        """
        if not self.latest_state:
            print("Waiting for robot data...")
            return
        
        state = self.latest_state
        
        # Clear screen and print header
        print("\033[2J\033[H")  # Clear screen
        print("=" * 100)
        print(f"G1 JOINT POSITIONS - Updates: {self.update_count}")
        print("=" * 100)
        
        if mode == 'compact':
            self._print_compact(state)
        else:
            self._print_detailed(state, mode)
        
        print("\n" + "=" * 100)
        print("Press Ctrl+C to stop")
    
    def _print_detailed(self, state, mode):
        """Print detailed joint information"""
        print(f"\n{'ID':<4} {'Joint Name':<25} {'Position (rad)':<16} {'Position (deg)':<16} {'Velocity (rad/s)':<18} {'Accel (rad/s²)'}")
        print("-" * 100)
        
        for i in range(35):
            # Filter by mode
            if mode == 'legs' and i >= 12:
                continue
            elif mode == 'arms' and not (17 <= i <= 30):
                continue
            
            motor = state.motor_state[i]
            joint_name = JOINT_NAMES.get(i, f"motor_{i}")
            
            position_rad = motor.q
            position_deg = np.degrees(position_rad)
            velocity = motor.dq
            acceleration = motor.ddq
            
            print(f"{i:<4} {joint_name:<25} {position_rad:+8.4f}        {position_deg:+8.2f}°        "
                  f"{velocity:+8.4f}          {acceleration:+8.4f}")
    
    def _print_compact(self, state):
        """Print compact view - positions only"""
        print("\n🦿 LEGS:")
        for i in range(12):
            motor = state.motor_state[i]
            joint_name = JOINT_NAMES.get(i, f"motor_{i}")
            print(f"  [{i:2d}] {joint_name:<20} {motor.q:+7.4f} rad  ({np.degrees(motor.q):+7.2f}°)")
        
        print("\n🦴 WAIST:")
        for i in range(12, 15):
            motor = state.motor_state[i]
            joint_name = JOINT_NAMES.get(i, f"motor_{i}")
            print(f"  [{i:2d}] {joint_name:<20} {motor.q:+7.4f} rad  ({np.degrees(motor.q):+7.2f}°)")
        
        print("\n🗣️  HEAD:")
        for i in range(15, 17):
            motor = state.motor_state[i]
            joint_name = JOINT_NAMES.get(i, f"motor_{i}")
            print(f"  [{i:2d}] {joint_name:<20} {motor.q:+7.4f} rad  ({np.degrees(motor.q):+7.2f}°)")
        
        print("\n🦾 LEFT ARM:")
        for i in range(17, 24):
            motor = state.motor_state[i]
            joint_name = JOINT_NAMES.get(i, f"motor_{i}")
            print(f"  [{i:2d}] {joint_name:<20} {motor.q:+7.4f} rad  ({np.degrees(motor.q):+7.2f}°)")
        
        print("\n🦾 RIGHT ARM:")
        for i in range(24, 31):
            motor = state.motor_state[i]
            joint_name = JOINT_NAMES.get(i, f"motor_{i}")
            print(f"  [{i:2d}] {joint_name:<20} {motor.q:+7.4f} rad  ({np.degrees(motor.q):+7.2f}°)")
        
        print("\n👋 HANDS:")
        for i in range(31, 35):
            motor = state.motor_state[i]
            joint_name = JOINT_NAMES.get(i, f"motor_{i}")
            print(f"  [{i:2d}] {joint_name:<20} {motor.q:+7.4f} rad  ({np.degrees(motor.q):+7.2f}°)")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <network_interface> [mode]")
        print(f"Example: python3 {sys.argv[0]} en7 compact")
        print(f"\nModes:")
        print(f"  all      - Show all joints with full details (default)")
        print(f"  compact  - Organized by body part")
        print(f"  legs     - Only leg joints")
        print(f"  arms     - Only arm joints")
        sys.exit(1)
    
    network_interface = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'compact'
    
    print("=" * 100)
    print("G1 JOINT POSITION READER")
    print("=" * 100)
    print(f"Initializing DDS on {network_interface}...")
    ChannelFactoryInitialize(0, network_interface)
    
    # Create reader
    reader = JointReader()
    
    # Subscribe to low-level state
    print("Subscribing to rt/lowstate...")
    lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
    lowstate_sub.Init(reader.state_handler, 10)
    
    print("\nWaiting for data...")
    time.sleep(1)
    
    print(f"Starting monitor (mode: {mode})...\n")
    
    try:
        while True:
            reader.print_positions(mode)
            time.sleep(0.1)  # Update display at 10 Hz
    except KeyboardInterrupt:
        print("\n\n" + "=" * 100)
        print("Reader stopped")
        print("=" * 100)


if __name__ == "__main__":
    main()

