#!/usr/bin/env python
"""
Basic control example for OpenLoong humanoid robot.

This example demonstrates how to:
1. Connect to the OpenLoong robot (simulation or physical)
2. Get observations (joint states, IMU data)
3. Send position commands
4. Reset to default position
"""

import time
import numpy as np

from lerobot.robots.openloong import OpenLoong, OpenLoongConfig


def main():
    """Run basic control example."""
    # Create configuration for simulation
    config = OpenLoongConfig(
        is_simulation=True,
        control_dt=1.0 / 500.0,  # 500Hz control rate
        verbose=True,
    )
    
    # Initialize robot
    print("Initializing OpenLoong robot...")
    robot = OpenLoong(config)
    
    # Connect to robot
    print("Connecting to robot...")
    robot.connect(calibrate=True)
    
    try:
        # Get initial observation
        print("\nGetting initial observation...")
        obs = robot.get_observation()
        
        print("\nObservation keys:")
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
        
        # Print some joint positions
        print("\nInitial joint positions (sample):")
        for joint_name in ["kLeftHipPitch", "kRightHipPitch", "kLeftKnee", "kRightKnee"]:
            print(f"  {joint_name}: {obs.get(f'{joint_name}.q', 'N/A'):.4f} rad")
        
        # Print IMU data
        print("\nIMU data:")
        print(f"  Roll: {obs.get('imu.rpy.roll', 0):.4f} rad")
        print(f"  Pitch: {obs.get('imu.rpy.pitch', 0):.4f} rad")
        print(f"  Yaw: {obs.get('imu.rpy.yaw', 0):.4f} rad")
        
        # Send some actions
        print("\nSending position commands...")
        
        # Crouch motion
        print("  Crouching...")
        for i in range(100):
            # Bend knees
            action = {
                "kLeftKnee.q": 0.5 + 0.5 * np.sin(i * 0.1),
                "kRightKnee.q": 0.5 + 0.5 * np.sin(i * 0.1),
                "kLeftHipPitch.q": -0.3 - 0.3 * np.sin(i * 0.1),
                "kRightHipPitch.q": -0.3 - 0.3 * np.sin(i * 0.1),
            }
            robot.send_action(action)
            time.sleep(config.control_dt)
        
        # Stand up
        print("  Standing up...")
        for i in range(100):
            alpha = i / 100.0
            action = {
                "kLeftKnee.q": 1.0 * (1 - alpha) + 0.3 * alpha,
                "kRightKnee.q": 1.0 * (1 - alpha) + 0.3 * alpha,
                "kLeftHipPitch.q": -0.6 * (1 - alpha) + 0.0 * alpha,
                "kRightHipPitch.q": -0.6 * (1 - alpha) + 0.0 * alpha,
            }
            robot.send_action(action)
            time.sleep(config.control_dt)
        
        # Reset to default
        print("\nResetting to default position...")
        robot.reset()
        
        print("\nFinal observation:")
        obs = robot.get_observation()
        print(f"  Base height: {obs.get('base.pos.z', 0):.4f} m")
        
    finally:
        # Disconnect
        print("\nDisconnecting...")
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
