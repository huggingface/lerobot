#!/usr/bin/env python

"""
Example script demonstrating how to use the Widow AI Follower robot.

This script shows how to:
1. Connect to a Trossen Widow AI arm
2. Read joint positions
3. Send joint commands
4. Safely disconnect

Make sure you have the trossen_arm package installed and the arm is connected to the network.
"""

import time
from lerobot.common.robots.widow_ai_follower import WidowAIFollower, WidowAIFollowerConfig


def main():
    # Configuration for the Widow AI Follower
    config = WidowAIFollowerConfig(
        id="widow_ai_1",
        port="192.168.1.3",
        model="V0_FOLLOWER",
        use_degrees=True,      # Use degrees for joint positions
        max_relative_target=30,  # Safety limit: max 30 degrees movement per command
    )
    
    # Create the robot instance
    robot = WidowAIFollower(config)
    
    try:
        print("Connecting to Widow AI Follower...")
        robot.connect(calibrate=False)  # Skip calibration for this example
        
        print("Robot connected successfully!")
        print(f"Robot type: {robot.robot_type}")
        print(f"Robot ID: {robot.id}")
        
        # Get initial observation
        print("\nGetting initial joint positions...")
        obs = robot.get_observation()
        print("Current joint positions:")
        for joint, pos in obs.items():
            if joint.endswith('.pos'):
                print(f"  {joint}: {pos:.2f} degrees")
        
        # Example: Move to a new position
        print("\nMoving joints to new positions...")
        target_positions = {
            "shoulder_pan.pos": 45.0,    # 45 degrees
            "shoulder_lift.pos": 30.0,   # 30 degrees
            "elbow_flex.pos": -20.0,     # -20 degrees
            "wrist_1.pos": 15.0,         # 15 degrees
            "wrist_2.pos": 0.0,          # 0 degrees
            "wrist_3.pos": 0.0,          # 0 degrees
            "gripper.pos": 50.0,         # 50% open
        }
        
        # Send the action
        sent_action = robot.send_action(target_positions)
        print("Action sent:")
        for joint, pos in sent_action.items():
            print(f"  {joint}: {pos:.2f} degrees")
        
        # Wait for movement to complete
        print("\nWaiting for movement to complete...")
        time.sleep(3.0)
        
        # Get final positions
        print("\nGetting final joint positions...")
        final_obs = robot.get_observation()
        print("Final joint positions:")
        for joint, pos in final_obs.items():
            if joint.endswith('.pos'):
                print(f"  {joint}: {pos:.2f} degrees")
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure:")
        print("1. The trossen_arm package is installed")
        print("2. The arm is connected to the network")
        print("3. The IP address in the config is correct")
        print("4. The arm model (V0_LEADER/V0_FOLLOWER) is correct")
    
    finally:
        # Always disconnect safely
        if robot.is_connected:
            print("\nDisconnecting...")
            robot.disconnect()
            print("Robot disconnected.")


if __name__ == "__main__":
    main() 