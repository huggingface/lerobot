#!/usr/bin/env python
"""
Simple test script to debug OpenArm leader-follower teleoperation.
Only moves joint_3 to isolate issues.
"""

import time
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from lerobot.teleoperators.openarm_leader import OpenArmLeader, OpenArmLeaderConfig

def main():
    leader_config = OpenArmLeaderConfig(
        port="can0",
        id="test_leader",
        manual_control=True,
    )
    
    follower_config = OpenArmFollowerConfig(
        port="can1",
        side="right",
        id="test_follower",
    )
    
    # Connect leader
    leader = OpenArmLeader(leader_config)
    leader.connect()

    # Connect follower
    follower = OpenArmFollower(follower_config)
    follower.connect()

    print("Starting teleoperation loop (Ctrl+C to stop)")
    try:
        while True:
            leader_action = leader.get_action()
            
            # Send only joint_3 to follower for testing
            action_to_send = {
                "joint_3.pos": leader_action.get("joint_3.pos", 0.0),
            }
            
            # Send action to follower
            follower.send_action(action_to_send)
            
            # Read follower observation
            follower_obs = follower.get_observation()
            
            leader_pos = leader_action.get("joint_3.pos", 0.0)
            follower_pos = follower_obs.get("joint_3.pos", 0.0)
            
            print(f"\rLeader: {leader_pos:7.2f}°  →  Follower: {follower_pos:7.2f}°", end="", flush=True)
            
            time.sleep(0.02)  # 50Hz
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        print("\nDisconnecting...")
        follower.disconnect()
        leader.disconnect()

if __name__ == "__main__":
    main()

