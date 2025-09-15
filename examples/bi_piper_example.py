#!/usr/bin/env python3

"""
Example script showing how to use the BiPiper robot for recording data.

This script demonstrates the configuration and usage of the BiPiper robot
with the LeRobot framework.

Usage:
    python -m lerobot.record \
        --robot.type=bi_piper \
        --robot.left_arm_can_port=can_0 \
        --robot.right_arm_can_port=can_1 \
        --robot.id=arm \
        --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, left: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}" \
        --dataset.repo_id=your_username/bimanual-piper-dataset \
        --dataset.num_episodes=10 \
        --dataset.single_task="Pick and place task" \
        --dataset.episode_time_s=30 \
        --dataset.reset_time_s=10

Requirements:
    - Install piper_sdk: pip install piper_sdk
    - Install lerobot with piper support: pip install -e ".[piper]"
    - Connect Piper arms to CAN ports (can_0 and can_1)
    - Connect cameras to the specified indices
"""

from lerobot.robots.bi_piper.config_bi_piper import BiPiperConfig


def create_bi_piper_config():
    """Create a sample BiPiper configuration"""
    
    # Basic configuration for BiPiper robot
    config = BiPiperConfig(
        type="bi_piper",
        left_arm_can_port="can_0",  # Adjust to your actual CAN port
        right_arm_can_port="can_1", # Adjust to your actual CAN port
        cameras={
            "front": {
                "type": "opencv",
                "index_or_path": 0,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "right": {
                "type": "opencv", 
                "index_or_path": 1,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "top": {
                "type": "opencv",
                "index_or_path": 2, 
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "left": {
                "type": "opencv",
                "index_or_path": 3,
                "width": 640,
                "height": 480,
                "fps": 30
            }
        }
    )
    
    return config


if __name__ == "__main__":
    # Example usage
    config = create_bi_piper_config()
    print("BiPiper configuration created:")
    print(f"  Left arm CAN port: {config.left_arm_can_port}")
    print(f"  Right arm CAN port: {config.right_arm_can_port}")
    print(f"  Number of cameras: {len(config.cameras)}")
    print(f"  Camera names: {list(config.cameras.keys())}")
    
    print("\nTo record data with this robot, run:")
    print("python -m lerobot.record --robot.type=bi_piper --robot.left_arm_can_port=can_0 --robot.right_arm_can_port=can_1 ...")
