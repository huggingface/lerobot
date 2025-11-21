#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OpenArms Dataset Replay Example

Replays position actions from a recorded dataset on an OpenArms follower robot.
Only position commands (ending with .pos) are replayed, not velocity or torque.

Example usage:
    python examples/openarms/replay.py
"""

import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

# Configuration
EPISODE_IDX = 0
DATASET_REPO_ID = "lerobot-data-collection/replay-this-2025-11-02-17-58"  # TODO: Replace with your dataset
DATASET_ROOT = None  # Use default cache location, or specify custom path

# Robot configuration - adjust these to match your setup
ROBOT_CONFIG = OpenArmsFollowerConfig(
    port_left="can2",      # CAN interface for left arm
    port_right="can3",     # CAN interface for right arm
    can_interface="socketcan", 
    id="openarms_follower",
    disable_torque_on_disconnect=True,
    max_relative_target=10.0,  # Safety limit: max degrees to move per step
)


def main():
    """Main replay function."""
    print("=" * 70)
    print("OpenArms Dataset Replay")
    print("=" * 70)
    print(f"\nDataset: {DATASET_REPO_ID}")
    print(f"Episode: {EPISODE_IDX}")
    print(f"Robot: {ROBOT_CONFIG.id}")
    print(f"  Left arm: {ROBOT_CONFIG.port_left}")
    print(f"  Right arm: {ROBOT_CONFIG.port_right}")
    print("\n" + "=" * 70)
    
    # Initialize the robot
    print("\n[1/3] Initializing robot...")
    robot = OpenArmsFollower(ROBOT_CONFIG)
    
    # Load the dataset
    print(f"\n[2/3] Loading dataset '{DATASET_REPO_ID}'...")
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=DATASET_ROOT,
        episodes=[EPISODE_IDX]
    )
    
    # Filter dataset to only include frames from the specified episode
    # (required for dataset V3.0 where episodes are chunked)
    episode_frames = dataset.hf_dataset.filter(
        lambda x: x["episode_index"] == EPISODE_IDX
    )
    
    if len(episode_frames) == 0:
        raise ValueError(
            f"No frames found for episode {EPISODE_IDX} in dataset {DATASET_REPO_ID}"
        )
    
    print(f"  Found {len(episode_frames)} frames in episode {EPISODE_IDX}")
    
    # Extract action features from dataset
    action_features = dataset.features.get(ACTION, {})
    action_names = action_features.get("names", [])
    
    # Filter to only position actions (ending with .pos)
    position_action_names = [name for name in action_names if name.endswith(".pos")]
    
    if not position_action_names:
        raise ValueError(
            f"No position actions found in dataset. Action names: {action_names}"
        )
    
    print(f"  Found {len(position_action_names)} position actions to replay")
    print(f"  Actions: {', '.join(position_action_names[:5])}{'...' if len(position_action_names) > 5 else ''}")
    
    # Select only action columns from dataset
    actions = episode_frames.select_columns(ACTION)
    
    # Connect to the robot
    print(f"\n[3/3] Connecting to robot...")
    robot.connect(calibrate=False)  # Skip calibration for replay
    
    if not robot.is_connected:
        raise RuntimeError("Robot failed to connect!")
    
    print("\n" + "=" * 70)
    print("Ready to replay!")
    print("=" * 70)
    print("\nThe robot will replay the recorded positions.")
    print("Press Ctrl+C to stop at any time.\n")
    
    input("Press ENTER to start replaying...")
    
    # Replay loop
    log_say(f"Replaying episode {EPISODE_IDX}", blocking=True)
    
    try:
        for idx in range(len(episode_frames)):
            loop_start = time.perf_counter()
            
            # Extract action array from dataset
            action_array = actions[idx][ACTION]
            
            # Build action dictionary, but only include position actions
            action = {}
            for i, name in enumerate(action_names):
                # Only include position actions (ending with .pos)
                if name.endswith(".pos"):
                    action[name] = float(action_array[i])
            
            # Send action to robot
            robot.send_action(action)
            
            # Maintain replay rate (use dataset fps)
            loop_duration = time.perf_counter() - loop_start
            dt_s = 1.0 / dataset.fps - loop_duration
            busy_wait(dt_s)
            
            # Progress indicator every 100 frames
            if (idx + 1) % 100 == 0:
                progress = (idx + 1) / len(episode_frames) * 100
                print(f"Progress: {idx + 1}/{len(episode_frames)} frames ({progress:.1f}%)")
        
        print(f"\n✓ Successfully replayed {len(episode_frames)} frames")
        log_say("Replay complete", blocking=True)
        
    except KeyboardInterrupt:
        print("\n\nReplay interrupted by user")
    finally:
        # Disconnect robot
        print("\nDisconnecting robot...")
        robot.disconnect()
        print("✓ Replay complete!")


if __name__ == "__main__":
    main()

