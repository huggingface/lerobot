#!/usr/bin/env python3
"""
Example script for multi-dataset recording using the new multi_record function.

This example shows how to record data for multiple datasets sequentially within the same episode.
For instance, you can record "pick" and "place" motions as separate datasets but within the same
continuous episode.

Usage:
```shell
python examples/multi_dataset_recording_example.py
```

Or run the multi_record function directly:
```shell
python -m lerobot.record --config=examples/multi_dataset_recording_example.py
```
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.record import DatasetRecordConfig, MultiDatasetRecordConfig, MultiRecordConfig
from lerobot.robots.so100_follower import SO100FollowerConfig
from lerobot.teleoperators.so100_leader import SO100LeaderConfig


def create_multi_record_config():
    """Create a configuration for multi-dataset recording."""

    # Define the robot configuration
    robot_config = SO100FollowerConfig(
        type="so100_follower",
        port="/dev/ttyUSB0",  # Adjust to your robot's port
        cameras={
            "laptop": OpenCVCameraConfig(
                type="opencv",
                camera_index=0,
                width=640,
                height=480,
                fps=30,
            ),
        },
        id="follower_robot",
    )

    # Define teleoperator configuration (optional)
    teleop_config = SO100LeaderConfig(
        type="so100_leader",
        port="/dev/ttyUSB1",  # Adjust to your teleoperator's port
        id="leader_robot",
    )

    # Define multiple dataset configurations for different stages
    dataset_configs = [
        DatasetRecordConfig(
            repo_id="username/pick_motion_dataset",
            single_task="Pick up the object from the table",
            fps=30,
            episode_time_s=30,  # 30 seconds for pick motion
            reset_time_s=10,
            num_episodes=50,
            video=True,
            push_to_hub=False,  # Set to True if you want to upload
        ),
        DatasetRecordConfig(
            repo_id="username/place_motion_dataset",
            single_task="Place the object in the target location",
            fps=30,
            episode_time_s=30,  # 30 seconds for place motion
            reset_time_s=10,
            num_episodes=50,
            video=True,
            push_to_hub=False,  # Set to True if you want to upload
        ),
    ]

    # Define multi-dataset configuration
    multi_dataset_config = MultiDatasetRecordConfig(
        datasets=dataset_configs,
        stage_switch_keys=["space", "tab"],  # Space for pick, Tab for place
    )

    # Create the complete multi-record configuration
    config = MultiRecordConfig(
        robot=robot_config,
        multi_dataset=multi_dataset_config,
        teleop=teleop_config,
        policy=None,  # No policy, using teleop
        display_data=True,
        play_sounds=True,
        resume=False,
    )

    return config


def main():
    """Example of how to use multi_record function."""
    from lerobot.record import multi_record

    # Create configuration
    config = create_multi_record_config()

    print("Starting multi-dataset recording...")
    print("Instructions:")
    print("- Press SPACE to record 'pick' motion to the first dataset")
    print("- Press TAB to record 'place' motion to the second dataset")
    print("- Press RIGHT ARROW to finish current episode")
    print("- Press LEFT ARROW to re-record current episode")
    print("- Press ESC to stop recording completely")
    print()

    # Start multi-dataset recording
    datasets = multi_record(config)

    print(f"Recording completed! Created {len(datasets)} datasets:")
    for i, dataset in enumerate(datasets):
        print(f"  Dataset {i}: {dataset.repo_id} with {dataset.num_episodes} episodes")


if __name__ == "__main__":
    main()
