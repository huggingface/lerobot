#!/usr/bin/env python

"""
Example script to load and inspect the converted Arena GR1 dataset.

Usage:
    python example_load_arena_dataset.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    print("=" * 80)
    print("Loading Arena GR1 Microwave Manipulation Dataset")
    print("=" * 80)

    # Load the dataset
    dataset = LeRobotDataset("arena/gr1_microwave_manipulation")

    # Print dataset info
    print(f"\n📊 Dataset Information:")
    print(f"  Total episodes: {dataset.num_episodes}")
    print(f"  Total frames: {len(dataset)}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Robot type: {dataset.meta.robot_type}")

    # Print features
    print(f"\n🎯 Available Features:")
    for key in sorted(dataset.meta.features.keys()):
        feature = dataset.meta.features[key]
        if key not in [
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]:
            print(f"  {key:40s} {feature['shape']}")

    # Load a single frame
    print(f"\n🔍 Inspecting First Frame:")
    frame = dataset[0]

    print(f"\n  Action:")
    print(f"    Shape: {frame['action'].shape}")
    print(f"    Range: [{frame['action'].min():.3f}, " f"{frame['action'].max():.3f}]")

    print(f"\n  Robot State (joint positions):")
    print(f"    Shape: {frame['observation.state'].shape}")
    print(
        f"    Range: [{frame['observation.state'].min():.3f}, "
        f"{frame['observation.state'].max():.3f}]"
    )

    print(f"\n  Camera Image:")
    print(f"    Shape: {frame['observation.images.robot_pov_cam'].shape}")
    print(f"    Dtype: {frame['observation.images.robot_pov_cam'].dtype}")

    print(f"\n  Left End-Effector Position:")
    print(f"    Value: {frame['observation.left_eef_pos'].numpy()}")

    print(f"\n  Right End-Effector Position:")
    print(f"    Value: {frame['observation.right_eef_pos'].numpy()}")

    # Inspect an episode
    print(f"\n📹 Episode Information:")
    episode_0 = dataset.meta.episodes[0]
    print(f"  Episode 0 length: {episode_0['length']} frames")
    print(f"  Episode 0 tasks: {episode_0['tasks']}")

    # Show video info
    print(f"\n🎥 Video Information:")
    print(f"  Video keys: {dataset.meta.video_keys}")
    if len(dataset.meta.video_keys) > 0:
        video_key = dataset.meta.video_keys[0]
        print(f"  Video resolution: " f"{dataset.meta.features[video_key]['shape']}")

    print(f"\n✅ Dataset inspection complete!")
    print(f"📁 Dataset location: {dataset.root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
