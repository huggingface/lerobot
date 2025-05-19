import argparse
import os
from datasets import load_dataset
from collections import defaultdict
from lerobot.common.robot_devices.robots import build_robot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

GOAL_SQUARE = "D4"
PIECE_TYPE = "rook"
IS_WHITE = True
DEFAULT_GRASP_TYPE = "top"

def is_valid_square(square):
    return square in [f"{c}{r}" for c in "ABCDEFGH" for r in "12345678"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="7jep7/rook_to_d4_v3", help="Source and target HF dataset repo")
    parser.add_argument("--local-dir", default="labeled_data", help="Temporary output dir before pushing")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"\n📦 Loading dataset from Hugging Face: {args.repo_id}")
    ds = load_dataset(args.repo_id, split="train")

    # Group by episode_index
    episodes = defaultdict(list)
    for row in ds:
        episodes[row["episode_index"]].append(row)

    print(f"Found {len(episodes)} episodes.")

    # Reconstruct dataset using LeRobotDataset
    print(f"\n🛠️ Creating new labeled dataset in: {args.local_dir}")
    robot = build_robot("so101")
    dset = LeRobotDataset.create(args.local_dir, fps=30, robot)

    for ep_idx, frames in sorted(episodes.items()):
        while True:
            start_square = input(f"📝 Enter start square for episode {ep_idx}: ").strip().upper()
            if is_valid_square(start_square):
                break
            print("❌ Invalid square format (expected e.g. A2).")

        for frame in frames:
            # Inject label metadata
            frame["start_square"] = start_square
            frame["goal_square"] = GOAL_SQUARE
            frame["piece_type"] = PIECE_TYPE
            frame["grasp_type"] = DEFAULT_GRASP_TYPE
            frame["is_white"] = IS_WHITE

            # Add frame to dataset
            dset.add_frame(frame)

        dset.end_episode()

    # Save and upload
    print("\n💾 Saving dataset...")
    dset.save()
    print("☁️ Pushing to Hugging Face Hub...")
    dset.push_to_hub(args.repo_id)
    print("✅ Labeled dataset uploaded successfully.")

if __name__ == "__main__":
    main()
