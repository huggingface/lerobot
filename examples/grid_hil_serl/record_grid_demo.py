#!/usr/bin/env python

"""
Grid Position Prediction Recording Script

This script records demonstrations for the grid position prediction task.
It automatically randomizes cube positions and records image observations
with corresponding grid position labels.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image

import sys

# Ensure this script can import the local 'lerobot' package whether installed or not.
# We add both the repository root and the 'src' directory (common layout: src/lerobot/...)
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from grid_position_env import GridPositionLeRobotEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def record_demonstrations_from_config(config_path: str):
    """
    Record demonstrations using a LeRobot-style JSON config.

    Expects a config with keys similar to gym_manipulator:
      - env.task (we use GridPositionPrediction-v0)
      - env.fps (optional, default 10)
      - dataset.repo_id, dataset.root, dataset.num_episodes_to_record, dataset.push_to_hub, dataset.task
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Defaults and required fields
    env_cfg = cfg.get("env", {})
    dataset_cfg = cfg.get("dataset", {})

    task_name = env_cfg.get("task", "GridPositionPrediction-v0")
    fps = int(env_cfg.get("fps", 10))

    repo_id = dataset_cfg.get("repo_id", "username/grid-position-prediction")
    root = dataset_cfg.get("root", "./recording_grid_position_lerobot")
    num_episodes = int(dataset_cfg.get("num_episodes_to_record", 10))
    steps_per_episode = int(dataset_cfg.get("steps_per_episode", 1))
    push_to_hub = bool(dataset_cfg.get("push_to_hub", False))
    task_string = dataset_cfg.get("task", "grid_position_prediction")

    root_path = Path(root)
    if root_path.exists():
        base = root_path
        counter = 1
        while True:
            candidate = base.parent / f"{base.name}_{counter:03d}"
            if not candidate.exists():
                root_path = candidate
                break
            counter += 1

    print(f"Recording {num_episodes} episodes to {root_path}")
    print("=" * 50)

    # Create environment
    env = GridPositionLeRobotEnv()

    # Target image size for training (keep aspect ratio via center-square crop, then resize)
    target_h, target_w = 128, 128

    # Helper: center-square crop and resize to (128,128)
    def _center_crop_resize(image_hwc: np.ndarray, out_h: int = 128, out_w: int = 128) -> np.ndarray:
        h, w, _ = image_hwc.shape
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        crop = image_hwc[top : top + side, left : left + side]
        img = Image.fromarray(crop)
        img = img.resize((out_w, out_h), resample=Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    # Prime env and process one frame to validate pipeline
    obs, info = env.reset()
    image_hwc = obs["observation.image"]  # H, W, C uint8
    image_hwc = _center_crop_resize(image_hwc, target_h, target_w)

    # Define features for LeRobot dataset
    # Observation: input image only; Action: normalized grid coordinates in [-1, 1]
    # Include reward/done to be compatible with RL replay buffer expectations
    features = {
        "observation.image": {"dtype": "image", "shape": (3, target_h, target_w), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }

    # Initialize LeRobot dataset (images saved as files, rows in parquet)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root_path,
        use_videos=False,
    )

    total_frames = 0

    for ep in range(num_episodes):
        print(f"\nRecording episode {ep + 1}/{num_episodes}")

        # Ensure a fresh episode
        if dataset.episode_buffer is not None:
            dataset.clear_episode_buffer()

        # Reset env for each episode
        obs, info = env.reset()
        current_position = int(info["grid_position"])  # 0..63

        for step in range(steps_per_episode):
            # Save current image to img.jpg (for quick visual check)
            image_array = obs["observation.image"]  # HWC uint8
            image_array = _center_crop_resize(image_array, target_h, target_w)
            Image.fromarray(image_array).save("img.jpg")

            # Prepare frame for dataset: action stores normalized coordinates
            grid_x = current_position % 8
            grid_y = current_position // 8
            action_xy = np.array(
                [
                    (grid_x / 7.0) * 2.0 - 1.0,
                    (grid_y / 7.0) * 2.0 - 1.0,
                ],
                dtype=np.float32,
            )
            # Offline pretraining: binary reward with single-step episodes; mark done on last step
            is_last = (step == steps_per_episode - 1)
            reward_value = 1.0  # Offline recording always logs the ground-truth action (correct prediction)
            frame = {
                "observation.image": image_array,  # LeRobot will write image file
                "action": action_xy,
                "next.reward": np.array([reward_value], dtype=np.float32),
                "next.done": np.array([is_last], dtype=bool),
            }

            dataset.add_frame(frame=frame, task=task_string)
            total_frames += 1

            print(
                f"  Step {step + 1}: Position {current_position} "
                f"(action_norm=[{action_xy[0]:.3f}, {action_xy[1]:.3f}] -> x={grid_x}, y={grid_y})"
            )

            # Advance environment using ground-truth action to satisfy env API
            obs, reward, terminated, truncated, info = env.step(action_xy)
            current_position = int(info["grid_position"])  # next label
            print(
                f"    Next cube position -> index={current_position} "
                f"(x={current_position % 8}, y={current_position // 8})"
            )

        # Save episode to parquet + meta
        dataset.save_episode()
        print(f"  Saved episode {ep:06d} to dataset")

    # Optionally push to hub
    if push_to_hub:
        print("\nPushing dataset to Hugging Face Hub...")
        dataset.push_to_hub()

    env.close()

    print(f"\nRecording complete! Total frames: {total_frames}")
    print(f"Dataset saved to: {root_path}")


def main():
    parser = argparse.ArgumentParser(description="Record Grid Position Prediction Demonstrations (LeRobot format)")
    parser.add_argument("--config_path", type=str, required=True, help="Path to LeRobot-style recording JSON config")

    args = parser.parse_args()

    try:
        record_demonstrations_from_config(args.config_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
