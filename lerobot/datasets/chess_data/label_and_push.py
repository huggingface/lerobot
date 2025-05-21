import argparse
from collections import defaultdict
import numpy as np
import torch
import traceback

from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import decode_video_frames


# === CONFIGURATION ===
GOAL_SQUARE = "D4"
PIECE_TYPE = "rook"
IS_WHITE = True
DEFAULT_GRASP_TYPE = "top"
TASK_DESCRIPTION = "Move the white rook to D4 on the chessboard."


def is_valid_square(square: str) -> bool:
    return square in [f"{c}{r}" for c in "ABCDEFGH" for r in "12345678"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID (e.g. 7jep7/rook_to_d4_v4)")
    parser.add_argument("--local-dir", required=True, help="Output path for new dataset")
    parser.add_argument("--target-repo-id", required=True, help="New HF repo to push labeled dataset")
    return parser.parse_args()


def normalize_frame(raw, image, cam_key):
    """Return a normalized frame dictionary suitable for LeRobotDataset.add_frame()"""
    return {
        "action": np.array(raw["action"], dtype=np.float32),
        "observation.state": np.array(raw["observation.state"], dtype=np.float32),
        "timestamp": np.array([float(raw["timestamp"])], dtype=np.float32),
        "task": TASK_DESCRIPTION,
        f"observation.images.follower_wrist": image  # fixed target key
    }


def main():
    args = parse_args()

    print(f"\n📦 Loading source dataset: {args.repo_id}")
    source_ds = LeRobotDataset(repo_id=args.repo_id)

    print(f"🎥 Detected video keys: {source_ds.meta.video_keys}")
    if not source_ds.meta.video_keys:
        raise ValueError("❌ No video keys found in the source dataset.")
    cam_key = source_ds.meta.video_keys[0]

    print(f"🛠️ Creating new labeled dataset at: {args.local_dir}")
    robot = make_robot("so101")
    labeled_ds = LeRobotDataset.create(
        repo_id=args.target_repo_id,
        root=args.local_dir,
        fps=source_ds.fps,
        robot=robot,
    )

    # Group frames by episode
    grouped = defaultdict(list)
    for frame in source_ds.hf_dataset:
        grouped[frame["episode_index"].item()].append(frame)

    # Process each episode
    for ep_idx, frames in sorted(grouped.items()):
        print(f"\n▶ Episode {ep_idx} ({len(frames)} frames)")

        while True:
            start_square = input(f"📝 Enter start square for episode {ep_idx} (e.g. A2): ").strip().upper()
            if is_valid_square(start_square):
                break
            print("❌ Invalid square format.")

        video_path = source_ds.root / source_ds.meta.get_video_file_path(ep_idx, cam_key)

        added = 0
        last_logged_second = -1
        for raw in frames:
            try:
                # Decode image frame at timestamp
                timestamp = raw["timestamp"].item() if hasattr(raw["timestamp"], "item") else raw["timestamp"]
                
                t_sec = int(timestamp)
                if t_sec != last_logged_second:
                    print(f"  ⏱️  {t_sec}s... decoding frame {raw['frame_index']}")
                    last_logged_second = t_sec
            
                # Decode with a more forgiving tolerance
                frames = decode_video_frames(
                    str(video_path),
                    [timestamp],
                    tolerance_s=0.5,  # loosened to match more timestamps
                )

                # If decoding fails, skip this frame
                if frames is None or len(frames) == 0:
                    print(f"❌ No video frame found at t={timestamp:.3f}s in {video_path.name}")
                    continue

                image = frames[0]

                if isinstance(image, torch.Tensor):
                    image = image.permute(1, 2, 0).numpy()  # convert from (C, H, W) → (H, W, C)
                
                frame = normalize_frame(raw, image, cam_key)
                
                assert isinstance(frame["timestamp"], np.ndarray)
                assert frame["timestamp"].shape == (1,)
                assert frame["timestamp"].dtype == np.float32

                # Add metadata to the frame; include piece, color, start square, end square.
                frame["task"] = f"Move the {'white' if IS_WHITE else 'black'} {PIECE_TYPE} from {start_square} to {GOAL_SQUARE} on the chessboard."
                
                if added == 0:
                    print("🔍 Sample frame keys:", list(frame.keys()))
                    print("🔍 Sample image shape:", frame["observation.images.follower_wrist"].shape)
                    print("🔍 Timestamp type:", type(frame["timestamp"]), "shape:", getattr(frame["timestamp"], "shape", None))
                
                labeled_ds.add_frame(frame)
                added += 1
            except Exception as e:
                print(f"⚠️ Skipping frame {raw['frame_index']}")
                traceback.print_exc()

        if added > 0:
            labeled_ds.save_episode()
            print(f"✅ Saved episode {ep_idx} with {added} frames")
        else:
            print(f"❌ Skipped episode {ep_idx}, no valid frames")

    print("\n☁️ Uploading labeled dataset to Hugging Face...")
    labeled_ds.push_to_hub()
    print("✅ Done!")


if __name__ == "__main__":
    main()
