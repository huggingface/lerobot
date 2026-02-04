import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def inspect_episode(repo_id, episode_idx, root=None, output_dir="my_scripts/inspection"):
    """
    Dumps images and vectors for every timestep of an episode for inspection.
    """
    print(f"Loading dataset: {repo_id} (Episode {episode_idx})")

    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Ensure episodes metadata is loaded
    if dataset.meta.episodes is None:
        try:
            dataset.meta.get_data_file_path(0)
        except Exception:
            try:
                from lerobot.datasets.utils import load_episodes

                dataset.meta.episodes = load_episodes(dataset.root)
            except Exception as e:
                print(f"Could not load episode metadata: {e}")
                return

    if episode_idx >= len(dataset.meta.episodes):
        print(f"Error: Episode index {episode_idx} out of range (Total: {len(dataset.meta.episodes)})")
        return

    ep_meta = dataset.meta.episodes[episode_idx]

    # Handle possible variations in types
    from_idx = int(
        ep_meta["dataset_from_index"]
        if not isinstance(ep_meta["dataset_from_index"], list)
        else ep_meta["dataset_from_index"][0]
    )
    to_idx = int(
        ep_meta["dataset_to_index"]
        if not isinstance(ep_meta["dataset_to_index"], list)
        else ep_meta["dataset_to_index"][0]
    )

    length = to_idx - from_idx
    print(f"Episode {episode_idx} frames: {from_idx} to {to_idx} (Length: {length})")

    # Prepare output directory
    clean_repo = repo_id.replace("/", "_").replace(".", "")
    episode_dir = Path(output_dir) / clean_repo / f"episode_{episode_idx}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {episode_dir}")

    # Process each step
    print(f"Processing {length} steps...")

    for i in range(from_idx, to_idx):
        relative_step = i - from_idx

        try:
            # Load item using dataset __getitem__ to ensure videos/images are loaded
            item = dataset[i]
        except Exception as e:
            print(f"Error loading step {relative_step} (global index {i}): {e}")
            continue

        step_prefix = f"step_{relative_step:04d}"

        # Save Vectors
        step_data = {}

        # Find observation keys
        for key, value in item.items():
            if "observation.state" in key or "action" in key:
                # Convert tensor/numpy to list
                val = value
                if hasattr(val, "tolist"):
                    val = val.tolist()
                elif hasattr(val, "numpy"):
                    val = val.numpy().tolist()
                step_data[key] = val

        # Dump vectors to a single json per step (or we could consolidate, but per step is requested "store every time step")
        with open(episode_dir / f"{step_prefix}_vectors.json", "w") as f:
            json.dump(step_data, f, indent=2)

        # Save Images
        # Check for image keys in the loaded item
        image_keys = [k for k in item.keys() if "image" in k]

        for img_key in image_keys:
            img_data = item[img_key]

            # Helper to convert to PIL
            img = None
            if isinstance(img_data, dict) and "bytes" in img_data:
                import io

                img = Image.open(io.BytesIO(img_data["bytes"]))
            else:
                # Assume torch tensor (C, H, W) float scaled ?? or uint8
                # LeRobotDataset transforms usually output float 0-1 (C, H, W)
                arr = img_data
                if hasattr(arr, "numpy"):
                    arr = arr.numpy()

                if arr.ndim == 3:
                    # Check if C H W
                    if arr.shape[0] <= 4:
                        arr = np.transpose(arr, (1, 2, 0))

                    if arr.dtype == np.float32 or arr.dtype == np.float64:
                        arr = (arr * 255).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)

                img = Image.fromarray(arr)

            if img:
                clean_key = img_key.replace("observation.images.", "")
                img.save(episode_dir / f"{step_prefix}_cam_{clean_key}.png")

        if relative_step % 10 == 0:
            print(f"Processed step {relative_step}/{length}", end="\r")

    print(f"\nFinished processing {length} steps.")
    print(f"Inspect results in: {episode_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect LeRobot episode data.")
    parser.add_argument("repo_id", type=str, help="Dataset repository ID")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--root", type=str, default=None, help="Dataset root")
    parser.add_argument("--output-dir", type=str, default="my_scripts/inspection", help="Output directory")

    args = parser.parse_args()

    inspect_episode(args.repo_id, args.episode, args.root, args.output_dir)
