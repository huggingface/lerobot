from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# Constants
PRETRAINED_MODEL_PATH = Path(
    "/home/droplab/workspace/lerobot_piper/outputs/train/lerobot_pick_and_place_50/checkpoints/last/pretrained_model"
)
DATASET_ROOT = Path("/home/droplab/.cache/huggingface/lerobot/local/lerobot_pick_and_place")
DATASET_ID = "lerobot_pick_and_place"
OUTPUT_NPZ = Path("my_scripts/action_chunks.npz")


def smooth_chunk(chunk, alpha=0.5):
    """
    Applies Exponential Moving Average (EMA) smoothing to the chunk.
    chunk: (time, dims)
    alpha: smoothing factor (0 < alpha <= 1). Lower alpha = more smoothing.
    """
    smoothed = np.zeros_like(chunk)
    smoothed[0] = chunk[0]
    for t in range(1, len(chunk)):
        smoothed[t] = alpha * chunk[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def main():
    print(f"Loading model from {PRETRAINED_MODEL_PATH}...")
    policy = ACTPolicy.from_pretrained(PRETRAINED_MODEL_PATH)
    policy.eval()
    print("Model loaded successfully.")

    print(f"Loading dataset {DATASET_ID} from {DATASET_ROOT}...")
    dataset = LeRobotDataset(repo_id=DATASET_ID, root=DATASET_ROOT)
    print("Dataset loaded successfully.")

    # Pick a random episode
    num_episodes = len(dataset.meta.episodes)
    # episode_idx = np.random.randint(num_episodes)
    episode_idx = 45
    print(f"Selected random episode index: {episode_idx}")

    episode_data = dataset.meta.episodes[episode_idx]

    if "index" in episode_data:
        start_index = episode_data["index"]
    elif "dataset_from_index" in episode_data:
        val = episode_data["dataset_from_index"]
        if not isinstance(val, (int, float, np.integer, np.floating)):
            start_index = val.item() if hasattr(val, "item") else val[0]
        else:
            start_index = val
    else:
        raise KeyError("Could not find start index in episode data")

    length = episode_data["length"]
    print(f"Episode length: {length}")

    # Create processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=PRETRAINED_MODEL_PATH,
        dataset_stats=dataset.meta.stats,
    )

    device = next(policy.parameters()).device

    merged_actions = []

    # Iterate through the episode with stride 100
    stride = 100
    print(f"Processing episode with stride {stride}...")

    for t in range(0, length, stride):
        frame_idx = start_index + t
        # Ensure we don't go out of bounds of the dataset
        if frame_idx >= len(dataset):
            break

        print(f"Processing frame {t}/{length} (Index: {frame_idx})")
        item = dataset[frame_idx]

        # Prepare batch
        batch = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0).to(device)

        # Preprocess
        batch = preprocessor(batch)

        # Inference
        with torch.inference_mode():
            action_chunk = policy.predict_action_chunk(batch)

        # Postprocess
        action_chunk_unnormalized = postprocessor(action_chunk)
        action_data = action_chunk_unnormalized[0].cpu().numpy()  # (chunk_size, dims)

        merged_actions.append(action_data)

    # Merge all chunks
    if merged_actions:
        full_trajectory = np.concatenate(merged_actions, axis=0)  # Concatenate along time axis
        print(f"Final merged trajectory shape: {full_trajectory.shape}")

        # Apply smoothing
        print("Applying smoothing...")
        smoothed_trajectory = smooth_chunk(full_trajectory, alpha=0.2)

        # Save to NPZ
        print(f"Saving to {OUTPUT_NPZ}...")
        np.savez(OUTPUT_NPZ, action=full_trajectory, smoothed_action=smoothed_trajectory)
        print("Done.")
    else:
        print("No actions generated.")


if __name__ == "__main__":
    main()
