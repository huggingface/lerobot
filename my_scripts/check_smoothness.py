from pathlib import Path

import matplotlib.pyplot as plt
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
DATASET_ID = "lerobot_pick_and_place"
OUTPUT_PLOT = Path("my_scripts/smoothness_check.png")
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
    episode_idx = np.random.randint(num_episodes)
    print(f"Selected random episode index: {episode_idx}")

    # Get frame selection (middle of the episode)
    episode_data = dataset.meta.episodes[episode_idx]
    print(f"Episode data keys: {episode_data.keys()}")

    if "index" in episode_data:
        start_index = episode_data["index"]
    elif "dataset_from_index" in episode_data:
        val = episode_data["dataset_from_index"]
        # Handle case where it might be a list or tensor
        if not isinstance(val, (int, float, np.integer, np.floating)):
            start_index = val.item() if hasattr(val, "item") else val[0]
        else:
            start_index = val
    else:
        raise KeyError(
            "Could not find start index in episode data (checked 'index' and 'dataset_from_index')"
        )

    length = episode_data["length"]
    # middle frame index relative to dataset
    # frame_idx = int(start_index + length // 2)
    frame_idx = start_index + 90
    print(f"Selected frame index: {frame_idx} (middle of episode length {length})")

    item = dataset[frame_idx]

    # Create processors
    print("Creating processors...")
    # Note: dataset.meta.stats keys might match dataset keys.
    # We pass dataset_stats directly.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=PRETRAINED_MODEL_PATH,
        dataset_stats=dataset.meta.stats,
    )

    # Prepare batch
    batch = {}
    device = next(policy.parameters()).device
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Preprocess batch
    print("Preprocessing batch...")
    batch = preprocessor(batch)

    # Run inference to get chunk
    print("Running inference to get action chunk...")
    with torch.inference_mode():
        # ACTPolicy.predict_action_chunk returns the action chunk (batch, chunk_size, action_dim)
        action_chunk = policy.predict_action_chunk(batch)

    print(f"Raw action chunk shape: {action_chunk.shape}")

    # Postprocess (unnormalize)
    print("Postprocessing (unnormalizing)...")
    # postprocessor expects a PolicyAction (Tensor)
    action_chunk_unnormalized = postprocessor(action_chunk)

    # Convert to numpy for plotting
    # Shape is (batch, chunk_size, action_dim). We take batch index 0.
    action_data = action_chunk_unnormalized[0].cpu().numpy()

    print(f"Unnormalized action chunk shape: {action_data.shape}")

    print(f"Unnormalized action chunk shape: {action_data.shape}")

    # Smoothing
    print("Applying smoothing...")
    smoothed_action_data = smooth_chunk(action_data, alpha=0.2)  # Adjust alpha as needed

    # Plotting
    print("Plotting...")

    # Identify image keys
    image_keys = [k for k in item.keys() if "image" in k]
    num_images = len(image_keys)

    if num_images > 0:
        # Create figure with 2 rows: images on top, trajectories below
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, num_images, height_ratios=[1, 3])

        # Plot images
        for idx, key in enumerate(sorted(image_keys)):
            ax_img = fig.add_subplot(gs[0, idx])
            img_tensor = item[key]
            # Convert (C, H, W) -> (H, W, C) numpy
            if isinstance(img_tensor, torch.Tensor):
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                # Normalize if needed (assuming float 0-1 or int 0-255)
                # If float and > 1, maybe it's 0-255? usually lerobot is 0-1 float.
                # Just in case, clip to valid range for imshow if float.
                if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                    img_np = np.clip(img_np, 0, 1)
                elif img_np.dtype == np.uint8:
                    pass  # Imshow handles it

                ax_img.imshow(img_np)
                ax_img.set_title(key.replace("observation.images.", ""))
                ax_img.axis("off")

        # Plot Trajectories
        ax_plot = fig.add_subplot(gs[1, :])
    else:
        # Fallback to single plot if no images
        fig, ax_plot = plt.subplots(figsize=(15, 10))

    chunk_size, action_dim = action_data.shape
    steps = np.arange(chunk_size)

    for i in range(action_dim):
        # Plot original faintly
        ax_plot.plot(steps, action_data[:, i], alpha=0.3, color=f"C{i}", linestyle="--")
        # Plot smoothed boldly
        ax_plot.plot(steps, smoothed_action_data[:, i], label=f"Dim {i} (Smoothed)", color=f"C{i}")

    ax_plot.set_title(f"Action Chunk Trajectories (Episode {episode_idx}, Frame {frame_idx})")
    ax_plot.set_xlabel("Chunk Step")
    ax_plot.set_ylabel("Action Value")
    # Move legend outside
    ax_plot.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_plot.grid(True)
    plt.tight_layout()

    print(f"Saving plot to {OUTPUT_PLOT}...")
    plt.savefig(OUTPUT_PLOT)

    print(f"Saving action chunks to {OUTPUT_NPZ}...")
    print(action_data.shape)
    np.savez(OUTPUT_NPZ, action=action_data, smoothed_action=smoothed_action_data)
    print("Done.")


if __name__ == "__main__":
    main()
