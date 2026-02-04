import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def plot_obs_action(repo_id, episode_idx, root=None, output_dir="my_scripts/plots"):
    """
    Plots observation and action values side-by-side for each state dimension.
    """
    print(f"Loading dataset: {repo_id} (Episode {episode_idx})")

    # helper for local paths, if root is not specified but repo_id looks like a path
    if root is None and (repo_id.startswith("/") or repo_id.startswith("./") or "local/" in repo_id):
        # If it's a local path string but not explicitly passed as root, let's try to handle it standardly first.
        # However, LeRobotDataset usually expects repo_id to be the ID.
        pass

    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset Loaded. Total episodes: {dataset.num_episodes}")

    # Access episode metadata to get frame range
    # Ensure episodes metadata is loaded
    if dataset.meta.episodes is None:
        print("Loading episode metadata...")
        # Trigger lazy load or import utility if needed.
        # Using a getter that triggers load as side-effect is safest if internal API changes,
        # but get_data_file_path triggers it.
        try:
            dataset.meta.get_data_file_path(0)
        except Exception:
            # If that fails (e.g. empty dataset), try manual load
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
    # Handle possible variations in types (list vs scalar from parquet)
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

    print(f"Episode {episode_idx} frames: {from_idx} to {to_idx} (Length: {to_idx - from_idx})")

    # Slice the dataset to get the sequences
    # dataset.hf_dataset supports slicing, which returns a dict of batched values
    try:
        # We use a range to slice
        data_batch = dataset.hf_dataset[from_idx:to_idx]
    except Exception as e:
        print(f"Error slicing dataset: {e}")
        return

    # Identify keys
    obs_key = "observation.state"
    if obs_key not in data_batch:
        print(f"Warning: '{obs_key}' not found. Available keys: {list(data_batch.keys())}")
        # Try to find a fallback
        possibles = [k for k in data_batch.keys() if "state" in k or "joint_pos" in k]
        if possibles:
            obs_key = possibles[0]
            print(f"Using fallback observation key: {obs_key}")
        else:
            print("Error: No suitable observation state key found.")
            return

    action_key = "action"
    if action_key not in data_batch:
        print(f"Error: '{action_key}' key not found.")
        return

    obs_data = data_batch[obs_key]
    action_data = data_batch[action_key]

    # Convert to numpy if they are tensors (which they should be due to transform)
    if not isinstance(obs_data, np.ndarray):
        # likely torch tensor or list
        import torch

        if isinstance(obs_data, torch.Tensor):
            obs_data = obs_data.numpy()
        elif isinstance(obs_data, list):
            # If list of tensors
            if len(obs_data) > 0 and isinstance(obs_data[0], torch.Tensor):
                obs_data = torch.stack(obs_data).numpy()
            else:
                obs_data = np.array(obs_data)
        else:
            obs_data = np.array(obs_data)

    if not isinstance(action_data, np.ndarray):
        import torch

        if isinstance(action_data, torch.Tensor):
            action_data = action_data.numpy()
        elif isinstance(action_data, list):
            if len(action_data) > 0 and isinstance(action_data[0], torch.Tensor):
                action_data = torch.stack(action_data).numpy()
            else:
                action_data = np.array(action_data)
        else:
            action_data = np.array(action_data)

    print(f"Observation shape: {obs_data.shape}")
    print(f"Action shape: {action_data.shape}")

    num_dims = obs_data.shape[1]

    # Create plots
    # We want num_dims plots. Let's arrange them gracefully.
    cols = 2
    rows = (num_dims + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)
    axes = axes.flatten()

    steps = np.arange(obs_data.shape[0])

    for i in range(num_dims):
        ax = axes[i]
        ax.plot(steps, obs_data[:, i], label=f"Obs (Dim {i})", color="blue")

        # Check if action has this dimension
        if i < action_data.shape[1]:
            ax.plot(steps, action_data[:, i], label=f"Action (Dim {i})", color="orange", linestyle="--")

        ax.set_title(f"Dimension {i}")
        ax.legend()
        ax.grid(True)

    # Hide unused subplots
    for i in range(num_dims, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    clean_repo = repo_id.replace("/", "_").replace(".", "")
    save_file = out_path / f"episode_{episode_idx}_{clean_repo}.png"
    plt.savefig(save_file)
    print(f"Plot saved to: {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot observation vs action for a LeRobot episode.")
    parser.add_argument("repo_id", type=str, help="Dataset repository ID (e.g. 'local/my_dataset')")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to plot")
    parser.add_argument("--root", type=str, default=None, help="Root directory for datasets (optional)")

    args = parser.parse_args()

    plot_obs_action(args.repo_id, args.episode, args.root)
