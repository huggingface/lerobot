import argparse

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def print_obs_action(repo_id, episode_idx, root=None):
    """
    Prints observation and action values for each time step of the specified episode.
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

    print(f"Episode {episode_idx} frames: {from_idx} to {to_idx} (Length: {to_idx - from_idx})")

    try:
        data_batch = dataset.hf_dataset[from_idx:to_idx]
    except Exception as e:
        print(f"Error slicing dataset: {e}")
        return

    # Identify keys
    obs_key = "observation.state"
    if obs_key not in data_batch:
        possibles = [k for k in data_batch.keys() if "state" in k or "joint_pos" in k]
        if possibles:
            obs_key = possibles[0]
        else:
            print("Error: No suitable observation state key found.")
            return

    action_key = "action"
    if action_key not in data_batch:
        print(f"Error: '{action_key}' key not found.")
        return

    obs_data = data_batch[obs_key]
    action_data = data_batch[action_key]
    for x in range(len(obs_data)):
        if not torch.equal(obs_data[x], action_data[x]):
            print(torch.equal(obs_data[x], action_data[x]))

    # Convert to numpy
    if not isinstance(obs_data, np.ndarray):
        if isinstance(obs_data, torch.Tensor):
            obs_data = obs_data.numpy()
        elif isinstance(obs_data, list):
            if len(obs_data) > 0 and isinstance(obs_data[0], torch.Tensor):
                obs_data = torch.stack(obs_data).numpy()
            else:
                obs_data = np.array(obs_data)
        else:
            obs_data = np.array(obs_data)

    if not isinstance(action_data, np.ndarray):
        if isinstance(action_data, torch.Tensor):
            action_data = action_data.numpy()
        elif isinstance(action_data, list):
            if len(action_data) > 0 and isinstance(action_data[0], torch.Tensor):
                action_data = torch.stack(action_data).numpy()
            else:
                action_data = np.array(action_data)
        else:
            action_data = np.array(action_data)

    num_steps = obs_data.shape[0]
    print("-" * 80)
    print(f"{'Step':<6} | {'Observation':<35} | {'Action':<35}")
    print("-" * 80)

    for i in range(num_steps):
        # Format vectors as strings with limited precision
        obs_str = np.array2string(
            obs_data[i], precision=3, suppress_small=True, separator=", ", max_line_width=100
        )
        act_str = np.array2string(
            action_data[i], precision=3, suppress_small=True, separator=", ", max_line_width=100
        )
        if obs_str == act_str:
            continue
        print(f"{i:<6} | {obs_str:<35} | {act_str:<35}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print observation and action vectors for a LeRobot episode."
    )
    parser.add_argument("repo_id", type=str, help="Dataset repository ID (e.g. 'local/my_dataset')")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to print")
    parser.add_argument("--root", type=str, default=None, help="Root directory for datasets (optional)")

    args = parser.parse_args()

    print_obs_action(args.repo_id, args.episode, args.root)
