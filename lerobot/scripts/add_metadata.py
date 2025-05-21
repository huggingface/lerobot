import argparse
from pathlib import Path
from huggingface_hub import HfApi
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import json

# === CONFIGURATION ===
GOAL_SQUARE = "D4"
PIECE_TYPE = "rook"
IS_WHITE = True
DEFAULT_GRASP_TYPE = "top"
TASK_DESCRIPTION = "Move the white rook to D4 on the chessboard."

def add_metadata_to_dataset(repo_id: str):
    # Load the dataset directly from Hugging Face
    dataset = LeRobotDataset(repo_id=repo_id)
    print(f"Loaded dataset: {dataset}")

    # Debugging: Print total episodes and loaded episodes
    print(f"Total episodes in metadata: {dataset.meta.total_episodes}")
    print(f"Loaded episodes: {list(dataset.meta.episodes.keys())}")

    # Iterate through actual episodes
    for episode_index in dataset.meta.episodes.keys():
        print(f"Processing episode {episode_index}...")

        # Prompt user for start square
        start_square = input(f"Enter the start square for episode {episode_index} (e.g., 'a7'): ").strip()

        # Create a task description that includes all fields
        # task_description = (
        #     f"Move the {'white' if IS_WHITE else 'black'} {PIECE_TYPE} "
        #     f"from {start_square.upper()} to {GOAL_SQUARE}."
        # )

        task_description = json.dumps({
            "piece": PIECE_TYPE,
            "color": "white" if IS_WHITE else "black",
            "start_square": start_square.upper(),
            "goal_square": GOAL_SQUARE
        })

        # Add the task description to the episode
        episode = dataset.meta.episodes[episode_index]
        episode["tasks"] = [task_description]  # Overwrite tasks with the new description

        # Save updated metadata
        dataset.meta.episodes[episode_index] = episode
        dataset.meta.save_episode(
            episode_index=episode_index,
            episode_length=episode["length"],
            episode_tasks=episode["tasks"],
            episode_stats=dataset.meta.episodes_stats[episode_index],
        )

    # Save episodes metadata to JSONL file
    save_episodes_metadata(dataset)

    # Push updated metadata to Hugging Face
    print("Pushing updated dataset to Hugging Face...")
    dataset.push_to_hub(tags=["updated", "metadata"])
    print("Dataset updated successfully!")

def save_episodes_metadata(dataset):
    episodes_path = dataset.meta.root / "meta/episodes.jsonl"
    print(f"Saving episodes metadata to: {episodes_path}")
    episodes_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    with open(episodes_path, "w") as f:
        for episode_index, episode in dataset.meta.episodes.items():
            f.write(f"{json.dumps(episode)}\n")
    print(f"Saved episodes metadata to {episodes_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add metadata to a LeRobotDataset.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo ID of the dataset.")
    parser.add_argument("--root", type=str, default=None, help="Local root directory of the dataset.")
    args = parser.parse_args()

    add_metadata_to_dataset(repo_id=args.repo_id)

    # # Ensure the meta directory exists
    # meta_dir = Path("/home/jonas-petersen/.cache/huggingface/lerobot/data/meta")
    # meta_dir.mkdir(parents=True, exist_ok=True)

    # # Push the updated metadata to Hugging Face
    # api = HfApi()
    # api.upload_folder(
    #     repo_id="7jep7/rook_to_d4_v4",
    #     folder_path=str(meta_dir),
    #     repo_type="dataset",
    #     allow_patterns=["episodes.jsonl"]
    # )