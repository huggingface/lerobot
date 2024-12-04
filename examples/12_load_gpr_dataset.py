"""
This script demonstrates loading and testing the GPR (General Purpose Robot) as a Lerobot dataset locally (not working yet)
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pprint import pprint
import shutil

from lerobot.common.datasets.push_dataset_to_hub.gpr_h5_format import from_raw_to_lerobot_format
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

GPR_FEATURES = {
    "observation.joint_pos": {
        "dtype": "float32",
        "shape": (12,),  # Adjust based on your robot's DOF
        "names": ["joint_positions"],
    },
    "observation.joint_vel": {
        "dtype": "float32",
        "shape": (12,),
        "names": ["joint_velocities"],
    },
    "observation.ang_vel": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["angular_velocity"],
    },
    "observation.euler_rotation": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["euler_angles"],
    },
    "action": {
        "dtype": "float32",
        "shape": (12,),
        "names": ["joint_commands"],
    },
}

def test_gpr_dataset():
    # Setup paths
    raw_dir = Path("/home/kasm-user/ali_repos/sim/runs/h5_out/stompypro/2024-12-02_20-04-51/all_h5")  # Directory containing your H5 files
    videos_dir = Path("data/temp")  # Required but not used
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary repo_id for local testing
    repo_id = "gpr_test"
    
    # Convert raw data to LeRobot format
    print("Converting raw data to LeRobot format...")
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(
        raw_dir=raw_dir,
        videos_dir=videos_dir,
        fps=50,  # Your simulation fps
        video=False  # No video data
    )

    # Delete the existing dataset folder if it exists
    dataset_path = Path.home() / ".cache/huggingface/lerobot/gpr_test"
    if dataset_path.exists():
        print(f"Deleting existing dataset folder: {dataset_path}")
        shutil.rmtree(dataset_path)
    
    # Create dataset instance
    print("\nCreating dataset...")
    dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    
    
    # Print dataset information
    print("\nDataset Overview:")
    print(f"Number of episodes: {dataset.num_episodes}")
    print(f"Number of frames: {dataset.num_frames}")
    print(f"FPS: {dataset.fps}")
    
    print("\nFeatures available:")
    pprint(dataset.features)
    
    # Test accessing single frame
    print("\nTesting single frame access...")
    frame_0 = dataset[0]
    print("First frame keys:", frame_0.keys())
    print("Shapes:")
    for key, value in frame_0.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
            
    # Test accessing an episode
    print("\nTesting episode access...")
    episode_index = 0
    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()
    print(f"Episode {episode_index} frames: {to_idx - from_idx}")
    
    # Test with history
    print("\nTesting dataset with history...")
    delta_timestamps = {
        "observation.joint_pos": [-0.1, -0.05, 0],  # Last 3 frames
        "observation.joint_vel": [-0.1, -0.05, 0],
        "observation.ang_vel": [-0.1, -0.05, 0],
        "observation.euler_rotation": [-0.1, -0.05, 0],
        "action": [0, 0.02, 0.04]  # Current and 2 future frames
    }
    
    dataset_with_history = LeRobotDataset(
        repo_id=repo_id,
        delta_timestamps=delta_timestamps,
        local_files_only=True
    )
    
    frame_with_history = dataset_with_history[0]
    print("\nShapes with history:")
    for key, value in frame_with_history.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
            
    # Test DataLoader
    print("\nTesting DataLoader...")
    dataloader = DataLoader(
        dataset_with_history,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    print("Loading first batch...")
    batch = next(iter(dataloader))
    print("\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")

if __name__ == "__main__":
    test_gpr_dataset() 