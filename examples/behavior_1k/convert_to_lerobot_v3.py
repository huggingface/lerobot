#!/usr/bin/env python

import argparse
import h5py
import numpy as np
import os
import torch as th
from pathlib import Path
from tqdm import tqdm
import logging

from .behavior_lerobot_dataset_v3 import BehaviorLeRobotDatasetV3
from .behaviour_1k_constants import TASK_NAMES_TO_INDICES, TASK_INDICES_TO_NAMES, BEHAVIOR_DATASET_FEATURES

from lerobot.utils.utils import init_logging

init_logging()

def load_hdf5_episode(hdf5_path: str, episode_id: int = 0) -> dict:
    """
    Load episode data from HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
        episode_id: Episode ID to load (default: 0)
        
    Returns:
        Dictionary containing episode data
    """
    episode_data = {}
    
    with h5py.File(hdf5_path, "r") as f:
        # Find the episode with most samples if episode_id not specified
        if episode_id == -1:
            num_samples = [f["data"][key].attrs["num_samples"] for key in f["data"].keys()]
            episode_id = num_samples.index(max(num_samples))
        
        demo_key = f"demo_{episode_id}"
        if demo_key not in f["data"]:
            raise ValueError(f"Episode {episode_id} not found in {hdf5_path}")
        
        demo_data = f["data"][demo_key]
        
        # Load actions
        episode_data["action"] = np.array(demo_data["action"][:])
        
        # Load observations
        episode_data["obs"] = {}
        for key in demo_data["obs"].keys():
            episode_data["obs"][key] = np.array(demo_data["obs"][key][:])
        
        # Load attributes
        episode_data["attrs"] = {}
        for attr_name in demo_data.attrs:
            episode_data["attrs"][attr_name] = demo_data.attrs[attr_name]
        
        # Add global attributes
        for attr_name in f["data"].attrs:
            episode_data["attrs"][f"global_{attr_name}"] = f["data"].attrs[attr_name]
    
    return episode_data


def convert_episode(
    data_folder: str,
    output_repo_id: str,
    task_id: int,
    demo_id: int,
    dataset: BehaviorLeRobotDatasetV3,
    include_videos: bool = True,
    include_segmentation: bool = True,
) -> None:
    """
    Convert a single episode from HDF5 to LeRobotDataset v3.0 format.
    
    Args:
        data_folder: Base data folder containing HDF5 files
        output_repo_id: Output repository ID for the dataset
        task_id: Task ID
        demo_id: Demo ID (episode ID)
        dataset: BehaviorLeRobotDatasetV3 instance to add data to
        include_videos: Whether to include video data
        include_segmentation: Whether to include segmentation data
    """
    # Construct paths
    task_name = TASK_INDICES_TO_NAMES[task_id]
    hdf5_path = f"{data_folder}/2025-challenge-rawdata/task-{task_id:04d}/episode_{demo_id:08d}.hdf5"
    
    if not os.path.exists(hdf5_path):
        logging.error(f"HDF5 file not found: {hdf5_path}")
        return
    
    logging.info(f"Converting episode {demo_id} from task {task_name}")
    
    # Load episode data
    try:
        episode_data = load_hdf5_episode(hdf5_path, episode_id=0)
    except Exception as e:
        logging.error(f"Failed to load episode data: {e}")
        return
    
    # Filter out segmentation if not requested
    if not include_segmentation:
        keys_to_remove = [k for k in episode_data["obs"].keys() if "seg_instance_id" in k]
        for key in keys_to_remove:
            del episode_data["obs"][key]
    
    # Add episode to dataset
    dataset.add_episode_from_hdf5(
        hdf5_data=episode_data,
        task_id=task_id,
        episode_id=demo_id,
        include_videos=include_videos,
    )


def convert_dataset(
    data_folder: str,
    output_repo_id: str,
    task_names: list = None,
    episode_ids: list = None,
    max_episodes_per_task: int = None,
    include_videos: bool = True,
    include_segmentation: bool = True,
    fps: int = 30,
    batch_encoding_size: int = 1,
    image_writer_processes: int = 0,
    image_writer_threads: int = 4,
    push_to_hub: bool = False,
) -> None:
    """
    Convert BEHAVIOR-1K dataset from HDF5 to LeRobotDataset v3.0 format.
    
    Args:
        data_folder: Base folder containing HDF5 data
        output_repo_id: Output repository ID (e.g., "username/dataset-name")
        task_names: List of task names to convert (None = all tasks)
        episode_ids: Specific episode IDs to convert (None = all episodes)
        max_episodes_per_task: Maximum episodes per task to convert
        include_videos: Whether to include video data
        include_segmentation: Whether to include segmentation data
        fps: Frames per second
        batch_encoding_size: Number of episodes to batch before encoding
        image_writer_processes: Number of processes for image writing
        image_writer_threads: Number of threads for image writing
        push_to_hub: Whether to push to HuggingFace Hub
    """
    # Create output directory
    output_dir = Path.home() / ".cache/huggingface/lerobot" / output_repo_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Converting dataset to: {output_dir}")
    
    # Initialize dataset
    dataset = BehaviorLeRobotDatasetV3.create(
        repo_id=output_repo_id,
        root=output_dir,
        fps=fps,
        robot_type="R1Pro",
        use_videos=include_videos,
        video_backend="pyav",
        batch_encoding_size=batch_encoding_size,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )
    
    # Determine which tasks to process
    if task_names is None:
        task_names = list(TASK_NAMES_TO_INDICES.keys())
    
    task_ids = [TASK_NAMES_TO_INDICES[name] for name in task_names]
    
    # Process each task
    total_episodes = 0
    for task_id in tqdm(task_ids, desc="Processing tasks"):
        task_name = TASK_INDICES_TO_NAMES[task_id]
        task_folder = f"{data_folder}/2025-challenge-rawdata/task-{task_id:04d}"
        
        if not os.path.exists(task_folder):
            logging.warning(f"Task folder not found: {task_folder}")
            continue
        
        # Find all episodes for this task
        if episode_ids is not None:
            # Use specified episode IDs
            task_episode_ids = [eid for eid in episode_ids if eid // 10000 == task_id]
        else:
            # Find all episodes in the task folder
            task_episode_ids = []
            for filename in os.listdir(task_folder):
                if filename.startswith("episode_") and filename.endswith(".hdf5"):
                    eid = int(filename.split("_")[1].split(".")[0])
                    task_episode_ids.append(eid)
            task_episode_ids.sort()
        
        # Limit episodes if requested
        if max_episodes_per_task is not None:
            task_episode_ids = task_episode_ids[:max_episodes_per_task]
        
        logging.info(f"Processing {len(task_episode_ids)} episodes for task {task_name}")
        
        # Convert each episode
        for demo_id in tqdm(task_episode_ids, desc=f"Task {task_name}", leave=False):
            try:
                convert_episode(
                    data_folder=data_folder,
                    output_repo_id=output_repo_id,
                    task_id=task_id,
                    demo_id=demo_id,
                    dataset=dataset,
                    include_videos=include_videos,
                    include_segmentation=include_segmentation,
                )
                total_episodes += 1
            except Exception as e:
                logging.error(f"Failed to convert episode {demo_id}: {e}")
                continue
    
    logging.info(f"Converted {total_episodes} episodes total")
    
    # Finalize dataset
    logging.info("Finalizing dataset...")
    dataset.finalize()
    
    # Push to hub if requested
    if push_to_hub:
        logging.info("Pushing dataset to HuggingFace Hub...")
        dataset.push_to_hub(
            private=True,
            license="apache-2.0",
        )
    
    logging.info("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert BEHAVIOR-1K data to LeRobotDataset v3.0")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--output_repo_id", type=str, required=True, 
                       help="Output repository ID (e.g., 'username/behavior-dataset-v3')")
    parser.add_argument("--task_names", type=str, nargs="+", default=None,
                       help="Task names to convert (default: all)")
    parser.add_argument("--episode_ids", type=int, nargs="+", default=None,
                       help="Specific episode IDs to convert")
    parser.add_argument("--max_episodes_per_task", type=int, default=None,
                       help="Maximum episodes per task to convert")
    parser.add_argument("--no_videos", action="store_true",
                       help="Exclude video data")
    parser.add_argument("--no_segmentation", action="store_true",
                       help="Exclude segmentation data")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second (default: 30)")
    parser.add_argument("--batch_encoding_size", type=int, default=1,
                       help="Number of episodes to batch before encoding videos")
    parser.add_argument("--image_writer_processes", type=int, default=0,
                       help="Number of processes for async image writing")
    parser.add_argument("--image_writer_threads", type=int, default=4,
                       help="Number of threads for image writing")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push dataset to HuggingFace Hub")
    
    args = parser.parse_args()
    
    # Convert dataset
    convert_dataset(
        data_folder=args.data_folder,
        output_repo_id=args.output_repo_id,
        task_names=args.task_names,
        episode_ids=args.episode_ids,
        max_episodes_per_task=args.max_episodes_per_task,
        include_videos=not args.no_videos,
        include_segmentation=not args.no_segmentation,
        fps=args.fps,
        batch_encoding_size=args.batch_encoding_size,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        push_to_hub=args.push_to_hub,
    )
    


if __name__ == "__main__":
    main()
