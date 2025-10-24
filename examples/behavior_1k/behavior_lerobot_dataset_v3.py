#!/usr/bin/env python

import json
import numpy as np
import torch as th
from pathlib import Path
from typing import Dict, Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .behaviour_1k_constants import (
    TASK_INDICES_TO_NAMES,
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    BEHAVIOR_DATASET_FEATURES,
)

import logging
from lerobot.utils.utils import init_logging

init_logging()

class BehaviorLeRobotDatasetV3(LeRobotDataset):
    """
    Extends LeRobotDataset v3.0 for BEHAVIOR-1K specific requirements.
    Handles task-based episode organization and BEHAVIOR-1K metadata.
    """
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int = 30,
        root: str | Path | None = None,
        robot_type: str = "R1Pro",
        use_videos: bool = True,
        video_backend: str = "pyav",
        batch_encoding_size: int = 1,
        image_writer_processes: int = 0,
        image_writer_threads: int = 4,
    ) -> "BehaviorLeRobotDatasetV3":
        """
        Create a new BEHAVIOR-1K dataset in v3.0 format.
        
        Args:
            repo_id: HuggingFace repository ID
            fps: Frames per second (default: 30)
            root: Local directory for the dataset
            robot_type: Robot type (default: "R1Pro")
            use_videos: Whether to encode videos (default: True)
            video_backend: Video backend to use (default: "pyav")
            batch_encoding_size: Number of episodes to batch before encoding videos
            image_writer_processes: Number of processes for async image writing
            image_writer_threads: Number of threads per process for image writing
            
        Returns:
            BehaviorLeRobotDatasetV3 instance
        """
        # Create the dataset using parent class method with BEHAVIOR-1K features
        obj = super().create(
            repo_id=repo_id,
            fps=fps,
            features=BEHAVIOR_DATASET_FEATURES,
            root=root,
            robot_type=robot_type,
            use_videos=use_videos,
            tolerance_s=1e-4,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )
        
        # Convert to BehaviorLeRobotDatasetV3 instance
        obj.__class__ = cls
        
        # Initialize BEHAVIOR-1K specific attributes
        obj.task_episode_mapping = {}  # Maps task_id to list of episode indices
        obj.episode_task_mapping = {}  # Maps episode_index to task info
        
        # Additional metadata for BEHAVIOR-1K
        obj.behavior_metadata = {
            "robot_type": robot_type,
            "task_names": TASK_INDICES_TO_NAMES,
            "proprioception_indices": PROPRIOCEPTION_INDICES[robot_type],
            "camera_names": ROBOT_CAMERA_NAMES[robot_type],
        }
        
        logging.info(f"Created BehaviorLeRobotDatasetV3 with repo_id: {repo_id}")
        return obj
    
    def __init__(self, *args, **kwargs):
        """
        Initialize from existing dataset.
        Use the create() classmethod to create a new dataset.
        """
        super().__init__(*args, **kwargs)
        
        # Initialize BEHAVIOR-1K specific attributes for loading existing datasets
        self.task_episode_mapping = {}
        self.episode_task_mapping = {}
        self.behavior_metadata = {}
        
        # Try to load BEHAVIOR-1K metadata if it exists
        metadata_path = self.root / "meta" / "behavior_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                stored_metadata = json.load(f)
                self.behavior_metadata = stored_metadata
                self.task_episode_mapping = stored_metadata.get("task_episode_mapping", {})
                self.episode_task_mapping = stored_metadata.get("episode_task_mapping", {})
    
    def add_episode_from_hdf5(
        self,
        hdf5_data: Dict[str, Any],
        task_id: int,
        episode_id: int,
        include_videos: bool = True,
    ) -> None:
        """
        Add an episode from HDF5 data to the dataset.
        
        Args:
            hdf5_data: Dictionary containing the HDF5 episode data
            task_id: Task ID for this episode
            episode_id: Episode ID (should be task_id * 10000 + local_episode_id)
            include_videos: Whether to include video data
        """
        task_name = TASK_INDICES_TO_NAMES[task_id]
        num_frames = len(hdf5_data["action"])
        
        logging.info(f"Adding episode {episode_id} (task: {task_name}) with {num_frames} frames")
        
        # Process each frame
        for frame_idx in range(num_frames):
            frame_data = {
                "action": hdf5_data["action"][frame_idx],
                "observation.state": hdf5_data["obs"]["robot_r1::proprio"][frame_idx],
                "observation.cam_rel_poses": hdf5_data["obs"]["robot_r1::cam_rel_poses"][frame_idx],
                "observation.task_info": hdf5_data["obs"]["task::low_dim"][frame_idx],
                "task": task_name,
                "timestamp": frame_idx / self.fps,
            }
            
            # Add video frames if requested
            if include_videos:
                for modality in ["rgb", "depth_linear", "seg_instance_id"]:
                    # Map depth_linear to depth for consistency
                    output_modality = "depth" if modality == "depth_linear" else modality
                    
                    for camera_name, robot_camera_name in ROBOT_CAMERA_NAMES[self.robot_type].items():
                        key = f"observation.images.{output_modality}.{camera_name}"
                        hdf5_key = f"{robot_camera_name}::{modality}"
                        
                        if hdf5_key in hdf5_data["obs"]:
                            # Get the frame data
                            frame = hdf5_data["obs"][hdf5_key][frame_idx]
                            
                            # Handle different data types
                            if isinstance(frame, th.Tensor):
                                frame = frame.numpy()
                            
                            # Ensure correct shape
                            if modality == "seg_instance_id" and len(frame.shape) == 2:
                                # Add channel dimension for grayscale
                                frame = np.expand_dims(frame, axis=-1)
                            elif modality == "depth_linear" and len(frame.shape) == 2:
                                frame = np.expand_dims(frame, axis=-1)
                            
                            frame_data[key] = frame
            
            # Add frame to dataset
            self.add_frame(frame_data)
        
        # Save episode with metadata
        episode_metadata = {
            "task_id": task_id,
            "task_name": task_name,
            "original_episode_id": episode_id,
        }
        
        # Add any additional HDF5 attributes as metadata
        if "attrs" in hdf5_data:
            for attr_name, attr_value in hdf5_data["attrs"].items():
                if isinstance(attr_value, (list, np.ndarray)):
                    episode_metadata[attr_name] = list(attr_value)
                else:
                    episode_metadata[attr_name] = attr_value
        
        # Save the episode
        self.save_episode(episode_data=None)
        
        # Track task-episode mapping
        if task_id not in self.task_episode_mapping:
            self.task_episode_mapping[task_id] = []
        self.task_episode_mapping[task_id].append(self.num_episodes - 1)
        self.episode_task_mapping[self.num_episodes - 1] = {
            "task_id": task_id,
            "task_name": task_name,
            "original_episode_id": episode_id,
        }
    
    def finalize(self) -> None:
        """Finalize the dataset and save additional BEHAVIOR-1K metadata."""
        # Save BEHAVIOR-1K specific metadata
        metadata_path = self.root / "meta" / "behavior_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.behavior_metadata.update({
            "task_episode_mapping": self.task_episode_mapping,
            "episode_task_mapping": self.episode_task_mapping,
            "total_tasks": len(self.task_episode_mapping),
            "total_episodes": self.num_episodes,
            "total_frames": self.num_frames,
        })
        
        with open(metadata_path, "w") as f:
            json.dump(self.behavior_metadata, f, indent=2)
        
        # Finalize the parent dataset
        super().finalize()
        
        logging.info(f"Finalized dataset with {self.num_episodes} episodes "
                   f"and {self.num_frames} frames across {len(self.task_episode_mapping)} tasks")
