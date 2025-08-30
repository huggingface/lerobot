#!/usr/bin/env python

"""
Convert video dataset to image dataset for faster training.
This pre-extracts all frames from MP4 files to PNG images.
"""

import argparse
from pathlib import Path
import logging
import shutil

def convert_dataset_videos_to_images(repo_id: str, root: str | None = None):
    """Convert all videos in a LeRobot dataset to individual image files."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import decode_video_frames
    import torch
    
    # Load dataset
    dataset = LeRobotDataset(repo_id, root=root, download_videos=True)
    
    total_frames_processed = 0
    
    for ep_idx in range(dataset.meta.total_episodes):
        logging.info(f"Processing episode {ep_idx}/{dataset.meta.total_episodes}")
        
        for vid_key in dataset.meta.video_keys:
            video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
            
            if not video_path.exists():
                logging.warning(f"Video not found: {video_path}")
                continue
                
            # Create image directory  
            img_dir = dataset.root / f"images/chunk-{dataset.meta.get_episode_chunk(ep_idx)}/{vid_key}"
            img_dir.mkdir(parents=True, exist_ok=True)
            
            # Decode all frames from video
            # Get episode length to decode all frames
            ep_length = dataset.meta.episodes[ep_idx]["length"]
            timestamps = [i / dataset.fps for i in range(ep_length)]
            
            try:
                frames = decode_video_frames(video_path, timestamps, dataset.tolerance_s, dataset.video_backend)
                
                # Save each frame as PNG
                for i, frame in enumerate(frames.squeeze(0)):
                    img_path = img_dir / f"episode_{ep_idx:06d}_{i:06d}.png"
                    # Convert tensor to PIL and save
                    import torchvision.transforms as T
                    to_pil = T.ToPILImage()
                    pil_frame = to_pil(frame)
                    pil_frame.save(img_path)
                    
                total_frames_processed += len(frames.squeeze(0))
                logging.info(f"  Extracted {len(frames.squeeze(0))} frames to {img_dir}")
                
            except Exception as e:
                logging.error(f"Failed to process {video_path}: {e}")
                continue
    
    logging.info(f"Conversion complete! Processed {total_frames_processed} total frames")
    logging.info(f"You can now use download_videos=False to use the extracted images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LeRobot video dataset to images")
    parser.add_argument("repo_id", help="Dataset repo ID (e.g., 'kenmacken/record-test-2')")
    parser.add_argument("--root", help="Local root directory", default=None)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    convert_dataset_videos_to_images(args.repo_id, args.root)
