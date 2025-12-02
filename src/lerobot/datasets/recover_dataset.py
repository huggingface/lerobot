#!/usr/bin/env python3
"""
Recover and finalize a dataset that was recorded but not finalized/uploaded.

Usage:
    python recover_dataset.py --repo_id=jetson/test_direct_policy
"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

init_logging()
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Recover and finalize a dataset")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repo_id (e.g., jetson/test_direct_policy)")
    parser.add_argument("--root", type=str, default=None, help="Custom root directory (optional)")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload to HuggingFace hub after finalizing")
    parser.add_argument("--private", action="store_true", help="Make the dataset private on hub")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    
    if not root.exists():
        logging.error(f"Dataset not found at: {root}")
        logging.info(f"Available datasets in {HF_LEROBOT_HOME}:")
        for item in HF_LEROBOT_HOME.iterdir():
            if item.is_dir() and item.name != "calibration":
                logging.info(f"  - {item}")
        return
    
    logging.info(f"Found dataset at: {root}")
    logging.info("Loading dataset...")
    
    try:
        # Load the dataset
        dataset = LeRobotDataset(args.repo_id, root=args.root)
        logging.info(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
        
        # Finalize the dataset
        logging.info("Finalizing dataset (closing parquet writers, writing metadata)...")
        dataset.finalize()
        logging.info("✅ Dataset finalized successfully!")
        
        if args.push_to_hub:
            logging.info(f"Uploading dataset to HuggingFace hub: {args.repo_id}")
            dataset.push_to_hub(private=args.private)
            logging.info("✅ Dataset uploaded successfully!")
        else:
            logging.info("Dataset finalized but not uploaded. Use --push_to_hub to upload.")
            
    except Exception as e:
        logging.error(f"Error recovering dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

