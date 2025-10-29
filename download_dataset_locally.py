#!/usr/bin/env python

"""
Download script for ywu67/keychain dataset using LeRobot.
This script downloads all data and files from the Hugging Face Hub dataset.
"""

import logging
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('download_log.txt')
    ]
)

def download_keychain_dataset(
    repo_id: str = "ywu67/keychain",
    root_dir: str | Path | None = None,
    download_videos: bool = True,
    force_cache_sync: bool = False
):
    """
    Download the ywu67/keychain dataset from Hugging Face Hub.
    
    Args:
        repo_id: The repository ID on Hugging Face Hub
        root_dir: Local directory to store the dataset (defaults to HF_LEROBOT_HOME)
        download_videos: Whether to download video files
        force_cache_sync: Whether to force sync and refresh local files
    """
    
    if root_dir is None:
        root_dir = HF_LEROBOT_HOME / repo_id.replace("/", "_")
    else:
        root_dir = Path(root_dir)
    
    logging.info(f"Starting download of dataset: {repo_id}")
    logging.info(f"Download location: {root_dir}")
    logging.info(f"Download videos: {download_videos}")
    logging.info(f"Force cache sync: {force_cache_sync}")
    
    try:
        # Create the dataset object - this will trigger the download
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root_dir,
            download_videos=download_videos,
            force_cache_sync=force_cache_sync
        )
        
        logging.info("Dataset downloaded successfully!")
        logging.info(f"Dataset info:")
        logging.info(f"  - Total episodes: {dataset.num_episodes}")
        logging.info(f"  - Total frames: {dataset.num_frames}")
        logging.info(f"  - FPS: {dataset.fps}")
        logging.info(f"  - Features: {list(dataset.features.keys())}")
        logging.info(f"  - Camera keys: {dataset.meta.camera_keys}")
        logging.info(f"  - Video keys: {dataset.meta.video_keys}")
        logging.info(f"  - Dataset size on disk: {get_directory_size(root_dir):.2f} MB")
        
        return dataset
        
    except Exception as e:
        logging.error(f"Error downloading dataset: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        raise

def get_directory_size(path: Path) -> float:
    """Calculate the total size of a directory in MB."""
    total_size = 0
    try:
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    except Exception as e:
        logging.warning(f"Could not calculate directory size: {e}")
        return 0.0

def main():
    """Main function to run the download script."""
    
    # Configuration
    REPO_ID = "ywu67/record-test30"
    DOWNLOAD_DIR = Path("./downloaded_dataset")  # Download to current directory
    
    logging.info("="*60)
    logging.info("LeRobot Dataset Downloader")
    logging.info("="*60)
    
    try:
        # Download the dataset
        dataset = download_keychain_dataset(
            repo_id=REPO_ID,
            root_dir=DOWNLOAD_DIR,
            download_videos=True,
            force_cache_sync=False  # Set to True to re-download even if files exist
        )
        
        logging.info("="*60)
        logging.info("Download completed successfully!")
        logging.info(f"Dataset available at: {DOWNLOAD_DIR}")
        logging.info("="*60)
        
        # Print final summary
        print(f"\n✓ Successfully downloaded {REPO_ID}")
        print(f"📁 Location: {DOWNLOAD_DIR.absolute()}")
        print(f"📊 Episodes: {dataset.num_episodes}")
        print(f"🎬 Frames: {dataset.num_frames}")
        print(f"📹 Videos included: {len(dataset.meta.video_keys) > 0}")
        
    except KeyboardInterrupt:
        logging.info("Download interrupted by user")
        print("\n⚠️ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Download failed: {e}")
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()