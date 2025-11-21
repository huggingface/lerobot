"""
Script to convert Unitree json data to the LeRobot dataset v2.0 format.

python unitree_lerobot/utils/sort_and_rename_folders.py --data_dir $HOME/datasets/g1_grabcube_double_hand
"""

import os
import tyro
import uuid
from pathlib import Path


def sort_and_rename_folders(data_dir: Path) -> None:
    # Get the list of folders sorted by name
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    temp_mapping = {}

    # First, rename all folders to unique temporary names
    for folder in folders:
        temp_name = str(uuid.uuid4())
        original_path = os.path.join(data_dir, folder)
        temp_path = os.path.join(data_dir, temp_name)
        os.rename(original_path, temp_path)
        temp_mapping[temp_name] = folder

    # Then, rename them to the final target names
    start_number = 0
    for temp_name, original_folder in temp_mapping.items():
        new_folder_name = f"episode_{start_number:04d}"
        temp_path = os.path.join(data_dir, temp_name)
        new_path = os.path.join(data_dir, new_folder_name)
        os.rename(temp_path, new_path)
        start_number += 1

    print("The folders have been successfully renamed.")


if __name__ == "__main__":
    tyro.cli(sort_and_rename_folders)
