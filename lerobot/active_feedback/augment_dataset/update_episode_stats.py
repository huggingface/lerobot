#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to update metadata for a dataset with new camera views (raw and masked).
This updates the episodes_stats.jsonl and info.json files.
"""

import os
import json
import argparse
import jsonlines
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a jsonl file."""
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for item in reader:
            data.append(item)
    return data


def save_jsonl(data: List[Dict], file_path: str) -> None:
    """Save data to a jsonl file."""
    with jsonlines.open(file_path, 'w') as writer:
        for item in data:
            writer.write(item)


def load_json(file_path: str) -> Dict:
    """Load data from a json file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str) -> None:
    """Save data to a json file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def update_info_file(original_info_path: str, updated_info_path: str) -> None:
    """
    Update the info.json file to reflect the new camera structure.
    
    Args:
        original_info_path: Path to the original info.json file
        updated_info_path: Path to save the updated info.json file
    """
    # Load the original info
    info = load_json(original_info_path)
    
    # Update the features section to reflect the new camera structure
    features = info.get("features", {})
    
    # Check if we already have the raw/masked features
    if "observation.images.front_raw" in features and "observation.images.front_masked" in features:
        print("Info file already has raw/masked features. No update needed.")
        save_json(info, updated_info_path)
        return
    
    # If we have front/overhead without raw/masked, we need to update
    if "observation.images.front" in features:
        # Create front_raw from front
        front_feature = features.pop("observation.images.front")
        features["observation.images.front_raw"] = copy.deepcopy(front_feature)
        features["observation.images.front_masked"] = copy.deepcopy(front_feature)
        
        # Update video paths if needed
        if "info" in features["observation.images.front_raw"]:
            features["observation.images.front_raw"]["info"]["video.is_depth_map"] = False
            features["observation.images.front_masked"]["info"]["video.is_depth_map"] = False
    
    if "observation.images.overhead" in features:
        # Create overhead_raw from overhead
        overhead_feature = features.pop("observation.images.overhead")
        features["observation.images.overhead_raw"] = copy.deepcopy(overhead_feature)
        features["observation.images.overhead_masked"] = copy.deepcopy(overhead_feature)
        
        # Update video paths if needed
        if "info" in features["observation.images.overhead_raw"]:
            features["observation.images.overhead_raw"]["info"]["video.is_depth_map"] = False
            features["observation.images.overhead_masked"]["info"]["video.is_depth_map"] = False
    
    # Update total videos (double the count if we're adding masked versions)
    if "total_videos" in info:
        info["total_videos"] = info["total_videos"] * 2
    
    # Save the updated info
    save_json(info, updated_info_path)
    print(f"Updated info file with new camera structure")


def update_episodes_stats_file(original_stats_path: str, updated_stats_path: str) -> None:
    """
    Update the episodes_stats.jsonl file to reflect the new camera structure.
    
    Args:
        original_stats_path: Path to the original episodes_stats.jsonl file
        updated_stats_path: Path to save the updated episodes_stats.jsonl file
    """
    # Load the original stats
    original_stats = load_jsonl(original_stats_path)
    updated_stats = []
    
    # Process each episode's stats
    for episode_stat in original_stats:
        stats = episode_stat.get("stats", {})
        
        # Check if we need to update the camera stats
        if "observation.images.front" in stats and "observation.images.front_raw" not in stats:
            # Create front_raw and front_masked from front
            front_stats = stats.pop("observation.images.front")
            stats["observation.images.front_raw"] = copy.deepcopy(front_stats)
            stats["observation.images.front_masked"] = copy.deepcopy(front_stats)
        
        if "observation.images.overhead" in stats and "observation.images.overhead_raw" not in stats:
            # Create overhead_raw and overhead_masked from overhead
            overhead_stats = stats.pop("observation.images.overhead")
            stats["observation.images.overhead_raw"] = copy.deepcopy(overhead_stats)
            stats["observation.images.overhead_masked"] = copy.deepcopy(overhead_stats)
        
        # Add the updated episode stats
        updated_stats.append(episode_stat)
    
    # Save the updated stats
    save_jsonl(updated_stats, updated_stats_path)
    print(f"Updated episodes stats file with new camera structure: {len(updated_stats)} episodes")


def main():
    parser = argparse.ArgumentParser(description="Update metadata for dataset with new camera structure")
    parser.add_argument("--meta-dir", type=str, required=True, 
                        help="Directory containing metadata files")
    parser.add_argument("--output-dir", type=str, required=True, 
                        help="Directory to save updated metadata files")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        print(f"Metadata directory: {args.meta_dir}")
        print(f"Output directory: {args.output_dir}")
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define paths
    original_info_path = os.path.join(args.meta_dir, "info.json")
    original_stats_path = os.path.join(args.meta_dir, "episodes_stats.jsonl")
    original_episodes_path = os.path.join(args.meta_dir, "episodes.jsonl")
    original_tasks_path = os.path.join(args.meta_dir, "tasks.jsonl")
    
    updated_info_path = os.path.join(args.output_dir, "info.json")
    updated_stats_path = os.path.join(args.output_dir, "episodes_stats.jsonl")
    updated_episodes_path = os.path.join(args.output_dir, "episodes.jsonl")
    updated_tasks_path = os.path.join(args.output_dir, "tasks.jsonl")
    
    # Update info.json
    update_info_file(original_info_path, updated_info_path)
    
    # Update episodes_stats.jsonl
    update_episodes_stats_file(original_stats_path, updated_stats_path)
    
    # Copy episodes.jsonl and tasks.jsonl as they don't need changes
    if os.path.exists(original_episodes_path):
        episodes = load_jsonl(original_episodes_path)
        save_jsonl(episodes, updated_episodes_path)
        print(f"Copied episodes file: {len(episodes)} episodes")
    
    if os.path.exists(original_tasks_path):
        tasks = load_jsonl(original_tasks_path)
        save_jsonl(tasks, updated_tasks_path)
        print(f"Copied tasks file: {len(tasks)} tasks")
    
    print("Metadata update complete!")


if __name__ == "__main__":
    main()
