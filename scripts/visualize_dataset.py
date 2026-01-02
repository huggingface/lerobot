#!/usr/bin/env python3
import os
import torch
from pathlib import Path
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def save_frame(dataset, frame_index, output_dir):
    sample = dataset[frame_index]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract images
    # LeRobotDataset returns images as tensors (C, H, W) in range [0, 1]
    for key in sample:
        if "observation.images" in key:
            img_tensor = sample[key]
            # Convert to PIL Image
            # (C, H, W) -> (H, W, C)
            img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
            img = Image.fromarray(img_array)
            
            # Save image
            camera_name = key.split(".")[-1]
            filename = f"frame_{frame_index}_{camera_name}.jpg"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            print(f"Saved {filepath}")

def main():
    dataset_repo_id = "NLTuan/up-down"
    output_dir = "visualizations"
    
    print(f"--- Visualizing Dataset: {dataset_repo_id} ---")
    
    try:
        dataset = LeRobotDataset(dataset_repo_id, video_backend="pyav")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Save a few frames from the beginning, middle, and end
    total_frames = dataset.meta.total_frames
    indices_to_save = [0, total_frames // 2, total_frames - 1]
    
    for idx in indices_to_save:
        print(f"Processing frame {idx}...")
        save_frame(dataset, idx, output_dir)

    print(f"\n--- Visualization frames saved to '{output_dir}/' ---")

if __name__ == "__main__":
    main()
