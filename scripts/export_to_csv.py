#!/usr/bin/env python3
import csv
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    dataset_repo_id = "NLTuan/up-down"
    output_file = "dataset_data.csv"
    
    print(f"--- Exporting Dataset to CSV: {dataset_repo_id} ---")
    
    try:
        dataset = LeRobotDataset(dataset_repo_id, video_backend="pyav")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    total_frames = dataset.meta.total_frames
    print(f"Total frames to process: {total_frames}")

    with open(output_file, mode='w', newline='') as csvfile:
        # Define headers
        # action has 9 values, observation.state has 6 values
        headers = ['index', 'episode_index', 'frame_index', 'timestamp']
        headers += [f'action_{i}' for i in range(9)]
        headers += [f'state_{i}' for i in range(6)]
        
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for i in range(total_frames):
            if i % 100 == 0:
                print(f"Processing frame {i}/{total_frames}...")
            
            sample = dataset[i]
            
            row = {
                'index': sample['index'].item(),
                'episode_index': sample['episode_index'].item(),
                'frame_index': sample['frame_index'].item(),
                'timestamp': sample['timestamp'].item(),
            }
            
            # Add actions
            actions = sample['action'].numpy()
            for j, val in enumerate(actions):
                row[f'action_{j}'] = val
                
            # Add states
            states = sample['observation.state'].numpy()
            for j, val in enumerate(states):
                row[f'state_{j}'] = val
                
            writer.writerow(row)

    print(f"\n--- Export Complete: '{output_file}' ---")

if __name__ == "__main__":
    main()
