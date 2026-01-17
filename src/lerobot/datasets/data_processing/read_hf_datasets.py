import pandas as pd
import json
import glob
import os

local_dir = "downloaded_dataset"

# Load info.json
info_path = os.path.join(local_dir, "meta", "info.json")
with open(info_path, 'r') as f:
    info_data = json.load(f)

print("Dataset Info:")
print(f"Total Episodes: {info_data['total_episodes']}")
print(f"Total Frames: {info_data['total_frames']}")
print(f"FPS: {info_data['fps']}")
print(f"Total Approximate Duration: {info_data['total_frames'] / info_data['fps']} seconds")

# Load stats.json (if exists)
stats_path = os.path.join(local_dir, "meta", "stats.json")
if os.path.exists(stats_path):
    with open(stats_path, 'r') as f:
        stats_data = json.load(f)
    print("\nAggregated Stats:")
    print(stats_data)
else:
    print("\nNo stats.json found.")

# Load all data Parquet files
data_pattern = os.path.join(local_dir, "data", "**", "*.parquet")
data_files = glob.glob(data_pattern, recursive=True)
df_list = []
for file in data_files:
    df_chunk = pd.read_parquet(file)
    df_list.append(df_chunk)
df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

if not df.empty:
    # Group by episode_index
    grouped = df.groupby('episode_index')
    
    for episode, group in grouped:
        num_frames = len(group)
        max_timestamp = group['timestamp'].max()
        min_timestamp = group['timestamp'].min()
        duration = max_timestamp - min_timestamp
        print(f"\nEpisode {episode}:")
        print(f"Number of Frames: {num_frames}")
        print(f"Video Duration: {duration} seconds")
        print(f"Frames per Second (calculated): {num_frames / duration if duration > 0 else 'N/A'}")
else:
    print("\nNo data Parquet files found or empty.")
