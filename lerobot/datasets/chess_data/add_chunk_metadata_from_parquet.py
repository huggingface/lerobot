import os
import glob
import pandas as pd
from datasets import Dataset, concatenate_datasets

# Path to all parquet episodes
parquet_dir = "/home/jonas-petersen/.cache/huggingface/lerobot/7jep7/rook_to_d4/data/chunk-000"
parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "episode_*.parquet")))

if not parquet_files:
    print("❌ No episode parquet files found.")
    exit()

print(f"\nFound {len(parquet_files)} episodes.")

all_rows = []

# Loop over files and add metadata
for idx, file in enumerate(parquet_files):
    df = pd.read_parquet(file)

    while True:
        square = input(f"Enter start square for {os.path.basename(file)} (e.g., A1): ").strip().upper()
        if len(square) == 2 and square[0] in "ABCDEFGH" and square[1] in "12345678":
            break
        print("❌ Invalid square. Format must be like A1–H8.")

    df["start_square"] = square
    df["goal_square"] = "D4"
    df["piece_type"] = "rook"
    df["grasp_type"] = "top"
    df["is_white"] = True

    all_rows.append(Dataset.from_pandas(df))

# Merge and save updated dataset
merged = concatenate_datasets(all_rows)
output_path = parquet_dir + "_with_metadata"
merged.save_to_disk(output_path)

print(f"\n✅ Dataset with metadata saved to: {output_path}")
