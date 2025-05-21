import json
import os
import re

# === CONFIGURATION ===

DATASET_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/7jep7/rook_to_d4_v3/data/chunk-000")

PIECE_TYPE = "rook"
IS_WHITE = True
GOAL_SQUARE = "D4"
DEFAULT_GRASP_TYPE = "top"

# === FUNCTIONS ===


def is_valid_square(square):
    return re.match(r"^[A-H][1-8]$", square.upper()) is not None


def get_existing_episodes(path):
    return sorted([f for f in os.listdir(path) if f.startswith("episode_") and f.endswith(".parquet")])


def load_existing_metadata(parquet_path):
    metadata_path = parquet_path.replace(".parquet", ".json")
    return os.path.exists(metadata_path)


def write_metadata(parquet_path, start_square):
    metadata_path = parquet_path.replace(".parquet", ".json")
    metadata = {
        "start_square": start_square.upper(),
        "goal_square": GOAL_SQUARE,
        "piece_type": PIECE_TYPE,
        "grasp_type": DEFAULT_GRASP_TYPE,
        "is_white": IS_WHITE,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


# === MAIN LOOP ===


def main():
    episodes = get_existing_episodes(DATASET_DIR)
    print(f"\nFound {len(episodes)} episodes in: {DATASET_DIR}\n")

    for ep_filename in episodes:
        parquet_path = os.path.join(DATASET_DIR, ep_filename)
        metadata_path = parquet_path.replace(".parquet", ".json")

        if os.path.exists(metadata_path):
            print(f"✅ Skipping {ep_filename} (already has metadata)")
            continue

        while True:
            start_square = input(f"📝 Enter start square for {ep_filename} (e.g. A2): ").strip().upper()
            if is_valid_square(start_square):
                break
            print("❌ Invalid square. Use format like A1–H8.")

        write_metadata(parquet_path, start_square)
        print(f"📦 Metadata saved for {ep_filename}: {start_square} → {GOAL_SQUARE}\n")


if __name__ == "__main__":
    main()
