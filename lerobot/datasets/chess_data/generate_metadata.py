import os
import json
import re

# === CONFIGURATION ===

DATASET_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/7jep7/rook_to_d4/data")

# Defaults — you can override via script args or later extension
PIECE_TYPE = "rook"
IS_WHITE = True
GOAL_SQUARE = "D4"
DEFAULT_GRASP_TYPE = "top"

# === FUNCTIONS ===

def is_valid_square(square):
    return re.match(r"^[A-H][1-8]$", square.upper()) is not None

def get_existing_episodes(path):
    return sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d.startswith("episode_")
    ])

def load_existing_metadata(ep_path):
    return os.path.exists(os.path.join(ep_path, "metadata.json"))

def write_metadata(ep_path, start_square):
    metadata = {
        "start_square": start_square.upper(),
        "goal_square": GOAL_SQUARE,
        "piece_type": PIECE_TYPE,
        "grasp_type": DEFAULT_GRASP_TYPE,
        "is_white": IS_WHITE
    }
    with open(os.path.join(ep_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

# === MAIN LOOP ===

def main():
    episodes = get_existing_episodes(DATASET_DIR)
    print(f"\nFound {len(episodes)} episodes in: {DATASET_DIR}\n")

    for ep in episodes:
        ep_path = os.path.join(DATASET_DIR, ep)
        metadata_path = os.path.join(ep_path, "metadata.json")

        if os.path.exists(metadata_path):
            print(f"✅ Skipping {ep} (already has metadata)")
            continue

        # Prompt for start square
        while True:
            start_square = input(f"📝 Enter start square for {ep} (e.g. A2): ").strip().upper()
            if is_valid_square(start_square):
                break
            print("❌ Invalid square. Use format like A1–H8.")

        write_metadata(ep_path, start_square)
        print(f"📦 Metadata saved for {ep}: {start_square} → {GOAL_SQUARE}\n")

if __name__ == "__main__":
    main()
