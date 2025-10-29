from pathlib import Path
from huggingface_hub import HfApi, upload_folder

REPO_ID = "zcp/record-test-20-grab"
REPO_TYPE = "dataset"   # change to "model" or "space" if needed
LOCAL_DIR = Path("/Users/chenz/.cache/huggingface/lerobot/ywu67/grab1-record-test")

p = LOCAL_DIR.expanduser().resolve()
if not p.exists() or not p.is_dir():
    raise ValueError(f"Not a directory: {p}")

api = HfApi()
# Create the repo if needed (safe if it already exists)
api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True)

# Upload the *contents* of LOCAL_DIR to the repo root
upload_folder(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    folder_path=str(p),
    path_in_repo="",                        # upload at repo root (no extra top-level folder)
    # use either allow/ignore or include/exclude depending on your hf-hub version:
    allow_patterns=["**/*"],
)
print("Done.")
