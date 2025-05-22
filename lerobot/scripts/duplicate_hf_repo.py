"""
duplicate_hf_repo.py

📦 Utility script to duplicate a Hugging Face dataset repository,
creating a copy with a `_labeled` suffix under your username.

✅ What it does:
- Downloads an existing dataset from the Hugging Face Hub
- Copies it into a temporary folder
- Uploads the full dataset (including .parquet, .jsonl, video) to a new repo with `_labeled` appended

🔧 Usage:
    python duplicate_hf_repo.py --source 7jep7/rook_to_d4_v6 --user 7jep7

This creates:
    https://huggingface.co/datasets/7jep7/rook_to_d4_v6_labeled

Make sure you're logged in via `huggingface-cli login` before running this script.
"""

from huggingface_hub import HfApi, snapshot_download, upload_folder
import shutil
import tempfile
import argparse

def duplicate_repo(source_repo: str, user: str):
    target_repo = f"{user}/{source_repo.split('/')[-1]}_labeled"

    print(f"📥 Downloading snapshot from: {source_repo}")
    src_dir = snapshot_download(repo_id=source_repo, repo_type="dataset", local_dir_use_symlinks=False)

    print(f"📦 Creating local copy in a temp folder")
    with tempfile.TemporaryDirectory() as tmp_dir:
        shutil.copytree(src_dir, f"{tmp_dir}/copy")

        print(f"🚀 Uploading to: {target_repo}")
        upload_folder(
            folder_path=f"{tmp_dir}/copy",
            path_in_repo=".",
            repo_id=target_repo,
            repo_type="dataset",
        )

    print(f"✅ Done! New dataset: https://huggingface.co/datasets/{target_repo}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Source HF dataset repo (e.g. 7jep7/rook_to_d4_v6)")
    parser.add_argument("--user", type=str, required=True, help="Your HF username (e.g. 7jep7)")
    args = parser.parse_args()

    duplicate_repo(source_repo=args.source, user=args.user)
