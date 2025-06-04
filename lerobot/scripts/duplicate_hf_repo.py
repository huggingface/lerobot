"""
duplicate_hf_repo.py

📦 Duplicates a Hugging Face dataset under a new name with `_labeled` appended.
Also replicates the version tag from the original repo (e.g. `v2.1`), based on info.json.

🔧 Usage:
    python duplicate_hf_repo.py --source 7jep7/rook_to_d4_v6 --user 7jep7
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, upload_folder


def duplicate_repo(source_repo: str, user: str):
    target_repo = f"{user}/{source_repo.split('/')[-1]}_labeled"
    api = HfApi()

    print(f"📥 Downloading snapshot from: {source_repo}")
    src_dir = snapshot_download(repo_id=source_repo, repo_type="dataset", local_dir_use_symlinks=False)

    print("📦 Creating local copy in a temp folder...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_copy = f"{tmp_dir}/copy"
        shutil.copytree(src_dir, tmp_copy)

        print("🔍 Reading version tag from info.json...")
        info_path = Path(tmp_copy) / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        version_tag = info.get("_version_")
        if not version_tag:
            print("⚠️ No _version_ found in info.json — skipping tag creation.")
        else:
            print(f"✅ Found version tag: {version_tag}")

        print(f"🚀 Uploading dataset to: {target_repo}")
        upload_folder(
            folder_path=tmp_copy,
            path_in_repo=".",
            repo_id=target_repo,
            repo_type="dataset",
        )

        if version_tag:
            print(f"🔁 Creating Git tag {version_tag} on new dataset...")
            api.create_tag(repo_id=target_repo, tag=version_tag, repo_type="dataset")

    print(f"✅ Done! New dataset available at: https://huggingface.co/datasets/{target_repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, required=True, help="Source HF dataset repo (e.g. 7jep7/rook_to_d4_v6)"
    )
    parser.add_argument("--user", type=str, required=True, help="Your HF username (e.g. 7jep7)")
    args = parser.parse_args()

    duplicate_repo(source_repo=args.source, user=args.user)
