import shutil
from pathlib import Path


def save_policy_to_safetensors(output_dir, repo_id="lerobot/pusht"):
    ...
    repo_dir = Path(output_dir) / repo_id

    if repo_dir.exists():
        shutil.rmtree(repo_dir)
