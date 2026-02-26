"""HuggingFace service for authentication and repo management."""

import subprocess
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, list_datasets

from lerobot.webui.backend.models.recording import HFRepoInfo
from lerobot.webui.backend.models.system import HFLoginStatus


class HuggingFaceService:
    """Service for HuggingFace CLI integration."""

    def __init__(self):
        """Initialize HuggingFaceService."""
        self.api = HfApi()

    def check_login(self) -> HFLoginStatus:
        """Check if user is logged into HuggingFace CLI.

        Returns:
            HFLoginStatus with login information.
        """
        try:
            result = subprocess.run(
                ["huggingface-cli", "whoami"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                username = result.stdout.strip()
                return HFLoginStatus(is_logged_in=True, username=username)

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return HFLoginStatus(is_logged_in=False, username=None)

    def list_repos(self, username: str) -> List[HFRepoInfo]:
        """List user's dataset repositories.

        Args:
            username: HuggingFace username.

        Returns:
            List of HFRepoInfo objects.
        """
        try:
            datasets = list_datasets(author=username)

            return [
                HFRepoInfo(
                    repo_id=dataset.id,
                    repo_type="dataset",
                    private=dataset.private,
                    url=f"https://huggingface.co/datasets/{dataset.id}",
                )
                for dataset in datasets
            ]

        except Exception as e:
            print(f"Error listing repos: {e}")
            return []

    def create_repo(self, username: str, repo_name: str, private: bool = False) -> Optional[HFRepoInfo]:
        """Create a new dataset repository.

        Args:
            username: HuggingFace username.
            repo_name: Repository name (without username prefix).
            private: Whether to create private repository.

        Returns:
            HFRepoInfo for created repo, or None if failed.
        """
        repo_id = f"{username}/{repo_name}"

        try:
            # Use huggingface-cli to create repo
            cmd = ["huggingface-cli", "repo", "create", repo_id, "--type", "dataset"]
            if private:
                cmd.append("--private")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return HFRepoInfo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=private,
                    url=f"https://huggingface.co/datasets/{repo_id}",
                )

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Error creating repo: {e}")

        return None

    def get_cache_path(self, repo_id: str) -> Optional[str]:
        """Get the local cache path for a dataset.

        Args:
            repo_id: HuggingFace repo ID (username/dataset_name).

        Returns:
            Path to cached dataset, or None if not exists.
        """
        cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id

        if cache_dir.exists():
            return str(cache_dir)

        return None
