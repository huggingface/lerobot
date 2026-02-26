"""Configuration management service for persistent config storage."""

import json
from pathlib import Path
from typing import Optional

from lerobot.webui.backend.models.config import Config


class ConfigManager:
    """Manages loading and saving configuration to/from JSON file."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize ConfigManager.

        Args:
            config_path: Path to config file. Defaults to webui_config.json in repo root.
        """
        if config_path is None:
            # Default to repo root
            repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
            config_path = repo_root / "webui_config.json"

        self.config_path = config_path

    def load_config(self) -> Config:
        """Load configuration from JSON file.

        Returns:
            Config object. Returns default empty config if file doesn't exist.
        """
        if not self.config_path.exists():
            return Config()

        try:
            with open(self.config_path) as f:
                data = json.load(f)
            return Config(**data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            return Config()

    def save_config(self, config: Config) -> None:
        """Save configuration to JSON file.

        Args:
            config: Config object to save.
        """
        with open(self.config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

    def reset_config(self) -> Config:
        """Reset configuration to defaults.

        Returns:
            Fresh default Config object.
        """
        if self.config_path.exists():
            self.config_path.unlink()

        return Config()

    def config_exists(self) -> bool:
        """Check if config file exists.

        Returns:
            True if config file exists, False otherwise.
        """
        return self.config_path.exists()
