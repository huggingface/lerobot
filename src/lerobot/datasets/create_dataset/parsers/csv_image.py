"""Parser for CSV trajectory data with associated images."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.datasets.create_dataset.parsers.parse_data import DataParser
from lerobot.datasets.create_dataset.parsers.utils import (
    extract_episode_number,
    find_sample_image,
    get_image_dimensions,
    load_image,
)


class CSVImageParser(DataParser):
    """Parser for CSV trajectory data with associated images."""

    def get_episode_files(self) -> list[Path]:
        """Find all CSV files matching the pattern."""
        pattern = self.config.csv_pattern.replace("{episode}", "*")
        csv_files = list(self.config.input_dir.glob(pattern))

        # Sort by episode number first
        sorted_files = sorted(csv_files, key=lambda x: extract_episode_number(x))
        self.logger.info(f"Found {len(sorted_files)} episode files")

        # Take first N episodes if in test mode
        if self.config.test_mode:
            sorted_files = sorted_files[: self.config.max_test_episodes]
            self.logger.info(f"Test mode: using first {len(sorted_files)} episodes")

        return sorted_files

    def parse_episode(self, episode_file: Path) -> dict[str, list[Any]]:
        """Parse CSV file and load associated images."""
        # Extract episode number from filename
        episode_num = extract_episode_number(episode_file)

        # Read CSV data
        self.logger.info(f"Reading CSV file: {episode_file}")
        df = pd.read_csv(episode_file)
        self.logger.debug(f"Episode {episode_num}: {len(df)} frames")

        # Debug first frame state data
        if episode_num == 0:
            self.logger.info(f"Episode 0 first row: {df.iloc[0][self.config.state_columns].values}")

        episode_data = {"actions": [], "states": [], "images": {}, "timestamps": [], "tasks": []}

        # Initialize image lists for each camera
        for img_key in self.config.image_keys:
            episode_data["images"][img_key] = []

        # Process each frame
        for frame_idx, row in df.iterrows():
            self._process_frame(
                row=row, frame_idx=frame_idx, episode_num=episode_num, episode_data=episode_data
            )

        return episode_data

    def get_features(self) -> dict[str, dict]:
        """Define features based on configuration."""
        features = {}

        # Action features
        if self.config.action_columns:
            features["action"] = {
                "dtype": "float32",
                "shape": (len(self.config.action_columns),),
                "names": self.config.action_columns,
            }

        # State features
        if self.config.state_columns:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (len(self.config.state_columns),),
                "names": self.config.state_columns,
            }

        # Image features
        for img_key in self.config.image_keys:
            # Get image dimensions from a sample image
            sample_img_path = find_sample_image(
                self.config.input_dir, self.config.image_pattern, self.config.image_extension
            )
            if sample_img_path:
                height, width, channels = get_image_dimensions(sample_img_path)
                features[img_key] = {
                    "dtype": "video" if self.config.use_videos else "image",
                    "shape": (height, width, channels),
                    "names": ["height", "width", "channels"],
                }

        return features

    def _process_frame(
        self, row: pd.Series, frame_idx: int, episode_num: int, episode_data: dict[str, list[Any]]
    ) -> None:
        """Process a single frame from CSV data."""
        # Extract action and state data
        if self.config.action_columns:
            action = row[self.config.action_columns].values.astype(np.float32)
            episode_data["actions"].append(action)

        if self.config.state_columns:
            # Keep all state columns including mass
            state = row[self.config.state_columns].values.astype(np.float32)
            if frame_idx == 0:  # Debug first frame
                self.logger.info(f"First frame state values: {state}")
                self.logger.info(f"State columns: {self.config.state_columns}")
                self.logger.info(f"Raw row values: {row[self.config.state_columns].values}")
                # Add more detailed debugging
                self.logger.info(f"Full row data: {row.to_dict()}")
            episode_data["states"].append(state)

        # Load images
        for img_key in self.config.image_keys:
            img_path = self._get_image_path(episode_num, frame_idx)
            if img_path.exists():
                image = load_image(img_path)
                episode_data["images"][img_key].append(image)
            else:
                self.logger.warning(f"Image not found: {img_path}")
                # Use a placeholder or skip frame
                episode_data["images"][img_key].append(None)

        # Handle timestamps
        if "time" in row.index:
            episode_data["timestamps"].append(float(row["time"]))
        else:
            # Throw error
            raise ValueError("Timestamp column not found in CSV file")

        # Add task information
        episode_data["tasks"].append(self.config.task_name)

    def _get_image_path(self, episode_num: int, frame_idx: int) -> Path:
        """Construct image file path."""
        filename = (
            self.config.image_pattern.format(episode=episode_num, frame=frame_idx)
            + self.config.image_extension
        )
        return self.config.input_dir / filename
