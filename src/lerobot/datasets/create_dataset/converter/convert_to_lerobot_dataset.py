"""Main converter class for dataset conversion."""

import logging
from typing import Any, Dict, Optional

from tqdm import tqdm

from lerobot.datasets.create_dataset.config.dataset_config import DatasetConfig
from lerobot.datasets.create_dataset.parsers.csv_image import CSVImageParser
from lerobot.datasets.create_dataset.parsers.parse_data import DataParser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import _validate_feature_names, validate_frame


class DatasetConverter:
    """Main converter class that orchestrates the conversion process."""

    def __init__(self, config: DatasetConfig, parser: Optional[DataParser] = None):
        self.config = config
        self.parser = parser or CSVImageParser(config)
        self.logger = self._setup_logging()
        self.dataset = None

    def _setup_logging(self) -> logging.Logger:
        """Configure logging based on debug setting."""
        level = logging.DEBUG if self.config.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)

    def convert(self) -> LeRobotDataset:
        """Main conversion function."""
        self.logger.info(f"Starting conversion for repo_id: {self.config.repo_id}")

        # Create empty LeRobotDataset
        features = self.parser.get_features()

        _validate_feature_names(features)

        self.logger.info(f"Creating dataset with features: {list(features.keys())}")

        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            root=self.config.output_dir,
            robot_type=self.config.robot_type,
            features=features,
            use_videos=self.config.use_videos,
            image_writer_processes=self.config.image_writer_processes,
            image_writer_threads=self.config.image_writer_threads,
            tolerance_s=self.config.tolerance_s
        )

        # Get episode files
        episode_files = self.parser.get_episode_files()

        if not episode_files:
            raise ValueError(f"No episode files found in {self.config.input_dir}")

        # Process each episode
        for episode_file in tqdm(episode_files, desc="Converting episodes"):
            try:
                episode_data = self.parser.parse_episode(episode_file)
                self._add_episode_to_dataset(episode_data)
            except Exception as e:
                self.logger.error(f"Error processing {episode_file}: {e}")
                if not self.config.debug:
                    continue
                else:
                    raise

        self.logger.info(f"Conversion completed. Dataset saved to: {self.config.output_dir}")

        # Push to hub if requested
        if self.config.push_to_hub:
            self.dataset.push_to_hub(
                private=self.config.private_repo,
                push_videos=self.config.use_videos
            )

        return self.dataset

    def _add_episode_to_dataset(self, episode_data: Dict[str, Any]) -> None:
        """Add parsed episode data to the dataset."""
        num_frames = len(episode_data["timestamps"])

        for frame_idx in range(num_frames):
            frame = self._create_frame(episode_data, frame_idx)

            # Validate frame if enabled
            if self.config.validate_data:
                validate_frame(frame, self.dataset.features)

            # Add frame to dataset with task and timestamp
            task = episode_data["tasks"][frame_idx]
            timestamp = episode_data["timestamps"][frame_idx]
            self.dataset.add_frame(frame, task, timestamp)

        # Save the complete episode
        self.dataset.save_episode()
        self.logger.debug(f"Saved episode with {num_frames} frames")

    def _create_frame(self, episode_data: Dict[str, Any], frame_idx: int) -> Dict[str, Any]:
        """Create a frame dictionary from episode data."""
        frame = {}

        # Add action data
        if episode_data.get("actions"):
            frame["action"] = episode_data["actions"][frame_idx]

        # Add state data
        if episode_data.get("states"):
            frame["observation.state"] = episode_data["states"][frame_idx]

        # Add image data
        for img_key in self.config.image_keys:
            if episode_data["images"][img_key][frame_idx] is not None:
                frame[img_key] = episode_data["images"][img_key][frame_idx]

        return frame
