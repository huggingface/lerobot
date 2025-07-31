"""Base class for dataset parsers."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from lerobot.datasets.create_dataset.config.dataset_config import DatasetConfig


class DataParser(ABC):
    """Abstract base class for parsing different data formats."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_episode_files(self) -> List[Path]:
        """Return list of episode files to process."""
        pass

    @abstractmethod
    def parse_episode(self, episode_file: Path) -> Dict[str, List[Any]]:
        """Parse a single episode file and return structured data.

        Returns:
            Dict containing:
                - actions: List of action arrays
                - states: List of state arrays
                - images: Dict of image key to list of image arrays
                - timestamps: List of timestamps
                - tasks: List of task names
        """
        pass

    @abstractmethod
    def get_features(self) -> Dict[str, Dict]:
        """Return feature definitions for the dataset.

        Returns:
            Dict of feature definitions compatible with LeRobotDataset format:
            {
                "feature_name": {
                    "dtype": str,
                    "shape": tuple,
                    "names": list
                }
            }
        """
        pass
