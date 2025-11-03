"""
Encode pending videos from PNG images for a LeRobot dataset.

This script is useful when you recorded a dataset with defer_video_encoding=True
and encode_on_exit=False, leaving PNG images on disk that need to be encoded into videos.

Example:
```shell
lerobot-encode-videos \
    --dataset.repo_id=<my_username>/<my_dataset_name> \
    --dataset.root=<optional_custom_path>
```

You can also specify episode ranges:
```shell
lerobot-encode-videos \
    --dataset.repo_id=<my_username>/<my_dataset_name> \
    --start_episode=0 \
    --end_episode=10
```
"""

import logging
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


@dataclass
class EncodeVideosConfig:
    dataset: "DatasetConfig"  # type: ignore
    start_episode: int | None = None
    end_episode: int | None = None


@dataclass
class DatasetConfig:
    repo_id: str
    root: str | None = None


@parser.wrap()
def encode_videos(cfg: EncodeVideosConfig) -> None:
    """Encode pending videos from PNG images for episodes in the dataset."""
    init_logging()
    logging.info(f"Loading dataset: {cfg.dataset.repo_id}")

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        encode_on_exit=False,  # Not relevant here, but needed for initialization
    )

    logging.info(f"Dataset loaded. Total episodes: {dataset.num_episodes}")

    # Encode pending videos
    dataset.encode_pending_videos(
        start_episode=cfg.start_episode,
        end_episode=cfg.end_episode,
    )

    logging.info("Video encoding completed!")


def main():
    encode_videos()


if __name__ == "__main__":
    main()

