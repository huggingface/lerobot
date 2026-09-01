#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score a LeRobot dataset with a reward model and write a signal sidecar."""

import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.configs import parser
from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.factory import make_reward_model
from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
from lerobot.rewards.scoring import ScoringSummary, score_dataset_with_reward_model
from lerobot.utils.import_utils import require_package

logger = logging.getLogger(__name__)


@dataclass
class ScoreDatasetConfig:
    """Configuration for ``lerobot-score-dataset``."""

    dataset_repo_id: str
    reward_model_path: str
    dataset_root: Path | None = None
    dataset_revision: str | None = None
    reward_model_revision: str | None = None
    output_path: Path | None = None
    episodes: list[int] | None = None
    device: str | None = None
    image_key: str | None = None
    batch_size: int = 32
    num_subsampled_frames: int = 4
    resume: bool = True
    push_to_hub: bool = False
    hub_path: str = "reward_signals/robometer.parquet"

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_subsampled_frames < 1:
            raise ValueError(f"num_subsampled_frames must be >= 1, got {self.num_subsampled_frames}")


def run_score_dataset(cfg: ScoreDatasetConfig) -> ScoringSummary:
    """Load the requested dataset/model and run the shared scoring workflow."""
    require_package("datasets", extra="dataset")
    from lerobot.datasets import LeRobotDataset

    reward_config = RewardModelConfig.from_pretrained(
        cfg.reward_model_path,
        revision=cfg.reward_model_revision,
    )
    if not isinstance(reward_config, RobometerConfig):
        raise ValueError(f"lerobot-score-dataset currently supports RoboMeter, got {reward_config.type!r}")
    reward_config.pretrained_path = cfg.reward_model_path
    reward_config.pretrained_revision = cfg.reward_model_revision
    if cfg.device is not None:
        reward_config.device = cfg.device
    if cfg.image_key is not None:
        previous_image_key = reward_config.image_key
        reward_config.image_key = cfg.image_key
        previous_feature = reward_config.input_features.pop(previous_image_key, None)
        if previous_feature is not None:
            reward_config.input_features.setdefault(cfg.image_key, previous_feature)

    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
        revision=cfg.dataset_revision,
        download_videos=True,
    )
    reward_model = make_reward_model(reward_config)
    output_path = cfg.output_path or Path(dataset.root) / "reward_signals" / "robometer.parquet"

    summary = score_dataset_with_reward_model(
        dataset,
        reward_model,
        output_path=output_path,
        model_id=cfg.reward_model_path,
        model_revision=cfg.reward_model_revision,
        episode_indices=cfg.episodes,
        resume=cfg.resume,
        batch_size=cfg.batch_size,
        num_subsampled_frames=cfg.num_subsampled_frames,
    )
    logger.info(
        "Scored %d episode(s), %d frame(s); new=%d resumed=%d; artifact=%s",
        summary.episode_count,
        summary.frame_count,
        summary.new_episode_count,
        summary.resumed_episode_count,
        summary.artifact_path,
    )
    if cfg.push_to_hub:
        HfApi().upload_file(
            path_or_fileobj=str(summary.artifact_path),
            path_in_repo=cfg.hub_path,
            repo_id=cfg.dataset_repo_id,
            repo_type="dataset",
            commit_message="Add reward-model frame signals (lerobot-score-dataset)",
        )
        logger.info(
            "Uploaded scoring artifact to hf://datasets/%s/%s",
            cfg.dataset_repo_id,
            cfg.hub_path,
        )
    return summary


@parser.wrap()
def score_dataset_cli(cfg: ScoreDatasetConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_score_dataset(cfg)


def main() -> None:
    score_dataset_cli()


if __name__ == "__main__":
    main()
