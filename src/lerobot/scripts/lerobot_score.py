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

"""Score a LeRobot dataset and publish a frame-signal sidecar."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.configs import parser
from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards import make_frame_scorer, make_reward_model
from lerobot.utils.import_utils import _av_available, _datasets_available, require_package

if TYPE_CHECKING or (_datasets_available and _av_available):
    from lerobot.datasets import LeRobotDataset


@dataclass
class ScoreConfig:
    """Configuration for ``lerobot-score``."""

    dataset_repo_id: str
    reward_model_path: str
    dataset_root: Path | None = None
    dataset_revision: str | None = None
    reward_model_revision: str | None = None
    name: str | None = None
    episodes: list[int] | None = None
    device: str | None = None
    image_key: str | None = None
    batch_size: int = 32
    num_subsampled_frames: int = 4
    resume: bool = True
    overwrite: bool = False
    push_to_hub: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_subsampled_frames < 1:
            raise ValueError(f"num_subsampled_frames must be >= 1, got {self.num_subsampled_frames}")


def run_score(cfg: ScoreConfig) -> None:
    """Load the model and dataset, then publish a score sidecar."""
    require_package("datasets", extra="dataset")
    require_package("av", extra="dataset")

    overrides = []
    if cfg.device is not None:
        overrides.append(f"--device={cfg.device}")
    if cfg.image_key is not None:
        overrides.append(f"--image_key={cfg.image_key}")
    reward_config = RewardModelConfig.from_pretrained(
        cfg.reward_model_path,
        revision=cfg.reward_model_revision,
        cli_overrides=overrides,
    )
    model = make_reward_model(reward_config)
    scorer = make_frame_scorer(
        model,
        batch_size=cfg.batch_size,
        num_subsampled_frames=cfg.num_subsampled_frames,
    )

    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
        revision=cfg.dataset_revision,
        download_videos=True,
    )
    score_name = cfg.name or scorer.name
    dataset.add_score(
        scorer,
        name=score_name,
        episodes=cfg.episodes,
        resume=cfg.resume,
        overwrite=cfg.overwrite,
    )

    if cfg.push_to_hub:
        dataset.push_score_to_hub(score_name)


@parser.wrap()
def score_cli(cfg: ScoreConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_score(cfg)


def main() -> None:
    score_cli()  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
