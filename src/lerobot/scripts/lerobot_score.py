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

from lerobot.configs import parser
from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.factory import make_frame_scorer, make_reward_model
from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
from lerobot.rewards.rynnvalue.configuration_rynnvalue import RynnValueConfig
from lerobot.utils.import_utils import require_package


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
    batch_size: int | None = None
    default_task: str | None = None
    # RoboMeter
    num_subsampled_frames: int = 4
    # RynnValue
    inference_fps: float = 1.0
    max_frames: int | None = 8
    horizon_s: float | None = None
    robot_description: str | None = None
    camera_description: str | None = None
    use_meta: bool | None = None
    resume: bool = True
    overwrite: bool = False
    push_to_hub: bool = False

    def __post_init__(self) -> None:
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_subsampled_frames < 1:
            raise ValueError(f"num_subsampled_frames must be >= 1, got {self.num_subsampled_frames}")
        if self.inference_fps <= 0:
            raise ValueError(f"inference_fps must be > 0, got {self.inference_fps}")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1 or None, got {self.max_frames}")
        if self.horizon_s is not None and self.horizon_s <= 0:
            raise ValueError(f"horizon_s must be > 0 or None, got {self.horizon_s}")


def run_score(cfg: ScoreConfig) -> None:
    """Load the dataset and model, then publish a score sidecar."""
    require_package("datasets", extra="dataset")
    from lerobot.datasets import LeRobotDataset

    reward_config = RewardModelConfig.from_pretrained(
        cfg.reward_model_path,
        revision=cfg.reward_model_revision,
    )
    if not isinstance(reward_config, (RobometerConfig, RynnValueConfig)):
        raise ValueError(
            f"lerobot-score currently supports RoboMeter and RynnValue, got {reward_config.type!r}"
        )
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
    if cfg.default_task is not None:
        reward_config.default_task = cfg.default_task
    if isinstance(reward_config, RynnValueConfig):
        if cfg.robot_description is not None:
            reward_config.robot_description = cfg.robot_description
        if cfg.camera_description is not None:
            reward_config.camera_description = cfg.camera_description
        if cfg.use_meta is not None:
            reward_config.use_meta = cfg.use_meta

    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
        revision=cfg.dataset_revision,
        download_videos=True,
    )
    model = make_reward_model(reward_config)
    if isinstance(reward_config, RynnValueConfig):
        scorer = make_frame_scorer(
            model,
            dataset_fps=float(dataset.fps),
            batch_size=cfg.batch_size or 2,
            inference_fps=cfg.inference_fps,
            max_frames=cfg.max_frames,
            horizon_s=cfg.horizon_s,
        )
    else:
        scorer = make_frame_scorer(
            model,
            batch_size=cfg.batch_size or 32,
            num_subsampled_frames=cfg.num_subsampled_frames,
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
    score_cli()  # type: ignore[call-arg, unused-ignore]


if __name__ == "__main__":
    main()
