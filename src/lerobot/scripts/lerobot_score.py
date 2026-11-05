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
    # Loaded with --reward_model.path=..., or built with --reward_model.type=... for zero-shot models.
    reward_model: RewardModelConfig | None = None
    dataset_root: Path | None = None
    dataset_revision: str | None = None
    name: str | None = None
    episodes: list[int] | None = None
    # Scorer options: only the ones set are passed, so each scorer keeps its own defaults.
    batch_size: int | None = None
    num_subsampled_frames: int | None = None
    num_samples: int | None = None
    stride: int | None = None
    head_mode: str | None = None
    resume: bool = True
    overwrite: bool = False
    push_to_hub: bool = False

    def __post_init__(self) -> None:
        reward_model_path = parser.get_path_arg("reward_model")
        if reward_model_path:
            self.reward_model = RewardModelConfig.from_pretrained(
                reward_model_path,
                revision=parser.get_pretrained_revision("reward_model"),
                cli_overrides=parser.get_cli_overrides("reward_model"),
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Keys for draccus pretrained-path loading."""
        return ["reward_model"]

    @property
    def scorer_options(self) -> dict[str, int | str]:
        options = {
            "batch_size": self.batch_size,
            "num_subsampled_frames": self.num_subsampled_frames,
            "num_samples": self.num_samples,
            "stride": self.stride,
            "head_mode": self.head_mode,
        }
        return {name: value for name, value in options.items() if value is not None}


def run_score(cfg: ScoreConfig) -> None:
    """Load the model and dataset, then publish a score sidecar."""
    require_package("datasets", extra="dataset")
    require_package("av", extra="dataset")
    if cfg.reward_model is None:
        raise ValueError(
            "lerobot-score needs a reward model: pass --reward_model.path=... or --reward_model.type=..."
        )

    model = make_reward_model(cfg.reward_model)
    scorer = make_frame_scorer(model, **cfg.scorer_options)

    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
        revision=cfg.dataset_revision,
        episodes=cfg.episodes,
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
