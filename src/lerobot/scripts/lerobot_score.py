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

r"""Score a LeRobot dataset and save frame-aligned signals under ``scores/<name>.parquet``.

The dataset is read from, and the scores are written to, ``--dataset_root``, which defaults to
``$HF_LEROBOT_HOME/<dataset_repo_id>``. Replace the dataset ID and paths below with your own.

Examples:
    Score with a pretrained RoboMeter checkpoint:

        uv run lerobot-score \
            --dataset_repo_id=user/dataset \
            --reward_model.path=lerobot/Robometer-4B \
            --reward_model.device=cuda

    Score with a trained SARM checkpoint, evaluating every fifth frame and interpolating:

        uv run lerobot-score \
            --dataset_repo_id=user/dataset \
            --reward_model.path=/path/to/sarm-checkpoint \
            --reward_model.device=cuda \
            --reward_model.head_mode=sparse \
            --reward_model.stride=5

    Score with zero-shot TOPReward:

        uv run lerobot-score \
            --dataset_repo_id=user/dataset \
            --reward_model.type=topreward \
            --reward_model.device=cuda \
            --reward_model.num_samples=15

The score name defaults to the model type; override it with ``--name=my-run``.
Set ``--reward_model.image_key`` if your camera differs from the checkpoint's configured key.
Model-specific scoring options belong to each reward model's config, for example
``--reward_model.num_subsampled_frames``. ``--batch_size`` sets how many predictions run together.
Rerun the same command to resume an interrupted run. To discard staged work and replace
existing scores, add ``--overwrite=true --resume=false``. Add ``--push_to_hub=true`` to
upload the completed scores to the dataset repository; scores with the same name already on the
Hub are only replaced with ``--overwrite=true``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import HfApi

from lerobot.configs import RewardModelConfig, parser
from lerobot.rewards import make_frame_scorer, make_reward_model, score_dataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import _av_available, _datasets_available, require_package
from lerobot.utils.utils import init_logging

if TYPE_CHECKING or (_datasets_available and _av_available):
    from lerobot.datasets import SCORES_DIR, LeRobotDataset, push_scores_to_hub


@dataclass
class ScoreConfig:
    """Configuration for ``lerobot-score``."""

    dataset_repo_id: str
    # Loaded with --reward_model.path=..., or built with --reward_model.type=... for zero-shot models.
    reward_model: RewardModelConfig | None = None
    # Local dataset directory; scores are written under `<dataset_root>/scores/`. If not specified,
    # defaults to $HF_LEROBOT_HOME/dataset_repo_id. Missing dataset files are downloaded into it.
    dataset_root: Path | None = None
    dataset_revision: str | None = None
    name: str | None = None
    episodes: list[int] | None = None
    # Number of predictions computed together; lower it if the GPU runs out of memory.
    batch_size: int = 32
    resume: bool = True
    overwrite: bool = False
    push_to_hub: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        reward_model_path = parser.get_path_arg("reward_model")
        if reward_model_path:
            yaml_overrides = parser.get_yaml_overrides("reward_model")
            cli_overrides = parser.get_cli_overrides("reward_model") or []
            self.reward_model = RewardModelConfig.from_pretrained(
                reward_model_path,
                revision=parser.get_pretrained_revision("reward_model"),
                cli_overrides=yaml_overrides + cli_overrides,
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Keys for draccus pretrained-path loading."""
        return ["reward_model"]


def run_score(cfg: ScoreConfig) -> None:
    """Load the model and dataset, then score the dataset and save the scores to a Parquet file."""
    require_package("datasets", extra="dataset")
    require_package("av", extra="dataset")
    if cfg.reward_model is None:
        raise ValueError(
            "lerobot-score needs a reward model: pass --reward_model.path=... or --reward_model.type=..."
        )
    # Scores are written into the dataset folder, so it must be a writable folder rather than the
    # read-only Hub snapshot cache that `LeRobotDataset` uses when no root is given.
    root = Path(cfg.dataset_root) if cfg.dataset_root is not None else HF_LEROBOT_HOME / cfg.dataset_repo_id
    score_name = cfg.reward_model.type if cfg.name is None else cfg.name
    # Fail before loading the model and downloading the dataset.
    output_path = root / SCORES_DIR / f"{score_name}.parquet"
    if output_path.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"Scores {score_name!r} already exist at {output_path}; pass --overwrite=true or choose another --name"
        )
    if (
        cfg.push_to_hub
        and not cfg.overwrite
        and HfApi().file_exists(cfg.dataset_repo_id, f"{SCORES_DIR}/{output_path.name}", repo_type="dataset")
    ):
        raise FileExistsError(
            f"Scores {score_name!r} already exist in hf://datasets/{cfg.dataset_repo_id}; "
            "pass --overwrite=true or choose another --name"
        )

    model = make_reward_model(cfg.reward_model)
    scorer = make_frame_scorer(model, batch_size=cfg.batch_size)

    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=root,
        revision=cfg.dataset_revision,
        episodes=cfg.episodes,
        download_videos=True,
    )
    score_dataset(
        dataset,
        scorer,
        name=score_name,
        episodes=cfg.episodes,
        resume=cfg.resume,
        overwrite=cfg.overwrite,
    )

    if cfg.push_to_hub:
        push_scores_to_hub(dataset, score_name, overwrite=cfg.overwrite)


@parser.wrap()
def score_cli(cfg: ScoreConfig) -> None:
    init_logging()
    run_score(cfg)


def main() -> None:
    score_cli()  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
