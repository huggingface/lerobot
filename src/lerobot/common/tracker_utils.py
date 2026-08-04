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
"""Experiment-tracker abstraction over wandb and trackio.

Training scripts talk to a tracker through the small :class:`TrackerLogger` protocol
and obtain one from :func:`make_tracker`, which dispatches on ``cfg.tracker``
(``--tracker.type=wandb`` / ``--tracker.type=trackio``).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from lerobot.configs.default import TrackioTrackerConfig, WandBTrackerConfig
from lerobot.utils.import_utils import require_package

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


@runtime_checkable
class TrackerLogger(Protocol):
    """The tracker surface used by the training scripts."""

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ) -> None: ...

    def log_policy(self, checkpoint_dir: Path) -> None: ...

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None: ...


def cfg_to_group(
    cfg: "TrainPipelineConfig",
    return_list: bool = False,
    truncate_tags: bool = False,
    max_tag_length: int = 64,
) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""

    def _maybe_truncate(tag: str) -> str:
        """Truncate tag to max_tag_length characters if required.

        wandb rejects tags longer than 64 characters.
        See: https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py
        """
        if len(tag) <= max_tag_length:
            return tag
        return tag[:max_tag_length]

    if cfg.is_reward_model_training:
        trainable_tag = f"reward_model:{cfg.reward_model.type}"
    else:
        trainable_tag = f"policy:{cfg.policy.type}"
    lst = [
        trainable_tag,
        f"seed:{cfg.seed}",
    ]
    if cfg.dataset is not None:
        lst.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    if truncate_tags:
        lst = [_maybe_truncate(tag) for tag in lst]
    return lst if return_list else "-".join(lst)


def make_tracker(cfg: "TrainPipelineConfig") -> TrackerLogger | None:
    """Build the experiment tracker selected by ``cfg.tracker`` (None = tracking disabled).

    Only call on the main process of a distributed run.
    """
    if cfg.tracker is None:
        return None

    if isinstance(cfg.tracker, WandBTrackerConfig):
        require_package("wandb", extra="training")
        from lerobot.common.wandb_utils import WandBLogger

        return WandBLogger(cfg)

    if isinstance(cfg.tracker, TrackioTrackerConfig):
        require_package("trackio", extra="trackio")
        from lerobot.common.trackio_utils import TrackioLogger

        return TrackioLogger(cfg)

    raise ValueError(f"Unsupported tracker config: {type(cfg.tracker).__name__}")
