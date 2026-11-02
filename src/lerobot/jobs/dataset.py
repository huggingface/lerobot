# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Make a training dataset reachable from an HF Job pod.

The pod can't see the host's ~/.cache/huggingface/lerobot, so the dataset has to
live on the Hub: the pod downloads it by repo_id at train time (the forwarded
HF_TOKEN covers private datasets). A dataset already on the Hub is used as-is; a
local-only dataset is pushed to a PRIVATE repo first (never public).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub.errors import (
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.utils import INFO_PATH, resolve_episode_indices
from lerobot.utils.constants import HF_LEROBOT_HOME

if TYPE_CHECKING:
    from huggingface_hub import HfApi

    from lerobot.configs.default import DatasetConfig


def ensure_dataset_available(repo_id: str, *, api: HfApi, tags: list[str] | None = None) -> None:
    """Ensure repo_id resolves on the Hub, pushing a local-only dataset privately first.

    `tags` are attached to the dataset only when we push it (an already-on-Hub
    dataset is left untouched). Raises RuntimeError if the dataset is neither on
    the Hub nor in the local cache.
    """
    if api.repo_exists(repo_id, repo_type="dataset"):
        return

    local_present = (HF_LEROBOT_HOME / repo_id / "meta" / "info.json").is_file()
    if not local_present:
        raise RuntimeError(
            f"Dataset '{repo_id}' is not in the local cache ({HF_LEROBOT_HOME}) and could not be "
            f"reached on the Hub — it may not exist, or be private and inaccessible with your "
            f"token. Record or download it first, or run `hf auth login`."
        )

    print(f"[dataset] '{repo_id}' is local-only; pushing to a PRIVATE Hub repo...")
    LeRobotDataset(repo_id).push_to_hub(private=True, tags=tags)
    print(f"[dataset] '{repo_id}' uploaded (private). The job will download it by repo_id.")


def count_dataset_episodes(dataset: DatasetConfig, *, api: HfApi) -> int | None:
    """Count the episodes a training run selects from `dataset`, from its `meta/info.json` alone.

    `--dataset.episodes` / `--dataset.exclude_episodes` are applied; `--dataset.eval_split` hold-outs
    are not (they need the per-episode task table), so this is an upper bound on the training
    episodes. Reads the Hub copy (at `dataset.revision`, else the default branch), or the local
    cache for a dataset that isn't on the Hub yet.

    Returns:
        int | None: The episode count, or None when it can't be read without loading more than
            info.json (a bucket, a missing revision or info.json, or a dataset found neither on
            the Hub nor locally). Network and auth errors propagate.
    """
    if dataset.repo_type != "dataset":
        return None
    try:
        info_file = Path(
            api.hf_hub_download(dataset.repo_id, INFO_PATH, repo_type="dataset", revision=dataset.revision)
        )
    except RepositoryNotFoundError:
        info_file = HF_LEROBOT_HOME / dataset.repo_id / INFO_PATH
        if not info_file.is_file():
            return None  # `ensure_dataset_available` reports it
    except (RevisionNotFoundError, RemoteEntryNotFoundError):
        return None  # the trainer reports a bad revision or layout
    total = int(json.loads(info_file.read_text())["total_episodes"])
    selected = resolve_episode_indices(dataset.episodes, total, dataset.exclude_episodes)
    return total if selected is None else len(selected)
