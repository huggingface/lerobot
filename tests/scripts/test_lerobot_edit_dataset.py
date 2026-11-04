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

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import LeRobotDataset
from lerobot.scripts import lerobot_edit_dataset
from lerobot.scripts.lerobot_edit_dataset import (
    EditDatasetConfig,
    RecomputeStatsConfig,
    ReencodeVideosConfig,
    handle_recompute_stats,
    handle_reencode_videos,
)

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize(
    ("handler", "operation"),
    [
        (handle_recompute_stats, RecomputeStatsConfig()),
        (handle_reencode_videos, ReencodeVideosConfig()),
    ],
)
def test_copies_dataset_from_resolved_root(
    tmp_path, monkeypatch, lerobot_dataset_factory, handler, operation
):
    """Without --root, the input is wherever LeRobotDataset resolved it (the Hub snapshot cache for
    downloaded datasets), not $HF_LEROBOT_HOME/repo_id."""
    repo_id = "user/hub_dataset"
    snapshot_root = tmp_path / "hub" / "snapshot"
    lerobot_dataset_factory(root=snapshot_root, repo_id=repo_id, use_videos=False)

    monkeypatch.setattr(lerobot_edit_dataset, "HF_LEROBOT_HOME", tmp_path / "lerobot_home")
    monkeypatch.setattr(
        lerobot_edit_dataset,
        "LeRobotDataset",
        lambda repo_id, root=None: LeRobotDataset(repo_id, root=root or snapshot_root),
    )

    output_root = tmp_path / "output"
    handler(EditDatasetConfig(operation=operation, repo_id=repo_id, new_root=str(output_root)))

    assert (output_root / "meta" / "info.json").is_file()
