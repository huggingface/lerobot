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

from unittest.mock import patch

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_edit_dataset import AddFeatureConfig, EditDatasetConfig, handle_add_feature


@pytest.fixture
def sample_dataset(tmp_path, empty_lerobot_dataset_factory):
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        "observation.images.top": {"dtype": "image", "shape": (224, 224, 3), "names": None},
    }

    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "test_dataset",
        features=features,
    )

    for ep_idx in range(5):
        for _ in range(10):
            frame = {
                "action": np.random.randn(6).astype(np.float32),
                "observation.state": np.random.randn(4).astype(np.float32),
                "observation.images.top": np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
                "task": f"task_{ep_idx % 2}",
            }
            dataset.add_frame(frame)
        dataset.save_episode()

    dataset.finalize()
    return dataset


def test_handle_add_feature_from_npy(sample_dataset, tmp_path):
    reward_values = np.random.randn(sample_dataset.meta.total_frames, 1).astype(np.float32)
    feature_path = tmp_path / "reward.npy"
    np.save(feature_path, reward_values)

    output_dir = tmp_path / "with_reward"
    cfg = EditDatasetConfig(
        repo_id=sample_dataset.repo_id,
        root=str(sample_dataset.root),
        new_repo_id=f"{sample_dataset.repo_id}_with_reward",
        new_root=str(output_dir),
        operation=AddFeatureConfig(
            feature_name="reward",
            feature_values_path=str(feature_path),
            feature_dtype="float32",
            feature_shape=[1],
        ),
    )

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(output_dir)

        handle_add_feature(cfg)

    new_dataset = LeRobotDataset(cfg.new_repo_id, root=output_dir)
    assert "reward" in new_dataset.meta.features
    assert new_dataset.meta.features["reward"] == {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }
    assert len(new_dataset) == sample_dataset.meta.total_frames
    assert "reward" in new_dataset[0]


def test_handle_add_feature_requires_npy(sample_dataset, tmp_path):
    feature_path = tmp_path / "reward.txt"
    feature_path.write_text("not-a-feature-file", encoding="utf-8")

    cfg = EditDatasetConfig(
        repo_id=sample_dataset.repo_id,
        root=str(sample_dataset.root),
        new_repo_id=f"{sample_dataset.repo_id}_with_reward",
        new_root=str(tmp_path / "with_reward"),
        operation=AddFeatureConfig(
            feature_name="reward",
            feature_values_path=str(feature_path),
            feature_dtype="float32",
            feature_shape=[1],
        ),
    )

    with pytest.raises(ValueError, match="Only \\.npy files are currently supported"):
        handle_add_feature(cfg)
