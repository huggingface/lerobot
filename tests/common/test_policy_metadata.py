#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team.
# All rights reserved.
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

import numpy as np
import pytest
from huggingface_hub.errors import EntryNotFoundError

from lerobot.common.policy_metadata import (
    POLICY_DATASET_METADATA_NAME,
    load_policy_dataset_metadata,
    save_policy_dataset_metadata,
)
from lerobot.utils.constants import ACTION


def _make_dataset_metadata():
    features = {
        ACTION: {
            "dtype": "float32",
            "shape": (2,),
            "names": ["shoulder", "gripper"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["shoulder", "gripper"],
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        },
    }
    info_dict = {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": features,
        "robot_type": "test_bot",
        "total_episodes": 1,
        "total_frames": 4,
        "total_tasks": 1,
        "splits": {"train": "0:1"},
    }

    class Info:
        def to_dict(self):
            return info_dict

    class DatasetMetadata:
        repo_id = "user/dataset"
        revision = "v3.0"
        info = Info()
        stats = {
            ACTION: {
                "mean": np.array([0.1, 0.2], dtype=np.float32),
                "std": np.array([1.0, 1.5], dtype=np.float32),
            },
            "observation.state": {
                "mean": np.array([0.3, 0.4], dtype=np.float32),
                "std": np.array([2.0, 2.5], dtype=np.float32),
            },
        }

    return DatasetMetadata()


def test_policy_dataset_metadata_roundtrip(tmp_path):
    ds_meta = _make_dataset_metadata()

    save_policy_dataset_metadata(tmp_path / "policy", ds_meta)

    metadata_path = tmp_path / "policy" / POLICY_DATASET_METADATA_NAME
    assert metadata_path.is_file()

    loaded = load_policy_dataset_metadata(tmp_path / "policy")
    assert loaded is not None
    assert loaded.repo_id == ds_meta.repo_id
    assert loaded.revision == ds_meta.revision
    assert loaded.fps == ds_meta.info.to_dict()["fps"]
    assert loaded.robot_type == ds_meta.info.to_dict()["robot_type"]
    assert loaded.features.keys() == ds_meta.info.to_dict()["features"].keys()
    assert loaded.features[ACTION]["shape"] == ds_meta.info.to_dict()["features"][ACTION]["shape"]
    assert loaded.action_names == ds_meta.info.to_dict()["features"][ACTION]["names"]

    for feature_name, feature_stats in ds_meta.stats.items():
        for stat_name, stat in feature_stats.items():
            np.testing.assert_allclose(loaded.stats[feature_name][stat_name], stat)


def test_load_policy_dataset_metadata_missing_returns_none(tmp_path):
    assert load_policy_dataset_metadata(tmp_path) is None


def test_load_policy_dataset_metadata_missing_hub_file_returns_none(monkeypatch):
    def raise_missing_file(*args, **kwargs):
        raise EntryNotFoundError("missing")

    monkeypatch.setattr("lerobot.common.policy_metadata.hf_hub_download", raise_missing_file)

    assert load_policy_dataset_metadata("user/policy") is None


def test_load_policy_dataset_metadata_rejects_malformed_json(tmp_path):
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / POLICY_DATASET_METADATA_NAME).write_text("{not-json")

    with pytest.raises(ValueError, match=POLICY_DATASET_METADATA_NAME):
        load_policy_dataset_metadata(policy_dir)
