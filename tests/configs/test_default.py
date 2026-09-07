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

from lerobot.configs.default import DatasetConfig


def test_dataset_config_valid():
    DatasetConfig(repo_id="user/repo", episodes=[0, 1, 2])


def test_dataset_config_negative_episodes():
    with pytest.raises(ValueError, match="non-negative"):
        DatasetConfig(repo_id="user/repo", episodes=[0, -1, 2])


def test_dataset_config_duplicate_episodes():
    with pytest.raises(ValueError, match="duplicates"):
        DatasetConfig(repo_id="user/repo", episodes=[0, 1, 1, 2])


def test_dataset_config_none_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=None)


def test_dataset_config_empty_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=[])


def test_dataset_config_ignores_negative_excluded_episodes(caplog):
    config = DatasetConfig(repo_id="user/repo", exclude_episodes=[-2, 1, -1, 3])

    assert config.exclude_episodes == [1, 3]
    assert "Ignoring negative exclude_episodes entries: [-2, -1]" in caplog.text


def test_dataset_config_bucket_ok():
    # Both allowed at config level. The dataset factory raises for storage
    # formats that only support streaming access on buckets.
    DatasetConfig(repo_id="user/repo", repo_type="bucket", streaming=True)
    DatasetConfig(repo_id="user/repo", repo_type="bucket")


def test_dataset_config_invalid_repo_type():
    with pytest.raises(ValueError, match="repo_type"):
        DatasetConfig(repo_id="user/repo", repo_type="model")


def test_dataset_config_eval_split():
    # map-style access on a bucket is fine; streaming access is not, anywhere
    DatasetConfig(repo_id="user/repo", repo_type="bucket", eval_split=0.1)
    with pytest.raises(ValueError, match="streaming"):
        DatasetConfig(repo_id="user/repo", streaming=True, eval_split=0.1)
