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
import re

import draccus
import pytest

from lerobot.configs.dataset import DatasetRecordConfig

TIMESTAMP_SUFFIX = re.compile(r"_\d{8}_\d{6}$")


def test_stamp_repo_id_appends_timestamp():
    cfg = DatasetRecordConfig(repo_id="user/rollout_my_task_r1")
    cfg.stamp_repo_id()
    assert cfg.repo_id.startswith("user/rollout_my_task_r1_")
    assert TIMESTAMP_SUFFIX.search(cfg.repo_id)


def test_stamp_repo_id_no_stamp_keeps_repo_id():
    cfg = DatasetRecordConfig(repo_id="user/rollout_my_task_r1", no_stamp=True)
    cfg.stamp_repo_id()
    assert cfg.repo_id == "user/rollout_my_task_r1"


@pytest.mark.parametrize("no_stamp", [False, True])
def test_stamp_repo_id_empty_repo_id_unchanged(no_stamp):
    cfg = DatasetRecordConfig(no_stamp=no_stamp)
    cfg.stamp_repo_id()
    assert cfg.repo_id == ""


def test_no_stamp_parses_from_cli():
    cfg = draccus.parse(DatasetRecordConfig, args=["--repo_id=user/rollout_my_task_r1", "--no_stamp=true"])
    assert cfg.no_stamp is True
    cfg.stamp_repo_id()
    assert cfg.repo_id == "user/rollout_my_task_r1"
