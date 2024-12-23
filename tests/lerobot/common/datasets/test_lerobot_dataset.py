#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.exceptions import DatasetExistError
from tests.utils import make_robot


def test_create_dataset(tmp_path):
    repo_id = "test_repo"
    fps = 30
    root = tmp_path / repo_id
    mocked_robot = make_robot("koch", mock=True)

    dataset = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot=mocked_robot,
        use_videos=True,
    )

    assert dataset.repo_id == repo_id
    assert dataset.info["fps"] == fps
    assert dataset.info["robot_type"] == mocked_robot.robot_type


def test_create_dataset_when_meta_cache_already_exists(tmp_path):
    repo_id = "test_repo"
    fps = 30
    root = tmp_path / repo_id

    # To test the DatasetExistException lets create dataset two times
    # The first time the dataset should be created successfully
    # The second time the DatasetExistException should be raised
    LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot=make_robot("koch", mock=True),
        use_videos=True,
    )

    with pytest.raises(DatasetExistError):
        LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot=make_robot("koch", mock=True),
            use_videos=True,
        )
