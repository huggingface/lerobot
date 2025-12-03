#!/usr/bin/env python

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

"""
Visual Feature Consistency Tests

This module tests the `validate_visual_features_consistency` function,
which ensures that visual features (camera observations) in a dataset/env
match the expectations defined in a policy configuration.

The purpose of this check is to prevent mismatches between what a policy expects
(e.g., `observation.images.camera1`, `camera2`, `camera3`) and what a dataset or
environment actually provides (e.g., `observation.images.top`, `side`, or fewer cameras).
"""

from pathlib import Path

import numpy as np
import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy_config
from lerobot.scripts.lerobot_train import train
from lerobot.utils.utils import auto_select_torch_device

pytest.importorskip("transformers")

DUMMY_REPO_ID = "dummy/repo"


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


DUMMY_STATE_DIM = 6
DUMMY_ACTION_DIM = 6
IMAGE_SIZE = 8
DEVICE = auto_select_torch_device()


def make_dummy_dataset(camera_keys, tmp_path):
    """Creates a minimal dummy dataset for testing rename_mapping logic."""
    features = {
        "action": {"dtype": "float32", "shape": (DUMMY_ACTION_DIM,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (DUMMY_STATE_DIM,), "names": None},
    }
    for cam in camera_keys:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "names": ["height", "width", "channel"],
        }
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=30,
        features=features,
        root=tmp_path / "_dataset",
    )
    root = tmp_path / "_dataset"
    for ep_idx in range(2):
        for _ in range(3):
            frame = {
                "action": np.random.randn(DUMMY_ACTION_DIM).astype(np.float32),
                "observation.state": np.random.randn(DUMMY_STATE_DIM).astype(np.float32),
            }
            for cam in camera_keys:
                frame[f"observation.images.{cam}"] = np.random.randint(
                    0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
                )
            frame["task"] = f"task_{ep_idx}"
            dataset.add_frame(frame)
        dataset.save_episode()

    dataset.finalize()
    return dataset, root


def custom_validate(train_config: TrainPipelineConfig, policy_path: str, empty_cameras: int):
    train_config.policy = PreTrainedConfig.from_pretrained(policy_path)
    train_config.policy.pretrained_path = Path(policy_path)
    # override empty_cameras and push_to_hub for testing
    train_config.policy.empty_cameras = empty_cameras
    train_config.policy.push_to_hub = False
    if train_config.use_policy_training_preset:
        train_config.optimizer = train_config.policy.get_optimizer_preset()
        train_config.scheduler = train_config.policy.get_scheduler_preset()
    return train_config


@pytest.mark.skip(reason="Skipping this test as it results OOM")
@pytest.mark.parametrize(
    "camera_keys, empty_cameras, rename_map, expect_success",
    [
        # case 1: dataset has fewer cameras than policy (3 instead of 4), but we specify empty_cameras=1 for smolvla, pi0, pi05
        (["camera1", "camera2", "camera3"], 1, {}, True),
        # case 2: dataset has 2 cameras with different names, rename_mapping provided
        (
            ["top", "side"],
            0,
            {
                "observation.images.top": "observation.images.camera1",
                "observation.images.side": "observation.images.camera2",
            },
            True,
        ),
        # case 3: dataset has 2 cameras, policy expects 3, names do not match, no empty_cameras
        (["top", "side"], 0, {}, False),
        # TODO: case 4: dataset has 2 cameras, policy expects 3, no rename_map, no empty_cameras, should raise for smolvla
        # (["camera1", "camera2"], 0, {}, False),
    ],
)
def test_train_with_camera_mismatch(camera_keys, empty_cameras, rename_map, expect_success, tmp_path):
    """Tests that training works or fails depending on camera/feature alignment."""

    _dataset, root = make_dummy_dataset(camera_keys, tmp_path)
    pretrained_path = "lerobot/smolvla_base"
    dataset_config = DatasetConfig(repo_id=DUMMY_REPO_ID, root=root)
    policy_config = make_policy_config(
        "smolvla",
        optimizer_lr=0.01,
        push_to_hub=False,
        pretrained_path=pretrained_path,
        device=DEVICE,
    )
    policy_config.empty_cameras = empty_cameras
    train_config = TrainPipelineConfig(
        dataset=dataset_config,
        policy=policy_config,
        rename_map=rename_map,
        output_dir=tmp_path / "_output",
        steps=1,
    )
    train_config = custom_validate(train_config, policy_path=pretrained_path, empty_cameras=empty_cameras)
    # HACK: disable the internal CLI validation step for tests, we did it with custom_validate
    train_config.validate = lambda: None
    if expect_success:
        train(train_config)
    else:
        with pytest.raises(ValueError):
            train(train_config)
