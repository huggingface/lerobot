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

from types import SimpleNamespace

import torch

from lerobot.datasets import factory
from lerobot.utils.constants import IMAGENET_STATS


class _FakeTrainableConfig:
    reward_delta_indices = None
    action_delta_indices = None
    observation_delta_indices = None


class _FakeMeta:
    fps = 30
    features = {"observation.images.front": {"dtype": "video"}}
    camera_keys = ["observation.images.front"]

    def __init__(self):
        self.stats = {}


class _FakeDataset:
    def __init__(self):
        self.meta = _FakeMeta()


def test_make_dataset_adds_missing_imagenet_stats_entries(monkeypatch):
    fake_dataset = _FakeDataset()

    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", lambda *args, **kwargs: _FakeMeta())
    monkeypatch.setattr(factory, "LeRobotDataset", lambda *args, **kwargs: fake_dataset)

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            repo_id="local/dataset",
            root="/tmp/local-dataset",
            revision=None,
            episodes=None,
            image_transforms=SimpleNamespace(enable=False),
            streaming=False,
            video_backend="pyav",
            use_imagenet_stats=True,
        ),
        trainable_config=_FakeTrainableConfig(),
        tolerance_s=1e-4,
        num_workers=0,
    )

    dataset = factory.make_dataset(cfg)

    assert dataset is fake_dataset
    assert "observation.images.front" in dataset.meta.stats
    for stats_type, stats in IMAGENET_STATS.items():
        torch.testing.assert_close(
            dataset.meta.stats["observation.images.front"][stats_type],
            torch.tensor(stats, dtype=torch.float32),
        )
