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
import numpy as np
import pytest
import torch

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tests.fixtures.constants import DUMMY_REPO_ID


@pytest.fixture
def empty_state_dataset(tmp_path):
    features = {
        "state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["shoulder_pan", "shoulder_lift"],
        },
    }
    dataset = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=30, features=features, root=tmp_path / "dummy")
    return dataset


@pytest.fixture
def empty_image_dataset(tmp_path):
    features = {
        "image": {
            "dtype": "video",
            "shape": (3, 96, 128),
            "names": [
                "channels",
                "height",
                "width",
            ],
        },
    }
    dataset = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=30, features=features, root=tmp_path / "dummy")
    return dataset


# TODO: simplify these tests by using `compute_episode_stats` instead of `save_episode`
def test_compute_episode_stats_one_state(empty_state_dataset):
    ds = empty_state_dataset
    ds.add_frame({"state": torch.tensor([1, 2])})
    ds.save_episode(task="lol")
    # stats = compute_episode_stats(ds.episode_buffer, ds.features)
    # TODO: assert state min, max, mean, std


def test_compute_episode_stats_two_states(empty_state_dataset):
    ds = empty_state_dataset
    ds.add_frame({"state": torch.tensor([1, 2])})
    ds.add_frame({"state": torch.tensor([4, 5])})
    ds.save_episode(task="lol")
    # stats = compute_episode_stats(ds.episode_buffer, ds.features)
    # TODO: assert state min, max, mean, std


def test_compute_episode_stats_one_image(empty_image_dataset):
    ds = empty_image_dataset
    ds.add_frame({"image": torch.ones(3, 96, 128)})
    ds.save_episode(task="lol")
    # TODO: assert state min, max, mean, std


def test_compute_episode_stats_two_images(empty_image_dataset):
    ds = empty_image_dataset
    ds.add_frame({"image": torch.ones(3, 96, 128)})
    ds.add_frame({"image": torch.ones(3, 96, 128) * 0.5})
    ds.save_episode(task="lol")
    # TODO: assert state min, max, mean, std


def test_aggregate_stats():
    all_stats = [
        {
            "observation.image": {
                "min": [1, 2, 3],
                "max": [10, 20, 30],
                "mean": [5.5, 10.5, 15.5],
                "std": [2.87, 5.87, 8.87],
                "count": 10,
            },
            "observation.state": {"min": 1, "max": 10, "mean": 5.5, "std": 2.87, "count": 10},
            "extra_key_0": {"min": 5, "max": 25, "mean": 15, "std": 6, "count": 6},
        },
        {
            "observation.image": {
                "min": [2, 1, 0],
                "max": [15, 10, 5],
                "mean": [8.5, 5.5, 2.5],
                "std": [3.42, 2.42, 1.42],
                "count": 15,
            },
            "observation.state": {"min": 2, "max": 15, "mean": 8.5, "std": 3.42, "count": 15},
            "extra_key_1": {"min": 0, "max": 20, "mean": 10, "std": 5, "count": 5},
        },
    ]

    expected_agg_stats = {
        "observation.image": {
            "min": [1, 1, 0],
            "max": [15, 20, 30],
            "mean": [7.3, 7.5, 7.7],
            "std": [3.5317, 4.8267, 8.5581],
            "count": 25,
        },
        "observation.state": {
            "min": 1,
            "max": 15,
            "mean": 7.3,
            "std": 3.5317,
            "count": 25,
        },
        "extra_key_0": {
            "min": 5,
            "max": 25,
            "mean": 15.0,
            "std": 6.0,
            "count": 6,
        },
        "extra_key_1": {
            "min": 0,
            "max": 20,
            "mean": 10.0,
            "std": 5.0,
            "count": 5,
        },
    }

    # cast to numpy
    for ep_stats in all_stats:
        for fkey, stats in ep_stats.items():
            for k in stats:
                stats[k] = np.array(stats[k], dtype=np.int64 if k == "count" else np.float32)
                if fkey == "observation.image" and k != "count":
                    stats[k] = stats[k].reshape(3, 1, 1)  # for normalization on image channels
                else:
                    stats[k] = stats[k].reshape(1)

    # cast to numpy
    for fkey, stats in expected_agg_stats.items():
        for k in stats:
            stats[k] = np.array(stats[k], dtype=np.int64 if k == "count" else np.float32)
            if fkey == "observation.image" and k != "count":
                stats[k] = stats[k].reshape(3, 1, 1)  # for normalization on image channels
            else:
                stats[k] = stats[k].reshape(1)

    results = aggregate_stats(all_stats)

    for fkey in expected_agg_stats:
        np.testing.assert_allclose(results[fkey]["min"], expected_agg_stats[fkey]["min"])
        np.testing.assert_allclose(results[fkey]["max"], expected_agg_stats[fkey]["max"])
        np.testing.assert_allclose(results[fkey]["mean"], expected_agg_stats[fkey]["mean"])
        np.testing.assert_allclose(
            results[fkey]["std"], expected_agg_stats[fkey]["std"], atol=1e-04, rtol=1e-04
        )
        np.testing.assert_allclose(results[fkey]["count"], expected_agg_stats[fkey]["count"])
