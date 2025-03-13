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
from datasets import Dataset

from lerobot.common.datasets.push_dataset_to_hub.utils import calculate_episode_data_index
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)


def test_drop_n_first_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index, drop_n_first_frames=1)
    assert sampler.indices == [1, 4, 5]
    assert len(sampler) == 3
    assert list(sampler) == [1, 4, 5]


def test_drop_n_last_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index, drop_n_last_frames=1)
    assert sampler.indices == [0, 3, 4]
    assert len(sampler) == 3
    assert list(sampler) == [0, 3, 4]


def test_episode_indices_to_use():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index, episode_indices_to_use=[0, 2])
    assert sampler.indices == [0, 1, 3, 4, 5]
    assert len(sampler) == 5
    assert list(sampler) == [0, 1, 3, 4, 5]


def test_shuffle():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index, shuffle=False)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert list(sampler) == [0, 1, 2, 3, 4, 5]
    sampler = EpisodeAwareSampler(episode_data_index, shuffle=True)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert set(sampler) == {0, 1, 2, 3, 4, 5}
