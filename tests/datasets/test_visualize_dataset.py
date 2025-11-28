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
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import EpisodeSampler, visualize_dataset


def test_episode_sampler_with_filtered_dataset(empty_lerobot_dataset_factory, tmp_path):
    """Test that EpisodeSampler produces valid indices when dataset is filtered by episodes.

    This is a regression test for a bug where the sampler would use global dataset indices
    from metadata even when the dataset was filtered, causing IndexError.
    """
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False)

    frames_per_episode = [50, 100, 75]
    for ep_idx in range(3):
        for _ in range(frames_per_episode[ep_idx]):
            dataset.add_frame({"state": torch.randn(2), "task": f"task_{ep_idx}"})
        dataset.save_episode()

    dataset.finalize()

    filtered_dataset = LeRobotDataset(dataset.repo_id, root=dataset.root, episodes=[1])
    assert filtered_dataset.episodes == [1]
    assert len(filtered_dataset) == frames_per_episode[1]

    sampler = EpisodeSampler(filtered_dataset, episode_index=1)

    assert len(sampler) == frames_per_episode[1]
    sample_indices = list(sampler)
    assert sample_indices == list(range(frames_per_episode[1]))
    assert max(sample_indices) < len(filtered_dataset), "Sampler indices must be within dataset bounds"

    multi_filtered = LeRobotDataset(dataset.repo_id, root=dataset.root, episodes=[0, 2])
    assert multi_filtered.episodes == [0, 2]
    assert len(multi_filtered) == frames_per_episode[0] + frames_per_episode[2]

    sampler_ep0 = EpisodeSampler(multi_filtered, episode_index=0)
    assert len(sampler_ep0) == frames_per_episode[0]
    sample_indices_ep0 = list(sampler_ep0)
    assert sample_indices_ep0 == list(range(0, frames_per_episode[0]))
    assert max(sample_indices_ep0) < len(multi_filtered)

    sampler_ep2 = EpisodeSampler(multi_filtered, episode_index=2)
    assert len(sampler_ep2) == frames_per_episode[2]
    sample_indices_ep2 = list(sampler_ep2)
    expected_start = frames_per_episode[0]
    expected_end = frames_per_episode[0] + frames_per_episode[2]
    assert sample_indices_ep2 == list(range(expected_start, expected_end))
    assert max(sample_indices_ep2) < len(multi_filtered)

    with pytest.raises(ValueError, match="Episode 1 not in filtered dataset"):
        EpisodeSampler(multi_filtered, episode_index=1)

    full_dataset = LeRobotDataset(dataset.repo_id, root=dataset.root)
    assert full_dataset.episodes is None
    assert len(full_dataset) == sum(frames_per_episode)

    sampler_full = EpisodeSampler(full_dataset, episode_index=1)

    ep1_from = full_dataset.meta.episodes["dataset_from_index"][1]
    ep1_to = full_dataset.meta.episodes["dataset_to_index"][1]
    assert len(sampler_full) == frames_per_episode[1]
    sample_indices_full = list(sampler_full)
    assert sample_indices_full == list(range(ep1_from, ep1_to))
    assert max(sample_indices_full) < len(full_dataset), "Sampler indices must be within dataset bounds"


@pytest.mark.skip("TODO: add dummy videos")
def test_visualize_local_dataset(tmp_path, lerobot_dataset_factory):
    root = tmp_path / "dataset"
    output_dir = tmp_path / "outputs"
    dataset = lerobot_dataset_factory(root=root)
    rrd_path = visualize_dataset(
        dataset,
        episode_index=0,
        batch_size=32,
        save=True,
        output_dir=output_dir,
    )
    assert rrd_path.exists()
