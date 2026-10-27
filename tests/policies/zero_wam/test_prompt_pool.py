# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest
import torch

from lerobot.policies.zero_wam.prompt_pool import ICLPromptPools


def member(episode, start=0, end=100, name=None):
    return {
        "episode_index": episode,
        "start_frame": start,
        "end_frame": end,
        "latent_path": name or f"episode-{episode}.pth",
    }


def write_pools(tmp_path, rows):
    path = tmp_path / "pools.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def pool(members, split="train", pool_id="task-a", dataset="abc"):
    return {"pool_id": pool_id, "dataset": dataset, "split": split, "members": members}


def test_samples_other_episodes_and_all_alternatives(tmp_path):
    path = write_pools(tmp_path, [pool([member(0), member(0, 100, 200, "self-2.pth"), member(2), member(5)])])
    sampler = ICLPromptPools(path)
    torch.manual_seed(12)
    for target, expected in [
        (0, {"episode-2.pth", "episode-5.pth"}),
        (2, {"episode-0.pth", "self-2.pth", "episode-5.pth"}),
        (5, {"episode-0.pth", "self-2.pth", "episode-2.pth"}),
    ]:
        assert {sampler.sample(target, 10, 90) for _ in range(128)} == expected


def test_checks_full_target_window_and_filters_split(tmp_path):
    rows = [
        pool([member(0, 10, 31), member(1)], pool_id="subtask"),
        pool([member(20), member(21)], split="validation"),
    ]
    sampler = ICLPromptPools(write_pools(tmp_path, rows))
    assert sampler.sample(0, 10, 30) == "episode-1.pth"
    assert sampler.sample(0, 11, 31) is None
    assert sampler.sample(0, 9, 20) is None
    assert sampler.sample(20, 10, 30) is None
    with pytest.raises(ValueError, match="Invalid target"):
        sampler.sample(0, -1, 10)


def test_rejects_cross_split_episode_reuse(tmp_path):
    rows = [pool([member(0), member(1)]), pool([member(1), member(2)], split="validation")]
    with pytest.raises(ValueError, match="more than one prompt split"):
        ICLPromptPools(write_pools(tmp_path, rows))


@pytest.mark.parametrize(
    "members",
    [[member(0), member(0)], [member(0, 10, 10), member(1)], [{**member(0), "latent_path": ""}, member(1)]],
)
def test_rejects_malformed_members(tmp_path, members):
    with pytest.raises(ValueError):
        ICLPromptPools(write_pools(tmp_path, [pool(members)]))


def test_rejects_mixed_dataset_or_missing_split(tmp_path):
    with pytest.raises(ValueError, match="one dataset"):
        ICLPromptPools(
            write_pools(
                tmp_path, [pool([member(0), member(1)]), pool([member(2), member(3)], dataset="other")]
            )
        )
    with pytest.raises(ValueError, match="requires a split"):
        ICLPromptPools(write_pools(tmp_path, [{"pool_id": "a", "dataset": "abc", "members": []}]))
    with pytest.raises(ValueError, match="No prompt pools selected"):
        ICLPromptPools(write_pools(tmp_path, [pool([member(0), member(1)])]), split="typo")


def test_index_stores_shared_pools_without_pair_expansion(tmp_path):
    n = 1000
    sampler = ICLPromptPools(write_pools(tmp_path, [pool([member(ep) for ep in range(n)])]))
    assert sampler.num_members == n and sampler.num_pools == 1
    assert len(sampler._by_episode) == n
    assert sum(len(pools) for pools in sampler._by_episode.values()) == n
    assert sampler._by_episode[0][0] is sampler._by_episode[n - 1][0]
