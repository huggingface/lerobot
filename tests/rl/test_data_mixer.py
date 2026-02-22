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
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for RL data mixing (DataMixer, OnlineOfflineMixer)."""

import torch

from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.data_sources import OnlineOfflineMixer
from lerobot.utils.constants import OBS_STATE


def _make_buffer(capacity: int = 100, state_dim: int = 4) -> ReplayBuffer:
    buf = ReplayBuffer(
        capacity=capacity,
        device="cpu",
        state_keys=[OBS_STATE],
        storage_device="cpu",
        use_drq=False,
    )
    for i in range(capacity):
        buf.add(
            state={OBS_STATE: torch.randn(state_dim)},
            action=torch.randn(2),
            reward=1.0,
            next_state={OBS_STATE: torch.randn(state_dim)},
            done=bool(i % 10 == 9),
            truncated=False,
        )
    return buf


def test_online_only_mixer_sample():
    """OnlineOfflineMixer with no offline buffer returns online-only batches."""
    buf = _make_buffer(capacity=50)
    mixer = OnlineOfflineMixer(online_buffer=buf, offline_buffer=None, online_ratio=0.5)
    batch = mixer.sample(batch_size=8)
    assert batch["state"][OBS_STATE].shape[0] == 8
    assert batch["action"].shape[0] == 8
    assert batch["reward"].shape[0] == 8


def test_online_only_mixer_ratio_one():
    """OnlineOfflineMixer with online_ratio=1.0 and no offline is equivalent to online-only."""
    buf = _make_buffer(capacity=50)
    mixer = OnlineOfflineMixer(online_buffer=buf, offline_buffer=None, online_ratio=1.0)
    batch = mixer.sample(batch_size=10)
    assert batch["state"][OBS_STATE].shape[0] == 10


def test_online_offline_mixer_sample():
    """OnlineOfflineMixer with two buffers returns concatenated batches."""
    online = _make_buffer(capacity=50)
    offline = _make_buffer(capacity=50)
    mixer = OnlineOfflineMixer(
        online_buffer=online,
        offline_buffer=offline,
        online_ratio=0.5,
    )
    batch = mixer.sample(batch_size=10)
    assert batch["state"][OBS_STATE].shape[0] == 10
    assert batch["action"].shape[0] == 10
    # 5 from online, 5 from offline (approx)
    assert batch["reward"].shape[0] == 10


def test_online_offline_mixer_iterator():
    """get_iterator yields batches of the requested size."""
    buf = _make_buffer(capacity=50)
    mixer = OnlineOfflineMixer(online_buffer=buf, offline_buffer=None)
    it = mixer.get_iterator(batch_size=4, async_prefetch=False)
    batch1 = next(it)
    batch2 = next(it)
    assert batch1["state"][OBS_STATE].shape[0] == 4
    assert batch2["state"][OBS_STATE].shape[0] == 4
