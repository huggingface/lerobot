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

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402

from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.rl.buffer import ReplayBuffer  # noqa: E402
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, TRUNCATED  # noqa: E402

pytestmark = pytest.mark.dataset


def test_replay_buffer_dataset_preserves_truncation(tmp_path):
    buffer = ReplayBuffer(4, device="cpu", state_keys=[OBS_STATE], use_drq=False)
    flags = [(False, False), (False, True), (True, False), (True, True)]
    for i, (done, truncated) in enumerate(flags):
        state = {OBS_STATE: torch.tensor([float(i), float(i + 1)])}
        buffer.add(state, torch.zeros(2), float(i), state, done, truncated)

    dataset = buffer.to_lerobot_dataset("test/replay", root=tmp_path / "dataset")

    assert TRUNCATED in dataset.features
    assert dataset.num_episodes == 3
    assert [bool(dataset[i][TRUNCATED]) for i in range(len(dataset))] == [truncated for _, truncated in flags]
    restored = ReplayBuffer.from_lerobot_dataset(dataset, device="cpu", state_keys=[OBS_STATE], use_drq=False)
    torch.testing.assert_close(restored.dones, buffer.dones)
    torch.testing.assert_close(restored.truncateds, buffer.truncateds)


@pytest.mark.parametrize("include_done", [False, True])
@pytest.mark.parametrize("include_truncated", [False, True])
def test_import_dataset_episode_flags(tmp_path, include_done, include_truncated):
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (2,)},
        ACTION: {"dtype": "float32", "shape": (2,)},
        REWARD: {"dtype": "float32", "shape": (1,)},
    }
    if include_done:
        features[DONE] = {"dtype": "bool", "shape": (1,)}
    if include_truncated:
        features[TRUNCATED] = {"dtype": "bool", "shape": (1,)}
    dataset = LeRobotDataset.create(
        "test/replay", fps=1, root=tmp_path / "dataset", features=features, use_videos=False
    )
    # A time-limited episode followed by a naturally terminated episode.
    for done, truncated in [(False, True), (True, False)]:
        frame = {OBS_STATE: torch.zeros(2), ACTION: torch.zeros(2), REWARD: torch.zeros(1), "task": "test"}
        if include_done:
            frame[DONE] = torch.tensor([done])
        if include_truncated:
            frame[TRUNCATED] = torch.tensor([truncated])
        dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()

    restored = ReplayBuffer.from_lerobot_dataset(dataset, device="cpu", state_keys=[OBS_STATE], use_drq=False)

    expected_done = [False, True] if include_done or include_truncated else [True, True]
    expected_truncated = [True, False] if include_truncated else [False, False]
    assert restored.dones.tolist() == expected_done
    assert restored.truncateds.tolist() == expected_truncated
