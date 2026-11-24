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
"""``observation.action``: the commands executed at past frames, read from the action column."""

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.utils.constants import ACTION, OBS_ACTION
from tests.fixtures.constants import DEFAULT_FPS

pytestmark = pytest.mark.dataset


def _offsets(*frames):
    return [i / DEFAULT_FPS for i in frames]


@pytest.mark.parametrize("action_window", [None, (0, 1, 2)])
def test_observation_action_matches_the_action_column(tmp_path, lerobot_dataset_factory, action_window):
    delta_timestamps = {OBS_ACTION: _offsets(-2, -1)}
    if action_window is not None:
        delta_timestamps[ACTION] = _offsets(*action_window)
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=2,
        total_frames=20,
        use_videos=False,
        delta_timestamps=delta_timestamps,
    )
    actions = torch.stack([dataset.reader.hf_dataset[i][ACTION] for i in range(dataset.num_frames)])
    episodes = torch.tensor([int(dataset.reader.hf_dataset[i]["episode_index"]) for i in range(len(actions))])

    for index in (0, 1, 5, len(actions) - 1):
        item = dataset[index]
        episode = episodes == episodes[index]
        first, last = int(episode.nonzero()[0]), int(episode.nonzero()[-1])
        past = [min(max(index + offset, first), last) for offset in (-2, -1)]
        assert torch.equal(item[OBS_ACTION], actions[past])
        assert item[f"{OBS_ACTION}_is_pad"].tolist() == [index + offset < first for offset in (-2, -1)]
        if action_window is None:
            # The action keeps its current value, without a time axis.
            assert torch.equal(item[ACTION], actions[index])
            assert f"{ACTION}_is_pad" not in item
        else:
            future = [min(index + offset, last) for offset in action_window]
            assert torch.equal(item[ACTION], actions[future])
            assert item[f"{ACTION}_is_pad"].tolist() == [index + offset > last for offset in action_window]
