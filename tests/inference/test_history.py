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

import numpy as np
import pytest
import torch

from lerobot.inference import ChunkHistory, ExecutionRecord, FeatureHistory

STATE = "observation.state"
FPS = 10


def run_episode(record: ExecutionRecord, frames: int, times: list[float] | None = None) -> None:
    """Tick ``frames`` times; tick ``t`` observes ``t`` and dispatches command ``100 + t``."""
    for t in range(frames):
        record.observe(t / FPS if times is None else times[t], {STATE: np.array([t], dtype=np.float32)})
        record.dispatched(np.array([100 + t], dtype=np.float32))


def padded_command() -> np.ndarray:
    return np.array([-1], dtype=np.float32)


@pytest.mark.dataset
# Request and sampled ticks both jitter, so frames match exactly below a quarter interval each.
@pytest.mark.parametrize("jitter", [0.0, 0.2])
def test_live_history_matches_the_dataset_window(tmp_path, empty_lerobot_dataset_factory, jitter):
    """At every frame, live sampling returns the frames and padding a training batch holds."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    offsets, command_offsets = (-6, -3, 0), (-4, -2, -1)
    features = {
        STATE: {"dtype": "float32", "shape": (1,), "names": ["t"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["t"]},
    }
    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "ds", features=features, use_videos=False, fps=FPS
    )
    frames = 12
    for t in range(frames):
        dataset.add_frame(
            {STATE: torch.tensor([t], dtype=torch.float32), "action": torch.tensor([100.0 + t]), "task": "x"}
        )
    dataset.save_episode()
    dataset.finalize()
    loaded = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        delta_timestamps={STATE: [o / FPS for o in offsets], "action": [o / FPS for o in command_offsets]},
    )
    record = ExecutionRecord(
        ChunkHistory(observations=(FeatureHistory(STATE, offsets),), actions=command_offsets),
        interval=1 / FPS,
        action_padding_value=padded_command,
    )
    rng = np.random.default_rng(0)
    times = [(t + rng.uniform(-jitter, jitter)) / FPS for t in range(frames)]
    for t in range(frames):
        record.observe(times[t], {STATE: np.array([t], dtype=np.float32)})
        history = record.sample(times[t])
        expected = loaded[t]
        past = [value.item() for value in history.observations[STATE]]
        assert past == expected[STATE].reshape(len(offsets), -1)[:-1, 0].tolist()
        assert list(history.observation_padding[STATE]) == expected[f"{STATE}_is_pad"][:-1].tolist()
        assert list(history.action_padding) == expected["action_is_pad"].tolist()
        real = [
            value.item()
            for value, pad in zip(history.actions, history.action_padding, strict=True)
            if not pad
        ]
        dataset_commands = expected["action"].reshape(len(command_offsets), -1)[:, 0].tolist()
        pads = expected["action_is_pad"].tolist()
        assert real == [value for value, pad in zip(dataset_commands, pads, strict=True) if not pad]
        assert history.max_skew_s <= 2 * jitter / FPS + 1e-9
        record.dispatched(np.array([100 + t], dtype=np.float32))


def test_a_gap_uses_the_nearest_frame_and_reports_its_distance():
    record = ExecutionRecord(
        ChunkHistory(observations=(FeatureHistory(STATE, (-10, -5, 0)),)),
        interval=1 / FPS,
    )
    times = [t / FPS for t in range(20) if not 7 <= t <= 13]  # a 0.7 s stall
    for time in times:
        record.observe(time, {STATE: np.array([round(time * FPS)], dtype=np.float32)})
    history = record.sample(times[-1])
    assert [v.item() for v in history.observations[STATE]] == [6, 14]
    assert history.max_skew_s == pytest.approx(0.3)
    assert history.observation_padding[STATE] == (False, False)


def test_commands_are_the_ones_in_effect_and_exclude_the_current_tick():
    record = ExecutionRecord(
        ChunkHistory(actions=(-3, -2, -1)),
        interval=1 / FPS,
        action_padding_value=padded_command,
    )
    for t in range(5):
        record.observe(t / FPS, {})
        if t != 3:  # tick 3 held: the command in effect stays tick 2's
            record.dispatched(np.array([100 + t], dtype=np.float32))
    record.observe(5 / FPS, {})
    record.dispatched(np.array([105], dtype=np.float32))
    history = record.sample(5 / FPS)
    assert [v.item() for v in history.actions] == [102, 102, 104]
    assert history.action_padding == (False, False, False)
    record.clear()
    record.observe(0.0, {})
    history = record.sample(0.0)
    assert history.action_padding == (True, True, True)
    assert [v.item() for v in history.actions] == [-1, -1, -1]
