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

import numpy as np
import pytest

from lerobot.scripts.lerobot_check_calibration import (
    VERDICT_MILD,
    VERDICT_OK,
    VERDICT_SIGNIFICANT,
    check_calibration,
    compute_episode_deltas,
    group_pairs_by_episode,
    summarize_calibration,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def make_episode_with_offset(
    offset: np.ndarray, n_moving: int = 20, n_stable: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (actions, states) pair: a fast ramp followed by a hold, with actions = states + offset."""
    num_motors = len(offset)
    ramp = np.linspace(0.0, 50.0, n_moving)[:, None] * np.ones(num_motors)
    hold = np.full((n_stable, num_motors), 50.0)
    states = np.concatenate([ramp, hold])
    actions = states + offset
    return actions, states


class TestComputeEpisodeDeltas:
    def test_constant_offset_recovered_on_stable_frames(self):
        offset = np.array([2.0, -1.5, 0.0])
        actions, states = make_episode_with_offset(offset)
        stable, every = compute_episode_deltas(actions, states, vel_threshold=0.5)

        assert len(every) == len(actions) - 1
        assert len(stable) > 0
        np.testing.assert_allclose(np.mean(stable, axis=0), offset, atol=1e-9)

    def test_no_stable_frames_when_always_moving(self):
        offset = np.zeros(3)
        actions, states = make_episode_with_offset(offset, n_moving=50, n_stable=0)
        stable, every = compute_episode_deltas(actions, states, vel_threshold=0.5)

        assert len(stable) == 0
        assert len(every) == 49

    def test_short_episode_returns_empty(self):
        actions = np.zeros((1, 3))
        states = np.zeros((1, 3))
        stable, every = compute_episode_deltas(actions, states, vel_threshold=0.5)

        assert len(stable) == 0
        assert len(every) == 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_episode_deltas(np.zeros((10, 3)), np.zeros((10, 4)), vel_threshold=0.5)


class TestSummarizeCalibration:
    def test_verdicts(self):
        offsets = np.array([0.5, 2.0, 5.0])
        deltas = np.tile(offsets, (100, 1))
        summary = summarize_calibration(deltas, deltas, ["a", "b", "c"])

        assert [m["verdict"] for m in summary] == [VERDICT_OK, VERDICT_MILD, VERDICT_SIGNIFICANT]
        assert [m["motor"] for m in summary] == ["a", "b", "c"]
        np.testing.assert_allclose([m["mean_stable"] for m in summary], offsets)

    def test_negative_offset_uses_absolute_value(self):
        deltas = np.full((100, 1), -5.0)
        summary = summarize_calibration(deltas, deltas, ["a"])
        assert summary[0]["verdict"] == VERDICT_SIGNIFICANT
        assert summary[0]["mean_stable"] == pytest.approx(-5.0)

    def test_custom_thresholds(self):
        deltas = np.full((10, 1), 2.0)
        summary = summarize_calibration(deltas, deltas, ["a"], ok_threshold=2.5, warn_threshold=5.0)
        assert summary[0]["verdict"] == VERDICT_OK


class TestGroupPairsByEpisode:
    def test_groups_and_sorts_by_frame_index(self):
        datasets = pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

        hf_dataset = datasets.Dataset.from_dict(
            {
                "episode_index": [1, 0, 0, 1],
                "frame_index": [1, 1, 0, 0],
                ACTION: [[10.0], [1.0], [0.0], [9.0]],
                OBS_STATE: [[10.5], [1.5], [0.5], [9.5]],
            }
        )
        pairs = group_pairs_by_episode(hf_dataset)

        assert set(pairs) == {0, 1}
        actions_ep0, states_ep0 = pairs[0]
        np.testing.assert_allclose(actions_ep0, [[0.0], [1.0]])
        np.testing.assert_allclose(states_ep0, [[0.5], [1.5]])
        actions_ep1, _ = pairs[1]
        np.testing.assert_allclose(actions_ep1, [[9.0], [10.0]])


class FakeMeta:
    def __init__(self, features):
        self.features = features


class FakeDataset:
    def __init__(self, hf_dataset, motor_names):
        self.hf_dataset = hf_dataset
        self.meta = FakeMeta(
            {
                ACTION: {"dtype": "float32", "shape": (len(motor_names),), "names": motor_names},
                OBS_STATE: {"dtype": "float32", "shape": (len(motor_names),), "names": motor_names},
            }
        )


def make_fake_dataset(offset: np.ndarray, motor_names: list[str], n_moving: int = 20, n_stable: int = 30):
    datasets = pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

    actions, states = make_episode_with_offset(offset, n_moving=n_moving, n_stable=n_stable)
    hf_dataset = datasets.Dataset.from_dict(
        {
            "episode_index": [0] * len(actions),
            "frame_index": list(range(len(actions))),
            ACTION: actions.tolist(),
            OBS_STATE: states.tolist(),
        }
    )
    return FakeDataset(hf_dataset, motor_names)


class TestCheckCalibration:
    def test_end_to_end_report(self):
        offset = np.array([0.2, 4.0])
        dataset = make_fake_dataset(offset, ["shoulder_pan", "elbow_flex"])
        report = check_calibration(dataset, vel_threshold=0.5)

        assert report["num_episodes"] == 1
        assert report["num_stable_frames"] > 0
        verdicts = {m["motor"]: m["verdict"] for m in report["motors"]}
        assert verdicts == {"shoulder_pan": VERDICT_OK, "elbow_flex": VERDICT_SIGNIFICANT}

    def test_missing_state_feature_raises(self):
        dataset = make_fake_dataset(np.zeros(2), ["a", "b"])
        del dataset.meta.features[OBS_STATE]
        with pytest.raises(ValueError, match="must contain both"):
            check_calibration(dataset)

    def test_no_stable_frames_raises(self):
        dataset = make_fake_dataset(np.zeros(2), ["a", "b"], n_moving=50, n_stable=0)
        with pytest.raises(ValueError, match="No stable frames"):
            check_calibration(dataset, vel_threshold=0.5)
