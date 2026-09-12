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

from lerobot.scripts.lerobot_dataset_quality import (
    compute_episode_metrics,
    detect_outliers,
    evaluate_dataset_quality,
    group_actions_by_episode,
)
from lerobot.utils.constants import ACTION

FPS = 30


def smooth_episode(n_frames: int = 90, num_motors: int = 3) -> np.ndarray:
    """A smooth sinusoidal trajectory: low jerk, no static frames."""
    t = np.linspace(0, np.pi, n_frames)[:, None]
    return 10.0 * np.sin(t) * np.ones(num_motors)


class TestComputeEpisodeMetrics:
    def test_smooth_trajectory_has_low_jerk(self):
        metrics = compute_episode_metrics(smooth_episode(), episode_index=0, fps=FPS)

        assert metrics["episode"] == 0
        assert metrics["n_frames"] == 90
        assert metrics["duration_s"] == 3.0
        assert metrics["median_jerk"] < 0.1
        assert metrics["static_fraction"] < 0.2

    def test_spike_increases_p95_jerk(self):
        actions = smooth_episode(n_frames=40)
        spiked = actions.copy()
        spiked[20] += 20.0  # single sharp correction
        smooth = compute_episode_metrics(actions, episode_index=0, fps=FPS)
        jerky = compute_episode_metrics(spiked, episode_index=1, fps=FPS)

        assert jerky["p95_jerk"] > smooth["p95_jerk"]
        assert jerky["max_velocity"] > smooth["max_velocity"]

    def test_hold_increases_static_fraction(self):
        moving = smooth_episode(n_frames=45)
        hold = np.tile(moving[-1], (45, 1))
        with_hold = np.concatenate([moving, hold])
        metrics = compute_episode_metrics(with_hold, episode_index=0, fps=FPS)

        assert metrics["static_fraction"] > 0.4

    def test_short_episode_returns_defaults(self):
        actions = np.ones((2, 3))
        metrics = compute_episode_metrics(actions, episode_index=5, fps=FPS)

        assert metrics["episode"] == 5
        assert metrics["n_frames"] == 2
        assert metrics["median_jerk"] == 0.0
        assert metrics["final_action"] == [1.0, 1.0, 1.0]

    def test_final_action_is_last_action(self):
        actions = smooth_episode()
        metrics = compute_episode_metrics(actions, episode_index=0, fps=FPS)
        np.testing.assert_allclose(metrics["final_action"], actions[-1])


class TestDetectOutliers:
    def make_uniform_metrics(self, n: int = 20) -> list[dict]:
        return [compute_episode_metrics(smooth_episode(), episode_index=ep, fps=FPS) for ep in range(n)]

    def test_uniform_dataset_has_no_outliers(self):
        outliers = detect_outliers(self.make_uniform_metrics())
        assert outliers == {}

    def test_long_episode_flagged_as_duration_outlier(self):
        metrics = self.make_uniform_metrics()
        metrics.append(compute_episode_metrics(smooth_episode(n_frames=900), episode_index=99, fps=FPS))
        outliers = detect_outliers(metrics)

        assert 99 in outliers
        assert "duration_high" in outliers[99]

    def test_divergent_final_pose_flagged(self):
        metrics = self.make_uniform_metrics()
        divergent = smooth_episode()
        divergent[-1] += 50.0
        metrics.append(compute_episode_metrics(divergent, episode_index=99, fps=FPS))
        outliers = detect_outliers(metrics)

        assert "final_state_high" in outliers.get(99, set())


class FakeMeta:
    def __init__(self, num_motors):
        self.features = {ACTION: {"dtype": "float32", "shape": (num_motors,), "names": None}}


class FakeDataset:
    def __init__(self, hf_dataset, num_motors, fps=FPS):
        self.hf_dataset = hf_dataset
        self.meta = FakeMeta(num_motors)
        self.fps = fps


def make_fake_dataset(episodes: dict[int, np.ndarray]):
    datasets = pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

    episode_index, frame_index, actions = [], [], []
    for ep, ep_actions in episodes.items():
        episode_index += [ep] * len(ep_actions)
        frame_index += list(range(len(ep_actions)))
        actions += ep_actions.tolist()
    hf_dataset = datasets.Dataset.from_dict(
        {"episode_index": episode_index, "frame_index": frame_index, ACTION: actions}
    )
    num_motors = next(iter(episodes.values())).shape[1]
    return FakeDataset(hf_dataset, num_motors)


class TestGroupActionsByEpisode:
    def test_groups_and_sorts_by_frame_index(self):
        datasets = pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

        hf_dataset = datasets.Dataset.from_dict(
            {
                "episode_index": [1, 0, 0, 1],
                "frame_index": [1, 1, 0, 0],
                ACTION: [[10.0], [1.0], [0.0], [9.0]],
            }
        )
        actions_by_ep = group_actions_by_episode(hf_dataset)

        assert set(actions_by_ep) == {0, 1}
        np.testing.assert_allclose(actions_by_ep[0], [[0.0], [1.0]])
        np.testing.assert_allclose(actions_by_ep[1], [[9.0], [10.0]])


class TestEvaluateDatasetQuality:
    def test_end_to_end_flags_bad_episode(self):
        episodes = {ep: smooth_episode() for ep in range(20)}
        episodes[20] = smooth_episode(n_frames=900)  # 10x longer than the rest
        dataset = make_fake_dataset(episodes)

        report = evaluate_dataset_quality(dataset)

        assert len(report["metrics"]) == 21
        assert report["metrics"][0]["episode"] == 0
        assert "duration_high" in report["outliers"][20]

    def test_missing_action_feature_raises(self):
        dataset = make_fake_dataset({0: smooth_episode()})
        del dataset.meta.features[ACTION]
        with pytest.raises(ValueError, match="must contain"):
            evaluate_dataset_quality(dataset)
