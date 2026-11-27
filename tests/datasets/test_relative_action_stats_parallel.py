# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression coverage for bounded, ordered relative-action statistics."""

import concurrent.futures
import weakref

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import compute_stats
from lerobot.utils.constants import ACTION, OBS_STATE

pytestmark = pytest.mark.dataset


def _make_data(n_frames):
    rng = np.random.default_rng(42)
    actions = rng.standard_normal((n_frames, 2), dtype=np.float32)
    actions *= np.linspace(0.1, 4.0, n_frames, dtype=np.float32)[:, None]
    states = rng.standard_normal((n_frames, 2), dtype=np.float32)
    data = {ACTION: actions, OBS_STATE: states, "episode_index": np.arange(n_frames) // 400}
    features = {
        key: {"dtype": "float32", "shape": (2,), "names": ["joint_a", "joint_b"]}
        for key in (ACTION, OBS_STATE)
    }
    return data, features


@pytest.fixture
def observed_executor(monkeypatch):
    """Make completion order controlled, and count live result-bearing futures."""
    observations = {"submitted": 0, "peak_live": 0, "consumed": []}
    live = weakref.WeakSet()

    class ObservedFuture(concurrent.futures.Future):
        def __init__(self, index, value):
            super().__init__()
            self.index = index
            self.set_result(value)

        def result(self, timeout=None):
            observations["consumed"].append(self.index)
            return super().result(timeout=timeout)

    class ObservedExecutor:
        def __init__(self, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, *args):
            index = observations["submitted"]
            future = ObservedFuture(index, fn(*args))
            observations["submitted"] += 1
            live.add(future)
            observations["peak_live"] = max(observations["peak_live"], len(live))
            return future

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", ObservedExecutor)
    monkeypatch.setattr(concurrent.futures, "as_completed", lambda futures: iter(reversed(list(futures))))
    return observations


def test_relative_stats_bounds_live_futures(observed_executor):
    # Seven 50k-frame batches with two workers; a submit-all implementation
    # keeps more result-bearing futures than the configured four-batch window.
    data, features = _make_data(300_001)
    compute_stats.compute_relative_action_stats(data, features, chunk_size=1, num_workers=2)
    assert observed_executor["submitted"] == 7
    assert observed_executor["peak_live"] <= 4


def test_relative_stats_reduces_in_dataset_order(observed_executor):
    data, features = _make_data(100_001)
    compute_stats.compute_relative_action_stats(data, features, chunk_size=1, num_workers=2)
    assert observed_executor["submitted"] == 3
    assert observed_executor["consumed"] == [0, 1, 2]


@pytest.mark.parametrize("workers", [2, 4])
def test_relative_stats_matches_serial(workers):
    data, features = _make_data(120_003)
    kwargs = {"chunk_size": 2, "exclude_joints": ["joint_b"]}
    expected = compute_stats.compute_relative_action_stats(data, features, num_workers=0, **kwargs)
    actual = compute_stats.compute_relative_action_stats(data, features, num_workers=workers, **kwargs)
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize("failing_stage", ["worker", "reducer"])
def test_relative_stats_cancels_queued_work_on_error(monkeypatch, failing_stage):
    futures = []
    failure = RuntimeError(f"{failing_stage} failed")

    class Executor:
        def __init__(self, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, *args):
            future = concurrent.futures.Future()
            if not futures:
                if failing_stage == "worker":
                    future.set_exception(failure)
                else:
                    future.set_result(fn(*args))
            elif len(futures) == 1:
                assert future.set_running_or_notify_cancel()
            futures.append(future)
            return future

    def fail_update(*args):
        raise failure

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", Executor)
    if failing_stage == "reducer":
        monkeypatch.setattr(compute_stats.RunningQuantileStats, "update", fail_update)
    data, features = _make_data(300_001)
    with pytest.raises(RuntimeError) as caught:
        compute_stats.compute_relative_action_stats(data, features, chunk_size=1, num_workers=2)

    assert caught.value is failure
    assert len(futures) == 4
    assert futures[1].running()
    assert all(future.cancelled() for future in futures[2:])
