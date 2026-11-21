"""Tests for the duck-typed LeRobot adapter (lerobot_data_quality.py).

These tests use a lightweight fake dataset class that mimics the LeRobotDataset
v3 surface (``.meta.episodes``, ``.meta.tasks``, and index-based frame access)
without importing torch or lerobot.

The script under test is loaded by absolute file path via ``importlib`` so this
test file never imports the ``lerobot`` package (which would pull in torch and
the repo's ``tests/conftest.py`` fixtures) and stays runnable in isolation.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "lerobot"
    / "scripts"
    / "lerobot_data_quality.py"
)
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_spec = importlib.util.spec_from_file_location("lerobot_data_quality", _SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"could not load script module from {_SCRIPT_PATH}")
_lerobot_data_quality = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lerobot_data_quality)

from_lerobot_dataset = _lerobot_data_quality.from_lerobot_dataset
make_demo_dataset = _lerobot_data_quality.make_demo_dataset
task_conditioned_quality = _lerobot_data_quality.task_conditioned_quality


class FakeLeRobotDataset:
    """Duck-typed stand-in for LeRobotDataset v3 (no torch/lerobot import)."""

    def __init__(self, trajectories, task_ids, feature="action", features=None):
        dim = trajectories[0].shape[1] if trajectories else 1
        self.features = features if features is not None else {
            feature: {"dtype": "float32", "shape": (dim,)},
            "observation.images.front": {"dtype": "video", "shape": (3, 48, 64)},
            "language_instruction": {"dtype": "string", "shape": (1,)},
        }
        self._frames = []
        episodes = []
        offset = 0
        for traj, tid in zip(trajectories, task_ids):
            for row in traj:
                self._frames.append({feature: np.asarray(row, dtype=float), "task_index": int(tid)})
            episodes.append(
                {"dataset_from_index": offset, "dataset_to_index": offset + len(traj), "task_index": int(tid)}
            )
            offset += len(traj)
        self.meta = types.SimpleNamespace(
            episodes=episodes,
            tasks={int(tid): f"task-{tid}" for tid in set(task_ids)},
        )

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return list(self._frames[idx])
        return self._frames[idx]


def _traj(start, end, length=10, dim=3):
    """Linearly interpolated (length, dim) trajectory from scalar start to end."""
    t = np.linspace(start, end, length)[:, None]
    return np.repeat(t, dim, axis=1)


def test_from_lerobot_dataset_multi_episode():
    """Correctly slices each episode's frame range and extracts action + task."""
    trajs = [_traj(0.0, 1.0, 10, 3), _traj(0.0, 1.0, 12, 3), _traj(1.0, 2.0, 10, 3)]
    task_ids = [0, 0, 1]
    ds = FakeLeRobotDataset(trajs, task_ids)

    out_trajs, out_tasks = from_lerobot_dataset(ds)

    assert out_tasks == [0, 0, 1]
    assert len(out_trajs) == 3
    assert [t.shape[0] for t in out_trajs] == [10, 12, 10]
    assert all(t.shape[1] == 3 for t in out_trajs)
    # Episode boundaries are respected: episode 2 starts at frame index 22.
    np.testing.assert_allclose(out_trajs[0], trajs[0])
    np.testing.assert_allclose(out_trajs[2], trajs[2])


def test_from_lerobot_dataset_empty():
    """An empty dataset adapts to ([], []) and signals insufficient data."""
    ds = FakeLeRobotDataset([], [])
    out_trajs, out_tasks = from_lerobot_dataset(ds)
    assert out_trajs == []
    assert out_tasks == []

    report = task_conditioned_quality(out_trajs, out_tasks)
    assert report["insufficient_data"] is True
    assert report["generalizable_quality"] == pytest.approx(0.0)


def test_from_lerobot_dataset_dimension_mismatch_raises():
    """Episodes with differing action dim D must raise a descriptive error."""
    trajs = [_traj(0.0, 1.0, 10, 2), _traj(1.0, 2.0, 10, 3)]
    ds = FakeLeRobotDataset(trajs, [0, 1])

    with pytest.raises(ValueError) as exc:
        from_lerobot_dataset(ds)
    msg = str(exc.value)
    assert "feature dimension D" in msg
    assert "episode 1" in msg


def test_skips_image_video_string_features():
    """Image/video/string features are skipped even when listed first."""
    trajs = [_traj(0.0, 1.0, 8, 3)]
    ds = FakeLeRobotDataset(
        trajs,
        [0],
        feature="action",
        features={
            "observation.images.front": {"dtype": "video", "shape": (3, 48, 64)},
            "language_instruction": {"dtype": "string", "shape": (1,)},
            "action": {"dtype": "float32", "shape": (3,)},
        },
    )
    out_trajs, out_tasks = from_lerobot_dataset(ds)
    assert out_tasks == [0]
    assert out_trajs[0].shape == (8, 3)


def test_falls_back_to_numeric_vector_when_action_missing():
    """If ``action`` is absent, fall back to the first numeric vector feature."""
    trajs = [_traj(0.0, 1.0, 8, 2)]
    ds = FakeLeRobotDataset(trajs, [0], feature="observation.state")

    out_trajs, out_tasks = from_lerobot_dataset(ds)  # default feature="action"

    assert out_tasks == [0]
    assert out_trajs[0].shape == (8, 2)


def test_raises_when_no_numeric_vector_feature():
    """A dataset with only image/video/string features has nothing to score."""
    trajs = [_traj(0.0, 1.0, 8, 3)]
    ds = FakeLeRobotDataset(
        trajs,
        [0],
        feature="action",
        features={
            "observation.images.front": {"dtype": "video", "shape": (3, 48, 64)},
            "language_instruction": {"dtype": "string", "shape": (1,)},
        },
    )
    with pytest.raises(ValueError) as exc:
        from_lerobot_dataset(ds)
    assert "no numeric vector feature" in str(exc.value)


def test_demo_dataset_runs_end_to_end():
    """The built-in demo dataset drives the full adapter -> core pipeline."""
    ds = make_demo_dataset()
    out_trajs, out_tasks = from_lerobot_dataset(ds)

    assert 3 <= len(out_trajs) <= 5
    assert len(out_tasks) == len(out_trajs)
    assert len(set(out_tasks)) >= 2  # multiple distinct tasks

    report = task_conditioned_quality(out_trajs, out_tasks)
    assert "insufficient_data" not in report
    assert 0.0 <= report["generalizable_quality"] <= 1.0
