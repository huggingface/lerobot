"""Tests for task-conditioned consistency + cross-task diversity."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "lerobot" / "scripts"))

from goal_conditioned_quality import (  # noqa: E402
    _resample,
    cross_task_diversity,
    task_conditioned_quality,
    within_group_consistency,
)


def _linear_traj(start, end, length=50, dim=2):
    """Linearly interpolated (length, dim) trajectory from scalar start to end."""
    t = np.linspace(start, end, length)[:, None]  # (length, 1)
    return np.repeat(t, dim, axis=1)


def test_goal_variation_is_not_flagged():
    """Two distinct tasks, clean within-task repeats -> high quality, no outliers.

    This is the exact case both #3762 and #2936 get wrong: unconditional
    consistency would flag every episode of task B as an "outlier" relative to
    task A. Task-conditioning must NOT flag them.
    """
    eps = [_linear_traj(0.0, 1.0, 50) for _ in range(5)]
    eps += [_linear_traj(1.0, 2.0, 50) for _ in range(5)]
    eps = [e + np.random.RandomState(i).randn(50, 2) * 1e-3 for i, e in enumerate(eps)]
    task_ids = [0] * 5 + [1] * 5

    report = task_conditioned_quality(eps, task_ids)

    # No within-task outliers (repeats are clean).
    assert report["groups"][0]["outliers"] == []
    assert report["groups"][1]["outliers"] == []
    # Repeatability is high.
    assert report["precision"] > 0.95
    # The two tasks are genuinely different -> diversity is high.
    assert report["diversity"] > 0.9
    assert report["generalizable_quality"] > 0.9


def test_sloppy_repetition_flagged():
    """Within one task, one bad demonstration should lower precision."""
    rng = np.random.RandomState(42)
    eps = [_linear_traj(0.0, 1.0, 50) for _ in range(10)]
    eps = [e + rng.randn(50, 2) * 1e-3 for e in eps]
    eps[5] = rng.randn(50, 2) * 100  # grossly different demonstration

    rep = within_group_consistency(eps)

    assert 5 in rep["outliers"]
    assert rep["precision"] < 1.0


def test_collapsed_tasks_penalized():
    """If 'different' tasks produce identical motion, diversity must be low.

    This is the inverse failure mode: a dataset where every task is the same
    trajectory has no generalization coverage, and unconditional consistency
    would reward it. We penalize it via low diversity.
    """
    eps = [_linear_traj(0.0, 1.0, 50) for _ in range(20)]
    task_ids = list(range(20))  # pretend they're 20 distinct tasks

    report = task_conditioned_quality(eps, task_ids)

    assert report["precision"] > 0.9  # "repeatable"
    assert report["diversity"] < 0.1  # ...but all identical -> useless coverage
    assert report["generalizable_quality"] < 0.1


def test_single_task_is_neutral_on_diversity():
    """One task cannot show diversity; do not penalize (diversity -> 1.0)."""
    eps = [_linear_traj(0.0, 1.0, 50) for _ in range(8)]
    task_ids = [0] * 8

    report = task_conditioned_quality(eps, task_ids)

    assert report["diversity"] == pytest.approx(1.0)
    assert report["generalizable_quality"] == pytest.approx(report["precision"], abs=1e-6)


def test_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        task_conditioned_quality([np.zeros((10, 2))], task_ids=[0, 1])


def test_cross_task_diversity_direct():
    """Between/within ratio is high for separated tasks, ~0 for identical."""
    separated = cross_task_diversity(
        [_linear_traj(0.0, 1.0, 50), _linear_traj(1.0, 2.0, 50)],
        task_ids=np.array([0, 1]),
    )
    identical = cross_task_diversity(
        [_linear_traj(0.0, 1.0, 50), _linear_traj(0.0, 1.0, 50)],
        task_ids=np.array([0, 1]),
    )
    assert separated > 0.9
    assert identical < 0.1


# ---------------------------------------------------------------------------
# Edge / robustness coverage (added for the LeRobot PR review).
# ---------------------------------------------------------------------------


def test_empty_dataset_signals_insufficient_data():
    """Empty input is NOT a perfect score: it must return an explicit no-data flag."""
    report = task_conditioned_quality([], [])
    assert report["insufficient_data"] is True
    assert report["generalizable_quality"] == pytest.approx(0.0)
    assert report["precision"] == pytest.approx(0.0)
    assert report["diversity"] == pytest.approx(0.0)
    assert report["groups"] == {}
    assert report["error"]


def test_empty_trajectory_raises():
    """An episode with zero frames is malformed and must fail fast."""
    with pytest.raises(ValueError):
        task_conditioned_quality([np.empty((0, 2)), np.zeros((5, 2))], [0, 0])


def test_single_frame_episodes_all_equal_length():
    """All episodes being single-frame (equal shape) is well-defined, no crash."""
    eps = [np.zeros((1, 2)) for _ in range(5)]
    rep = within_group_consistency(eps)
    assert rep["outliers"] == []
    assert rep["precision"] == pytest.approx(1.0)


def test_zero_variance_is_not_flagged_as_outlier():
    """Identical repeats have zero std; the outlier rule must not false-positive."""
    eps = [_linear_traj(0.0, 1.0, 50) for _ in range(10)]
    rep = within_group_consistency(eps)
    assert rep["outliers"] == []
    assert rep["std_dist"] == pytest.approx(0.0)
    assert rep["precision"] == pytest.approx(1.0)


def test_all_zero_trajectories_single_task():
    """All-zero data still produces a well-defined (if uninformative) score."""
    eps = [np.zeros((10, 2)) for _ in range(6)]
    report = task_conditioned_quality(eps, [0] * 6)
    assert report["precision"] == pytest.approx(1.0)
    assert report["diversity"] == pytest.approx(1.0)  # single task -> neutral
    assert report["generalizable_quality"] == pytest.approx(1.0)


def test_string_task_ids_supported():
    """Non-integer (string) task labels are part of the public contract."""
    eps = [_linear_traj(0.0, 1.0, 50) for _ in range(3)]
    eps += [_linear_traj(1.0, 2.0, 50) for _ in range(3)]
    report = task_conditioned_quality(eps, ["grab"] * 3 + ["push"] * 3)
    assert report["precision"] > 0.95
    assert report["diversity"] > 0.9
    assert "grab" in report["groups"] and "push" in report["groups"]


def test_mixed_feature_dimensions_fail_fast():
    """Trajectories with differing D cannot be stacked; must raise, not corrupt."""
    with pytest.raises(ValueError) as exc:
        task_conditioned_quality([np.zeros((5, 2)), np.zeros((5, 3))], [0, 1])
    assert "feature dimension D" in str(exc.value)
    assert "episode 1" in str(exc.value)


def test_resample_extreme_target_lengths():
    """Resampling down to one frame or up to many frames stays well-formed."""
    traj = np.linspace(0.0, 1.0, 50)[:, None].repeat(2, axis=1)
    down = _resample(traj, 1)
    assert down.shape == (1, 2)
    assert down[0, 0] == pytest.approx(traj[0, 0])  # t=0 -> first frame
    up = _resample(traj, 1000)
    assert up.shape == (1000, 2)
    assert up[-1, 0] == pytest.approx(traj[-1, 0])  # t=1 -> last frame


def test_single_frame_mixed_with_longer_episodes():
    """Mixing a 1-frame episode with longer ones must resample the short one."""
    eps = [np.zeros((1, 2)), np.ones((10, 2)), np.zeros((10, 2))]
    rep = within_group_consistency(eps)
    assert rep["n"] == 3
    assert 0.0 <= rep["precision"] <= 1.0


def test_nan_episode_is_detected():
    """A NaN episode must be flagged (or raise), not silently scored perfect."""
    eps = [_linear_traj(0.0, 1.0, 50) for _ in range(5)]
    eps[2] = np.full((50, 2), np.nan)
    with np.errstate(all="ignore"):
        rep = within_group_consistency(eps)
    assert np.isfinite(rep["mean_dist"])
    assert rep["precision"] < 1.0 or rep["outliers"] == [2]


def test_within_group_consistency_empty_list():
    rep = within_group_consistency([])
    assert rep["n"] == 0
    assert rep["precision"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Observation A/B regression tests (REVIEW_NOTES low-priority observations).
# ---------------------------------------------------------------------------


def test_empty_dataset_is_not_confused_with_perfect_score():
    """Empty dataset must be distinguishable from a genuinely perfect dataset."""
    empty = task_conditioned_quality([], [])
    perfect = task_conditioned_quality([_linear_traj(0.0, 1.0, 50)], [0])
    assert empty["insufficient_data"] is True
    assert "insufficient_data" not in perfect
    assert empty["generalizable_quality"] != perfect["generalizable_quality"]
    assert empty["generalizable_quality"] == pytest.approx(0.0)


def test_dimension_mismatch_error_names_episode_and_dims():
    """Mixed D -> ValueError that names episode index, expected D, actual D."""
    with pytest.raises(ValueError) as exc:
        task_conditioned_quality(
            [np.zeros((5, 2)), np.zeros((5, 3)), np.zeros((5, 2))], [0, 1, 2]
        )
    msg = str(exc.value)
    assert "feature dimension D" in msg
    assert "episode 1" in msg
    assert "D=3" in msg
    assert "D=2" in msg


def test_within_group_dimension_mismatch_raises_clear_error():
    """Direct helper call with mixed D must also raise a descriptive error."""
    with pytest.raises(ValueError) as exc:
        within_group_consistency([np.zeros((5, 2)), np.zeros((5, 3))])
    assert "episode 1" in str(exc.value)
    assert "D=3" in str(exc.value)


def test_cross_task_dimension_mismatch_raises_clear_error():
    """Direct helper call with mixed D must also raise a descriptive error."""
    with pytest.raises(ValueError) as exc:
        cross_task_diversity(
            [np.zeros((5, 2)), np.zeros((5, 3))], task_ids=np.array([0, 1])
        )
    assert "episode 1" in str(exc.value)
    assert "D=3" in str(exc.value)