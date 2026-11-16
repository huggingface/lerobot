"""Task-conditioned consistency + cross-task diversity for robot datasets.

Why this exists
---------------
Two in-flight LeRobot PRs (#3762 and #2936) both measure inter-episode
"consistency" *unconditionally*: every episode is compared to a single mean
trajectory, and episodes that deviate are flagged as outliers.

That is wrong for goal-conditioned / multi-task datasets (``lerobot/pusht``,
ALOHA, and essentially everything in Open X-Embodiment), where different
episodes are *supposed* to end at different poses or follow different
trajectories. Unconditional consistency:

* systematically flags legitimate task variation as "dirty" data, and
* rewards datasets that memorize a single trajectory — the opposite of the
  generalization coverage you actually want.

The fix is to condition consistency on the task/goal:

1. **Precision** — consistency *within* a task (repetition quality). Two
   demonstrations of the same task should agree; sloppy repeats are outliers.
2. **Diversity** — separation *across* tasks (coverage). Distinct tasks should
   produce distinct trajectories; if they all collapse to the same motion, the
   dataset is not generalizable.

Both are unit-free 0-1 scores. Their product is a single
``generalizable_quality`` score.

This module is pure NumPy and has no LeRobot dependency, so it can be unit
tested in isolation and dropped into either PR as an additional analyzer.

Usage
-----
>>> from goal_conditioned_quality import task_conditioned_quality
>>> task_conditioned_quality([np.zeros((50, 2)), np.ones((50, 2))],
...                          task_ids=[0, 1])
"""

import numpy as np


def _resample(trajectory: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a (T, D) trajectory to ``target_length`` via linear interp."""
    if len(trajectory) == 0:
        raise ValueError("trajectory must have at least one frame")
    if len(trajectory) == 1:
        if target_length == 1:
            return trajectory.astype(float)
        return np.repeat(trajectory.astype(float), target_length, axis=0)
    if target_length == len(trajectory):
        return trajectory.astype(float)

    t_orig = np.linspace(0.0, 1.0, len(trajectory))
    t_new = np.linspace(0.0, 1.0, target_length)
    return np.stack(
        [np.interp(t_new, t_orig, trajectory[:, d]) for d in range(trajectory.shape[1])],
        axis=1,
    )


def _check_consistent_feature_dim(all_actions: list[np.ndarray], context: str = "") -> None:
    """Validate that every episode is 2D ``(T, D)`` and shares the same ``D``.

    Raises a descriptive ``ValueError`` (naming the offending episode index and
    the expected vs actual feature dimension ``D``) instead of letting a later
    ``np.stack`` fail with the opaque "all input arrays must have the same
    shape" message.
    """
    if len(all_actions) == 0:
        return
    expected_d = None
    for i, a in enumerate(all_actions):
        arr = np.asarray(a)
        if arr.ndim != 2:
            raise ValueError(
                f"{context}episode {i} must be a 2D (T, D) trajectory, got "
                f"{arr.ndim}D array with shape {arr.shape}"
            )
        d = arr.shape[1]
        if expected_d is None:
            expected_d = d
        elif d != expected_d:
            raise ValueError(
                f"{context}all episodes must share the same feature dimension D; "
                f"episode {i} has D={d} (shape {arr.shape}), expected D={expected_d}"
            )


def within_group_consistency(episodes: list[np.ndarray], outlier_std: float = 2.0) -> dict:
    """Consistency of demonstrations *within one task*.

    Args:
        episodes: List of (T_i, D) action arrays for the same task.
        outlier_std: Number of standard deviations above the mean distance at
            which a demonstration is flagged as a sloppy outlier.

    Returns:
        dict with ``precision`` (0-1), ``outliers`` (list of episode-local
        indices), ``mean_dist``, ``std_dist``, ``n``.
    """
    n = len(episodes)
    if n == 0:
        return {
            "n": 0,
            "mean_dist": 0.0,
            "std_dist": 0.0,
            "outliers": [],
            "precision": 1.0,
        }

    _check_consistent_feature_dim(episodes, context="within_group_consistency: ")

    lengths = [len(a) for a in episodes]
    target_len = max(int(np.median(lengths)), 1)

    finite = [bool(np.all(np.isfinite(a))) for a in episodes]
    clean = [i for i, ok in enumerate(finite) if ok]
    nan_outliers = [i for i, ok in enumerate(finite) if not ok]

    resampled = np.stack([_resample(a, target_len) for a in episodes])  # (N, T, D)
    dists: list[float] = []
    if clean:
        mean_traj = np.mean(resampled[clean], axis=0)
        dists = [
            float(np.mean(np.linalg.norm(resampled[i] - mean_traj, axis=1)))
            for i in clean
        ]

    mean_dist = float(np.mean(dists)) if dists else 0.0
    std_dist = float(np.std(dists)) if dists else 0.0

    outliers: list[int] = list(nan_outliers)
    if std_dist >= 1e-12:
        threshold = mean_dist + outlier_std * std_dist
        outliers += [clean[i] for i, d in enumerate(dists) if d > threshold]

    outlier_ratio = len(outliers) / n
    precision = float(np.clip(1.0 - outlier_ratio, 0.0, 1.0))

    return {
        "n": n,
        "mean_dist": mean_dist,
        "std_dist": std_dist,
        "outliers": outliers,
        "precision": precision,
    }


def cross_task_diversity(all_actions: list[np.ndarray], task_ids: np.ndarray) -> float:
    """Separation between distinct tasks, as a between/within ratio (0-1).

    This is the trajectory-space analogue of an ANOVA between/within variance
    ratio: diverse tasks should have group means far apart relative to the
    spread *within* a task.

    A single task (or empty input) cannot demonstrate diversity, so we return
    1.0 (neutral) rather than penalizing it.
    """
    if len(all_actions) == 0:
        return 1.0

    _check_consistent_feature_dim(all_actions, context="cross_task_diversity: ")

    unique = np.unique(task_ids)
    if len(unique) < 2:
        return 1.0

    lengths = [len(a) for a in all_actions]
    target_len = max(int(np.median(lengths)), 1)
    resampled = np.stack([_resample(a, target_len) for a in all_actions])  # (N, T, D)

    means = {}
    within_sum = 0.0
    within_count = 0
    for t in unique:
        idxs = np.where(task_ids == t)[0]
        means[t] = np.mean(resampled[idxs], axis=0)
        for i in idxs:
            within_sum += float(np.mean(np.linalg.norm(resampled[i] - means[t], axis=1)))
            within_count += 1

    within_avg = within_sum / max(within_count, 1)

    between = []
    for i, t1 in enumerate(unique):
        for t2 in unique[i + 1 :]:
            between.append(float(np.mean(np.linalg.norm(means[t1] - means[t2], axis=1))))

    between_avg = float(np.mean(between)) if between else 0.0

    diversity = between_avg / (between_avg + within_avg + 1e-8)
    return float(np.clip(diversity, 0.0, 1.0))


def task_conditioned_quality(
    all_actions: list[np.ndarray],
    task_ids,
    outlier_std: float = 2.0,
) -> dict:
    """Primary entry point: precision x diversity = generalizable quality.

    Args:
        all_actions: List of (T_i, D) action arrays, one per episode.
        task_ids: Iterable of int/str task labels, one per episode. For LeRobot
            datasets this is ``dataset.meta.tasks`` / the per-episode
            ``task_index`` (or an excerpt-averaged ``task`` instruction hash).
        outlier_std: See ``within_group_consistency``.

    Returns:
        {
            "precision":          0-1 within-task repeatability,
            "diversity":          0-1 cross-task separation,
            "generalizable_quality": precision * diversity,
            "groups": {task_id: within_group_consistency report},
        }

        For an **empty dataset** (0 episodes) the result is an explicit
        "no data / not evaluable" signal, NOT a perfect score::

            {
                "precision": 0.0,
                "diversity": 0.0,
                "generalizable_quality": 0.0,
                "insufficient_data": True,
                "error": "empty dataset: 0 episodes provided ...",
                "groups": {},
            }

    Raises:
        ValueError: if ``len(all_actions) != len(task_ids)``; if any episode is
            not a 2D ``(T, D)`` array; or if episodes do not all share the same
            feature dimension ``D`` (the message names the offending episode
            index and the expected vs actual ``D``).
    """
    all_actions = list(all_actions)
    task_ids = np.asarray(task_ids)

    if len(all_actions) != len(task_ids):
        raise ValueError(
            f"len(all_actions)={len(all_actions)} != len(task_ids)={len(task_ids)}"
        )

    if len(all_actions) == 0:
        return {
            "precision": 0.0,
            "diversity": 0.0,
            "generalizable_quality": 0.0,
            "insufficient_data": True,
            "groups": {},
            "error": "empty dataset: 0 episodes provided; quality is undefined "
            "(not 1.0) — nothing to evaluate",
        }

    _check_consistent_feature_dim(all_actions, context="task_conditioned_quality: ")

    unique = np.unique(task_ids)
    group_to_idxs = {t: np.where(task_ids == t)[0].tolist() for t in unique}

    groups: dict = {}
    precisions: list[float] = []
    for t, idxs in group_to_idxs.items():
        eps = [all_actions[i] for i in idxs]
        if len(eps) < 2:
            groups[t] = {
                "n": len(eps),
                "mean_dist": 0.0,
                "std_dist": 0.0,
                "outliers": [],
                "precision": 1.0,
            }
            precisions.append(1.0)
            continue
        rep = within_group_consistency(eps, outlier_std)
        groups[t] = rep
        precisions.append(rep["precision"])

    precision = float(np.mean(precisions)) if precisions else 1.0
    diversity = cross_task_diversity(all_actions, task_ids)
    score = float(np.clip(precision * diversity, 0.0, 1.0))

    return {
        "precision": precision,
        "diversity": diversity,
        "generalizable_quality": score,
        "groups": groups,
    }