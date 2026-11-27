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

"""Standalone CLI + thin adapter for goal-conditioned data-quality scoring.

This module measures a robot dataset's *generalizable quality* as the product of
two task-conditioned scores computed over per-episode action trajectories:

* **precision** — consistency *within* a task (repetition quality). Two
  demonstrations of the same task should agree; sloppy repeats are outliers.
* **diversity** — separation *across* tasks (coverage). Distinct tasks should
  produce distinct trajectories; if they all collapse to the same motion, the
  dataset is not generalizable.

The scoring core lives in :mod:`goal_conditioned_quality` (pure NumPy, no
LeRobot/torch dependency). On top of it this module layers:

* ``from_lerobot_dataset(dataset)`` — a duck-typed adapter that converts a
  LeRobotDataset v3 object into the ``(list[np.ndarray], list[int])`` format the
  core expects, without importing ``lerobot`` or ``torch`` at module import time.
* an ``argparse`` CLI (``main()``) with a ``--dataset`` mode and a
  dependency-free ``--demo`` mode that runs the whole pipeline on a synthetic
  dataset.

The adapter only reads the public, duck-typed surface of the dataset object:

* ``dataset.meta.episodes[i]["dataset_from_index"] / ["dataset_to_index"]`` — the
  absolute frame range of episode ``i``;
* ``dataset[from:to]`` — the ``__getitem__`` slice protocol returning one frame
  dict per row (each carrying the selected vector feature and a ``task_index``);
* ``dataset.features`` — optional feature metadata used to pick a numeric vector
  feature and to skip image/video/string/language features.

No torch, no lerobot, no models. Everything here is serial and memory-light.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    # Package mode (e.g. ``python -m lerobot.scripts.lerobot_data_quality``): the
    # sibling module is importable as a relative import.
    from .goal_conditioned_quality import _check_consistent_feature_dim, task_conditioned_quality
except ImportError:
    # Standalone execution (e.g. ``python src/lerobot/scripts/lerobot_data_quality.py --demo``):
    # the scripts directory is on sys.path instead of the package, so fall back
    # to a flat import.
    from goal_conditioned_quality import (  # type: ignore[no-redef]
        _check_consistent_feature_dim,
        task_conditioned_quality,
    )


# ---------------------------------------------------------------------------
# Adapter: LeRobotDataset v3 -> (list[np.ndarray], list[int]).
# ---------------------------------------------------------------------------

# Feature dtypes that are not numeric vectors and therefore can never be used to
# build a (T, D) action trajectory.
_NON_VECTOR_DTYPES = frozenset({"image", "video", "string", "language"})


def _is_numeric_vector(info: dict[str, Any]) -> bool:
    """True if a ``dataset.features`` entry is a numeric 1-D vector feature."""
    dtype = info.get("dtype")
    shape = tuple(info.get("shape", ()))
    return dtype not in _NON_VECTOR_DTYPES and len(shape) == 1


def _select_vector_feature(dataset, feature: str) -> str:
    """Pick the feature to score, skipping image/video/string/language features.

    If ``dataset.features`` is unavailable (pure duck typing), trust the caller's
    ``feature`` name. Otherwise require the requested feature to be a numeric
    vector, fall back to the first numeric vector feature when it is absent, and
    raise a descriptive error when nothing qualifies.
    """
    features = getattr(dataset, "features", None)
    if features is None:
        return feature

    if feature is not None and feature in features:
        if _is_numeric_vector(features[feature]):
            return feature
        raise ValueError(
            f"requested feature {feature!r} is not a numeric vector "
            f"(dtype={features[feature].get('dtype')!r}, "
            f"shape={tuple(features[feature].get('shape', ()))!r}); "
            "cannot build a (T, D) trajectory from an image/video/string/language feature"
        )

    candidates = [k for k, info in features.items() if _is_numeric_vector(info)]
    if not candidates:
        raise ValueError(
            "no numeric vector feature found in dataset.features "
            "(image/video/string/language features were skipped); nothing to score"
        )
    return candidates[0]


def from_lerobot_dataset(dataset, feature: str = "action") -> tuple[list[np.ndarray], list[int]]:
    """Convert a LeRobotDataset v3 object into core-function inputs.

    Duck-typed: does not import ``lerobot``/``torch``; only touches
    ``dataset.meta.episodes``, ``dataset.features``, and the ``dataset[from:to]``
    slice protocol.

    Args:
        dataset: A LeRobotDataset v3 object (or any duck-typed stand-in exposing
            ``meta.episodes``, optional ``features``, and ``__getitem__``).
        feature: Vector feature to build each ``(T_i, D)`` trajectory from
            (default ``"action"``). Must be a numeric vector feature; image/video/
            string/language features are skipped.

    Returns:
        ``(trajectories, task_ids)`` where ``trajectories[i]`` is a ``(T_i, D)``
        float array and ``task_ids[i]`` is the integer ``task_index`` of that
        episode (read from the per-frame ``task_index`` column).

    Raises:
        ValueError: if no numeric vector feature exists; if the frame range of an
            episode is empty; if a frame range is missing; or if episodes do not
            all share the same feature dimension ``D``.
    """
    episodes = dataset.meta.episodes
    feature = _select_vector_feature(dataset, feature)

    trajectories: list[np.ndarray] = []
    task_ids: list[int] = []

    for i in range(len(episodes)):
        ep = episodes[i]
        try:
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"episode {i} is missing dataset_from_index/dataset_to_index; "
                f"got keys {list(ep.keys()) if hasattr(ep, 'keys') else ep!r}"
            ) from exc

        if end <= start:
            raise ValueError(
                f"episode {i} has an empty frame range [{start}, {end}); "
                "cannot build a trajectory from zero frames"
            )

        frames = dataset[start:end]
        if len(frames) == 0:
            raise ValueError(f"episode {i} produced no frames for range [{start}, {end})")

        try:
            col = [np.asarray(frame[feature], dtype=float) for frame in frames]
        except KeyError as exc:
            raise ValueError(
                f"feature {feature!r} is missing from the frames of episode {i}; "
                f"available frame keys: {list(frames[0].keys()) if frames else []}"
            ) from exc

        arr = np.stack(col, axis=0)
        if arr.ndim == 1:  # scalar-valued frames -> (T, 1)
            arr = arr[:, None]
        elif arr.ndim != 2:
            raise ValueError(
                f"episode {i} feature {feature!r} is not a (T, D) vector; "
                f"got shape {arr.shape} (image/video features must be skipped)"
            )

        trajectories.append(arr)

        # Task label comes from the per-frame ``task_index`` column. All frames
        # of a single episode share one task by construction, so take the first.
        task_idx_values = [int(frame["task_index"]) for frame in frames]
        task_ids.append(task_idx_values[0])

    # Reuse the core's dimension validation so cross-episode D mismatches raise
    # the same descriptive error as task_conditioned_quality itself.
    _check_consistent_feature_dim(trajectories, context="from_lerobot_dataset: ")

    return trajectories, task_ids


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into JSON-serializable natives."""
    if isinstance(obj, dict):
        return {_jsonable(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class _SyntheticDataset:
    """Minimal duck-typed stand-in for LeRobotDataset v3 (used by --demo)."""

    def __init__(self, trajectories: list[np.ndarray], task_ids: list[int], dim: int) -> None:
        self.features = {
            "action": {"dtype": "float32", "shape": (dim,)},
            "observation.state": {"dtype": "float32", "shape": (dim,)},
            "observation.images.front": {"dtype": "video", "shape": (3, 48, 64)},
            "language_instruction": {"dtype": "string", "shape": (1,)},
        }
        self._frames: list[dict[str, Any]] = []
        episodes: list[dict[str, int]] = []
        offset = 0
        for traj, tid in zip(trajectories, task_ids, strict=False):
            n = len(traj)
            for row in traj:
                self._frames.append({"action": np.asarray(row, dtype=float), "task_index": int(tid)})
            episodes.append(
                {"dataset_from_index": offset, "dataset_to_index": offset + n, "task_index": int(tid)}
            )
            offset += n
        self.meta = types.SimpleNamespace(
            episodes=episodes,
            tasks={int(tid): f"task-{tid}" for tid in set(task_ids)},
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(idx, slice):
            return list(self._frames[idx])
        return self._frames[idx]


def make_demo_dataset() -> _SyntheticDataset:
    """Build a small synthetic dataset for the dependency-free ``--demo`` path.

    Produces 4 episodes across 3 tasks with a fixed feature dimension D=3:
    task 0 has two (slightly noisy) repeats so within-task precision is
    meaningful, tasks 1 and 2 have one episode each so cross-task diversity is
    meaningful.
    """
    rng = np.random.RandomState(0)
    dim = 3
    targets = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0]),
        2: np.array([0.0, 0.0, 1.0]),
    }
    plan = [(0, 50), (0, 52), (1, 50), (2, 55)]

    trajectories: list[np.ndarray] = []
    task_ids: list[int] = []
    for tid, length in plan:
        t = np.linspace(0.0, 1.0, length)[:, None]
        traj = t * targets[tid][None, :]
        traj = traj + rng.randn(length, dim) * 1e-3
        trajectories.append(traj)
        task_ids.append(tid)

    return _SyntheticDataset(trajectories, task_ids, dim)


def _load_lerobot_dataset(dataset_id: str) -> Any:
    """Lazily import and construct a LeRobotDataset (only for --dataset mode)."""
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "cannot load --dataset: lerobot is not installed in this environment. "
            "The adapter itself is dependency-free and duck-typed; use --demo to "
            f"verify the pipeline without LeRobot. Original error: {exc}"
        ) from exc
    return LeRobotDataset(dataset_id)


def _print_report(report: dict, trajectories: list[np.ndarray], task_ids: list[int]) -> int:
    """Print the data-quality JSON, or an explicit insufficient-data signal.

    Returns the process exit code (0 = scored, 2 = insufficient data).
    """
    if report.get("insufficient_data"):
        print(
            "insufficient_data: the dataset has no evaluable episodes, so quality "
            "is UNDEFINED (not a perfect 1.0 score).",
            file=sys.stderr,
        )
        print(f"reason: {report.get('error')}", file=sys.stderr)
        print(
            json.dumps(
                {
                    "data_quality": {
                        "insufficient_data": True,
                        "generalizable_quality": 0.0,
                        "precision": 0.0,
                        "diversity": 0.0,
                        "groups": {},
                    }
                },
                indent=2,
            )
        )
        return 2

    payload = {
        "data_quality": _jsonable(report),
        "num_episodes": len(trajectories),
        "num_tasks": int(len(set(task_ids))),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "LeRobot data-quality analyzer: task-conditioned consistency (precision) "
            "x cross-task diversity = generalizable quality."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path or repo id to a LeRobotDataset v3 (requires lerobot installed)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the full pipeline on a built-in synthetic dataset (no LeRobot needed)",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="action",
        help="Numeric vector feature to score (default: action); image/video/string features are skipped",
    )
    parser.add_argument(
        "--outlier-std",
        type=float,
        default=2.0,
        help="Outlier threshold in std above the within-task mean distance (default: 2.0)",
    )
    args = parser.parse_args(argv)

    if not args.dataset and not args.demo:
        parser.error("one of --dataset or --demo is required")

    if args.demo:
        print("running demo on a synthetic LeRobot-like dataset...", file=sys.stderr)
        dataset = make_demo_dataset()
    else:
        dataset = _load_lerobot_dataset(args.dataset)

    trajectories, task_ids = from_lerobot_dataset(dataset, feature=args.feature)
    report = task_conditioned_quality(trajectories, task_ids, outlier_std=args.outlier_std)
    return _print_report(report, trajectories, task_ids)


if __name__ == "__main__":
    sys.exit(main())
