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

"""Compute the normalization statistics of a LeRobot dataset for a Praxis-1 fine-tune from the base.

The statistics are those of the ``normalization`` section of a single-arm praxis checkpoint
(``praxis.train.norm_stats``): the 1 % and 99 % quantiles of the relative actions and the mean
and standard deviation of the proprio, over random training windows drawn as the fine-tune
draws them (an episode uniformly, then the window start uniformly in time). The dataset's
state and action columns hold the DROID pose and gripper ``[x, y, z, roll, pitch, yaw, gripper]``
per frame (the ``observation.state.cartesian`` and ``action.cartesian`` columns of DROID, or your
rig's); no video is read.

    python examples/praxis/compute_norm_stats.py --repo-id lerobot/droid_1.0.1 --revision refs/pr/11 \\
        --state-key observation.state.cartesian --action-key action.cartesian \\
        --episodes droid_episodes.json --n-windows 2000 --out stats.json

``init_heads.py --stats stats.json`` reads the file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa

from lerobot.policies.praxis.praxis_core.robot_io import FPS, contiguous_chunk_geometry, pose8_to_10d
from lerobot.policies.praxis.praxis_core.train.data import draw_anchor, droid_pose8, window_fits
from lerobot.policies.praxis.praxis_core.train.norm_stats import norm_stats

logger = logging.getLogger(__name__)

WINDOW_GEOMETRY = contiguous_chunk_geometry(24, 4)  # the chunks of a training window: (0, 21), (21, 24), ...
WINDOW_FRAMES = sum(WINDOW_GEOMETRY[-1])  # 93


def window_rows(observed7: np.ndarray, commanded7: np.ndarray, anchor: int) -> tuple[np.ndarray, np.ndarray]:
    """The relative actions ``(93, 10)`` and the proprio ``(4, 10)`` of the window at ``anchor``.

    ``observed7`` and ``commanded7`` are the per-frame DROID rows of the episode. The rows are
    those of a training window: the commanded pose of each frame relative to the observed pose at
    the start of its chunk, and the observed pose at each chunk start.
    """
    obs = droid_pose8(
        observed7[anchor : anchor + WINDOW_FRAMES, :6], observed7[anchor : anchor + WINDOW_FRAMES, 6]
    )
    cmd = droid_pose8(
        commanded7[anchor : anchor + WINDOW_FRAMES, :6], commanded7[anchor : anchor + WINDOW_FRAMES, 6]
    )
    actions = np.concatenate(
        [pose8_to_10d(cmd[off : off + ln], anchor8=obs[off]) for off, ln in WINDOW_GEOMETRY]
    )
    proprio = pose8_to_10d(obs[[off for off, _ in WINDOW_GEOMETRY]])
    return actions, proprio


def sample_window_stats(
    episodes: list[tuple[np.ndarray, np.ndarray]], n_windows: int, seed: int = 0, fps: float = FPS
) -> dict[str, list[float]]:
    """Draw ``n_windows`` windows over ``episodes`` (``(observed7, commanded7)`` per episode) and return the stats."""
    if n_windows <= 0:
        raise ValueError(f"n_windows must be > 0, got {n_windows}")
    pool = [i for i, (obs, _) in enumerate(episodes) if window_fits(obs.shape[0], WINDOW_FRAMES, fps)]
    if not pool:
        raise RuntimeError(f"no episode holds a window of {WINDOW_FRAMES} frames plus the margin")
    rng = np.random.default_rng(seed)
    actions, proprio = [], []
    for _ in range(n_windows):
        obs, cmd = episodes[pool[int(rng.integers(len(pool)))]]
        anchor = draw_anchor(obs.shape[0], WINDOW_FRAMES, fps, rng)
        a, p = window_rows(obs, cmd, anchor)
        actions.append(a)
        proprio.append(p)
    return norm_stats(np.concatenate(actions).astype(np.float64), np.concatenate(proprio).astype(np.float64))


def episode_slices(frame_index: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> list[slice]:
    """The rows of each episode in a table of frames, from the frame ranges of the episodes.

    ``frame_index`` is the ``index`` column of the table: the absolute frame index of each row,
    increasing (the frames of the selected episodes, in order). Episode ``i`` holds the frames
    ``[starts[i], ends[i])``, the ``dataset_from_index`` and ``dataset_to_index`` of its metadata.
    """
    frame_index = np.asarray(frame_index).reshape(-1)
    starts, ends = np.asarray(starts).reshape(-1), np.asarray(ends).reshape(-1)
    if frame_index.size > 1 and not bool(np.all(np.diff(frame_index) > 0)):
        raise ValueError("the frame index of the table must increase")
    lo = np.searchsorted(frame_index, starts)
    hi = np.searchsorted(frame_index, ends)
    if np.any(hi - lo != ends - starts):
        raise ValueError("the table does not hold every frame of the selected episodes")
    return [slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True)]


def _rows(table: pa.Table, key: str) -> np.ndarray:
    """The column ``key`` of an Arrow table as a numpy array, one row per frame (no per-row conversion)."""
    column = table.column(key).combine_chunks()
    if pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type):
        values = column.flatten().to_numpy(zero_copy_only=False)
        if len(column) and values.size % len(column):
            raise ValueError(f"{key} has rows of different lengths")
        return values.reshape(len(column), -1)
    return column.to_numpy(zero_copy_only=False)


def load_episodes(
    repo_id: str,
    *,
    state_key: str,
    action_key: str,
    root: str | None = None,
    revision: str | None = None,
    episodes: list[int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The ``(observed7, commanded7)`` rows of each episode of a LeRobot dataset, in episode order.

    The columns are read once, as Arrow; each episode is a slice of them given by the frame range
    that the episode metadata stores. No video is downloaded or read.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root, revision=revision, episodes=episodes, download_videos=False)
    table = dataset.hf_dataset.with_format("arrow").select_columns(["index", state_key, action_key])[:]
    state = _rows(table, state_key).astype(np.float32, copy=False)
    action = _rows(table, action_key).astype(np.float32, copy=False)
    if state.shape[1:] != (7,) or action.shape[1:] != (7,):
        raise ValueError(
            f"{state_key} and {action_key} must hold 7 values per frame, got {state.shape[1:]} and {action.shape[1:]}"
        )
    meta = dataset.meta.episodes.with_format("arrow").select_columns(
        ["episode_index", "dataset_from_index", "dataset_to_index"]
    )[:]
    episode_index = _rows(meta, "episode_index")
    keep = np.isin(episode_index, episodes) if episodes is not None else np.ones(len(episode_index), bool)
    rows = np.flatnonzero(keep)[np.argsort(episode_index[keep], kind="stable")]
    slices = episode_slices(
        _rows(table, "index"),
        _rows(meta, "dataset_from_index")[rows],
        _rows(meta, "dataset_to_index")[rows],
    )
    out = [(state[s], action[s]) for s in slices]
    logger.info("%d episodes, %d frames", len(out), len(state))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--episodes", default=None, help="a JSON list of episode indices (default: all)")
    parser.add_argument("--state-key", default="observation.state.cartesian")
    parser.add_argument("--action-key", default="action.cartesian")
    parser.add_argument("--n-windows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    selected = json.loads(Path(args.episodes).read_text()) if args.episodes else None
    episodes = load_episodes(
        args.repo_id,
        state_key=args.state_key,
        action_key=args.action_key,
        root=args.root,
        revision=args.revision,
        episodes=selected,
    )
    stats = sample_window_stats(episodes, args.n_windows, args.seed)
    stats["domain"] = args.repo_id
    Path(args.out).write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {args.out}: {args.n_windows} windows of {len(episodes)} episodes")


if __name__ == "__main__":
    main()
