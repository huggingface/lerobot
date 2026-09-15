#!/usr/bin/env python
"""Derive a per-frame "subgoal state" label for `lerobot/vlabench_unified` from data the dataset
already has -- no VLABench simulator access, no seeds, no new data collection.

Why: SafeDiff-VLA's state-predictor head needs a real target to predict (see
`lerobot.policies.safediff_vla`), and true object positions aren't recoverable for existing
episodes (VLABench's per-episode object layout isn't reproducible from `env.reset(seed=...)` --
verified empirically: the same seed gives a different target object and position on repeated
resets). But every episode's own `action` gripper channel already tells us, for free, exactly
when the demonstrated pick/place actually happens: the frame where it transitions open->close is
the grasp point, close->open is the release/place point. The `observation.state` at that frame is
a strong, zero-cost proxy for "where the arm needed to go" -- arguably more useful than a raw
object position, since it's literally where the expert-collected demonstration went to interact
with something.

For every frame t in an episode, the label is `observation.state` at the *next* gripper
transition after t (or the episode's last frame, if there is none) -- i.e. "what sub-goal is
this frame currently working towards".

Usage:
    uv run python examples/safediff_vla/compute_subgoal_labels.py
    uv run python examples/safediff_vla/compute_subgoal_labels.py --out outputs/data/vlabench_subgoal_labels/labels.parquet

Output columns: `index` (global row index, matches the dataset's own `index` feature),
`episode_index`, `frame_index`, `subgoal_state` (list[float], same width as `observation.state`),
`steps_to_subgoal` (int, diagnostic only).
"""

import argparse

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "lerobot/vlabench_unified"
GRIPPER_CHANNEL = 6
# Action channel is close to 0/1 but not exactly (see docs/source/vlabench.mdx quantiles) --
# open when the recent history sits above this, closed below it.
GRIPPER_OPEN_THRESHOLD = 0.5


def find_subgoal_indices(gripper: np.ndarray) -> np.ndarray:
    """For each frame, the index of the next gripper open<->close transition (or the last frame)."""
    is_open = gripper > GRIPPER_OPEN_THRESHOLD
    transitions = np.flatnonzero(is_open[1:] != is_open[:-1]) + 1  # frame indices where state flips
    n = len(gripper)
    if len(transitions) == 0:
        # No gripper transition anywhere in this episode (e.g. it never grasps): every frame's
        # subgoal defaults to the episode's last frame.
        return np.full(n, n - 1, dtype=np.int64)
    # For each t, the label is the first transition index strictly greater than t, or n - 1 if
    # none remain (episode's last frame -- the end of the demonstrated pick-then-place).
    positions = np.searchsorted(transitions, np.arange(n), side="right")
    subgoal = np.where(
        positions < len(transitions), transitions[np.minimum(positions, len(transitions) - 1)], n - 1
    )
    return subgoal.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=str, default="outputs/data/vlabench_subgoal_labels/labels.parquet")
    args = parser.parse_args()

    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    data_files = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))
    print(f"Found {len(data_files)} data files: {data_files}")

    frames = []
    for f in data_files:
        path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f)
        frames.append(
            pd.read_parquet(
                path, columns=["index", "episode_index", "frame_index", "observation.state", "action"]
            )
        )
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} frames across {df['episode_index'].nunique()} episodes")

    out_rows = []
    for episode_index, group in df.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        actions = np.stack(group["action"].to_numpy())
        states = np.stack(group["observation.state"].to_numpy())
        gripper = actions[:, GRIPPER_CHANNEL]
        subgoal_idx = find_subgoal_indices(gripper)
        frame_indices = group["frame_index"].to_numpy()
        global_indices = group["index"].to_numpy()
        for row_pos in range(len(group)):
            out_rows.append(
                {
                    "index": int(global_indices[row_pos]),
                    "episode_index": int(episode_index),
                    "frame_index": int(frame_indices[row_pos]),
                    "subgoal_state": states[subgoal_idx[row_pos]].tolist(),
                    "steps_to_subgoal": int(subgoal_idx[row_pos] - row_pos),
                }
            )

    out_df = pd.DataFrame(out_rows)
    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")
    print(out_df["steps_to_subgoal"].describe())


if __name__ == "__main__":
    main()
