#!/usr/bin/env python
"""LIBERO grasp-transition detection audit (see `adapter.py`'s
`libero_find_grasp_target`/`_libero_action_gripper_is_open`/`_libero_state_gripper_is_open`).

Runs the LIBERO-scoped transition detector over a real 20-50 episode sample spread across several
tasks, reports the detected first open->close transition index per episode, and extracts REAL
dataset video frames (not simulation) around a handful of them for visual confirmation --
before/at/after each detected transition -- so the transition label can be checked against what
the gripper is actually doing on screen, per the "20~50 episode, 영상/프레임으로 확인" requirement.

Does not touch `lerobot.policies.safediff_vla.utils.find_grasp_target` or any global registry.

Usage:
    uv run python examples/safediff_vla/libero_smoke_test/grasp_transition_audit.py \
        --dataset libero --n-episodes 40 --n-video-checks 6 \
        --output-dir outputs/libero_smoke_test/grasp_transition_audit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adapter import _libero_action_gripper_is_open, _libero_state_gripper_is_open

REPO_IDS = {"libero": "lerobot/libero", "libero_plus": "lerobot/libero_plus"}


def detect_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        a = torch.as_tensor(np.stack(g["action"].values))
        s = torch.as_tensor(np.stack(g["observation.state"].values))
        half_diff_state6 = (s[:, 6] - s[:, 7]) / 2.0  # raw (pre-collapse) state, same formula
        state_open_now = _libero_state_gripper_is_open(half_diff_state6[0:1])
        action_open = _libero_action_gripper_is_open(a[:, 6])
        is_open_seq = torch.cat([state_open_now, action_open])
        transitions = is_open_seq[:-1] & ~is_open_seq[1:]
        has_transition = bool(transitions.any())
        idx = int(transitions.to(torch.float32).argmax()) + 1 if has_transition else None
        rows.append(
            {
                "episode_index": int(ep),
                "task_index": int(g["task_index"].iloc[0]) if "task_index" in g else None,
                "n_steps": len(g),
                "has_transition": has_transition,
                "transition_idx": idx,
                "grip_before": float(a[idx - 1, 6]) if idx else None,
                "grip_after": float(a[idx, 6]) if idx else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("libero", "libero_plus"), required=True)
    parser.add_argument("--n-episodes", type=int, default=40)
    parser.add_argument("--n-video-checks", type=int, default=6)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_id = REPO_IDS[args.dataset]

    from huggingface_hub import HfFileSystem
    import pyarrow.dataset as ds

    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{repo_id}/data/**/*.parquet"))
    dset = ds.dataset(paths, filesystem=fs, format="parquet")
    # Sample the first N distinct episodes per a spread of task_index values for task diversity.
    tbl = dset.to_table(columns=["episode_index", "task_index", "timestamp", "observation.state", "action"])
    df = tbl.to_pandas()
    all_eps = df[["episode_index", "task_index"]].drop_duplicates()
    # round-robin across tasks for diversity, capped at n_episodes total
    chosen = (
        all_eps.groupby("task_index", group_keys=False)
        .apply(lambda g: g.head(2))
        .sort_values(["task_index", "episode_index"])
        .head(args.n_episodes)["episode_index"]
        .tolist()
    )
    df = df[df.episode_index.isin(chosen)]

    result = detect_transitions(df)
    result.to_csv(out_dir / "detected_transitions.csv", index=False)
    n_valid = int(result.has_transition.sum())
    print(f"[audit] dataset={args.dataset} n_episodes={len(result)} n_with_transition={n_valid} "
          f"({100 * n_valid / len(result):.1f}%)")
    print(result.to_string())

    # Extract real video frames for a handful, spread across different tasks.
    valid = result[result.has_transition].copy()
    picks = valid.drop_duplicates("task_index").head(args.n_video_checks)
    if picks.empty:
        print("[audit] no valid transitions to visually check")
        return

    from huggingface_hub import hf_hub_download

    ep_meta = pd.read_parquet(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/episodes/chunk-000/file-000.parquet"))
    img_key = "observation.images.image" if args.dataset == "libero" else "observation.images.front"
    fps = 10 if args.dataset == "libero" else 20

    for _, row in picks.iterrows():
        ep = int(row.episode_index)
        t = int(row.transition_idx)
        meta_row = ep_meta[ep_meta.episode_index == ep].iloc[0]
        chunk = int(meta_row[f"videos/{img_key}/chunk_index"])
        file_idx = int(meta_row[f"videos/{img_key}/file_index"])
        from_ts = float(meta_row[f"videos/{img_key}/from_timestamp"])
        video_path = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=f"videos/{img_key}/chunk-{chunk:03d}/file-{file_idx:03d}.mp4",
        )
        import subprocess

        for label, dt in (("before", -1.0), ("transition", 0.0), ("after", 1.0)):
            ts = from_ts + t / fps + dt
            out_png = out_dir / f"ep{ep}_task{int(row.task_index)}_t{t}_{label}.png"
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path, "-frames:v", "1", str(out_png)],
                capture_output=True,
            )
        print(f"[audit] wrote frames for episode={ep} task={int(row.task_index)} transition_idx={t}")


if __name__ == "__main__":
    main()
