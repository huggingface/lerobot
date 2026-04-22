#!/usr/bin/env python
"""Prepare SARM training data from success + failure demonstration datasets.

Merges two cropped+resized LeRobotDatasets (success + failures), adds
``dense_only`` subtask annotations, writes temporal proportions, splits
into train/val, and propagates annotations.

Output trio:
    <output>                (combined, annotated)
    <output>-train          (75% frames, annotated)
    <output>-val            (25% frames, annotated)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.reward_model.split_dataset import split_dataset
from lerobot.utils.constants import DONE, REWARD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SKIP_KEYS = {"task_index", "timestamp", "episode_index", "frame_index", "index", "task"}


def _build_frame(frame: dict) -> dict:
    out: dict = {}
    for k, v in frame.items():
        if k in _SKIP_KEYS:
            continue
        if k in (DONE, REWARD):
            v = v.unsqueeze(0) if hasattr(v, "unsqueeze") else v
        if k.startswith("complementary_info") and hasattr(v, "dim") and v.dim() == 0:
            v = v.unsqueeze(0)
        out[k] = v
    return out


def _annotate_episodes(
    root: Path,
    n_success: int,
    n_total: int,
    fps: float,
    idle_proportion: float = 0.001,
) -> None:
    meta_dir = root / "meta"
    tp_dense = {"idle": idle_proportion, "task": 1.0 - idle_proportion}
    tp_sparse = {"task": 1.0}
    (meta_dir / "temporal_proportions_dense.json").write_text(json.dumps(tp_dense, indent=2))
    (meta_dir / "temporal_proportions_sparse.json").write_text(json.dumps(tp_sparse, indent=2))

    pq_path = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    if not pq_path.exists():
        candidates = list((meta_dir / "episodes").rglob("*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No episode parquet found under {meta_dir / 'episodes'}")
        pq_path = candidates[0]

    df = pd.read_parquet(pq_path)
    names, starts, ends, st, et = [], [], [], [], []
    for i, row in df.iterrows():
        ep_len = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
        label = "task" if i < n_success else "idle"
        names.append([label])
        starts.append([0])
        ends.append([ep_len - 1])
        st.append([0.0])
        et.append([(ep_len - 1) / fps])

    df["dense_subtask_names"] = names
    df["dense_subtask_start_frames"] = starts
    df["dense_subtask_end_frames"] = ends
    df["dense_subtask_start_times"] = st
    df["dense_subtask_end_times"] = et
    df.to_parquet(pq_path)
    logger.info("Annotated %d episodes (%d task + %d idle) in %s",
                len(df), n_success, len(df) - n_success, pq_path)


def prepare(
    success_repo_id: str,
    failure_repo_id: str,
    output_repo_id: str | None = None,
    idle_proportion: float = 0.001,
    val_stride: int = 4,
) -> None:
    ds_s = LeRobotDataset(success_repo_id)
    ds_f = LeRobotDataset(failure_repo_id)
    assert ds_s.fps == ds_f.fps, f"FPS mismatch: {ds_s.fps} vs {ds_f.fps}"

    out_id = output_repo_id or f"{success_repo_id}_with_failures"
    out_root = Path(str(ds_s.root).rsplit("_cropped", 1)[0] + "_sarm_combined")
    if out_root.exists():
        logger.warning("Removing existing output at %s", out_root)
        shutil.rmtree(out_root)

    out_ds = LeRobotDataset.create(
        repo_id=out_id, root=out_root, fps=int(ds_s.fps),
        robot_type=ds_s.meta.robot_type,
        features=ds_s.meta.info["features"],
        use_videos=len(ds_s.meta.video_keys) > 0,
    )

    for label, ds in [("success", ds_s), ("failure", ds_f)]:
        for ep_idx in tqdm(range(ds.num_episodes), desc=f"{label} eps"):
            ep = ds.meta.episodes[ep_idx]
            s, e = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            for fidx in range(s, e):
                frame = ds[fidx]
                nf = _build_frame(frame)
                nf["task"] = frame.get("task", "")
                out_ds.add_frame(nf)
            out_ds.save_episode()

    out_ds.finalize()
    n_success = ds_s.num_episodes
    n_total = out_ds.num_episodes

    _annotate_episodes(out_root, n_success, n_total, ds_s.fps, idle_proportion)

    train_root = Path(str(out_root) + "-train")
    val_root = Path(str(out_root) + "-val")
    for p in (train_root, val_root):
        if p.exists():
            shutil.rmtree(p)

    train_ds, val_ds = split_dataset(
        src_repo_id=out_id,
        src_root=str(out_root),
        train_repo_id=f"{out_id}-train",
        val_repo_id=f"{out_id}-val",
        val_stride=val_stride,
    )

    for _suffix, split_root in [("-train", train_root), ("-val", val_root)]:
        for fn in ("temporal_proportions_dense.json", "temporal_proportions_sparse.json"):
            src = out_root / "meta" / fn
            dst = split_root / "meta" / fn
            shutil.copy2(src, dst)
        _annotate_episodes(split_root, n_success, n_total, ds_s.fps, idle_proportion)

    print(
        f"\nSARM data prepared:\n"
        f"  success eps: {n_success} ({success_repo_id})\n"
        f"  failure eps: {ds_f.num_episodes} ({failure_repo_id})\n"
        f"  combined:    {out_root}\n"
        f"  train:       {train_root}\n"
        f"  val:         {val_root}\n"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--success-repo-id", required=True)
    p.add_argument("--failure-repo-id", required=True)
    p.add_argument("--output-repo-id", default=None)
    p.add_argument("--idle-proportion", type=float, default=0.001)
    p.add_argument("--val-stride", type=int, default=4)
    args = p.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
