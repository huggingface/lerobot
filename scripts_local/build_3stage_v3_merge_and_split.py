"""Merge sim_3stage_v3_success + sim_3stage_v3_failures → with_partials.
Then frame-stride split (N=10) → train_fs / val_fs.

Sources:
  - domrachev03/sim_3stage_v3_success (59 full-success eps, 6/6)
  - domrachev03/sim_3stage_v3_failures (31 partials: 5 each 0/6-5/6, 6 in 2/6)

Outputs:
  - domrachev03/sim_3stage_v3_with_partials (90 eps)
  - domrachev03/sim_3stage_v3_train_fs
  - domrachev03/sim_3stage_v3_val_fs

Then patches subtask cols on splits + writes temporal_proportions.
"""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME, DONE, REWARD
import torch  # noqa: F401

SRC_SUCCESS = "domrachev03/sim_3stage_v3_success"
SRC_PARTIALS = "domrachev03/sim_3stage_v3_failures"
SRC_PARTIALS_FILTERED = "domrachev03/sim_3stage_v3_failures_clean"
DROP_PARTIAL_EPS = [21]  # extra 2/6 ep — bucket should be 5 not 6
OUT_FULL = "domrachev03/sim_3stage_v3_with_partials"
OUT_TRAIN = "domrachev03/sim_3stage_v3_train_fs"
OUT_VAL = "domrachev03/sim_3stage_v3_val_fs"
META_OUT = Path("outputs/sim_3stage_v3_merge_split_meta.json")
N_STRIDE = 10

_SKIP_KEYS = {"task_index", "timestamp", "episode_index", "frame_index", "index", "task"}


def _build_new_frame(frame: dict) -> dict:
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


def merge() -> int:
    full_root = HF_LEROBOT_HOME / OUT_FULL
    if full_root.exists():
        print(f"removing existing {full_root}")
        shutil.rmtree(full_root)
    ds_s = LeRobotDataset(repo_id=SRC_SUCCESS)

    # Drop accidental extra ep(s) from failures source.
    if DROP_PARTIAL_EPS:
        clean_root = HF_LEROBOT_HOME / SRC_PARTIALS_FILTERED
        if clean_root.exists():
            shutil.rmtree(clean_root)
        ds_p_full = LeRobotDataset(repo_id=SRC_PARTIALS)
        print(f"  src_partials_full: {ds_p_full.num_episodes} eps {ds_p_full.num_frames} fr → drop {DROP_PARTIAL_EPS}")
        ds_p = delete_episodes(
            dataset=ds_p_full,
            episode_indices=list(DROP_PARTIAL_EPS),
            repo_id=SRC_PARTIALS_FILTERED,
            output_dir=clean_root,
        )
        # delete_episodes does not preserve sparse_subtask_* cols — patch them back.
        src_meta = pd.concat([
            pq.read_table(f).to_pandas()
            for f in sorted((HF_LEROBOT_HOME / SRC_PARTIALS / "meta" / "episodes").rglob("*.parquet"))
        ], ignore_index=True).sort_values("episode_index").reset_index(drop=True)
        kept = src_meta[~src_meta["episode_index"].isin(DROP_PARTIAL_EPS)].reset_index(drop=True)
        dst_meta_path = clean_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        dst = pq.read_table(dst_meta_path).to_pandas().sort_values("episode_index").reset_index(drop=True)
        for col in ("sparse_subtask_names", "sparse_subtask_start_frames", "sparse_subtask_end_frames",
                    "dense_subtask_names", "dense_subtask_start_frames", "dense_subtask_end_frames"):
            if col in kept.columns:
                dst[col] = kept[col].values
        dst.to_parquet(dst_meta_path, index=False)
        # Reload with patched meta so num_episodes/num_frames reflect filtered ds.
        ds_p = LeRobotDataset(repo_id=SRC_PARTIALS_FILTERED, root=clean_root)
    else:
        ds_p = LeRobotDataset(repo_id=SRC_PARTIALS)

    print(f"  src_success: {ds_s.num_episodes} eps {ds_s.num_frames} fr")
    print(f"  src_partials: {ds_p.num_episodes} eps {ds_p.num_frames} fr")
    merged = merge_datasets(datasets=[ds_s, ds_p], output_repo_id=OUT_FULL)
    print(f"  merged: {merged.num_episodes} eps {merged.num_frames} fr")
    return merged.num_episodes


def split() -> tuple[int, int]:
    src = LeRobotDataset(repo_id=OUT_FULL)
    _meta_files = sorted((src.root / "meta" / "episodes").rglob("*.parquet"))
    src_meta = pd.concat([pq.read_table(p).to_pandas() for p in _meta_files], ignore_index=True)
    train_root = HF_LEROBOT_HOME / OUT_TRAIN
    val_root = HF_LEROBOT_HOME / OUT_VAL
    for d in (train_root, val_root):
        if d.exists():
            shutil.rmtree(d)

    common = dict(
        fps=int(src.fps),
        robot_type=src.meta.robot_type,
        features=src.meta.info["features"],
        use_videos=len(src.meta.video_keys) > 0,
    )
    train_ds = LeRobotDataset.create(repo_id=OUT_TRAIN, root=train_root, **common)
    val_ds = LeRobotDataset.create(repo_id=OUT_VAL, root=val_root, **common)

    new_subtasks = {"train": [], "val": []}
    n_train_eps = 0
    n_val_eps = 0
    src_meta_by_ep = {int(r["episode_index"]): r for _, r in src_meta.iterrows()}

    pbar = tqdm(total=len(src), desc="splitting")
    cur_ep = -1
    train_in_ep = 0
    val_in_ep = 0
    train_orig_to_new: list[int] = []
    val_orig_to_new: list[int] = []
    train_buf = False
    val_buf = False

    def _flush_ep(ep_idx: int) -> None:
        nonlocal train_in_ep, val_in_ep, train_buf, val_buf, n_train_eps, n_val_eps
        if not train_buf and not val_buf:
            return
        meta_row = src_meta_by_ep[ep_idx]
        names = meta_row.get("sparse_subtask_names")
        starts = meta_row.get("sparse_subtask_start_frames")
        ends = meta_row.get("sparse_subtask_end_frames")
        names = list(names) if names is not None else []
        starts = list(starts) if starts is not None else []
        ends = list(ends) if ends is not None else []

        def _adjust(orig_list, mapping):
            adj = []
            for f in orig_list:
                best = -1
                for i in range(min(int(f), len(mapping) - 1), -1, -1):
                    if mapping[i] >= 0:
                        best = mapping[i]
                        break
                adj.append(int(best))
            return adj

        if train_buf:
            tr_starts = _adjust(starts, train_orig_to_new)
            tr_ends = _adjust(ends, train_orig_to_new)
            new_subtasks["train"].append({
                "sparse_subtask_names": names if names else None,
                "sparse_subtask_start_frames": tr_starts if names else None,
                "sparse_subtask_end_frames": tr_ends if names else None,
                "dense_subtask_names": names if names else None,
                "dense_subtask_start_frames": tr_starts if names else None,
                "dense_subtask_end_frames": tr_ends if names else None,
            })
            train_ds.save_episode()
            train_buf = False
            n_train_eps += 1

        if val_buf:
            vl_starts = _adjust(starts, val_orig_to_new)
            vl_ends = _adjust(ends, val_orig_to_new)
            new_subtasks["val"].append({
                "sparse_subtask_names": names if names else None,
                "sparse_subtask_start_frames": vl_starts if names else None,
                "sparse_subtask_end_frames": vl_ends if names else None,
                "dense_subtask_names": names if names else None,
                "dense_subtask_start_frames": vl_starts if names else None,
                "dense_subtask_end_frames": vl_ends if names else None,
            })
            val_ds.save_episode()
            val_buf = False
            n_val_eps += 1

    for global_idx in range(len(src)):
        frame = src[global_idx]
        ep_idx = int(frame["episode_index"].item())
        if ep_idx != cur_ep:
            if cur_ep >= 0:
                _flush_ep(cur_ep)
            cur_ep = ep_idx
            train_in_ep = 0
            val_in_ep = 0
            train_orig_to_new = []
            val_orig_to_new = []

        frame_in_ep = len(train_orig_to_new)
        is_val = (frame_in_ep % N_STRIDE) == (N_STRIDE - 1)
        new_frame = _build_new_frame(frame)
        new_frame["task"] = frame.get("task", "")
        if is_val:
            val_ds.add_frame(new_frame)
            val_orig_to_new.append(val_in_ep)
            train_orig_to_new.append(-1)
            val_in_ep += 1
            val_buf = True
        else:
            train_ds.add_frame(new_frame)
            train_orig_to_new.append(train_in_ep)
            val_orig_to_new.append(-1)
            train_in_ep += 1
            train_buf = True
        pbar.update(1)

    _flush_ep(cur_ep)
    pbar.close()
    train_ds.finalize()
    val_ds.finalize()

    print(f"  train: {n_train_eps} eps {train_ds.num_frames} fr")
    print(f"  val:   {n_val_eps} eps {val_ds.num_frames} fr")

    for split_name, ds in (("train", train_ds), ("val", val_ds)):
        path = ds.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        df = pq.read_table(path).to_pandas()
        rows = new_subtasks[split_name]
        for col in ("sparse_subtask_names", "sparse_subtask_start_frames", "sparse_subtask_end_frames",
                    "dense_subtask_names", "dense_subtask_start_frames", "dense_subtask_end_frames"):
            df[col] = [r[col] for r in rows]
        df.to_parquet(path, index=False)
    return n_train_eps, train_ds.num_frames, n_val_eps, val_ds.num_frames


def main() -> None:
    print("== merge ==")
    n_total = merge()
    print("== split ==")
    n_tr_eps, n_tr_fr, n_vl_eps, n_vl_fr = split()
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps({
        "src_success": SRC_SUCCESS,
        "src_partials": SRC_PARTIALS,
        "out_full": OUT_FULL,
        "stride_n": N_STRIDE,
        "n_total_eps": n_total,
        "n_train_eps": n_tr_eps,
        "n_train_frames": n_tr_fr,
        "n_val_eps": n_vl_eps,
        "n_val_frames": n_vl_fr,
        "train_repo_id": OUT_TRAIN,
        "val_repo_id": OUT_VAL,
    }, indent=2))
    print(f"  meta → {META_OUT}")


if __name__ == "__main__":
    main()
