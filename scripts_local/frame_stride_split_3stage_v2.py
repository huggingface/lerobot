"""Frame-stride train/val split for the 3-stage SARM v2 dataset (158 eps = 100 succ + 58 partials).

Per-ep stride: every N-th frame (frame_in_ep % N == N-1) goes to val.
Re-encodes videos. Subtask annotations regenerated per-ep based on
counting train/val frames within each subtask range.

Source: domrachev03/sim_3stage_with_partials_v2
Outputs:
  - domrachev03/sim_3stage_v2_train_fs
  - domrachev03/sim_3stage_v2_val_fs
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME, DONE, REWARD
import torch  # noqa: F401

SRC = "domrachev03/sim_3stage_with_partials_v2"
N_STRIDE = 10
OUT_TRAIN = "domrachev03/sim_3stage_v2_train_fs"
OUT_VAL = "domrachev03/sim_3stage_v2_val_fs"
META_OUT = Path("outputs/sim_3stage_v2_split_meta.json")

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


def main() -> None:
    src = LeRobotDataset(repo_id=SRC)
    import pandas as _pd
    _meta_files = sorted((src.root / "meta" / "episodes").rglob("*.parquet"))
    src_meta_rows = _pd.concat([pq.read_table(p).to_pandas() for p in _meta_files], ignore_index=True)
    print(f"src: {src.num_episodes} eps, {src.num_frames} frames")

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

    src_meta_by_ep = {int(r["episode_index"]): r for _, r in src_meta_rows.iterrows()}

    pbar = tqdm(total=len(src), desc="splitting frames")
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

    print(f"  train: {n_train_eps} eps, {train_ds.num_frames} frames")
    print(f"  val:   {n_val_eps} eps, {val_ds.num_frames} frames")

    for split_name, ds in (("train", train_ds), ("val", val_ds)):
        path = ds.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        df = pq.read_table(path).to_pandas()
        rows = new_subtasks[split_name]
        for col in ("sparse_subtask_names", "sparse_subtask_start_frames", "sparse_subtask_end_frames",
                    "dense_subtask_names", "dense_subtask_start_frames", "dense_subtask_end_frames"):
            df[col] = [r[col] for r in rows]
        df.to_parquet(path, index=False)
        print(f"  patched subtask cols in {path}")

    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps({
        "src_repo_id": SRC,
        "stride_n": N_STRIDE,
        "n_train_eps": n_train_eps,
        "n_val_eps": n_val_eps,
        "n_train_frames": train_ds.num_frames,
        "n_val_frames": val_ds.num_frames,
        "train_repo_id": OUT_TRAIN,
        "val_repo_id": OUT_VAL,
    }, indent=2))
    print(f"  meta → {META_OUT}")


if __name__ == "__main__":
    main()
