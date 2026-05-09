r"""Build merged_v2 (filtered + _2 + _3, drop _3 eps {2,8,11,12}) and stratified train/val.

Strategy:
  1. Merge all 3 sources INCLUDING bad/dropped _3 eps → merged_v2_full (133 eps).
     (delete_episodes hits a video-length assertion on _3 eps 2/8/12 due to recording
      artifacts. aggregate path bypasses that.)
  2. Compute per-ep bucket from preserved sparse_subtask_names in merged meta.
  3. Stratified per-(bucket × source) split, EXCLUDING dropped eps from both train + val.

Outputs:
  - local/sim_assemble_sarm_merged_v2_full   (133 eps; canonical, kept for stats)
  - local/sim_assemble_sarm_merged_v2_train  (train split)
  - local/sim_assemble_sarm_merged_v2_val    (val split)
  - outputs/iter6_split_meta.json
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.dataset_tools import merge_datasets, split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SEED = 1000
SRC_FILTERED = "domrachev03/sim_assemble_sarm_multistage_two_stages_filtered"  # 51 eps
SRC_2 = "domrachev03/sim_assemble_sarm_multistage_two_stages_2"  # 52 eps
SRC_3 = "domrachev03/sim_assemble_sarm_multistage_two_stages_3"  # 30 eps
DROP_3_LOCAL = [2, 8, 11, 12]  # local _3 ep ids to drop (ep11 per user; 2/8/12 = video-length artifact)

OUT_FULL = "local/sim_assemble_sarm_merged_v2_full"
META_OUT = Path("outputs/iter6_split_meta.json")


def _bucket_of(sn) -> int:
    return 0 if sn is None else len(list(sn))


def _read_meta(ds_root: Path):
    rows = []
    for f in sorted((ds_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(f).to_pandas().to_dict("records"))
    return rows


def _stratified_val(eps: list[dict], val_frac: float, rng: np.random.Generator) -> set[int]:
    by_b: dict[int, list[int]] = {}
    for e in eps:
        by_b.setdefault(e["bucket"], []).append(e["merged_idx"])
    val: set[int] = set()
    for b, ids in by_b.items():
        n_val = max(1, round(len(ids) * val_frac))
        chosen = rng.choice(ids, size=n_val, replace=False)
        val.update(int(x) for x in chosen)
    return val


def main() -> None:
    print("[1/4] merge filtered + _2 + _3 (all 30 eps) ...")
    ds_f = LeRobotDataset(repo_id=SRC_FILTERED)
    ds_2 = LeRobotDataset(repo_id=SRC_2)
    ds_3 = LeRobotDataset(repo_id=SRC_3)
    n_f, n_2, n_3 = ds_f.num_episodes, ds_2.num_episodes, ds_3.num_episodes
    print(f"  source eps: filtered={n_f} _2={n_2} _3={n_3}")

    full_root = HF_LEROBOT_HOME / OUT_FULL
    if full_root.exists():
        shutil.rmtree(full_root)
    merged = merge_datasets([ds_f, ds_2, ds_3], output_repo_id=OUT_FULL, output_dir=full_root)
    print(f"  merged_v2_full: {merged.num_episodes} eps, {merged.num_frames} fr")

    # merge_datasets has an off-by-one: 2 boundary eps get meta/episodes/file_index=1
    # but only file-000.parquet exists. Patch in place so split_dataset can find them.
    import pandas as pd
    meta_path = full_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df_meta = pq.read_table(meta_path).to_pandas()
    bad = df_meta["meta/episodes/file_index"] != 0
    if bad.any():
        print(f"  patching {int(bad.sum())} stale file_index pointers → 0")
        df_meta["meta/episodes/file_index"] = 0
        df_meta["meta/episodes/chunk_index"] = 0
        df_meta.to_parquet(meta_path, index=False)
        # Reload merged so meta pointers are fresh.
        merged = LeRobotDataset(repo_id=OUT_FULL, root=full_root)

    print("[2/4] read merged meta + assign source/bucket ...")
    meta_rows = _read_meta(merged.root)
    drop_merged = {n_f + n_2 + i for i in DROP_3_LOCAL}
    eps_by_src = {"filtered": [], "_2": [], "_3": []}
    for r in meta_rows:
        i = int(r["episode_index"])
        if i in drop_merged:
            continue
        if i < n_f:
            src = "filtered"
        elif i < n_f + n_2:
            src = "_2"
        else:
            src = "_3"
        eps_by_src[src].append(dict(merged_idx=i, src=src, bucket=_bucket_of(r["sparse_subtask_names"]),
                                     length=int(r["length"])))

    print(f"  kept eps: filtered={len(eps_by_src['filtered'])} _2={len(eps_by_src['_2'])} _3={len(eps_by_src['_3'])}")
    for src in ("filtered", "_2", "_3"):
        b = {}
        for e in eps_by_src[src]:
            b[e["bucket"]] = b.get(e["bucket"], 0) + 1
        print(f"  {src} bucket counts: {sorted(b.items())}")

    print("[3/4] stratified split per (bucket × source) ...")
    rng = np.random.default_rng(SEED)
    val_f = _stratified_val(eps_by_src["filtered"], 0.20, rng)
    val_2 = _stratified_val(eps_by_src["_2"], 0.20, rng)
    val_3 = _stratified_val(eps_by_src["_3"], 0.22, rng)

    val_ids = sorted(val_f | val_2 | val_3)
    new_val_ids = sorted(val_3)
    all_kept = sorted({e["merged_idx"] for L in eps_by_src.values() for e in L})
    train_ids = sorted(set(all_kept) - set(val_ids))
    print(f"  total kept: {len(all_kept)}  val: {len(val_ids)} (filtered={len(val_f)} _2={len(val_2)} _3={len(val_3)})  train: {len(train_ids)}")

    # Verify val bucket coverage
    bucket_of = {int(r["episode_index"]): _bucket_of(r["sparse_subtask_names"]) for r in meta_rows}
    val_buckets: dict[int, int] = {}
    for i in val_ids:
        b = bucket_of[i]
        val_buckets[b] = val_buckets.get(b, 0) + 1
    print(f"  val bucket counts: {sorted(val_buckets.items())}")

    print("[4/4] write train + val splits ...")
    for sub in ("sim_assemble_sarm_merged_v2_full_train", "sim_assemble_sarm_merged_v2_full_val"):
        d = HF_LEROBOT_HOME / "local" / sub
        if d.exists():
            shutil.rmtree(d)
    splits = split_dataset(dataset=merged, splits={"train": train_ids, "val": val_ids})
    print(f"  train @ {splits['train'].root}: {splits['train'].num_episodes} eps, {splits['train'].num_frames} fr")
    print(f"  val   @ {splits['val'].root}: {splits['val'].num_episodes} eps, {splits['val'].num_frames} fr")

    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps({
        "seed": SEED,
        "merged_v2_full": OUT_FULL,
        "train_repo_id": "local/sim_assemble_sarm_merged_v2_full_train",
        "val_repo_id": "local/sim_assemble_sarm_merged_v2_full_val",
        "drop_3_local": DROP_3_LOCAL,
        "drop_merged_idx": sorted(drop_merged),
        "n_full": merged.num_episodes,
        "n_kept": len(all_kept),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "val_ep_ids_in_full": val_ids,
        "train_ep_ids_in_full": train_ids,
        "new_val_ep_ids_in_full": new_val_ids,
        "val_bucket_counts": val_buckets,
        "source_offsets": {"filtered": 0, "_2": n_f, "_3": n_f + n_2},
    }, indent=2))
    print(f"  meta → {META_OUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
