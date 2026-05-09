"""Build merged_v3 = filtered + _2 + only 3/4-plateau eps from _3 (local 25-29).

Step 1: delete_episodes(_3, [0..24])  → 5 eps remain (orig 25..29 → new 0..4)
Step 2: merge_datasets([filtered, _2, _3_3of4])  → merged_v3_full (108 eps)
Step 3: frame-stride split (every 10th frame → val) using existing splitter logic

Outputs:
  - local/_sim_assemble_sarm_3_3of4_only       (intermediate, 5 eps from _3 ep 25-29)
  - local/sim_assemble_sarm_merged_v3_full     (108 eps; canonical)
  - local/sim_assemble_sarm_merged_v3_train_fs (frame-stride train)
  - local/sim_assemble_sarm_merged_v3_val_fs   (frame-stride val)
  - outputs/iter7_split_meta.json
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC_FILTERED = "domrachev03/sim_assemble_sarm_multistage_two_stages_filtered"  # 51 eps
SRC_2 = "domrachev03/sim_assemble_sarm_multistage_two_stages_2"  # 52 eps
SRC_3 = "domrachev03/sim_assemble_sarm_multistage_two_stages_3"  # 30 eps; keep ONLY 25-29

# Local _3 ep ids to DELETE (keep only 25-29, the 3/4-plateau timeouts)
DROP_3_LOCAL = list(range(25))  # [0..24]

OUT_3OF4 = "local/_sim_assemble_sarm_3_3of4_only"
OUT_FULL = "local/sim_assemble_sarm_merged_v3_full"
META_OUT = Path("outputs/iter7_split_meta.json")


def main() -> None:
    print("[1/4] load source ds + extract _3 ep 25-29 ...")
    ds_f = LeRobotDataset(repo_id=SRC_FILTERED)
    ds_2 = LeRobotDataset(repo_id=SRC_2)
    ds_3 = LeRobotDataset(repo_id=SRC_3)
    n_f, n_2, n_3 = ds_f.num_episodes, ds_2.num_episodes, ds_3.num_episodes
    print(f"  source eps: filtered={n_f} _2={n_2} _3={n_3}")

    out_3of4_root = HF_LEROBOT_HOME / OUT_3OF4
    if out_3of4_root.exists():
        shutil.rmtree(out_3of4_root)
    ds_3_keep = delete_episodes(
        dataset=ds_3,
        episode_indices=DROP_3_LOCAL,
        output_dir=out_3of4_root,
        repo_id=OUT_3OF4,
    )
    print(f"  _3 (only 25-29): {ds_3_keep.num_episodes} eps, {ds_3_keep.num_frames} fr")

    print("[2/4] merge filtered + _2 + _3_3of4 → merged_v3_full ...")
    full_root = HF_LEROBOT_HOME / OUT_FULL
    if full_root.exists():
        shutil.rmtree(full_root)
    merged = merge_datasets([ds_f, ds_2, ds_3_keep], output_repo_id=OUT_FULL, output_dir=full_root)
    print(f"  merged_v3_full: {merged.num_episodes} eps, {merged.num_frames} fr")

    # Patch stale meta/episodes/file_index pointers (same bug as merged_v2)
    import pandas as pd  # noqa: F401
    meta_path = full_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df_meta = pq.read_table(meta_path).to_pandas()
    bad = df_meta["meta/episodes/file_index"] != 0
    if bad.any():
        print(f"  patching {int(bad.sum())} stale file_index pointers → 0")
        df_meta["meta/episodes/file_index"] = 0
        df_meta["meta/episodes/chunk_index"] = 0
        df_meta.to_parquet(meta_path, index=False)
        merged = LeRobotDataset(repo_id=OUT_FULL, root=full_root)

    print("[3/4] write split meta (frame-stride done by separate script) ...")
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps({
        "merged_v3_full": OUT_FULL,
        "n_eps": merged.num_episodes,
        "n_frames": merged.num_frames,
        "source_offsets": {"filtered": 0, "_2": n_f, "_3_3of4": n_f + n_2},
        "_3_kept_local": [25, 26, 27, 28, 29],
        "drop_3_local": DROP_3_LOCAL,
    }, indent=2))
    print(f"  meta → {META_OUT}")

    print("[4/4] DONE. Run frame_stride_split_v3.py to build train/val splits.")


if __name__ == "__main__":
    main()
