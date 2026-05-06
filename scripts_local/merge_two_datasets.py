"""Concat two HF lerobot datasets episode-wise. Renumbers indices.

Source A and B videos must be the SAME video keys + features (state/action shape).
Used here: merge fresh v2 demos (v2_extra) with v2_no01 to build success-only set.

Usage:
    uv run python scripts_local/merge_two_datasets.py \\
        --src-a-repo domrachev03/sim_3stage_v2_extra \\
        --src-b-repo domrachev03/sim_3stage_v2_no01_train_fs \\
        --dst-repo local/merged \\
        --keep-a-eps 0,1,...59 \\
        --keep-b-eps 5,12,17,...
"""
import argparse, json, logging, shutil
from pathlib import Path
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import pandas as pd
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src-a-repo", required=True)
    ap.add_argument("--src-b-repo", required=True)
    ap.add_argument("--dst-repo", required=True)
    ap.add_argument("--keep-a-eps", required=True)
    ap.add_argument("--keep-b-eps", required=True)
    args=ap.parse_args()
    keep_a={int(x) for x in args.keep_a_eps.split(",") if x}
    keep_b={int(x) for x in args.keep_b_eps.split(",") if x}
    src_a=Path(HF_LEROBOT_HOME)/args.src_a_repo
    src_b=Path(HF_LEROBOT_HOME)/args.src_b_repo
    dst=Path(HF_LEROBOT_HOME)/args.dst_repo
    if dst.exists(): shutil.rmtree(dst)
    logging.info(f"copy {src_a} -> {dst} (base for meta)")
    shutil.copytree(src_a, dst)
    cc=dst/"meta"/"clip_cache.npz"
    if cc.exists(): cc.unlink()

    # Copy B videos in alongside A under separate suffix is impossible — videos
    # are addressed by (chunk_index, file_index) per ep, all in one mp4 per cam.
    # So we re-encode video indices: but lerobot v3 stores per-ep timestamps,
    # and surviving rows preserve their original timestamp. Easier path: keep
    # A's videos as-is and ALSO copy B's video files alongside under a different
    # chunk_index (chunk-001).
    for cam in ["observation.images.front", "observation.images.wrist"]:
        b_vid=src_b/"videos"/cam/"chunk-000"/"file-000.mp4"
        if not b_vid.exists():
            logging.warning(f"missing {b_vid}; skipping cam {cam}")
            continue
        dst_vid_dir=dst/"videos"/cam/"chunk-001"
        dst_vid_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(b_vid, dst_vid_dir/"file-000.mp4")
        logging.info(f"copied B videos {cam} -> chunk-001")

    # ---- merge data parquet ----
    a_data=pq.read_table(src_a/"data/chunk-000/file-000.parquet").to_pandas()
    b_data=pq.read_table(src_b/"data/chunk-000/file-000.parquet").to_pandas()
    a_eps=pq.read_table(src_a/"meta/episodes/chunk-000/file-000.parquet").to_pandas()
    b_eps=pq.read_table(src_b/"meta/episodes/chunk-000/file-000.parquet").to_pandas()

    new_data_rows=[]; new_ep_rows=[]; gidx=0; new_ep_idx=0
    def add_eps(eps_df, data_df, keep, src_chunk_idx):
        nonlocal gidx, new_ep_idx
        for _,r in eps_df.iterrows():
            if int(r["episode_index"]) not in keep: continue
            T=int(r["length"]); df_=int(r["dataset_from_index"])
            sl=data_df.iloc[df_:df_+T].reset_index(drop=True).copy()
            sl["index"]=np.arange(gidx,gidx+T,dtype=np.int64)
            sl["episode_index"]=np.int64(new_ep_idx)
            new_data_rows.append(sl)
            nr=r.copy()
            nr["episode_index"]=np.int64(new_ep_idx)
            nr["dataset_from_index"]=np.int64(gidx); nr["dataset_to_index"]=np.int64(gidx+T)
            for vk in ["videos/observation.images.front","videos/observation.images.wrist"]:
                ck=vk+"/chunk_index"
                if ck in nr.index:
                    nr[ck]=np.int64(src_chunk_idx)
            new_ep_rows.append(nr)
            gidx+=T; new_ep_idx+=1

    add_eps(a_eps, a_data, keep_a, 0)
    add_eps(b_eps, b_data, keep_b, 1)

    new_df=pd.concat(new_data_rows, ignore_index=True)
    new_eps_df=pd.DataFrame(new_ep_rows)

    src_data_t=pq.read_table(src_a/"data/chunk-000/file-000.parquet")
    src_eps_t=pq.read_table(src_a/"meta/episodes/chunk-000/file-000.parquet")

    pq.write_table(pa.Table.from_pandas(new_df, schema=src_data_t.schema, preserve_index=False),
                   dst/"data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pandas(new_eps_df, schema=src_eps_t.schema, preserve_index=False),
                   dst/"meta/episodes/chunk-000/file-000.parquet")
    info=json.loads((dst/"meta/info.json").read_text())
    info["total_episodes"]=len(new_eps_df); info["total_frames"]=int(gidx)
    info["splits"]={"train": f"0:{len(new_eps_df)}"}
    (dst/"meta/info.json").write_text(json.dumps(info, indent=4))
    logging.info(f"DONE: {len(new_eps_df)} eps, {gidx} frames -> {dst}")


if __name__=="__main__": main()
