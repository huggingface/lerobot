"""Make a subset of an HF lerobot dataset by selecting specific episode indices.

Drops other episodes' rows + fixes meta (lengths, dataset_from/to_index, total_frames).
Videos are NOT re-encoded — surviving rows keep original timestamps.

Usage:
    uv run python scripts_local/subset_episodes.py \\
        --src-repo-id domrachev03/sim_3stage_v4_with_partials \\
        --dst-repo-id local/sim_3stage_v4_with_partials_139 \\
        --keep-eps 0,1,2,...
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
    ap.add_argument("--src-repo-id", required=True)
    ap.add_argument("--dst-repo-id", required=True)
    ap.add_argument("--keep-eps", required=True, help="comma-sep episode indices")
    args=ap.parse_args()
    keep=sorted(int(x) for x in args.keep_eps.split(","))
    src=Path(HF_LEROBOT_HOME)/args.src_repo_id
    dst=Path(HF_LEROBOT_HOME)/args.dst_repo_id
    if dst.exists():
        shutil.rmtree(dst)
    logging.info(f"copy {src}->{dst}")
    shutil.copytree(src, dst)
    cc=dst/"meta"/"clip_cache.npz"
    if cc.exists(): cc.unlink()
    data=pq.read_table(dst/"data/chunk-000/file-000.parquet").to_pandas()
    eps=pq.read_table(dst/"meta/episodes/chunk-000/file-000.parquet").to_pandas()
    keep_set=set(keep)
    new_rows=[]; new_eps=[]; gidx=0; new_ep_idx=0
    for _,r in eps.iterrows():
        if int(r["episode_index"]) not in keep_set: continue
        T=int(r["length"]); df_=int(r["dataset_from_index"])
        sl=data.iloc[df_:df_+T].reset_index(drop=True).copy()
        sl["index"]=np.arange(gidx,gidx+T,dtype=np.int64)
        sl["episode_index"]=np.int64(new_ep_idx)
        new_rows.append(sl)
        nr=r.copy()
        nr["episode_index"]=np.int64(new_ep_idx)
        nr["dataset_from_index"]=np.int64(gidx); nr["dataset_to_index"]=np.int64(gidx+T)
        new_eps.append(nr); gidx+=T; new_ep_idx+=1
    new_df=pd.concat(new_rows,ignore_index=True)
    new_eps_df=pd.DataFrame(new_eps)
    src_data_t=pq.read_table(src/"data/chunk-000/file-000.parquet")
    src_eps_t=pq.read_table(src/"meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pandas(new_df,schema=src_data_t.schema,preserve_index=False), dst/"data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pandas(new_eps_df,schema=src_eps_t.schema,preserve_index=False), dst/"meta/episodes/chunk-000/file-000.parquet")
    info=json.loads((dst/"meta/info.json").read_text())
    info["total_frames"]=int(gidx); info["total_episodes"]=len(new_eps)
    info["splits"]={"train": f"0:{len(new_eps)}"}
    (dst/"meta/info.json").write_text(json.dumps(info,indent=4))
    logging.info(f"DONE: {len(new_eps)} eps {gidx} frames")


if __name__=="__main__": main()
