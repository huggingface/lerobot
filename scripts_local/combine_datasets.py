"""Combine multiple HF lerobot datasets into one.

Handles datasets with multi-file data/episode-meta parquets and multi-file
videos (chunk-NNN/file-NNN.mp4). Each ep's video stays in its own original
mp4; we just renumber chunk_index across the combined set.

Usage:
    uv run python scripts_local/combine_datasets.py \\
        --src 'domrachev03/sim_3stage_v2_extra:exclude=4,84' \\
        --src 'domrachev03/sim_3stage_v2_extra_partial' \\
        --dst-repo local/sim_3stage_v2_extra_combined
"""
import argparse, json, logging, shutil
from pathlib import Path
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import pandas as pd
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_all(root: Path, sub: str):
    """Read all parquet shards under root/sub/chunk-*/file-*.parquet, concat preserving order."""
    parts = []
    for chunk_dir in sorted(root.glob(f"{sub}/chunk-*")):
        for f in sorted(chunk_dir.glob("file-*.parquet")):
            parts.append(pq.read_table(f).to_pandas())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True,
                    help="repo_id[:exclude=i,j,k]")
    ap.add_argument("--dst-repo", required=True)
    args=ap.parse_args()

    sources = []
    for s in args.src:
        if ":exclude=" in s:
            repo, exc = s.split(":exclude=", 1)
            exclude = {int(x) for x in exc.split(",") if x}
        else:
            repo, exclude = s, set()
        sources.append((repo, exclude))

    dst = Path(HF_LEROBOT_HOME)/args.dst_repo
    if dst.exists(): shutil.rmtree(dst)

    # Use first source's meta (info, stats, tasks, etc.) as base
    base_root = Path(HF_LEROBOT_HOME)/sources[0][0]
    logging.info(f"copy meta base from {base_root}")
    dst.mkdir(parents=True)
    (dst/"meta"/"episodes"/"chunk-000").mkdir(parents=True)
    (dst/"data"/"chunk-000").mkdir(parents=True)
    (dst/"videos").mkdir()
    for f in ["info.json","stats.json","tasks.parquet",
              "temporal_proportions_dense.json","temporal_proportions_sparse.json"]:
        if (base_root/"meta"/f).exists():
            shutil.copy2(base_root/"meta"/f, dst/"meta"/f)

    # Discover video keys
    vid_keys = [d.name for d in (base_root/"videos").iterdir() if d.is_dir()]

    new_rows=[]; new_eps=[]; gidx=0; new_ep_idx=0
    next_chunk_idx={vk:0 for vk in vid_keys}

    for repo, exclude in sources:
        root = Path(HF_LEROBOT_HOME)/repo
        data = load_all(root, "data")
        eps = load_all(root, "meta/episodes")
        logging.info(f"{repo}: {len(eps)} eps, {len(data)} rows, exclude={sorted(exclude)}")

        # Collect unique source video files per cam (chunk_index, file_index) -> assign new chunk_index
        vid_remap = {}  # (vk, src_chunk, src_file) -> new_chunk
        for vk in vid_keys:
            seen = set()
            for _,r in eps.iterrows():
                if int(r['episode_index']) in exclude: continue
                key = (vk, int(r[f'videos/{vk}/chunk_index']), int(r[f'videos/{vk}/file_index']))
                if key in seen: continue
                seen.add(key)
                new_chunk = next_chunk_idx[vk]
                next_chunk_idx[vk] += 1
                # copy video file
                src_vid = root/"videos"/vk/f"chunk-{key[1]:03d}"/f"file-{key[2]:03d}.mp4"
                dst_vid_dir = dst/"videos"/vk/f"chunk-{new_chunk:03d}"
                dst_vid_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_vid, dst_vid_dir/"file-000.mp4")
                vid_remap[key] = new_chunk

        # Iterate eps in order, renumber + remap
        for _,r in eps.iterrows():
            ep_idx_orig = int(r['episode_index'])
            if ep_idx_orig in exclude: continue
            T = int(r['length'])
            df_from = int(r['dataset_from_index'])
            df_to = int(r['dataset_to_index'])
            sl = data.iloc[df_from:df_to].reset_index(drop=True).copy()
            sl["index"] = np.arange(gidx, gidx+T, dtype=np.int64)
            sl["episode_index"] = np.int64(new_ep_idx)
            new_rows.append(sl)

            nr = r.copy()
            nr["episode_index"] = np.int64(new_ep_idx)
            nr["dataset_from_index"] = np.int64(gidx)
            nr["dataset_to_index"] = np.int64(gidx+T)
            nr["data/chunk_index"] = np.int64(0)
            nr["data/file_index"] = np.int64(0)
            for vk in vid_keys:
                key = (vk, int(r[f'videos/{vk}/chunk_index']), int(r[f'videos/{vk}/file_index']))
                nr[f'videos/{vk}/chunk_index'] = np.int64(vid_remap[key])
                nr[f'videos/{vk}/file_index'] = np.int64(0)
            new_eps.append(nr)
            gidx += T
            new_ep_idx += 1

    new_df = pd.concat(new_rows, ignore_index=True)
    new_eps_df = pd.DataFrame(new_eps)

    # Use schema from base source's first parquet shard
    base_data_t = pq.read_table(base_root/"data"/"chunk-000"/"file-000.parquet")
    base_eps_t = pq.read_table(base_root/"meta"/"episodes"/"chunk-000"/"file-000.parquet")
    pq.write_table(pa.Table.from_pandas(new_df, schema=base_data_t.schema, preserve_index=False),
                   dst/"data"/"chunk-000"/"file-000.parquet")
    pq.write_table(pa.Table.from_pandas(new_eps_df, schema=base_eps_t.schema, preserve_index=False),
                   dst/"meta"/"episodes"/"chunk-000"/"file-000.parquet")

    info = json.loads((dst/"meta"/"info.json").read_text())
    info["total_episodes"] = len(new_eps_df)
    info["total_frames"] = int(gidx)
    info["splits"] = {"train": f"0:{len(new_eps_df)}"}
    (dst/"meta"/"info.json").write_text(json.dumps(info, indent=4))
    logging.info(f"DONE: {len(new_eps_df)} eps, {gidx} frames -> {dst}")


if __name__=="__main__": main()
