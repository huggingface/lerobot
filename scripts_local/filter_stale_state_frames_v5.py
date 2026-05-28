"""Multi-file-aware destale w/ optional tail-pct restriction.

Drop frames where ||state[t] - state[t-1]||_2 < eps for >=N consecutive frames
WITHIN the last `tail_pct` fraction of each episode. First and last frames of
each episode + stage-boundary frames are always preserved.

Output: single-chunk parquet (file-000) regardless of input file count. Videos
are copied verbatim from src; surviving rows keep their original timestamps so
loader still seeks correct video frames. CLIP cache invalidated.

Per feedback_destale_surgical_only.md: full-ep destale breaks gripper coord.
tail-30 (tail_pct=0.3) is the proven setting.

Usage:
    uv run python scripts_local/filter_stale_state_frames_v5.py \\
        --src-repo-id domrachev03/sim_3stage_v5_success \\
        --dst-repo-id domrachev03/sim_3stage_v5_success_destale_t30 \\
        --tail-pct 0.3 --eps 1e-3 --min-run 3
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _build_drop_mask(
    state: np.ndarray,
    boundary_frames: set[int],
    eps: float,
    min_run: int,
    tail_pct: float,
) -> np.ndarray:
    """Return bool mask of length T; True = drop. Never drops boundary frames,
    first/last frame, or any frame outside the tail-pct window of the ep."""
    T = state.shape[0]
    if T < 2:
        return np.zeros(T, dtype=bool)
    delta = np.linalg.norm(np.diff(state, axis=0), axis=1)
    is_stale = np.zeros(T, dtype=bool)
    is_stale[1:] = delta < eps
    is_stale[0] = False
    is_stale[T - 1] = False
    for b in boundary_frames:
        if 0 <= b < T:
            is_stale[b] = False
    if tail_pct < 1.0:
        tail_start = int(np.ceil(T * (1.0 - tail_pct)))
        is_stale[:tail_start] = False
    drop = np.zeros(T, dtype=bool)
    i = 0
    while i < T:
        if not is_stale[i]:
            i += 1
            continue
        j = i
        while j < T and is_stale[j]:
            j += 1
        run_len = j - i
        if run_len >= min_run:
            drop[i:j] = True
        i = j
    return drop


def _remap_frames(old_frames: list[int], drop_mask: np.ndarray, old_to_new: np.ndarray) -> list[int]:
    out = []
    T = len(drop_mask)
    for f in old_frames:
        f = int(f)
        if f < 0:
            out.append(f)
            continue
        f_clip = min(max(f, 0), T - 1)
        cur = f_clip
        while cur < T and drop_mask[cur]:
            cur += 1
        if cur >= T:
            cur = T - 1
            while cur >= 0 and drop_mask[cur]:
                cur -= 1
        out.append(int(old_to_new[cur]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-repo-id", required=True)
    ap.add_argument("--dst-repo-id", required=True)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--min-run", type=int, default=3)
    ap.add_argument("--tail-pct", type=float, default=0.3, help="Only destale frames in last tail_pct of each ep. 1.0=full ep")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not 0.0 < args.tail_pct <= 1.0:
        raise SystemExit("--tail-pct must be in (0, 1]")

    src = Path(HF_LEROBOT_HOME) / args.src_repo_id
    dst = Path(HF_LEROBOT_HOME) / args.dst_repo_id
    if not src.exists():
        raise SystemExit(f"src missing: {src}")

    if not args.dry_run:
        if dst.exists():
            logging.info(f"removing existing dst {dst}")
            shutil.rmtree(dst)
        logging.info(f"copying {src} -> {dst}")
        shutil.copytree(src, dst)
        cc = dst / "meta" / "clip_cache.npz"
        if cc.exists():
            cc.unlink()
            logging.info(f"removed stale {cc}")

    data_files = sorted((src / "data").rglob("*.parquet"))
    eps_files = sorted((src / "meta" / "episodes").rglob("*.parquet"))
    if not data_files:
        raise SystemExit(f"no data parquet in {src}/data")
    if not eps_files:
        raise SystemExit(f"no meta/episodes parquet in {src}")

    logging.info(f"src: {len(data_files)} data file(s), {len(eps_files)} ep-meta file(s)")
    df = pd.concat([pq.read_table(p).to_pandas() for p in data_files], ignore_index=True)
    eps_df = pd.concat([pq.read_table(p).to_pandas() for p in eps_files], ignore_index=True)
    eps_df = eps_df.sort_values("episode_index").reset_index(drop=True)
    info = json.loads((src / "meta" / "info.json").read_text())

    total_in = len(df)
    total_drop = 0
    new_rows: list = []
    new_eps_rows: list[dict] = []
    new_global = 0

    for _, ep_row in eps_df.iterrows():
        ep_idx = int(ep_row["episode_index"])
        T = int(ep_row["length"])
        df_from = int(ep_row["dataset_from_index"])
        df_to = int(ep_row["dataset_to_index"])
        ep_chunk = df.iloc[df_from:df_to].reset_index(drop=True)
        assert len(ep_chunk) == T, f"ep {ep_idx} parquet rows {len(ep_chunk)} != length {T}"
        state = np.stack(ep_chunk["observation.state"].values).astype(np.float32)

        boundary = set()
        for fld in ("sparse_subtask_start_frames", "sparse_subtask_end_frames",
                    "dense_subtask_start_frames", "dense_subtask_end_frames"):
            arr = ep_row.get(fld)
            if arr is None:
                continue
            for v in arr:
                if v is not None and v >= 0:
                    boundary.add(int(v))

        drop = _build_drop_mask(state, boundary, args.eps, args.min_run, args.tail_pct)
        keep = ~drop
        new_T = int(keep.sum())
        n_drop = int(drop.sum())
        total_drop += n_drop

        old_to_new = np.full(T, -1, dtype=np.int64)
        nidx = 0
        for ofi in range(T):
            if keep[ofi]:
                old_to_new[ofi] = nidx
                nidx += 1
        assert nidx == new_T

        kept = ep_chunk[keep].reset_index(drop=True)
        kept["frame_index"] = np.arange(new_T, dtype=np.int64)
        kept["episode_index"] = np.int64(ep_idx)
        kept["index"] = np.arange(new_global, new_global + new_T, dtype=np.int64)
        new_rows.append(kept)

        ep_new = ep_row.copy()
        ep_new["length"] = np.int64(new_T)
        ep_new["dataset_from_index"] = np.int64(new_global)
        ep_new["dataset_to_index"] = np.int64(new_global + new_T)
        for fld in ("sparse_subtask_start_frames", "sparse_subtask_end_frames",
                    "dense_subtask_start_frames", "dense_subtask_end_frames"):
            arr = ep_new.get(fld)
            if arr is None:
                continue
            ep_new[fld] = list(np.asarray(_remap_frames(list(arr), drop, old_to_new), dtype=np.int32).tolist())
        # Force all eps into a single output file/chunk.
        for vk in info.get("features", {}).keys():
            if vk.startswith("observation.images."):
                pass
        # Rewrite video chunk_index/file_index to keep originals (videos copied as-is in dst)
        new_eps_rows.append(ep_new)

        new_global += new_T
        if ep_idx < 3 or ep_idx == len(eps_df) - 1:
            logging.info(f"ep {ep_idx}: T={T} -> {new_T} (drop {n_drop}, {n_drop/T*100:.1f}%) tail_pct={args.tail_pct}")

    new_total = new_global
    pct = 100 * total_drop / max(total_in, 1)
    logging.info(f"OVERALL: total_in={total_in} total_drop={total_drop} ({pct:.2f}%) total_out={new_total}")

    if args.dry_run:
        logging.info("dry-run: not writing")
        return

    # Wipe dst data + meta/episodes, then write SINGLE-file output (file-000 only)
    new_df = pd.concat(new_rows, ignore_index=True)
    dst_data_dir = dst / "data" / "chunk-000"
    for p in dst_data_dir.glob("file-*.parquet"):
        p.unlink()
    dst_data_dir.mkdir(parents=True, exist_ok=True)
    src_data_t = pq.read_table(data_files[0])
    out_data_t = pa.Table.from_pandas(new_df, schema=src_data_t.schema, preserve_index=False)
    pq.write_table(out_data_t, dst_data_dir / "file-000.parquet")
    logging.info(f"wrote {dst_data_dir / 'file-000.parquet'} ({len(new_df)} rows)")

    new_eps_df = pd.DataFrame(new_eps_rows)
    dst_eps_dir = dst / "meta" / "episodes" / "chunk-000"
    for p in dst_eps_dir.glob("file-*.parquet"):
        p.unlink()
    src_eps_t = pq.read_table(eps_files[0])
    out_eps_t = pa.Table.from_pandas(new_eps_df, schema=src_eps_t.schema, preserve_index=False)
    pq.write_table(out_eps_t, dst_eps_dir / "file-000.parquet")
    logging.info(f"wrote {dst_eps_dir / 'file-000.parquet'} ({len(new_eps_df)} eps)")

    info["total_frames"] = int(new_total)
    info["total_episodes"] = int(len(new_eps_df))
    info["splits"] = {"train": f"0:{len(new_eps_df)}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    logging.info(f"updated info.json total_frames={new_total} total_episodes={len(new_eps_df)}")

    logging.info("DONE.")


if __name__ == "__main__":
    main()
