"""Drop frames where ||state[t] - state[t-1]||_2 < eps for >=N consecutive frames.

Only "interior" stale frames are dropped: the first and last frame of each episode
are always preserved, and we never drop a frame that sits at a stage boundary
(i.e., stage_start_frames or stage_end_frames). After filtering, frame_index is
renumbered 0..K-1 within each episode and `index` 0..total_frames-1 globally;
`sparse_subtask_*_frames` / `dense_subtask_*_frames` are remapped to the new
indices, and `length` / `dataset_from_index` / `dataset_to_index` are rewritten.

Videos are NOT re-encoded — surviving rows keep their original `timestamp`, so
the loader still seeks the correct video frame. CLIP cache is invalidated.

Usage:
    uv run python scripts_local/filter_stale_state_frames.py \\
        --src-repo-id domrachev03/sim_3stage_v4_success_train_fs \\
        --dst-repo-id local/sim_3stage_v4_nostale \\
        --eps 1e-3 --min-run 3
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _build_drop_mask(
    state: np.ndarray, boundary_frames: set[int], eps: float, min_run: int
) -> np.ndarray:
    """Return bool mask of length T; True = drop. Never drops boundary frames."""
    T = state.shape[0]
    if T < 2:
        return np.zeros(T, dtype=bool)
    delta = np.linalg.norm(np.diff(state, axis=0), axis=1)  # length T-1; delta[i] = ||s[i+1]-s[i]||
    is_stale = np.zeros(T, dtype=bool)
    is_stale[1:] = delta < eps  # mark frame t (t>=1) as stale if no motion since t-1
    # never drop first/last of episode or any boundary frame
    is_stale[0] = False
    is_stale[T - 1] = False
    for b in boundary_frames:
        if 0 <= b < T:
            is_stale[b] = False
    # require >=min_run consecutive stale frames; only middle of run is dropped
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
            drop[i:j] = True  # whole run dropped; first/last/boundaries already excluded above
        i = j
    return drop


def _remap_frames(old_frames: list[int], drop_mask: np.ndarray, old_to_new: np.ndarray) -> list[int]:
    """Map a list of old frame indices to new indices (after dropping)."""
    out = []
    T = len(drop_mask)
    for f in old_frames:
        f = int(f)
        if f < 0:
            out.append(f)
            continue
        f_clip = min(max(f, 0), T - 1)
        # walk forward to first non-dropped frame >= f_clip
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
    ap.add_argument("--eps", type=float, default=1e-3, help="L2 norm of state delta below this = stale")
    ap.add_argument("--min-run", type=int, default=3, help="min consecutive stale frames to drop")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

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
        # invalidate CLIP cache; will be regenerated on training start
        cc = dst / "meta" / "clip_cache.npz"
        if cc.exists():
            cc.unlink()
            logging.info(f"removed stale {cc}")

    # ---- load source data + episodes ----
    data_fp = src / "data" / "chunk-000" / "file-000.parquet"
    eps_fp = src / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    data_t = pq.read_table(data_fp)
    eps_t = pq.read_table(eps_fp)
    df = data_t.to_pandas()
    eps_df = eps_t.to_pandas()
    info = json.loads((src / "meta" / "info.json").read_text())

    total_in = len(df)
    total_drop = 0
    new_rows: list[dict] = []
    new_eps_rows: list[dict] = []
    new_global = 0

    for _, ep_row in eps_df.iterrows():
        ep_idx = int(ep_row["episode_index"])
        T = int(ep_row["length"])
        df_from = int(ep_row["dataset_from_index"])
        df_to = int(ep_row["dataset_to_index"])
        ep_df = df.iloc[df_from:df_to].reset_index(drop=True)
        assert len(ep_df) == T, f"ep {ep_idx} parquet rows {len(ep_df)} != length {T}"
        state = np.stack(ep_df["observation.state"].values).astype(np.float32)

        # boundary frames (stage starts and ends)
        boundary = set()
        for fld in ("sparse_subtask_start_frames", "sparse_subtask_end_frames",
                    "dense_subtask_start_frames", "dense_subtask_end_frames"):
            arr = ep_row[fld]
            if arr is None:
                continue
            for v in arr:
                if v is not None and v >= 0:
                    boundary.add(int(v))

        drop = _build_drop_mask(state, boundary, args.eps, args.min_run)
        keep = ~drop
        new_T = int(keep.sum())
        n_drop = int(drop.sum())
        total_drop += n_drop

        # mapping old_frame_idx -> new_frame_idx (within episode)
        old_to_new = np.full(T, -1, dtype=np.int64)
        nidx = 0
        for ofi in range(T):
            if keep[ofi]:
                old_to_new[ofi] = nidx
                nidx += 1
        assert nidx == new_T

        # rewrite frame_index, episode_index, index for the kept rows
        kept = ep_df[keep].reset_index(drop=True)
        kept["frame_index"] = np.arange(new_T, dtype=np.int64)
        kept["episode_index"] = np.int64(ep_idx)
        kept["index"] = np.arange(new_global, new_global + new_T, dtype=np.int64)
        # timestamps: keep original (videos unchanged)
        new_rows.append(kept)

        # episodes-meta row remap
        ep_new = ep_row.copy()
        ep_new["length"] = np.int64(new_T)
        ep_new["dataset_from_index"] = np.int64(new_global)
        ep_new["dataset_to_index"] = np.int64(new_global + new_T)
        for fld in ("sparse_subtask_start_frames", "sparse_subtask_end_frames",
                    "dense_subtask_start_frames", "dense_subtask_end_frames"):
            arr = ep_new[fld]
            if arr is None:
                continue
            ep_new[fld] = list(np.asarray(_remap_frames(list(arr), drop, old_to_new), dtype=np.int32).tolist())
        new_eps_rows.append(ep_new)

        new_global += new_T
        if ep_idx < 3 or ep_idx == len(eps_df) - 1:
            logging.info(f"ep {ep_idx}: T={T} -> {new_T} (drop {n_drop}, {n_drop/T*100:.1f}%)")

    new_total = new_global
    pct = 100 * total_drop / max(total_in, 1)
    logging.info(f"OVERALL: total_in={total_in} total_drop={total_drop} ({pct:.2f}%) total_out={new_total}")

    if args.dry_run:
        logging.info("dry-run: not writing")
        return

    # ---- write new data parquet ----
    import pandas as pd
    new_df = pd.concat(new_rows, ignore_index=True)
    new_data_fp = dst / "data" / "chunk-000" / "file-000.parquet"
    new_data_fp.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(new_df, preserve_index=False), new_data_fp)
    logging.info(f"wrote {new_data_fp} ({len(new_df)} rows)")

    # ---- write new episodes meta parquet (preserve source schema) ----
    new_eps_df = pd.DataFrame(new_eps_rows)
    new_eps_fp = dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    new_eps_fp.parent.mkdir(parents=True, exist_ok=True)
    src_eps_t = pq.read_table(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    new_eps_t = pa.Table.from_pandas(new_eps_df, schema=src_eps_t.schema, preserve_index=False)
    pq.write_table(new_eps_t, new_eps_fp)
    logging.info(f"wrote {new_eps_fp} ({len(new_eps_df)} eps)")

    # ---- update info.json ----
    info["total_frames"] = int(new_total)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    logging.info(f"updated info.json total_frames={new_total}")

    logging.info("DONE. Run refresh_action_stats next:")
    logging.info(f"  uv run python -m lerobot.scripts.refresh_action_stats --repo-id {args.dst_repo_id}")


if __name__ == "__main__":
    main()
