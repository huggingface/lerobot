"""Rebuild chunk video files so each episode's frame count exactly matches its
parquet row count.

Background: a recorder bug produces per-episode chunk videos with N + extra
frames vs parquet length N. This script re-encodes each affected chunk video
per cam, taking exactly L_parquet[i] frames per episode in order, then rewrites
episode metadata's videos/*/from_timestamp + to_timestamp.

Handles multi-file chunks (resume sessions): groups eps by (cam, chunk_index,
file_index), rebuilds only files that contain mismatches. Backs up originals
to `.bak` suffix and writes new chunks in place.
"""

from __future__ import annotations

import argparse
import shutil
from fractions import Fraction
from pathlib import Path

import av
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def rebuild_cam_chunk(
    src_video: Path,
    dst_video: Path,
    ep_video_frame_counts_src: list[int],
    ep_keep_counts: list[int],
    fps: int,
) -> list[tuple[float, float]]:
    """Decode src in order, emit each ep's first `keep_counts[i]` frames into dst.

    `ep_video_frame_counts_src` and `ep_keep_counts` are aligned per-ep WITHIN
    this file (not global). Returns list of (from_ts, to_ts) per ep based on
    output frame indices (file-local seconds).
    """
    src = av.open(str(src_video))
    in_stream = src.streams.video[0]
    w, h = in_stream.codec_context.width, in_stream.codec_context.height

    dst_video.parent.mkdir(parents=True, exist_ok=True)
    dst = av.open(str(dst_video), mode="w", format="mp4", options={"movflags": "faststart"})
    out_stream = dst.add_stream("libsvtav1", rate=fps)
    out_stream.width = w
    out_stream.height = h
    out_stream.pix_fmt = "yuv420p"
    out_stream.time_base = Fraction(1, fps)
    out_stream.options = {"crf": "30", "preset": "8", "g": "2"}
    frame_tb = Fraction(1, fps)

    ep_video_frame_starts = []
    cum = 0
    for cnt in ep_video_frame_counts_src:
        ep_video_frame_starts.append(cum)
        cum += cnt

    cum_in = 0
    out_frame_idx = 0
    n_eps = len(ep_keep_counts)
    current_ep = 0
    frames_taken_in_ep = 0
    ep_first_out = [0] * n_eps
    ep_last_out = [0] * n_eps
    for packet in src.demux(in_stream):
        for frame in packet.decode():
            while current_ep < n_eps and cum_in >= (
                ep_video_frame_starts[current_ep] + ep_video_frame_counts_src[current_ep]
            ):
                current_ep += 1
                frames_taken_in_ep = 0
            if current_ep >= n_eps:
                cum_in += 1
                continue
            if frames_taken_in_ep < ep_keep_counts[current_ep]:
                if frames_taken_in_ep == 0:
                    ep_first_out[current_ep] = out_frame_idx
                frame.time_base = frame_tb
                frame.pts = out_frame_idx
                for opkt in out_stream.encode(frame):
                    dst.mux(opkt)
                out_frame_idx += 1
                ep_last_out[current_ep] = out_frame_idx
                frames_taken_in_ep += 1
            cum_in += 1

    for opkt in out_stream.encode(None):
        dst.mux(opkt)
    dst.close()
    src.close()

    return [(ep_first_out[i] / fps, ep_last_out[i] / fps) for i in range(n_eps)]


def _load_parquet_lens(root: Path) -> dict[int, int]:
    parq_dir = root / "data/chunk-000"
    dfs = [pd.read_parquet(p, columns=["episode_index"]) for p in sorted(parq_dir.glob("file-*.parquet"))]
    df = pd.concat(dfs)
    return df.groupby("episode_index").size().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.root) if args.root else LeRobotDataset(args.repo_id)
    fps = ds.fps
    root = Path(ds.root)

    parq = _load_parquet_lens(root)
    cams = list(ds.meta.video_keys)
    n_eps = ds.num_episodes

    # Group eps by (cam, chunk, file) — cams may share but we verify each
    per_file: dict[tuple[str, int, int], list[int]] = {}
    for i in range(n_eps):
        e = ds.meta.episodes[i]
        for cam in cams:
            key = (cam, int(e[f"videos/{cam}/chunk_index"]), int(e[f"videos/{cam}/file_index"]))
            per_file.setdefault(key, []).append(i)

    # Detect mismatches per file
    files_with_mismatch: dict[tuple[str, int, int], list[int]] = {}
    for (cam, chunk, fi), eps in per_file.items():
        mism = []
        for i in eps:
            e = ds.meta.episodes[i]
            f = float(e[f"videos/{cam}/from_timestamp"])
            t = float(e[f"videos/{cam}/to_timestamp"])
            src_n = round((t - f) * fps)
            if src_n != parq[i]:
                mism.append((i, src_n, parq[i]))
        if mism:
            files_with_mismatch[(cam, chunk, fi)] = eps
            print(f"file {cam} chunk={chunk} file={fi}: {len(mism)} mismatched eps")
            for m in mism[:5]:
                print(f"    ep{m[0]}: src={m[1]} keep={m[2]} drop={m[1]-m[2]}")

    if not files_with_mismatch:
        print("Nothing to fix.")
        return
    if args.dry_run:
        return

    # For each affected file: rebuild it
    new_ts_per_cam_ep: dict[str, dict[int, tuple[float, float]]] = {cam: {} for cam in cams}
    for (cam, chunk, fi), eps in files_with_mismatch.items():
        src = root / f"videos/{cam}/chunk-{chunk:03d}/file-{fi:03d}.mp4"
        bak = src.with_suffix(".mp4.bak")
        if not bak.exists():
            print(f"Backup {src.name} -> {bak.name}")
            shutil.copy(src, bak)
        else:
            print(f"Using existing backup {bak.name}.")

        # Determine source frame counts + keep counts per ep IN THIS FILE in order
        eps_sorted = sorted(eps)
        # Sanity: order in file should be increasing ep_idx, with from_timestamp ascending
        eps_sorted.sort(key=lambda i: float(ds.meta.episodes[i][f"videos/{cam}/from_timestamp"]))
        src_counts = []
        keep_counts = []
        for i in eps_sorted:
            e = ds.meta.episodes[i]
            f = float(e[f"videos/{cam}/from_timestamp"])
            t = float(e[f"videos/{cam}/to_timestamp"])
            src_counts.append(round((t - f) * fps))
            keep_counts.append(int(parq[i]))

        tmp_new = src.with_suffix(".mp4.new")
        if tmp_new.exists():
            tmp_new.unlink()
        print(f"Rebuilding {src.name} ({len(eps_sorted)} eps)...")
        ts = rebuild_cam_chunk(
            src_video=bak,
            dst_video=tmp_new,
            ep_video_frame_counts_src=src_counts,
            ep_keep_counts=keep_counts,
            fps=fps,
        )
        shutil.move(str(tmp_new), str(src))
        print(f"  wrote {src.name}")
        for ep_idx, (f_ts, t_ts) in zip(eps_sorted, ts, strict=True):
            new_ts_per_cam_ep[cam][ep_idx] = (f_ts, t_ts)

    # Update episode metadata parquet(s). Episodes meta may also be split across files.
    ep_meta_dir = root / "meta/episodes/chunk-000"
    for ep_parq in sorted(ep_meta_dir.glob("file-*.parquet")):
        ep_df = pd.read_parquet(ep_parq)
        modified = False
        for cam in cams:
            f_col = f"videos/{cam}/from_timestamp"
            t_col = f"videos/{cam}/to_timestamp"
            for ep_idx, (f_ts, t_ts) in new_ts_per_cam_ep[cam].items():
                mask = ep_df.episode_index == ep_idx
                if mask.any():
                    ep_df.loc[mask, f_col] = f_ts
                    ep_df.loc[mask, t_col] = t_ts
                    modified = True
        if modified:
            ep_bak = ep_parq.with_suffix(".parquet.bak")
            if not ep_bak.exists():
                shutil.copy(ep_parq, ep_bak)
            ep_df.to_parquet(ep_parq, index=False)
            print(f"Updated {ep_parq.name}")


if __name__ == "__main__":
    main()
