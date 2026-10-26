#!/usr/bin/env python
"""PHASE D0 step F (part 1): decode frame 0 of the front ("image") and "second_image" cameras for every
episode of the 6 audited tasks, downsample to 96x96 uint8, save. Read-only on the dataset."""
from __future__ import annotations

import sys, time
from pathlib import Path

import cv2, numpy as np, pandas as pd

from lerobot.datasets.video_utils import decode_video_frames

SNAP = Path("/home/user_lerobot/.cache/huggingface/lerobot/hub/datasets--lerobot--vlabench_unified/snapshots/f61ad898199e746c20af3112f278cc86eb3ca3c4")
OUT = Path("outputs/eval/safediff_vla_phase_d0_training_data_audit")
CAMS = ["observation.images.image", "observation.images.second_image"]


def main():
    ep = pd.read_json(OUT / "episodes.json")[["episode", "task"]]
    meta = pd.concat([pd.read_parquet(p) for p in sorted((SNAP / "meta/episodes").rglob("*.parquet"))]).set_index("episode_index")
    imgs = np.zeros((len(ep), len(CAMS), 96, 96, 3), np.uint8)
    t0 = time.time()
    for ci, cam in enumerate(CAMS):
        cols = [f"videos/{cam}/chunk_index", f"videos/{cam}/file_index", f"videos/{cam}/from_timestamp"]
        m = meta.loc[ep.episode, cols].reset_index(drop=True); m.columns = ["chunk", "file", "ts"]
        for (ch, fi), g in m.groupby(["chunk", "file"]):
            path = SNAP / f"videos/{cam}/chunk-{int(ch):03d}/file-{int(fi):03d}.mp4"
            frames = decode_video_frames(path, g.ts.tolist(), tolerance_s=0.05, return_uint8=True)  # [n,3,H,W]
            for row, fr in zip(g.index, frames):
                a = fr.permute(1, 2, 0).numpy() if hasattr(fr, "permute") else np.asarray(fr)
                imgs[row, ci] = cv2.resize(a.astype(np.uint8), (96, 96), interpolation=cv2.INTER_AREA)
            print(f"{cam} chunk{ch} file{fi}: {len(g)} eps  ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(OUT / "first_frames_96.npz", imgs=imgs, episode=ep.episode.to_numpy(), task=ep.task.to_numpy())
    print("saved", imgs.shape)


if __name__ == "__main__":
    main()
