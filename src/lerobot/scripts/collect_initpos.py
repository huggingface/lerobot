#!/usr/bin/env python
import argparse
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.visualize_dataset import EpisodeSampler

# --- EDIT for your robot layout ---
MOTOR_IDXS = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_roll": 4,
    "gripper_pos": 5,
}
# ----------------------------------

def collect_first_10s_episode(dataset: LeRobotDataset, episode_index: int,
                              seconds=10.0, use_state=True, dl_workers: int = 2):
    """Return (raw, avg) where:
       raw[motor] -> np.array(T,),  avg[motor] -> float
    """
    fps = float(dataset.meta.fps)
    n_frames = int(seconds * fps)

    sampler = EpisodeSampler(dataset, episode_index)
    loader = DataLoader(dataset,
                        sampler=sampler,
                        batch_size=1,
                        shuffle=False,
                        num_workers=dl_workers,
                        pin_memory=False)

    # incremental buffers per motor to avoid big reallocs
    vals = {m: [] for m in MOTOR_IDXS}

    for i, batch in enumerate(loader):
        if i >= n_frames:
            break
        vec = batch["observation.state"][0] if use_state else batch["action"][0]
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        for m, idx in MOTOR_IDXS.items():
            vals[m].append(float(vec[idx]))

    raw = {m: np.asarray(v, dtype=np.float32) for m, v in vals.items()}
    avg = {m: (float(v.mean()) if v.size else float("nan")) for m, v in raw.items()}
    return raw, avg

def invert_episode_major(d_ep_motor):
    if not d_ep_motor:
        return {}
    motors = next(iter(d_ep_motor.values())).keys()
    out = {m: {} for m in motors}
    for ep, m_dict in d_ep_motor.items():
        for m, v in m_dict.items():
            out[m][ep] = v
    return out

def save_dicts(raw_motor_major, avg_motor_major, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    raw_json = {m: {str(ep): arr.tolist() for ep, arr in eps.items()} for m, eps in raw_motor_major.items()}
    avg_json = {m: {str(ep): val for ep, val in eps.items()} for m, eps in avg_motor_major.items()}
    (outdir / "first10s_raw.json").write_text(json.dumps(raw_json))
    (outdir / "first10s_avg.json").write_text(json.dumps(avg_json))

def plot_episode_means(avg_motor_major, outpath: Path):
    motors = list(MOTOR_IDXS.keys())
    n = len(motors)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 2.2 * n))
    for i, m in enumerate(motors, 1):
        plt.subplot(n, 1, i)
        items = sorted(((int(ep), v) for ep, v in avg_motor_major.get(m, {}).items()), key=lambda x: x[0])
        if not items:
            plt.title(f"{m} (no data)"); continue
        xs = [ep for ep, _ in items]
        ys = [v for _, v in items]
        plt.scatter(xs, ys, s=14)
        plt.xlabel("episode_id"); plt.ylabel("mean @ first 10s"); plt.title(m)
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, type=str)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--use-state", action="store_true", help="Use observation.state (default)")
    ap.add_argument("--use-action", action="store_true", help="Use action instead of state")
    ap.add_argument("--outdir", type=Path, default=Path("initpos_stats"))
    ap.add_argument("--dl-workers", type=int, default=2, help="DataLoader workers PER EPISODE")
    ap.add_argument("--max-threads", type=int, default=max(1, os.cpu_count() // 2),
                    help="Max concurrent episodes")
    args = ap.parse_args()

    use_state = True
    if args.use_action: use_state = False
    if args.use_state:  use_state = True

    ds = LeRobotDataset(args.repo_id, root=args.root, tolerance_s=1e-4)
    n_eps = len(ds.episode_data_index["from"])

    # Parallel over episodes
    all_raw_ep_major, all_avg_ep_major = {}, {}
    with ThreadPoolExecutor(max_workers=args.max_threads) as ex:
        futures = {
            ex.submit(collect_first_10s_episode, ds, ep,
                      seconds=args.seconds, use_state=use_state,
                      dl_workers=args.dl_workers): ep
            for ep in range(n_eps)
        }
        for fut in as_completed(futures):
            ep = futures[fut]
            try:
                raw, avg = fut.result()
            except Exception as e:
                # Don't crash the run; record NaNs for this episode
                raw = {m: np.array([], dtype=np.float32) for m in MOTOR_IDXS}
                avg = {m: float("nan") for m in MOTOR_IDXS}
                print(f"[warn] episode {ep} failed: {e}")
            all_raw_ep_major[ep] = raw
            all_avg_ep_major[ep] = avg

    # Convert to motor-major for saving/plotting
    raw_motor_major = invert_episode_major(all_raw_ep_major)   # motor -> {ep: np.array}
    avg_motor_major = invert_episode_major(all_avg_ep_major)   # motor -> {ep: float}

    args.outdir.mkdir(parents=True, exist_ok=True)
    save_dicts(raw_motor_major, avg_motor_major, args.outdir)
    plot_episode_means(avg_motor_major, args.outdir / "episode_means.png")

    fps = float(ds.meta.fps)
    print(f"[done] fps={fps:.3f} | episodes={n_eps} | saved:")
    print(f"  - {args.outdir/'first10s_raw.json'}")
    print(f"  - {args.outdir/'first10s_avg.json'}")
    print(f"  - {args.outdir/'episode_means.png'}")

if __name__ == "__main__":
    main()
