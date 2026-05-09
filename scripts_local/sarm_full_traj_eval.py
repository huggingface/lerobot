"""Evaluate SARM on FULL trajectories (every frame, not strided val).

For comparing in-distribution full rollout behavior across ckpts.
Plots: GT stage, pred argmax, GT progress, pred progress per ep.
Also writes per-ep summary metrics (mean_mid, max, monotonicity).
"""
from __future__ import annotations
import argparse, sys
sys.path.insert(0, "/home/dom-iva/github.com/orel/lerobot/lerobot_policy_sarm/src")
sys.path.insert(0, "/home/dom-iva/github.com/orel/lerobot/lerobot/src")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.reward_model.sarm import SARMRewardConfig, SARMRewardProcessorStep
from lerobot_policy_sarm.eval_sarm_sim_assemble import run_episode, run_episode_multicam


def get_ep_subtasks(ds, ep_idx):
    df = ds.meta.episodes.to_pandas()
    row = df.loc[df["episode_index"] == ep_idx].iloc[0]
    rn, rs, re_ = row['sparse_subtask_names'], row['sparse_subtask_start_frames'], row['sparse_subtask_end_frames']
    if rn is None or not hasattr(rn, '__iter__') or isinstance(rn, str):
        return [], [], []
    return list(rn), [int(x) for x in list(rs)], [int(x) for x in list(re_)]


def gt_stage_per_frame(ep_len, names, starts, ends, global_names):
    arr = np.full(ep_len, -1, dtype=int)
    for n, s, e in zip(names, starts, ends):
        if n in global_names and s >= 0:
            arr[s:e+1] = global_names.index(n)
    return arr


def gt_progress(ep_len, starts, ends, props, global_names):
    """Cumulative-breakpoint GT progress per frame."""
    if isinstance(props, dict):
        ordered = [props[n] for n in global_names]
    else:
        ordered = list(props)
    cum = np.cumsum(ordered) / sum(ordered)
    out = np.zeros(ep_len)
    for s, (st, en) in enumerate(zip(starts, ends)):
        lo = 0.0 if s == 0 else cum[s-1]
        hi = cum[s]
        L = en - st + 1
        for t in range(st, min(en+1, ep_len)):
            tau = (t - st) / max(1, L - 1)
            out[t] = lo + tau * (hi - lo)
    return out


def pick_eps(ds, want_ids: list[int] | None = None, n_full: int = 4):
    df = ds.meta.episodes.to_pandas()
    cands_full = []
    cands_partial = []
    for _, r in df.iterrows():
        rn = r["sparse_subtask_names"]
        if rn is None or not hasattr(rn, '__iter__'):
            continue
        n = len(list(rn))
        if n >= 6 or (n >= 3 and "place_cover" in str(rn)):
            cands_full.append(int(r["episode_index"]))
        else:
            cands_partial.append(int(r["episode_index"]))
    if want_ids:
        return [i for i in want_ids if i in cands_full]
    # spread across full eps
    if len(cands_full) >= n_full:
        idxs = np.linspace(0, len(cands_full)-1, n_full, dtype=int)
        return [cands_full[i] for i in idxs]
    return cands_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to pretrained_model dir")
    ap.add_argument("--stats", default="domrachev03/sim_3stage_v4_success_train_fs")
    ap.add_argument("--full-ds", default="domrachev03/sim_3stage_v4_with_partials")
    ap.add_argument("--task", default="Three-stage assembly")
    ap.add_argument("--image-key", default="observation.images.wrist")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="Output dir")
    ap.add_argument("--eps", type=int, nargs="*", default=[2, 40, 80, 120])
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True, parents=True)

    cfg = SARMRewardConfig(
        type="sarm_ext", pretrained_path=args.ckpt, device=args.device, task=args.task,
        head_mode="sparse", reward_mode="dense", success_threshold=0.9,
        stats_dataset_repo_id=args.stats, verbose=False,
    )
    step = SARMRewardProcessorStep(config=cfg)
    global_names = list(step._model.config.sparse_subtask_names)
    print(f"global_names={global_names}")
    print(f"sparse_temporal_proportions={step._model.config.sparse_temporal_proportions}")

    ds = LeRobotDataset(repo_id=args.full_ds)
    ep_df = ds.meta.episodes.to_pandas()

    summary_lines = ["| ep | len | stages | mean_mid | max | mono | argmax_acc |", "|---|---|---|---|---|---|---|"]
    for ep in args.eps:
        if ep not in ep_df["episode_index"].values:
            continue
        row = ep_df.loc[ep_df["episode_index"] == ep].iloc[0]
        ep_start = int(row["dataset_from_index"])
        ep_end = int(row["dataset_to_index"])
        names, starts, ends = get_ep_subtasks(ds, ep)
        if not names:
            continue
        ep_len = ep_end - ep_start
        gt = gt_stage_per_frame(ep_len, names, starts, ends, global_names)

        image_keys = list(step._model.config.image_keys or [step._model.config.image_key])
        if len(image_keys) > 1:
            prog, sp, _ = run_episode_multicam(
                step._model, step._preprocess, image_keys,
                step._state_key, args.task, ds, ep_start, ep_end, head_mode="sparse",
            )
        else:
            prog, sp, _ = run_episode(step, ds, ep_start, ep_end, head_mode="sparse")
        argmax = sp.argmax(axis=1)
        gp = gt_progress(ep_len, starts, ends, step._model.config.sparse_temporal_proportions, global_names)

        # Metrics
        valid = gt >= 0
        mid = (np.abs(gp - 0.5) < 0.1) & valid
        mean_mid = float(prog[mid].mean()) if mid.any() else float('nan')
        max_p = float(prog.max())
        diffs = np.diff(prog)
        mono = float((diffs >= -0.01).mean()) if len(diffs) else 1.0
        acc = float((argmax[valid] == gt[valid]).mean()) if valid.any() else float('nan')
        summary_lines.append(f"| {ep} | {ep_len} | {len(names)} | {mean_mid:.3f} | {max_p:.3f} | {mono:.3f} | {acc:.3f} |")
        print(summary_lines[-1])

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        ax = axes[0]
        ax.plot(gt, label="GT stage", lw=2, color="black")
        ax.plot(argmax, label="pred argmax", lw=1.0, alpha=0.85, color="tab:red")
        ax.set_yticks(range(len(global_names)))
        ax.set_yticklabels([f"{i}:{n[:14]}" for i, n in enumerate(global_names)])
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"ep={ep} (len={ep_len}, stages={len(names)})  ckpt={Path(args.ckpt).parent.parent.name}")

        ax = axes[1]
        im = ax.imshow(sp.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(global_names)))
        ax.set_yticklabels([n[:14] for n in global_names])
        ax.set_ylabel("stage prob")
        plt.colorbar(im, ax=ax, fraction=0.02)

        ax = axes[2]
        ax.plot(gp, label="GT progress", lw=2, color="black")
        ax.plot(prog, label="pred progress", lw=1, alpha=0.85, color="tab:blue")
        ax.set_ylabel("progress")
        ax.set_xlabel("frame")
        ax.legend(loc="upper left")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        for st in starts:
            for axx in axes:
                axx.axvline(st, color="gray", lw=0.5, alpha=0.5)

        out_path = out_dir / f"full_ep{ep:03d}.png"
        fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
