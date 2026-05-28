"""Sanity: feed seen training frames through a trained ACT, compare predicted
action to recorded action. If ACT learned demos, predicted ≈ recorded.

Resets policy queue every frame so each call returns the FIRST action of a
fresh chunk — directly comparable to the recorded action at that frame.

Usage:
    DISPLAY=:1 CUDA_VISIBLE_DEVICES=0 uv run python scripts_local/act_on_train_sanity.py \\
        --pretrained outputs/act_v2_first4_bc_v11/checkpoints/080000/pretrained_model \\
        --train-ds local/sim_3stage_v2_first4_destale_tail30 \\
        --n-eps 3 --max-frames-per-ep 50
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/dom-iva/github.com/orel/lerobot/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--train-ds", required=True)
    ap.add_argument("--task", default="Three-stage assembly")
    ap.add_argument("--n-eps", type=int, default=3)
    ap.add_argument("--max-frames-per-ep", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Loading dataset %s", args.train_ds)
    ds = LeRobotDataset(args.train_ds)

    logging.info("Loading policy from %s", args.pretrained)
    cfg = PreTrainedConfig.from_pretrained(args.pretrained)
    policy = ACTPolicy.from_pretrained(args.pretrained, config=cfg).to(args.device).eval()
    logging.info("policy chunk=%s n_obs=%s", cfg.chunk_size, cfg.n_obs_steps)

    n_eps = min(args.n_eps, ds.num_episodes)
    all_diffs = []
    per_dim_diffs = []  # list of (5,)
    gripper_match = []

    with torch.no_grad():
        for ep in range(n_eps):
            ep_meta = ds.meta.episodes[ep]
            ep_from = int(ep_meta["dataset_from_index"])
            ep_to = int(ep_meta["dataset_to_index"])
            ep_len = ep_to - ep_from
            n_fr = min(ep_len, args.max_frames_per_ep)
            print(f"\n=== ep {ep} (len {ep_len}, sampling {n_fr} frames) ===")

            ep_diffs = []
            ep_per_dim = []
            ep_grip = []
            for k in range(n_fr):
                idx = ep_from + k
                item = ds[idx]
                # Build batch
                batch = {
                    "observation.images.front": item["observation.images.front"].unsqueeze(0).to(args.device),
                    "observation.images.wrist": item["observation.images.wrist"].unsqueeze(0).to(args.device),
                    "observation.state": item["observation.state"].unsqueeze(0).to(args.device),
                    "task": [args.task],
                }
                policy.reset()  # CLEAR queue → next select_action returns first of new chunk
                pred = policy.select_action(batch).squeeze(0).cpu().numpy()  # (5,)
                rec = item["action"].numpy()  # (5,)
                d = pred - rec
                l2 = float(np.linalg.norm(d))
                ep_diffs.append(l2)
                ep_per_dim.append(np.abs(d))
                # gripper: pred & rec are continuous [-1, 1]; match if same sign
                ep_grip.append(int(np.sign(pred[4]) == np.sign(rec[4]) or (abs(rec[4]) < 0.1 and abs(pred[4]) < 0.5)))

                if k < 5 or k == n_fr - 1:
                    print(f"  frame {k:3d}: pred={[f'{x:+.3f}' for x in pred]}  rec={[f'{x:+.3f}' for x in rec]}  L2={l2:.3f}")

            mean_l2 = float(np.mean(ep_diffs))
            mean_per_dim = np.stack(ep_per_dim).mean(0)
            grip_rate = float(np.mean(ep_grip))
            print(f"  ep summary: mean L2={mean_l2:.3f}  per-dim mean abs= {[f'{x:.3f}' for x in mean_per_dim]}  gripper_match={grip_rate:.2%}")
            all_diffs.append(mean_l2)
            per_dim_diffs.append(mean_per_dim)
            gripper_match.append(grip_rate)

    print("\n=== OVERALL ===")
    print(f"mean L2 across eps: {np.mean(all_diffs):.3f}  (std {np.std(all_diffs):.3f})")
    pd = np.stack(per_dim_diffs).mean(0)
    print(f"per-dim mean abs error (dx, dy, dz, dyaw, grip): {[f'{x:.3f}' for x in pd]}")
    print(f"gripper sign-match rate: {np.mean(gripper_match):.2%}")
    print()
    print("Action MIN_MAX normalization range is [-1, 1] (5-dim). So L2~0.1 means ~10% per-dim deviation.")
    print("Expected for healthy ACT on TRAINED frames: L2 < 0.2, per-dim < 0.1, gripper > 0.95")


if __name__ == "__main__":
    main()
