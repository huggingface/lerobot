"""Calibrate halfway CNN threshold for ground truth.

For each demo episode, compute max P(succ) over stage-4 frames (true positives).
For each fake-success ACT rollout, compute max P(succ) over entire rollout (false
positives — user-confirmed fakes from BC c80 long 160k).

Find threshold T such that:
  - TPR (% demos > T) is as high as possible (target 1.0)
  - FPR (% fakes > T) is as low as possible (target 0.0)
"""
from __future__ import annotations
import argparse
import sys
import glob

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.transforms.functional import resize as tv_resize, normalize as tv_normalize
import imageio.v2 as imageio

sys.path.insert(0, "/home/dom-iva/github.com/orel/lerobot/lerobot/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class CNNCls(nn.Module):
    def __init__(self):
        super().__init__()
        b = tvm.resnet18()
        b.fc = nn.Linear(b.fc.in_features, 2)
        self.net = b
    def forward(self, x): return self.net(x)


def cnn_prob(model, img_tensor_uint8_or_float):
    # img_tensor: torch (C,H,W) or (H,W,C) uint8 or float [0,1]
    if img_tensor_uint8_or_float.ndim == 3 and img_tensor_uint8_or_float.shape[0] not in (1, 3):
        img_tensor_uint8_or_float = img_tensor_uint8_or_float.permute(2, 0, 1)
    t = img_tensor_uint8_or_float
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    t = tv_resize(t, [224, 224], antialias=True)
    t = tv_normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    with torch.no_grad():
        logits = model(t.unsqueeze(0).cuda())
        return float(torch.softmax(logits, -1)[0, 1].cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-ckpt", default="outputs/cnn_halfway_v1/best.pt")
    ap.add_argument("--demo-ds", default="local/sim_3stage_v2_full_v2_succonly_destale_tail30")
    ap.add_argument("--fake-dir", default="outputs/act_bc_v2_chunk80_long_halfwayCNN")
    ap.add_argument("--stage-idx", type=int, default=3)
    ap.add_argument("--n-demos", type=int, default=50)
    args = ap.parse_args()

    m = CNNCls().cuda().eval()
    m.load_state_dict(torch.load(args.cnn_ckpt, map_location="cuda", weights_only=True))

    print(f"Loading demos {args.demo_ds}...")
    ds = LeRobotDataset(args.demo_ds)
    n_demos = min(args.n_demos, ds.num_episodes)

    # True positives: max P over stage-4 frames
    demo_max_probs = []
    print(f"Computing demo TPs (n={n_demos})...")
    for ep in range(n_demos):
        ep_meta = ds.meta.episodes[ep]
        ep_from = int(ep_meta["dataset_from_index"])
        ends = list(ep_meta["sparse_subtask_end_frames"])
        if args.stage_idx >= len(ends):
            continue
        s4_end = int(ends[args.stage_idx])
        # Use last 5 frames of stage 4
        max_p = 0.0
        for k in range(max(0, s4_end - 4), s4_end + 1):
            item = ds[ep_from + k]
            img = item["observation.images.front"]
            p = cnn_prob(m, img)
            max_p = max(max_p, p)
        demo_max_probs.append(max_p)
    demo_max_probs = np.array(demo_max_probs)

    # False positives: max P over fake rollouts
    print(f"Computing fake FPs from {args.fake_dir}...")
    fake_max_probs = []
    fake_files = sorted(glob.glob(f"{args.fake_dir}/*.mp4"))
    for vf in fake_files:
        v = imageio.get_reader(vf)
        n = v.count_frames()
        max_p = 0.0
        # Sample every 5 frames
        for i in range(0, n, 5):
            fr = v.get_data(i)
            h, w = fr.shape[:2]
            # Crop top 18% (overlay bar) — eval used raw obs; saved video adds overlay text
            top_crop = int(h * 0.18)
            front = fr[top_crop:, :w//2]
            ft = torch.from_numpy(front).permute(2, 0, 1).float() / 255.0
            p = cnn_prob(m, ft)
            max_p = max(max_p, p)
        fake_max_probs.append(max_p)
        print(f"  {vf}: max_p={max_p:.3f}")
    fake_max_probs = np.array(fake_max_probs)

    print(f"\n=== Demo P(succ) max distribution (n={len(demo_max_probs)}) ===")
    print(f"  mean={demo_max_probs.mean():.3f} std={demo_max_probs.std():.3f}")
    print(f"  min={demo_max_probs.min():.3f} q10={np.quantile(demo_max_probs, 0.10):.3f} q50={np.quantile(demo_max_probs, 0.50):.3f} max={demo_max_probs.max():.3f}")

    print(f"\n=== Fake P(succ) max distribution (n={len(fake_max_probs)}) ===")
    print(f"  mean={fake_max_probs.mean():.3f} std={fake_max_probs.std():.3f}")
    print(f"  min={fake_max_probs.min():.3f} max={fake_max_probs.max():.3f}")

    print(f"\n=== Threshold sweep ===")
    print(f"{'thr':>6} {'TPR':>6} {'FPR':>6} {'TP':>4} {'FP':>4}")
    for thr in [0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999]:
        tpr = (demo_max_probs >= thr).mean()
        fpr = (fake_max_probs >= thr).mean()
        tp = int((demo_max_probs >= thr).sum())
        fp = int((fake_max_probs >= thr).sum())
        print(f"{thr:>6.4f} {tpr:>6.2f} {fpr:>6.2f} {tp:>4d} {fp:>4d}")


if __name__ == "__main__":
    main()
