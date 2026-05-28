"""Reproduce training loss on saved ckpt to verify model integrity.

If saved ckpt produces same recon loss as final training step on a fresh batch,
the ckpt is intact and the issue is in select_action / inference path.
If recon loss is much higher, ckpt mismatched with stats or norms.

Also dumps normalized input/target distributions to spot stats issues.
"""
from __future__ import annotations
import argparse
import sys

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
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = LeRobotDataset(args.train_ds)
    cfg = PreTrainedConfig.from_pretrained(args.pretrained)
    policy = ACTPolicy.from_pretrained(args.pretrained, config=cfg).to(args.device)
    print(f"chunk={cfg.chunk_size} n_action_steps={cfg.n_action_steps} use_vae={cfg.use_vae} kl_weight={cfg.kl_weight}")

    # Sample frames from ep 0, build a chunk-shaped batch
    ep_from = int(ds.meta.episodes[0]["dataset_from_index"])
    ep_to = int(ds.meta.episodes[0]["dataset_to_index"])
    chunk = cfg.chunk_size

    # Build batch of N items, each a chunk-aligned slice from ep0
    states = []
    images_front = []
    images_wrist = []
    actions = []  # chunk_size actions
    for k in range(args.n):
        i = ep_from + k
        item = ds[i]
        states.append(item["observation.state"])
        images_front.append(item["observation.images.front"])
        images_wrist.append(item["observation.images.wrist"])
        # Get chunk-actions from i to i+chunk
        act_chunk = []
        for j in range(chunk):
            idx = min(i + j, ep_to - 1)
            act_chunk.append(ds[idx]["action"])
        actions.append(torch.stack(act_chunk))  # (chunk, 5)

    batch = {
        "observation.state": torch.stack(states).to(args.device),
        "observation.images.front": torch.stack(images_front).to(args.device),
        "observation.images.wrist": torch.stack(images_wrist).to(args.device),
        "action": torch.stack(actions).to(args.device),  # (N, chunk, 5)
        "action_is_pad": torch.zeros(args.n, chunk, dtype=torch.bool, device=args.device),
        "task": ["Three-stage assembly"] * args.n,
    }

    # Set eval mode
    policy.eval()
    with torch.no_grad():
        try:
            loss_dict = policy.forward(batch)
            print(f"eval mode loss_dict: { {k: float(v) if torch.is_tensor(v) else v for k, v in loss_dict.items() if not isinstance(v, dict)} }")
        except Exception as e:
            print(f"eval forward failed: {e}")

    # Set train mode for original loss recompute
    policy.train()
    with torch.no_grad():
        result = policy.forward(batch)
        print(f"train mode result type: {type(result)}")
        if isinstance(result, tuple):
            loss = result[0]
            extras = result[1] if len(result) > 1 else {}
            print(f"  loss (tensor): {float(loss):.4f}")
            if isinstance(extras, dict):
                for k, v in extras.items():
                    if torch.is_tensor(v):
                        print(f"  extras[{k}]: {float(v):.4f}")
                    else:
                        print(f"  extras[{k}]: {v}")
        elif isinstance(result, dict):
            for k, v in result.items():
                if torch.is_tensor(v):
                    print(f"  {k}: {float(v):.4f}")

    # Show normalized action stats
    if hasattr(policy, "normalize_targets"):
        with torch.no_grad():
            n_targets = policy.normalize_targets({"action": batch["action"]})
            na = n_targets["action"]
            print(f"normalized action: shape={tuple(na.shape)}  min={na.min().item():.3f}  max={na.max().item():.3f}  mean={na.mean().item():.3f}  std={na.std().item():.3f}")
            print(f"raw action ep0:    min={batch['action'].min().item():.3f}  max={batch['action'].max().item():.3f}  mean={batch['action'].mean().item():.3f}")


if __name__ == "__main__":
    main()
