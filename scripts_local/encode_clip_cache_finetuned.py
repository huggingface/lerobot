#!/usr/bin/env python3
"""Encode all dataset frames with a fine-tuned CLIP, save as clip_cache.npz.

Backs up existing clip_cache.npz first. After eval, restore via --restore.
"""
import argparse
import os
import sys
import shutil
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lerobot_policy_sarm/src"))

from transformers import CLIPModel, CLIPProcessor
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-ckpt", required=True, help="path to clip_model.safetensors")
    ap.add_argument("--dataset-repo", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--image-keys", nargs="+", default=["observation.images.front", "observation.images.wrist"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--restore", action="store_true", help="Restore original cache (.bak) and exit")
    args = ap.parse_args()

    ds = LeRobotDataset(repo_id=args.dataset_repo, root=args.dataset_root)
    cache_path = Path(ds.meta.root) / "meta" / "clip_cache.npz"
    bak_path = cache_path.with_suffix(".npz.bak")

    if args.restore:
        if bak_path.exists():
            shutil.move(str(bak_path), str(cache_path))
            print(f"[restore] {bak_path} → {cache_path}")
        else:
            print(f"[restore] no backup at {bak_path}")
        return

    # Backup
    if cache_path.exists() and not bak_path.exists():
        shutil.copy2(str(cache_path), str(bak_path))
        print(f"[backup] {cache_path} → {bak_path}")

    # Load fine-tuned CLIP
    clip_name = "openai/clip-vit-base-patch32"
    print(f"[load] CLIP {clip_name} + ft weights {args.clip_ckpt}")
    clip_model = CLIPModel.from_pretrained(clip_name).to(args.device)
    clip_proc = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    if Path(args.clip_ckpt).exists():
        sd = load_file(args.clip_ckpt)
        clip_model.load_state_dict(sd, strict=False)
        print(f"[load] loaded {len(sd)} tensors from finetune")
    else:
        print(f"[warn] no clip ckpt at {args.clip_ckpt} — using pretrained")
    clip_model.eval()

    n_total = ds.meta.total_frames
    print(f"[encode] {n_total} frames × {len(args.image_keys)} cams")

    out = {}
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=args.device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=args.device).view(1, 3, 1, 1)

    for k in args.image_keys:
        print(f"[encode] {k}")
        feats = np.zeros((n_total, 512), dtype=np.float32)
        with torch.no_grad():
            i = 0
            while i < n_total:
                batch_imgs = []
                for j in range(min(args.batch_size, n_total - i)):
                    item = ds[i + j]
                    img = item[k]  # tensor (C,H,W)
                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
                    if img.ndim == 4:
                        img = img[0]
                    if img.shape[0] in (1, 3):
                        img = img.transpose(1, 2, 0)
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    batch_imgs.append(Image.fromarray(img))
                proc = clip_proc(images=batch_imgs, return_tensors="pt")
                pixel_values = proc["pixel_values"].to(args.device)
                feat = clip_model.get_image_features(pixel_values=pixel_values)
                if hasattr(feat, "pooler_output"):
                    feat = feat.pooler_output
                feats[i : i + len(batch_imgs)] = feat.cpu().numpy()
                i += len(batch_imgs)
                if i % (args.batch_size * 20) == 0:
                    print(f"  {i}/{n_total}")
        out[k.replace(".", "__")] = feats

    np.savez(cache_path, **out)
    print(f"[save] {cache_path} ({sum(v.nbytes for v in out.values()) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
