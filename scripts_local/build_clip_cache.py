"""Pre-encode every frame of a LeRobotDataset with CLIP and dump to .npz.

Saves under ds.root/meta/clip_cache.npz. Keys: each image_key (e.g.
"observation.images.front", "observation.images.wrist") holds a (N_frames, 512)
float32 array indexed by global frame index in the dataset.

SARM CLIP is openai/clip-vit-base-patch32 (frozen). Caching is mathematically
identical to in-pipeline encoding (no augmentation in current cfg).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset


CLIP_NAME_DEFAULT = "openai/clip-vit-base-patch32"


@torch.no_grad()
def encode(ds: LeRobotDataset, image_keys: list[str], device: str, batch: int, clip_name: str) -> dict[str, np.ndarray]:
    proc = CLIPImageProcessor.from_pretrained(clip_name)
    model = CLIPModel.from_pretrained(clip_name).to(device).eval()
    feat_dim = int(model.config.projection_dim)
    n = len(ds)
    feats = {k: np.zeros((n, feat_dim), dtype=np.float32) for k in image_keys}

    pil_buf: dict[str, list[Image.Image]] = {k: [] for k in image_keys}
    idx_buf: list[int] = []

    def flush():
        if not idx_buf:
            return
        for k in image_keys:
            inp = proc(images=pil_buf[k], return_tensors="pt")
            inp = {kk: vv.to(device) for kk, vv in inp.items()}
            out = model.get_image_features(**inp)
            emb = (out.pooler_output if hasattr(out, "pooler_output") else out)
            feats[k][idx_buf] = emb.float().cpu().numpy()
        for k in image_keys:
            pil_buf[k].clear()
        idx_buf.clear()

    for i in tqdm(range(n), desc="encode"):
        sample = ds[i]
        for k in image_keys:
            img = sample[k]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            # Expect (C,H,W) channel-first
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.transpose(1, 2, 0)
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            pil_buf[k].append(Image.fromarray(img))
        idx_buf.append(i)
        if len(idx_buf) >= batch:
            flush()
    flush()
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--image-keys", default="observation.images.front,observation.images.wrist")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--clip-name", default=CLIP_NAME_DEFAULT)
    ap.add_argument("--out-name", default="clip_cache.npz")
    args = ap.parse_args()

    image_keys = [k.strip() for k in args.image_keys.split(",")]
    ds = LeRobotDataset(repo_id=args.repo_id)
    print(f"ds {args.repo_id}: {len(ds)} frames, image_keys={image_keys}, clip={args.clip_name}")

    feats = encode(ds, image_keys, args.device, args.batch, args.clip_name)
    out = ds.root / "meta" / args.out_name
    np.savez_compressed(out, **{k.replace(".", "__"): v for k, v in feats.items()})
    print(f"wrote {out}  ({sum(v.nbytes for v in feats.values()) / 1e6:.1f} MB raw)")


if __name__ == "__main__":
    main()
