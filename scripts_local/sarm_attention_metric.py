#!/usr/bin/env python3
"""Numeric attention sanity metric for SARM ckpts.

For N test frames sampled uniformly across the val dataset, computes pixel-grad
saliency on front + wrist cameras (same path as sarm_gradcam.py), pools to
7x7 ViT patch grid, then derives:

  - workspace_focus_rate : fraction of saliency mass in lower 60% of image
                            (the table/workspace region; objects + gripper).
  - cam_consistency      : Pearson correlation of front+wrist saliency
                            spatial patterns over the test set (high = both
                            cams agree on where the action is).
  - off_object_rate      : fraction of saliency mass on top corners + borders
                            (background/edges that shouldn't drive task progress).

Outputs JSON + prints summary.
"""
import argparse
import json
import math
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import pyarrow.parquet as pq
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lerobot_policy_sarm/src"))

from transformers import CLIPModel, CLIPProcessor
from lerobot_policy_sarm.modeling_sarm import SARMRewardModel
from lerobot_policy_sarm.configuration_sarm import SARMConfig


def load_sarm(ckpt_path: str, device: str = "cuda"):
    cfg_path = Path(ckpt_path) / "config.json"
    cfg_dict = json.loads(cfg_path.read_text())
    cfg = SARMConfig(**{k: v for k, v in cfg_dict.items() if k in SARMConfig.__dataclass_fields__})
    model = SARMRewardModel(cfg)
    sd = load_file(str(Path(ckpt_path) / "model.safetensors"))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, cfg


def decode_one(path: str, ts: float) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp = tf.name
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(ts),
           "-c:v", "libdav1d", "-i", path,
           "-frames:v", "1", "-update", "1", tmp]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        cmd2 = ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(ts),
                "-i", path, "-frames:v", "1", "-update", "1", tmp]
        subprocess.run(cmd2, check=True, capture_output=True)
    img = np.array(Image.open(tmp).convert("RGB"))
    os.unlink(tmp)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SARM pretrained_model dir")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--n-frames", type=int, default=64, help="number of test frames")
    ap.add_argument("--n-eps", type=int, default=8, help="distinct episodes to sample from")
    ap.add_argument("--n-context", type=int, default=8)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--out", default="outputs/sarm_attention_metric")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default="Three-stage assembly")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    sarm, cfg = load_sarm(args.ckpt, device)

    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_proc = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    # Use finetuned CLIP if available
    ft_clip_path = Path(args.ckpt) / "clip_model.safetensors"
    if not getattr(cfg, "freeze_clip", True) and getattr(sarm, "clip_model", None) is not None:
        clip_model = sarm.clip_model
        print(f"[attn] using SARM-attached fine-tuned CLIP")
    elif ft_clip_path.exists():
        sd = load_file(str(ft_clip_path))
        clip_model.load_state_dict(sd, strict=False)
        print(f"[attn] loaded standalone CLIP weights from {ft_clip_path}")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # Load dataset eps
    ds_root = Path(args.dataset_root).expanduser()
    files = sorted((ds_root / "meta" / "episodes").rglob("*.parquet"))
    import pandas as pd
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)

    # Sample frames evenly across n_eps episodes
    ep_indices = np.linspace(0, len(df) - 1, args.n_eps).astype(int)
    n_per_ep = max(1, args.n_frames // args.n_eps)
    samples = []  # list of (ep_idx, target_frame)
    for ep in ep_indices:
        ep_len = int(df.iloc[ep].length)
        fracs = np.linspace(0.1, 0.9, n_per_ep)
        for f in fracs:
            samples.append((int(ep), int(f * (ep_len - 1))))
    samples = samples[:args.n_frames]
    print(f"[attn] {len(samples)} samples from {args.n_eps} eps")

    fps_default = 20

    # CLIP normalization
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    metrics_per_frame = []

    for ep, target_frame in samples:
        ep_meta = df.iloc[ep]
        starts = [max(0, target_frame - i * args.gap) for i in range(args.n_context - 1, -1, -1)]
        cams = ["front", "wrist"]
        sal_per_cam = {}
        for cam in cams:
            ch = ep_meta[f"videos/observation.images.{cam}/chunk_index"]
            fi = ep_meta[f"videos/observation.images.{cam}/file_index"]
            ts0 = float(ep_meta[f"videos/observation.images.{cam}/from_timestamp"])
            vp = ds_root / f"videos/observation.images.{cam}/chunk-{ch:03d}/file-{fi:03d}.mp4"
            frames = []
            for idx in starts:
                ts = ts0 + idx / fps_default
                frames.append(decode_one(str(vp), ts))
            pil = [Image.fromarray(f) for f in frames]
            pix = clip_proc(images=pil, return_tensors="pt")["pixel_values"].to(device)
            pix.requires_grad_(True)
            sal_per_cam[cam] = (pix, frames)

        # Forward both cams + SARM
        f_pool = clip_model.vision_model(sal_per_cam["front"][0]).pooler_output
        w_pool = clip_model.vision_model(sal_per_cam["wrist"][0]).pooler_output
        f_emb = clip_model.visual_projection(f_pool).unsqueeze(0)  # (1,T,512)
        w_emb = clip_model.visual_projection(w_pool).unsqueeze(0)
        img_emb = torch.stack([f_emb.squeeze(0), w_emb.squeeze(0)], dim=0).unsqueeze(0)  # (1,N,T,512)

        tok = clip_proc.tokenizer([args.task], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            txt = clip_model.get_text_features(**tok)
            text_emb = (txt.pooler_output if hasattr(txt, "pooler_output") else txt)

        state = torch.zeros(1, args.n_context, sarm.config.max_state_dim, device=device)
        lengths = torch.tensor([args.n_context], dtype=torch.int32, device=device)
        stage_logits = sarm.stage_model(img_emb, text_emb, state, lengths, scheme="sparse")
        target_t = args.n_context - 1
        target_stage = stage_logits[0, target_t].argmax().item()
        target_logit = stage_logits[0, target_t, target_stage]
        target_logit.backward()

        ps = 32
        n_p = 224 // ps
        cam_grids = {}
        for cam in cams:
            pix, _ = sal_per_cam[cam]
            sal = pix.grad[target_t].abs().sum(dim=0).detach().cpu().numpy()  # (224,224)
            pooled = sal.reshape(n_p, ps, n_p, ps).mean(axis=(1, 3))  # (7,7)
            if pooled.max() > 0:
                pooled = pooled / pooled.max()
            cam_grids[cam] = pooled

        # Workspace focus: lower 60% of image vs upper 40%
        for cam in cams:
            grid = cam_grids[cam]
            lower_rows = grid[3:, :]  # rows 3-6 (lower 4/7 = 57%)
            ws_focus = lower_rows.sum() / max(grid.sum(), 1e-8)

            # Off-object: top corners + edge fringe
            corners = grid[0:2, 0:2].sum() + grid[0:2, -2:].sum()
            off_obj = corners / max(grid.sum(), 1e-8)

            # Center-of-mass coords (0..1 normalized)
            ys, xs = np.indices(grid.shape) / max(1, n_p - 1)
            mass = grid.sum()
            com_y = (grid * ys).sum() / max(mass, 1e-8)
            com_x = (grid * xs).sum() / max(mass, 1e-8)

            metrics_per_frame.append({
                "ep": ep, "frame": target_frame, "cam": cam,
                "workspace_focus": float(ws_focus),
                "off_object_corners": float(off_obj),
                "com_x": float(com_x), "com_y": float(com_y),
                "predicted_stage": target_stage,
            })

        # Cam pattern consistency: spearman between front+wrist 7x7 grids (flattened)
        from scipy.stats import spearmanr
        rho, _ = spearmanr(cam_grids["front"].flatten(), cam_grids["wrist"].flatten())
        metrics_per_frame.append({
            "ep": ep, "frame": target_frame, "cam": "both",
            "front_wrist_pattern_corr": float(rho if not math.isnan(rho) else 0.0),
            "predicted_stage": target_stage,
        })

    # Aggregate
    front = [m for m in metrics_per_frame if m["cam"] == "front"]
    wrist = [m for m in metrics_per_frame if m["cam"] == "wrist"]
    both = [m for m in metrics_per_frame if m["cam"] == "both"]
    summary = {
        "n_frames": len(samples),
        "front_workspace_focus_mean": float(np.mean([m["workspace_focus"] for m in front])),
        "wrist_workspace_focus_mean": float(np.mean([m["workspace_focus"] for m in wrist])),
        "front_off_corners_mean": float(np.mean([m["off_object_corners"] for m in front])),
        "wrist_off_corners_mean": float(np.mean([m["off_object_corners"] for m in wrist])),
        "front_com_y_mean": float(np.mean([m["com_y"] for m in front])),
        "wrist_com_y_mean": float(np.mean([m["com_y"] for m in wrist])),
        "front_com_y_std": float(np.std([m["com_y"] for m in front])),
        "wrist_com_y_std": float(np.std([m["com_y"] for m in wrist])),
        "cam_pattern_corr_mean": float(np.mean([m["front_wrist_pattern_corr"] for m in both])),
    }
    summary["sanity_score"] = float(np.mean([
        summary["front_workspace_focus_mean"],
        summary["wrist_workspace_focus_mean"],
        max(0.0, summary["cam_pattern_corr_mean"]),
        1.0 - summary["front_off_corners_mean"],
        1.0 - summary["wrist_off_corners_mean"],
    ]))

    print("\n=== ATTENTION SANITY ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")

    out_path = out_dir / f"{Path(args.ckpt).parent.parent.name}_attn.json"
    out_path.write_text(json.dumps({"summary": summary, "per_frame": metrics_per_frame}, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
