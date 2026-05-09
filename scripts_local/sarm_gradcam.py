#!/usr/bin/env python3
"""Grad-CAM visualization for SARM stage prediction.

Per-frame heatmap: which input pixels raise the predicted stage probability.
Bypasses CLIP cache, runs CLIP fresh, backprops through frozen ViT to input.
"""
import argparse
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pyarrow.parquet as pq
import cv2
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lerobot_policy_sarm/src"))

from transformers import CLIPModel, CLIPProcessor

from lerobot_policy_sarm.modeling_sarm import SARMRewardModel
from lerobot_policy_sarm.configuration_sarm import SARMConfig


def load_sarm(ckpt_path: str, device: str = "cuda") -> SARMRewardModel:
    cfg_path = Path(ckpt_path) / "config.json"
    cfg_dict = json.loads(cfg_path.read_text())
    cfg = SARMConfig(**{k: v for k, v in cfg_dict.items() if k in SARMConfig.__dataclass_fields__})
    model = SARMRewardModel(cfg)
    sd = {}
    from safetensors.torch import load_file
    sd = load_file(str(Path(ckpt_path) / "model.safetensors"))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def decode_video_frames(video_path: str, ts_from: float, ts_to: float, n_frames: int):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_f = int(round(ts_from * fps))
    end_f = int(round(ts_to * fps))
    total = end_f - start_f + 1
    indices = np.linspace(0, total - 1, n_frames).astype(int) + start_f
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed read frame {idx}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames  # list of HxWxC uint8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to SARM .../pretrained_model")
    ap.add_argument("--dataset-root", required=True, help="local dataset root, e.g. ~/.cache/.../sim_3stage_v2_full_v2_nostale")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--target-frame-frac", type=float, default=0.5, help="0..1 frame in episode to backprop")
    ap.add_argument("--cam", default="front", choices=["front", "wrist"])
    ap.add_argument("--out", default="outputs/sarm_gradcam")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-context", type=int, default=8, help="frames in window (n_obs_steps)")
    ap.add_argument("--gap", type=int, default=5, help="frame_gap")
    ap.add_argument("--target-stage", type=int, default=None, help="stage idx to backprop (None=argmax)")
    ap.add_argument("--task", default="Three-stage assembly")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device

    # Load SARM
    print(f"loading SARM from {args.ckpt}")
    sarm = load_sarm(args.ckpt, device)

    # Load CLIP fresh (NO cache) — must be on device + grad-enabled
    clip_name = "openai/clip-vit-base-patch32"
    print(f"loading CLIP {clip_name}")
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_proc = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)  # frozen, gradients flow but params not updated

    # Load episode metadata
    ds_root = Path(args.dataset_root).expanduser()
    ep_dir = ds_root / "meta" / "episodes"
    files = sorted(ep_dir.rglob("*.parquet"))
    import pandas as pd
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    ep = df.iloc[args.episode]
    print(f"ep={args.episode} task={ep.tasks[0]} length={ep.length}")

    # Get video paths for both cams
    vids = {}
    for cam in ["front", "wrist"]:
        ch = ep[f"videos/observation.images.{cam}/chunk_index"]
        fi = ep[f"videos/observation.images.{cam}/file_index"]
        ts0 = ep[f"videos/observation.images.{cam}/from_timestamp"]
        ts1 = ep[f"videos/observation.images.{cam}/to_timestamp"]
        vp = ds_root / f"videos/observation.images.{cam}/chunk-{ch:03d}/file-{fi:03d}.mp4"
        vids[cam] = (str(vp), ts0, ts1)

    # Sample n_context frames evenly + a target around target_frame_frac
    ep_len = int(ep.length)
    target_frame = int(args.target_frame_frac * (ep_len - 1))
    # Window of n_context frames ending at target_frame, with frame_gap spacing
    starts = [max(0, target_frame - i * args.gap) for i in range(args.n_context - 1, -1, -1)]
    indices = starts  # sorted ascending
    ts_per_frame = (vids[args.cam][2] - vids[args.cam][1]) / max(1, ep_len - 1)
    # decode frames using ffmpeg (libdav1d for AV1) at exact timestamps
    import subprocess, tempfile
    fps_default = 20  # lerobot ds fps
    frames_cam = {}
    for cam in ["front", "wrist"]:
        path, ts0, ts1 = vids[cam]
        out = []
        for idx in indices:
            ts = ts0 + idx / fps_default
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_png = tf.name
            cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(ts),
                   "-c:v", "libdav1d", "-i", path,
                   "-frames:v", "1", "-update", "1", tmp_png]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                # retry without dav1d (e.g. h264 video)
                cmd2 = ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(ts),
                        "-i", path, "-frames:v", "1", "-update", "1", tmp_png]
                subprocess.run(cmd2, check=True, capture_output=True)
            img = np.array(Image.open(tmp_png).convert("RGB"))
            os.unlink(tmp_png)
            out.append(img)
        frames_cam[cam] = out
    print(f"target_frame={target_frame} window_indices={indices}")

    # Process images via CLIP processor — keep raw tensors for gradient
    def prep_clip_inputs(frames_list):
        pil = [Image.fromarray(f) for f in frames_list]
        out = clip_proc(images=pil, return_tensors="pt")
        return out["pixel_values"].to(device)  # (T,3,224,224)

    pix_front = prep_clip_inputs(frames_cam["front"]).requires_grad_(True)
    pix_wrist = prep_clip_inputs(frames_cam["wrist"]).requires_grad_(True)

    # Hook on last vision encoder block to capture activations + gradients
    last_block = clip_model.vision_model.encoder.layers[-1]
    feat_front, feat_wrist = {}, {}
    grad_front, grad_wrist = {}, {}

    def make_hook(store_feat, store_grad, tag):
        def fwd_hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            store_feat["x"] = o
            o.retain_grad()
            def bwd_hook(grad):
                store_grad["x"] = grad
            o.register_hook(bwd_hook)
        return fwd_hook

    h1 = last_block.register_forward_hook(make_hook(feat_front, grad_front, "front"))
    print("running CLIP front")
    out_front = clip_model.vision_model(pix_front)
    feat_front["x_pooled"] = out_front.pooler_output if hasattr(out_front, "pooler_output") else out_front[1]
    img_emb_front = clip_model.visual_projection(feat_front["x_pooled"])  # (T, 512)
    h1.remove()

    h2 = last_block.register_forward_hook(make_hook(feat_wrist, grad_wrist, "wrist"))
    print("running CLIP wrist")
    out_wrist = clip_model.vision_model(pix_wrist)
    feat_wrist["x_pooled"] = out_wrist.pooler_output if hasattr(out_wrist, "pooler_output") else out_wrist[1]
    img_emb_wrist = clip_model.visual_projection(feat_wrist["x_pooled"])  # (T, 512)
    h2.remove()

    # Stack into (B=1, N=2, T, 512) — front first then wrist (matches train order)
    img_emb = torch.stack([img_emb_front, img_emb_wrist], dim=0).unsqueeze(0)  # (1,2,T,512)
    print(f"img_emb shape={img_emb.shape}")

    # Encode text fresh
    txt_in = clip_proc.tokenizer([args.task], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        txt_out = clip_model.get_text_features(**txt_in)
    text_emb = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out  # (1, 512)

    # State: zeros (we don't have it easily, plus state shouldn't drive vision-grad)
    state_dim = sarm.config.max_state_dim
    state_feat = torch.zeros(1, args.n_context, state_dim, device=device)

    # Run SARM stage_model (sparse)
    lengths = torch.tensor([args.n_context], dtype=torch.int32, device=device)
    stage_logits = sarm.stage_model(img_emb, text_emb, state_feat, lengths, scheme="sparse")
    # stage_logits: (1, T, num_classes)
    target_t = args.n_context - 1  # last (current) frame
    if args.target_stage is None:
        target_stage = stage_logits[0, target_t].argmax().item()
    else:
        target_stage = args.target_stage
    print(f"target_t={target_t} predicted_stage={stage_logits[0, target_t].argmax().item()} target_stage={target_stage} logit={stage_logits[0,target_t,target_stage].item():.3f}")

    # Backprop selected stage logit
    target_logit = stage_logits[0, target_t, target_stage]
    target_logit.backward()

    # Grad-CAM compute per cam
    def gradcam(feat_dict, grad_dict, raw_img_uint8, tag=""):
        # feat: (T, 1+P, D), grad: (T, 1+P, D)
        feat = feat_dict["x"]
        grad = grad_dict["x"]
        print(f"[{tag}] feat shape={feat.shape} feat range=[{feat.min():.3f},{feat.max():.3f}]")
        print(f"[{tag}] grad shape={grad.shape} grad range=[{grad.min():.6f},{grad.max():.6f}] nonzero={(grad!=0).any().item()}")
        f_t = feat[target_t]  # (1+P, D)
        g_t = grad[target_t]
        print(f"[{tag}] g_t range at target_t=[{g_t.min():.6f},{g_t.max():.6f}] g_t.abs.sum()={g_t.abs().sum().item():.4f}")
        f_p = f_t[1:]
        g_p = g_t[1:]
        # Direct contribution: per-patch (f_p * g_p).sum(D) — saliency-style
        prod = f_p * g_p
        print(f"[{tag}] prod stats: min={prod.min():.4e} max={prod.max():.4e} mean={prod.mean():.4e} std={prod.std():.4e}")
        cam_raw = prod.sum(dim=-1)
        print(f"[{tag}] cam_raw range=[{cam_raw.min():.4e},{cam_raw.max():.4e}] std={cam_raw.std():.4e}")
        # Try magnitude only via |grad|
        cam_grad_only = g_p.abs().sum(dim=-1)
        print(f"[{tag}] |grad|.sum range=[{cam_grad_only.min():.4e},{cam_grad_only.max():.4e}]")
        # Alternative: feature L2 norm weighted by grad L2
        cam_alt = (g_p.abs() * f_p.abs()).sum(dim=-1)
        print(f"[{tag}] |g|*|f| range=[{cam_alt.min():.4e},{cam_alt.max():.4e}]")
        # Use cam_alt as saliency (always positive)
        cam_raw = cam_alt
        cam = cam_raw.abs()  # use abs to capture both positive + negative contributions
        if cam.max() > 0:
            cam = cam / cam.max()
        P = cam.shape[0]
        side = int(math.sqrt(P))
        cam = cam.detach().cpu().numpy().reshape(side, side)
        H, W = raw_img_uint8.shape[:2]
        cam_up = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
        return cam_up

    # ViT pooler uses only CLS — patch gradients are zero.
    # Use input-pixel gradient saliency instead.
    def pixel_saliency(pix_grad_t, raw_img_uint8, tag=""):
        # pix_grad_t (3, 224, 224)
        sal = pix_grad_t.abs().sum(dim=0)  # (224,224)
        sal = sal.detach().cpu().numpy()
        # Pool to 7x7 patch grid (CLIP ViT-B/32 patch_size=32) then upsample smoothly
        ps = 32  # patch size
        n_p = 224 // ps  # 7
        pooled = sal.reshape(n_p, ps, n_p, ps).mean(axis=(1, 3))  # (7,7)
        if pooled.max() > 0:
            pooled = pooled / pooled.max()
        H, W = raw_img_uint8.shape[:2]
        sal_up = cv2.resize(pooled, (W, H), interpolation=cv2.INTER_CUBIC)
        sal_up = np.clip(sal_up, 0, 1)
        print(f"[{tag}] pooled (7x7) range=[{pooled.min():.3f},{pooled.max():.3f}] argmax_patch={pooled.argmax()}")
        return sal_up

    cam_front = pixel_saliency(pix_front.grad[target_t], frames_cam["front"][target_t], "front")
    cam_wrist = pixel_saliency(pix_wrist.grad[target_t], frames_cam["wrist"][target_t], "wrist")

    # Overlay + save
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (cam_name, raw, cam_map) in enumerate([
        ("front", frames_cam["front"][target_t], cam_front),
        ("wrist", frames_cam["wrist"][target_t], cam_wrist),
    ]):
        axes[i, 0].imshow(raw); axes[i, 0].set_title(f"{cam_name} raw"); axes[i, 0].axis("off")
        axes[i, 1].imshow(cam_map, cmap="jet"); axes[i, 1].set_title(f"{cam_name} grad-cam"); axes[i, 1].axis("off")
        # blend
        heat = (cm.jet(cam_map)[:, :, :3] * 255).astype(np.uint8)
        blend = (0.5 * raw + 0.5 * heat).astype(np.uint8)
        axes[i, 2].imshow(blend); axes[i, 2].set_title(f"{cam_name} overlay"); axes[i, 2].axis("off")
    fig.suptitle(f"ep{args.episode} frame{target_frame}/{ep_len-1} stage={target_stage} task={args.task}")
    out_png = Path(args.out) / f"ep{args.episode}_f{target_frame}_stage{target_stage}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
