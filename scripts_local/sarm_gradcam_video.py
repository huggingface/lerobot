#!/usr/bin/env python3
"""Per-frame Grad-CAM saliency video for a SARM episode.

For every frame t in episode, computes pixel saliency w.r.t. predicted stage at t,
overlays heatmap on front+wrist images, stitches into mp4.
"""
import argparse
import os
import sys
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm

import pyarrow.parquet as pq
import json
import cv2

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
    from safetensors.torch import load_file
    sd = load_file(str(Path(ckpt_path) / "model.safetensors"))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default="outputs/sarm_gradcam_video")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-context", type=int, default=8)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--task", default="Three-stage assembly")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5, help="overlay blend (0..1)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"loading SARM from {args.ckpt}")
    sarm = load_sarm(args.ckpt, device)

    clip_name = "openai/clip-vit-base-patch32"
    print(f"loading CLIP {clip_name}")
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_proc = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    ds_root = Path(args.dataset_root).expanduser()
    files = sorted((ds_root / "meta" / "episodes").rglob("*.parquet"))
    import pandas as pd
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    ep = df.iloc[args.episode]
    ep_len = int(ep.length)
    print(f"ep={args.episode} task={ep.tasks[0]} length={ep_len}")

    vids = {}
    for cam in ["front", "wrist"]:
        ch = ep[f"videos/observation.images.{cam}/chunk_index"]
        fi = ep[f"videos/observation.images.{cam}/file_index"]
        ts0 = ep[f"videos/observation.images.{cam}/from_timestamp"]
        ts1 = ep[f"videos/observation.images.{cam}/to_timestamp"]
        vp = ds_root / f"videos/observation.images.{cam}/chunk-{ch:03d}/file-{fi:03d}.mp4"
        vids[cam] = (str(vp), float(ts0), float(ts1))

    fps_default = 20

    # Cache decoded frames for entire episode
    print("decoding frames...")
    raw_frames = {}
    for cam in ["front", "wrist"]:
        path, ts0, _ = vids[cam]
        raw_frames[cam] = []
        for idx in range(ep_len):
            ts = ts0 + idx / fps_default
            raw_frames[cam].append(decode_one(path, ts))
    H, W = raw_frames["front"][0].shape[:2]
    print(f"decoded {ep_len} × 2 frames at {H}x{W}")

    # Encode text once
    txt_in = clip_proc.tokenizer([args.task], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        txt_out = clip_model.get_text_features(**txt_in)
    text_emb = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out

    state_dim = sarm.config.max_state_dim

    # Iterate every frame
    overlay_frames = []
    stage_per_t = []
    conf_per_t = []
    for t in range(ep_len):
        starts = [max(0, t - i * args.gap) for i in range(args.n_context - 1, -1, -1)]
        # Prep inputs
        def prep(cam):
            pil = [Image.fromarray(raw_frames[cam][i]) for i in starts]
            x = clip_proc(images=pil, return_tensors="pt")["pixel_values"].to(device)
            x.requires_grad_(True)
            return x

        pix_front = prep("front")
        pix_wrist = prep("wrist")

        with torch.enable_grad():
            f_out = clip_model.vision_model(pix_front)
            f_pool = f_out.pooler_output if hasattr(f_out, "pooler_output") else f_out[1]
            f_emb = clip_model.visual_projection(f_pool)
            w_out = clip_model.vision_model(pix_wrist)
            w_pool = w_out.pooler_output if hasattr(w_out, "pooler_output") else w_out[1]
            w_emb = clip_model.visual_projection(w_pool)

            img_emb = torch.stack([f_emb, w_emb], dim=0).unsqueeze(0)  # (1, 2, T, 512)
            state_feat = torch.zeros(1, args.n_context, state_dim, device=device)
            lengths = torch.tensor([args.n_context], dtype=torch.int32, device=device)
            stage_logits = sarm.stage_model(img_emb, text_emb, state_feat, lengths, scheme="sparse")
            target_t = args.n_context - 1
            stage_probs = F.softmax(stage_logits[0, target_t], dim=-1)
            target_stage = stage_logits[0, target_t].argmax().item()
            target_logit = stage_logits[0, target_t, target_stage]
            target_logit.backward()

        stage_per_t.append(target_stage)
        conf_per_t.append(stage_probs[target_stage].item())

        # Pixel-pooled saliency
        def saliency(pix_grad, raw):
            sal = pix_grad[target_t].abs().sum(dim=0)  # (224,224)
            sal = sal.detach().cpu().numpy()
            ps = 32
            n_p = 224 // ps
            pooled = sal.reshape(n_p, ps, n_p, ps).mean(axis=(1, 3))
            if pooled.max() > 0:
                pooled = pooled / pooled.max()
            up = cv2.resize(pooled, (W, H), interpolation=cv2.INTER_CUBIC)
            return np.clip(up, 0, 1)

        sal_front = saliency(pix_front.grad, raw_frames["front"][t])
        sal_wrist = saliency(pix_wrist.grad, raw_frames["wrist"][t])

        def overlay(raw, sal):
            heat = (cm.jet(sal)[:, :, :3] * 255).astype(np.uint8)
            return ((1 - args.alpha) * raw + args.alpha * heat).astype(np.uint8)

        ov_front = overlay(raw_frames["front"][t], sal_front)
        ov_wrist = overlay(raw_frames["wrist"][t], sal_wrist)

        # Compose 2x2 grid: front_raw, front_overlay, wrist_raw, wrist_overlay
        top = np.concatenate([raw_frames["front"][t], ov_front], axis=1)
        bot = np.concatenate([raw_frames["wrist"][t], ov_wrist], axis=1)
        grid = np.concatenate([top, bot], axis=0)

        # Annotation: stage + confidence + frame
        canvas = Image.fromarray(grid)
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        text = f"f{t}/{ep_len-1}  stage={target_stage}  p={stage_probs[target_stage].item():.2f}"
        draw.rectangle([(0, 0), (W * 2, 24)], fill=(0, 0, 0))
        draw.text((6, 4), text, fill=(255, 255, 255), font=font)
        overlay_frames.append(np.array(canvas))

        if t % 10 == 0:
            print(f"frame {t}/{ep_len-1} stage={target_stage} p={stage_probs[target_stage].item():.2f}")

    # Save mp4 via ffmpeg
    print(f"writing video ({len(overlay_frames)} frames)...")
    Hf, Wf = overlay_frames[0].shape[:2]
    mp4 = out_dir / f"ep{args.episode}_{Path(args.ckpt).parent.parent.name}.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{Wf}x{Hf}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        stdin=subprocess.PIPE,
    )
    for fr in overlay_frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"saved {mp4}")

    # Save stage trace as text
    import json as _json
    (out_dir / f"ep{args.episode}_stages.json").write_text(_json.dumps({
        "ep": args.episode, "task": ep.tasks[0], "length": ep_len,
        "stage_per_t": stage_per_t, "conf_per_t": conf_per_t,
    }))


if __name__ == "__main__":
    main()
