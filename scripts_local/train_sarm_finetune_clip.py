#!/usr/bin/env python3
"""End-to-end CLIP+SARM fine-tuning trainer (standalone, bypasses cache).

Loads:
- raw frames via LeRobotDataset with delta_timestamps (8 obs at frame_gap=5)
- SARM ckpt warm-start from --init-ckpt
- CLIP fresh from pretrained, unfrozen

Forward each step:
  raw_imgs (B,N,T,3,H,W) -> CLIP.vision -> img_emb (B,N,T,512)
  task_str -> CLIP.text -> txt_emb (B,512)
  sarm.stage_model + subtask_model (warm-started)
  loss = sw*CE(stage_logits, gt_stage) + plw*MSE(tau_pred, gt_tau)

Saves SARM+CLIP weights into output_dir/checkpoints/<step>/pretrained_model/.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lerobot_policy_sarm/src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_sarm.modeling_sarm import SARMRewardModel
from lerobot_policy_sarm.configuration_sarm import SARMConfig
from safetensors.torch import load_file, save_file


def build_delta_timestamps(n_obs: int, frame_gap: int, fps: int = 20):
    """Returns list of timestamps for n_obs observations at frame_gap spacing, ending at 0."""
    dt = 1.0 / fps
    return [-(n_obs - 1 - i) * frame_gap * dt for i in range(n_obs)]


class SARMSampleWrapper(torch.utils.data.Dataset):
    """Wraps LeRobotDataset to inject sparse_targets per window."""

    def __init__(self, ds: LeRobotDataset, n_obs: int, frame_gap: int, sparse_stage_names: list[str]):
        self.ds = ds
        self.n_obs = n_obs
        self.frame_gap = frame_gap
        self.sparse_stage_names = sparse_stage_names
        self.n_stages = len(sparse_stage_names)
        # Pre-extract per-episode subtask info
        eps_meta = ds.meta.episodes
        self._ep_stage_starts = {}
        self._ep_stage_ends = {}
        self._ep_stage_idx_arr = {}
        self._ep_lengths = {}
        for ep_idx in range(len(eps_meta)):
            ep = eps_meta[ep_idx]
            length = int(ep["length"])
            self._ep_lengths[ep_idx] = length
            sn = ep.get("sparse_subtask_names")
            ss = ep.get("sparse_subtask_start_frames")
            se = ep.get("sparse_subtask_end_frames")
            # Build per-frame stage idx + per-frame tau
            stage_idx = np.zeros(length, dtype=np.int64)
            tau = np.zeros(length, dtype=np.float32)
            if sn is None or ss is None or se is None:
                # Fallback: stage 0, linear progress
                stage_idx[:] = 0
                tau[:] = np.linspace(0, 1, length, endpoint=False)
            else:
                for s, e, name in zip(ss, se, sn):
                    if name not in sparse_stage_names:
                        continue
                    sid = sparse_stage_names.index(name)
                    stage_idx[s : e + 1] = sid
                    span = max(1, e - s)
                    for f in range(s, e + 1):
                        tau[f] = (f - s) / span
            self._ep_stage_idx_arr[ep_idx] = stage_idx
            self._ep_stage_ends[ep_idx] = {sparse_stage_names.index(n) if n in sparse_stage_names else 0: e for n, e in zip(sn or [], se or [])}
            # Build sparse_target = stage_idx + tau (float)
            self._ep_stage_starts[ep_idx] = (stage_idx, tau)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        ep_idx = int(item["episode_index"])
        global_idx = int(item["index"])
        # Items contain (T, ...) due to delta_timestamps. T = n_obs.
        # Frame indices for the window: anchor is current item's frame; deltas in time
        # Since we set delta_timestamps to deliver n_obs frames ending at current frame,
        # frame indices in window are [anchor - (n_obs-1)*gap, ..., anchor].
        # Compute targets from per-ep stage_idx + tau arrays.
        stage_idx_arr, tau_arr = self._ep_stage_starts[ep_idx]
        ep_meta = self.ds.meta.episodes[ep_idx]
        ep_start_global = int(ep_meta["dataset_from_index"])
        ep_end_global = int(ep_meta["dataset_to_index"])
        local_anchor = global_idx - ep_start_global
        # window local indices, clamp
        local_indices = np.clip(
            np.array([local_anchor - (self.n_obs - 1 - i) * self.frame_gap for i in range(self.n_obs)]),
            0, self._ep_lengths[ep_idx] - 1,
        )
        sparse_target = (stage_idx_arr[local_indices].astype(np.float32) + tau_arr[local_indices])  # (T,)
        item["sparse_target"] = torch.from_numpy(sparse_target)
        item["lengths"] = torch.tensor(self.n_obs, dtype=torch.int32)
        return item


def stage_tau_from_target(target: torch.Tensor, n_stages: int):
    """target = stage + tau (float). Returns int stage and float tau."""
    stage = target.floor().clamp(0, n_stages - 1).long()
    tau = (target - stage.float()).clamp(0, 1)
    return stage, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", required=True, help="warm-start SARM ckpt path (pretrained_model dir)")
    ap.add_argument("--dataset-repo", default="local/sim_3stage_v2_full_v2_nostale")
    ap.add_argument("--dataset-root", default=None, help="local root for dataset")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--frame-gap", type=int, default=5)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--clip-lr", type=float, default=5e-7)
    ap.add_argument("--sarm-lr", type=float, default=2e-5)
    ap.add_argument("--save-freq", type=int, default=2000)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--sw", type=float, default=10.0)
    ap.add_argument("--plw", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default="Three-stage assembly")
    ap.add_argument("--freeze-clip", action="store_true", help="If set, freeze CLIP weights (no finetune).")
    ap.add_argument("--stage-balanced", action="store_true", help="WeightedRandomSampler so each stage equally represented per batch.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    out_dir = Path(args.output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Load CLIP
    clip_name = "openai/clip-vit-base-patch32"
    print(f"[init] loading CLIP {clip_name} freeze={args.freeze_clip}")
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_proc = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    if args.freeze_clip:
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
    else:
        clip_model.train()
        for p in clip_model.parameters():
            p.requires_grad_(True)

    # Load SARM ckpt config + warm-start
    cfg_dict = json.loads((Path(args.init_ckpt) / "config.json").read_text())
    sarm_cfg = SARMConfig(**{k: v for k, v in cfg_dict.items() if k in SARMConfig.__dataclass_fields__})
    sarm_cfg.frame_gap = args.frame_gap
    sarm_cfg.n_obs_steps = args.n_obs
    print(f"[init] loading SARM from {args.init_ckpt}")
    sarm = SARMRewardModel(sarm_cfg).to(device)
    sd = load_file(str(Path(args.init_ckpt) / "model.safetensors"))
    sarm.load_state_dict(sd, strict=False)
    sarm.train()

    # Dataset
    print(f"[data] loading {args.dataset_repo}")
    delta_ts = build_delta_timestamps(args.n_obs, args.frame_gap, args.fps)
    delta_timestamps = {
        "observation.images.front": delta_ts,
        "observation.images.wrist": delta_ts,
        "observation.state": delta_ts,
    }
    ds = LeRobotDataset(
        repo_id=args.dataset_repo,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
    )
    sample_ds = SARMSampleWrapper(ds, args.n_obs, args.frame_gap, sarm_cfg.sparse_subtask_names)
    print(f"[data] dataset size={len(sample_ds)} stages={sarm_cfg.sparse_subtask_names}")

    def collate(batch):
        out = {}
        for k in batch[0]:
            v0 = batch[0][k]
            if isinstance(v0, torch.Tensor):
                out[k] = torch.stack([b[k] for b in batch])
            else:
                out[k] = [b[k] for b in batch]
        return out

    sampler = None
    if args.stage_balanced:
        # Compute per-frame stage by walking each ep's stage map
        print("[data] computing stage-balanced sampler weights...")
        from collections import Counter
        n = len(sample_ds)
        per_frame_stage = np.zeros(n, dtype=np.int64)
        for ep_idx in range(len(ds.meta.episodes)):
            ep_meta = ds.meta.episodes[ep_idx]
            ep_start = int(ep_meta["dataset_from_index"])
            ep_end = int(ep_meta["dataset_to_index"])
            stage_idx_arr, _ = sample_ds._ep_stage_starts[ep_idx]
            for local_i, gi in enumerate(range(ep_start, ep_end)):
                if local_i < len(stage_idx_arr):
                    per_frame_stage[gi] = stage_idx_arr[local_i]
        cnt = Counter(per_frame_stage.tolist())
        print(f"[data] stage frame counts: {dict(cnt)}")
        weights = np.zeros(n, dtype=np.float64)
        for s, c in cnt.items():
            weights[per_frame_stage == s] = 1.0 / max(1, c)
        weights = weights / weights.sum()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=n,
            replacement=True,
        )

    loader = DataLoader(
        sample_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # Optimizer
    sarm_params = list(sarm.parameters())
    if args.freeze_clip:
        optimizer = torch.optim.AdamW(sarm_params, lr=args.sarm_lr, weight_decay=1e-3)
        clip_params = []
        print(f"[opt] SARM params={sum(p.numel() for p in sarm_params)/1e6:.1f}M lr={args.sarm_lr} (CLIP frozen)")
    else:
        clip_params = list(clip_model.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": clip_params, "lr": args.clip_lr, "weight_decay": 1e-4},
                {"params": sarm_params, "lr": args.sarm_lr, "weight_decay": 1e-3},
            ]
        )
        print(f"[opt] CLIP params={sum(p.numel() for p in clip_params)/1e6:.1f}M lr={args.clip_lr}")
        print(f"[opt] SARM params={sum(p.numel() for p in sarm_params)/1e6:.1f}M lr={args.sarm_lr}")

    # Pre-encode text embedding (single task string)
    tok = clip_proc.tokenizer([args.task], return_tensors="pt", padding=True).to(device)

    n_classes = sarm_cfg.num_sparse_stages
    state_dim = sarm_cfg.max_state_dim

    # CLIP normalization (matches CLIPProcessor)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    step = 0
    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            front = batch["observation.images.front"].to(device)  # (B,T,3,H,W)
            wrist = batch["observation.images.wrist"].to(device)
            state = batch.get("observation.state")
            if state is not None:
                state = state.to(device)
            sparse_target = batch["sparse_target"].to(device)  # (B,T) float
            B, T = sparse_target.shape

            # Resize images to 224x224 for CLIP (expects 224 input)
            def to_clip(x):
                # x: (B,T,3,H,W); resize H,W → 224
                x = x.flatten(0, 1)  # (B*T,3,H,W)
                if x.dtype == torch.uint8:
                    x = x.float() / 255.0
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                x = (x - clip_mean) / clip_std
                return x

            pix_front = to_clip(front)
            pix_wrist = to_clip(wrist)
            from contextlib import nullcontext
            clip_ctx = torch.no_grad() if args.freeze_clip else nullcontext()
            with clip_ctx:
                f_pool = clip_model.vision_model(pix_front).pooler_output  # (B*T, 768)
                w_pool = clip_model.vision_model(pix_wrist).pooler_output
                f_emb = clip_model.visual_projection(f_pool).view(B, T, 512)
                w_emb = clip_model.visual_projection(w_pool).view(B, T, 512)
                img_emb = torch.stack([f_emb, w_emb], dim=1)  # (B,N=2,T,512)

                txt_out = clip_model.get_text_features(**tok)
                text_emb = (txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out).expand(B, -1)  # (B,512)

            # Pad/use state
            if state is None:
                state_feat = torch.zeros(B, T, state_dim, device=device)
            else:
                from lerobot_policy_sarm.modeling_sarm import pad_state_to_max_dim
                state_feat = pad_state_to_max_dim(state, state_dim)

            lengths = torch.full((B,), T, dtype=torch.int32, device=device)

            # Run SARM
            stage_logits = sarm.stage_model(img_emb, text_emb, state_feat, lengths, scheme="sparse")
            gt_stage, gt_tau = stage_tau_from_target(sparse_target, n_classes)

            # Use teacher forcing (GT stage one-hot for subtask)
            stage_onehot = F.one_hot(gt_stage, num_classes=n_classes).float().unsqueeze(1)  # (B,1,T,C)
            tau_pred = sarm.subtask_model(img_emb, text_emb, state_feat, lengths, stage_onehot, scheme="sparse")

            # Losses
            stage_logits_flat = stage_logits.reshape(B * T, n_classes)
            gt_stage_flat = gt_stage.reshape(B * T)
            stage_loss = F.cross_entropy(stage_logits_flat, gt_stage_flat)
            tau_loss = F.mse_loss(tau_pred.reshape(-1), gt_tau.reshape(-1))
            total_loss = args.sw * stage_loss + args.plw * tau_loss

            optimizer.zero_grad()
            total_loss.backward()
            if clip_params:
                torch.nn.utils.clip_grad_norm_(clip_params, 5.0)
            torch.nn.utils.clip_grad_norm_(sarm_params, 5.0)
            optimizer.step()
            step += 1

            if step % args.log_freq == 0:
                with torch.no_grad():
                    pred_stage = stage_logits_flat.argmax(-1)
                    acc = (pred_stage == gt_stage_flat).float().mean().item()
                eps = step / max(1, time.time() - t0)
                print(f"[step {step}/{args.steps}] loss={total_loss.item():.4f} stage={stage_loss.item():.4f} tau={tau_loss.item():.4f} stageAcc={acc:.3f} stepsPerSec={eps:.2f}")

            if step % args.save_freq == 0 or step == args.steps:
                ck_dir = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
                ck_dir.mkdir(parents=True, exist_ok=True)
                # Ensure contiguous for safetensors
                sarm_sd = {k: v.contiguous() for k, v in sarm.state_dict().items()}
                save_file(sarm_sd, str(ck_dir / "model.safetensors"))
                if not args.freeze_clip:
                    clip_sd = {k: v.contiguous().detach().clone() for k, v in clip_model.state_dict().items()}
                    save_file(clip_sd, str(ck_dir / "clip_model.safetensors"))
                # Save config
                (ck_dir / "config.json").write_text(json.dumps(cfg_dict))
                # Copy preprocessor config from init ckpt if exists
                for f in ["policy_postprocessor.json", "policy_preprocessor.json", "policy_preprocessor_step_2_normalizer_processor.safetensors", "train_config.json"]:
                    src = Path(args.init_ckpt) / f
                    if src.exists():
                        import shutil
                        shutil.copy2(src, ck_dir / f)
                print(f"[ckpt] saved {ck_dir}")


if __name__ == "__main__":
    main()
