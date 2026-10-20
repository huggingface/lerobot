#!/usr/bin/env python
"""Train the SafeDiff-VLA StagePlanHead (ONE primary run; no sweeps). Frozen backbone = the final learned checkpoint
`grounded_grasp_v2_20k/checkpoints/020000`; only `StagePlanHead` is trained. TRAIN split only (episode-disjoint validation
split = the repo's own last-10%-per-task-string hold-out); simulator eval seeds 3000-3019 are never used.

Checkpoint rule FIXED BEFORE TRAINING: the FINAL step (20k) is the model; validation loss is logged for reporting only.
Loss = L_xyz(SmoothL1, normalized action-xyz space, valid mask) + L_rot(MSE on unit sin/cos, valid mask) + L_release(BCE),
all weights 1.0 (no sweep). L_ref is NOT used: the dataset has no distractor annotations (see report).

Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/train_stageplan.py [--steps 20000] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.stageplan import NUM_STAGES, SafeDiffVLAStagePlanPolicy, StageConstants, StagePlanHead  # noqa: E402

CKPT = "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/020000/pretrained_model"
ROOT = Path("outputs/stageplan")
OUT = Path("outputs/train/safediff_stageplan_v1")


def label_tensors(labels: dict, eps: torch.Tensor, device) -> dict[str, torch.Tensor]:
    P = np.stack([np.stack([labels[int(e)][f"P{k}"] for k in range(NUM_STAGES)]) for e in eps.tolist()])  # [B,4,6] (P3 may be NaN)
    rel = np.array([labels[int(e)]["release_valid"] for e in eps.tolist()], dtype=np.float32)
    valid = np.ones((len(eps), NUM_STAGES), dtype=np.float32); valid[:, 3] = rel
    P = np.nan_to_num(P)
    return {"xyz": torch.as_tensor(P[..., :3], dtype=torch.float32, device=device), "euler": torch.as_tensor(P[..., 3:], dtype=torch.float32, device=device),
            "release": torch.as_tensor(rel, device=device), "valid": torch.as_tensor(valid, device=device)}


def stage_loss(out: dict, lab: dict, pos_mean: torch.Tensor, pos_std: torch.Tensor) -> tuple[torch.Tensor, dict]:
    xyz_gt = (lab["xyz"] - pos_mean) / pos_std
    sc_gt = torch.stack([torch.sin(lab["euler"]), torch.cos(lab["euler"])], dim=-1)  # [B,4,3,2]
    v = lab["valid"]
    l_xyz = (F.smooth_l1_loss(out["xyz"], xyz_gt, reduction="none").mean(-1) * v).sum() / v.sum()
    l_rot = (F.mse_loss(out["rot_sc"], sc_gt, reduction="none").mean((-1, -2)) * v).sum() / v.sum()
    l_rel = F.binary_cross_entropy_with_logits(out["release_logit"], lab["release"])
    return l_xyz + l_rot + l_rel, {"xyz": l_xyz.item(), "rot": l_rot.item(), "rel": l_rel.item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=2000); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    torch.manual_seed(1000); np.random.seed(1000)
    dev = "cuda"
    labels = pickle.load(open(ROOT / "labels.pkl", "rb")); sp = json.load(open(ROOT / "split_constants.json"))
    consts = {**sp["constants"], "seg_min": 2, "seg_max": 100}  # clip limits: mathematical min / generous max (not tuned)
    policy = SafeDiffVLAStagePlanPolicy.from_pretrained(CKPT).to(dev).eval()
    for p in policy.parameters(): p.requires_grad_(False)
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT, preprocessor_overrides={"device_processor": {"device": dev}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    head_cfg = {"d_model": 256, "n_heads": 4, "n_layers": 2, "n_cams": 3, "dropout": 0.1, "state_dim": 7}
    head = StagePlanHead(policy._multimodal_latent_dim(), **head_cfg).to(dev)
    print("head params:", sum(p.numel() for p in head.parameters()) / 1e6, "M", flush=True)
    train_eps = sp["train"]; val_eps = sp["val"]
    if a.smoke: train_eps, val_eps = train_eps[:40], val_eps[:20]
    mk = lambda eps: LeRobotDataset("lerobot/vlabench_unified", episodes=eps, image_transforms=None, delta_timestamps=None, video_backend="torchcodec", return_uint8=True)
    ds_tr, ds_va = mk(train_eps), mk(val_eps)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    g = torch.Generator().manual_seed(7); vidx = torch.randperm(len(ds_va), generator=g)[: (64 if a.smoke else 800)].tolist()
    dl_va = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_va, vidx), batch_size=a.batch, shuffle=False, num_workers=a.workers)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-6, betas=(0.9, 0.999), eps=1e-8)
    warm, total, peak, floor = 1000, 30000, 1e-4, 1e-5  # same cosine-with-warmup convention as the v2 run (decay horizon 30k)
    lr_at = lambda s: peak * (s + 1) / warm if s < warm else floor + 0.5 * (peak - floor) * (1 + np.cos(np.pi * min(1.0, (s - warm) / (total - warm))))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def forward(batch):
        for k in [k for k in batch if k.startswith("observation.images.") and batch[k].dtype == torch.uint8]:
            batch[k] = batch[k].to(dtype=torch.float32) / 255.0  # same uint8->float step as lerobot_train._preprocess_dataset_batch
        batch = pre(batch)
        with torch.no_grad():
            tokens, pad, mod = policy._encode_multimodal_latent(batch)
        state = policy._current_state(batch)
        return head(tokens.detach(), pad, mod, state), batch

    def validate() -> dict:
        head.eval(); tot, n, parts = 0.0, 0, {"xyz": 0.0, "rot": 0.0, "rel": 0.0}
        with torch.no_grad():
            for b in dl_va:
                eps = b["episode_index"]; o, _ = forward(b); lab = label_tensors(labels, eps, dev); l, p = stage_loss(o, lab, policy.action_pos_mean, policy.action_pos_std)
                tot += l.item() * len(eps); n += len(eps)
                for k in parts: parts[k] += p[k] * len(eps)
        head.train(); return {"val_loss": tot / n, **{f"val_{k}": v / n for k, v in parts.items()}}

    log = []; step = 0; t0 = time.time(); head.train()
    while step < a.steps:
        for batch in dl:
            eps = batch["episode_index"]
            for gp in opt.param_groups: gp["lr"] = lr_at(step)
            o, _ = forward(batch); lab = label_tensors(labels, eps, dev); loss, parts = stage_loss(o, lab, policy.action_pos_mean, policy.action_pos_std)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0); opt.step(); step += 1
            if step % 100 == 0: print(f"step {step} loss {loss.item():.4f} {parts} lr {lr_at(step):.2e} {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % a.val_every == 0 or step == a.steps:
                v = validate(); log.append({"step": step, "train_loss": loss.item(), **v}); print("VAL", log[-1], flush=True)
                (out / "train_log.json").write_text(json.dumps(log, indent=1))
            if step >= a.steps: break
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, out / "stageplan_head.safetensors")
    (out / "stageplan_config.json").write_text(json.dumps({"head": head_cfg, "constants": consts, "init_checkpoint": CKPT, "steps": a.steps, "batch": a.batch, "checkpoint_rule": "final step (fixed before training)",
                                                            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "label_config": "examples/safediff_vla/stageplan_labels.py"}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
