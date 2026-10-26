#!/usr/bin/env python
"""G1 RETENTION check (original dataset frames are used for EVALUATION ONLY; never for training).
cache : frozen-v2 VLM latents of held-out dataset frames (same held-out rule as v2 training: last ceil(10%) episodes of each task string) at
        k=30 steps before the anchor and frame 0, with the TRUE instruction and two swapped referents of the same task.
eval  : for each trained arm/seed (and untouched V2): median XY anchor error with the true text (pooled over the two frame types and per type),
        and instruction-swap sensitivity on dataset frames: mean prediction shift under swap + error to the ORIGINAL anchor under swap.
Criterion (catastrophic-forgetting guard, not a success criterion): median error worsening vs V2 within max(+2 cm, +25%)."""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
D0 = Path("outputs/eval/safediff_vla_phase_d0_training_data_audit"); OUTR = Path("outputs/eval/safediff_vla_g1_offline")


def build_cache():
    from _multitask_common import RENAME_MAP, REPO_ID
    from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
    from lerobot.processor.rename_processor import rename_batch_keys

    df = pd.read_json(D0 / "episodes.json"); df = df[df.has_anchor & df.task.isin(["select_poker", "select_fruit"])]
    held = []
    for _, g in df.groupby("instruction"):
        g = g.sort_values("episode"); held += g.episode.tolist()[len(g) - math.ceil(len(g) * 0.1):]
    ho = df[df.episode.isin(held)]; instr = {t: sorted(g.instruction.unique()) for t, g in df.groupby("task")}
    ds = LeRobotDataset(REPO_ID); from_idx = ds.meta.episodes["dataset_from_index"]; cams = ds.meta.camera_keys
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval()
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    rng = np.random.default_rng(0)
    for task, g in ho.groupby("task"):
        T, M, A, meta = [], [], [], []
        for _, e in g.iterrows():
            tc = int(e.t_close); others = [s for s in instr[task] if s != e.instruction]; texts = [e.instruction, *[rng.choice(others) for _ in range(2)]]
            for k in ("k30", "f0"):
                f = 0 if k == "f0" else tc - 30
                if f < 0: continue
                item = ds[int(from_idx[int(e.episode)]) + f]; tt, mm = [], []
                for text in texts:
                    it = dict(item); it["task"] = text
                    b = {kk: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for kk, v in it.items()}
                    for c in cams:
                        if c in b and b[c].dtype == torch.uint8: b[c] = b[c].to(torch.float32) / 255.0
                    with torch.no_grad():
                        tok, pad, mod = policy._encode_multimodal_latent(pre(rename_batch_keys(b, RENAME_MAP)))
                    mod = mod.clone(); mod[~pad.to(torch.bool)] = 3; tt.append(tok[0].half().cpu()); mm.append(mod[0].to(torch.int8).cpu())
                T.append(tt); M.append(mm); A.append(e.anchor); meta.append({"episode": int(e.episode), "frame": k, "texts": texts})
        nmax = max(t.shape[0] for tt in T for t in tt)
        tok = torch.zeros(len(T), 3, nmax, 960, dtype=torch.float16); mod = torch.full((len(T), 3, nmax), 3, dtype=torch.int8)
        for i, (tt, mm) in enumerate(zip(T, M)):
            for j, (t, m) in enumerate(zip(tt, mm)): tok[i, j, :t.shape[0]] = t; mod[i, j, :m.shape[0]] = m
        torch.save({"tokens": tok, "modality": mod, "anchors": torch.tensor(np.array(A), dtype=torch.float32), "meta": meta,
                    "consts": json.load(open(Path("outputs/data/safediff_vla_g1_counterfactual/policy_constants.json")))}, OUTR / f"retention_cache_{task}.pt")
        print(task, "retention cache", tuple(tok.shape), flush=True)


@torch.no_grad()
def eval_model(model, task, dev="cuda"):
    c = torch.load(OUTR / f"retention_cache_{task}.pt", weights_only=False); mean = torch.tensor(c["consts"]["pos_mean"], device=dev); std = torch.tensor(c["consts"]["pos_std"], device=dev)
    tok, mod, anc = c["tokens"], c["modality"].long(), c["anchors"]; n = tok.shape[0]; P = []
    model.eval()
    for i in range(0, n, 16):
        t = tok[i:i + 16].to(dev).float(); m = mod[i:i + 16].to(dev); S = t.shape[0]
        x, _ = model(t.reshape(S * 3, *t.shape[2:]), m.reshape(S * 3, -1)); P.append((x * std + mean).reshape(S, 3, 3).cpu())
    P = torch.cat(P).numpy(); A = anc.numpy(); fr = np.array([m["frame"] for m in c["meta"]])
    e_true = np.linalg.norm((P[:, 0] - A)[:, :2], axis=1); e_swap = np.linalg.norm((P[:, 1:] - A[:, None])[..., :2], axis=2).mean(1)
    shift = np.linalg.norm((P[:, 1:] - P[:, :1])[..., :2], axis=2).mean(1)
    out = {"n_frames": int(n), "median_xy_err_cm_pooled": float(np.median(e_true) * 100)}
    for k in ("k30", "f0"):
        s = fr == k; out[k] = {"median_xy_err_cm": float(np.median(e_true[s]) * 100), "mean_pred_shift_under_swap_cm": float(shift[s].mean() * 100),
                               "median_err_to_original_anchor_under_swap_cm": float(np.median(e_swap[s]) * 100)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); a = ap.parse_args(); OUTR.mkdir(parents=True, exist_ok=True)
    if a.build: build_cache()


if __name__ == "__main__":
    main()
