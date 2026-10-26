#!/usr/bin/env python
"""[StagePlan-v2.1] Train the counterfactual-referent-grounding head (ONE primary run, no sweeps). Frozen v2 backbone; only StagePlanV21Head trains.

PRE-DECLARED (before training): steps=20000, batch=16 (8 dataset samples + 8 pilot samples = 4 scenes x 2 DIFFERENT-referent instructions of the SAME scene),
seed=1000, AdamW lr 1e-4 (1000 warm-up, cosine to 1e-5 over a 30k horizon = v2's schedule), checkpoint = FINAL step.
  L = L_grasp + L_aux(rot,dir,pre,wp: dataset only, as v2) + 1.0*L_ref + 1.0*L_cf + 0.5*L_heat
  L_grasp : SmoothL1 on normalised grasp_xyz (dataset: expert P1; pilot: anchor + per-task expert EE-anchor offset, drink excluded)
  L_ref   : candidate-level CE; logit_i = -|ref_xyz - anchor_i|^2 / (2*0.04^2) over the scene's valid candidates (true = instructed referent), each sample's CE divided by ln(K) (scale normalisation; sigma/normalisation fixed after a 200-step smoke, before the primary run)
  L_cf    : same-scene pair (i,j): (1 - cos(d_pred, d_gt)) + 0.5*SmoothL1(proj(d_pred on unit d_gt)/0.1, |d_gt|/0.1)   [d = grasp_j - grasp_i]
  L_heat  : SmoothL1 between per-head soft-argmax coords and the target's projected token coord (fixed cams 0,1; token units)
Train data: TRAIN-split episodes of lerobot/vlabench_unified + TRAIN-split pilot scenes (scene-level split, seeds 100000-100049/task). No final seeds, no LIBERO.
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/train_stageplan_v21.py [--steps 20000] [--smoke]
"""
from __future__ import annotations
import argparse, json, pickle, subprocess, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).parent))
from train_stageplan_v2 import label_tensors, stage_loss  # noqa: E402
from v21_common import OUT as V21, XYZ_TASKS, cf_metrics, load_policy, offsets  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.safediff_vla.stageplan_v21 import StagePlanV21Head  # noqa: E402

ROOT = Path("outputs/stageplan"); OUT = Path("outputs/train/safediff_stageplan_v21")
LAM = {"ref": 1.0, "cf": 1.0, "mag": 0.5, "heat": 0.5}; SIGMA_REF = 0.04


def pilot_tables():
    rows = pickle.load(open(V21 / "referent_table.pkl", "rb")); split = json.load(open(V21 / "pilot_scene_split.json")); T = {}
    for r in rows:
        T[r["scene_id"]] = {"task": r["task"], "split": split[r["scene_id"]], "cands": [{"anchor": c["anchor"], "valid": c["valid"], "uv": [(p[0] / 60.0, p[1] / 60.0) if p else (np.nan, np.nan) for p in c["proj_uv_depth"][:2]]} for c in r["candidates"]]}
    return rows, split, T


def pilot_batch(cache_by, T, scene_ids, rng, dev, mean, std, off):
    E, meta = [], []
    for sid in scene_ids:
        ids = [i for i, c in enumerate(T[sid]["cands"]) if c["valid"]]; a, b = rng.choice(ids, size=2, replace=False)
        for ci in (a, b): E.append(cache_by[(sid, int(ci))]); meta.append((sid, int(ci)))
    tok = torch.stack([e["tok"] for e in E]).float().to(dev); pad = torch.stack([e["pad"] for e in E]).to(dev); mod = torch.stack([e["mod"] for e in E]).to(dev); st = torch.stack([e["state"] for e in E]).to(dev)
    anchors, tgt_xyz, has_xyz, uv, cand_sets, tgt_idx = [], [], [], [], [], []
    for sid, ci in meta:
        t = T[sid]; c = t["cands"][ci]; o = off.get(t["task"], np.zeros(3)) if t["task"] in XYZ_TASKS else np.zeros(3)
        anchors.append(c["anchor"]); tgt_xyz.append(c["anchor"] + o); has_xyz.append(t["task"] in XYZ_TASKS); uv.append(c["uv"])
        ids = [i for i, cc in enumerate(t["cands"]) if cc["valid"]]; cand_sets.append(np.stack([t["cands"][i]["anchor"] for i in ids])); tgt_idx.append(ids.index(ci))
    f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
    return {"tok": tok, "pad": pad, "mod": mod, "st": st, "anchor": f(anchors), "tgt_n": (f(tgt_xyz) - mean) / std, "has_xyz": torch.as_tensor(has_xyz, device=dev), "uv": f(uv), "cands": cand_sets, "tgt_idx": tgt_idx}


def pilot_losses(o, pb, mean, std):
    dev = o["grasp_xyz"].device; l_grasp = F.smooth_l1_loss(o["grasp_xyz"][pb["has_xyz"]], pb["tgt_n"][pb["has_xyz"]]) if pb["has_xyz"].any() else o["grasp_xyz"].sum() * 0
    ref = o["ref_xyz"] * std + mean; l_ref, acc = 0.0, []
    for i, (C, ti) in enumerate(zip(pb["cands"], pb["tgt_idx"])):
        Ct = torch.as_tensor(C, dtype=torch.float32, device=dev); lg = -((ref[i][None] - Ct) ** 2).sum(-1) / (2 * SIGMA_REF ** 2)
        l_ref = l_ref + F.cross_entropy(lg[None], torch.tensor([ti], device=dev)) / float(np.log(len(C))); acc.append(float(lg.argmax().item() == ti))
    l_ref = l_ref / len(pb["cands"])
    g = o["grasp_xyz"] * std + mean; g = g.view(-1, 2, 3); A = pb["anchor"].view(-1, 2, 3)
    dp, dg = g[:, 1] - g[:, 0], A[:, 1] - A[:, 0]; ng = dg.norm(dim=-1).clamp_min(1e-6); u = dg / ng[:, None]
    l_dir = (1 - F.cosine_similarity(dp, dg, dim=-1, eps=1e-6)).mean(); l_mag = F.smooth_l1_loss((dp * u).sum(-1) / 0.1, ng / 0.1)
    l_cf = l_dir + LAM["mag"] * l_mag
    co = o["ref_coords"][:, :2]  # [B,2cams,H,2] token units
    l_heat = F.smooth_l1_loss(co, pb["uv"][:, :, None, :].expand_as(co))
    return l_grasp, l_ref, l_cf, l_heat, {"ref_acc": float(np.mean(acc)), "dir": l_dir.item(), "mag": l_mag.item()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=2000); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=str(OUT)); a = ap.parse_args()
    torch.manual_seed(1000); np.random.seed(1000); rng = np.random.default_rng(1000); dev = "cuda"
    labels = pickle.load(open(ROOT / "labels_v2.pkl", "rb")); sp = json.load(open(ROOT / "split_constants.json")); off = offsets()
    pol, pre = load_policy(dev); mean, std = pol.action_pos_mean, pol.action_pos_std
    rows, split, T = pilot_tables(); cache = torch.load(V21 / "pilot_token_cache.pt", weights_only=False); cache_by = {(e["scene_id"], e["ci"]): e for e in cache}
    tr_scenes = {t: [s for s, v in T.items() if v["task"] == t and v["split"] == "train" and sum(c["valid"] for c in v["cands"]) >= 2] for t in sorted({v["task"] for v in T.values()})}
    print({t: len(v) for t, v in tr_scenes.items()}, flush=True)
    head_cfg = {"d_model": 256, "n_heads": 4, "n_cams": 3, "dropout": 0.1, "state_dim": 7}; head = StagePlanV21Head(pol._multimodal_latent_dim(), **head_cfg).to(dev)
    print("head params", sum(p.numel() for p in head.parameters()) / 1e6, "M", flush=True)
    train_eps, val_eps = sp["train"], sp["val"]
    if a.smoke: train_eps, val_eps = train_eps[:40], val_eps[:20]
    mk = lambda eps: LeRobotDataset("lerobot/vlabench_unified", episodes=eps, image_transforms=None, delta_timestamps=None, video_backend="torchcodec", return_uint8=True)
    ds_tr, ds_va = mk(train_eps), mk(val_eps); nd = a.batch // 2; npairs = (a.batch - nd) // 2
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=nd, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    g = torch.Generator().manual_seed(7); vidx = torch.randperm(len(ds_va), generator=g)[: (64 if a.smoke else 800)].tolist(); dl_va = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_va, vidx), batch_size=16, shuffle=False, num_workers=a.workers)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-6, betas=(0.9, 0.999), eps=1e-8)
    warm, total, peak, floor = 1000, 30000, 1e-4, 1e-5
    lr_at = lambda s: peak * (s + 1) / warm if s < warm else floor + 0.5 * (peak - floor) * (1 + np.cos(np.pi * min(1.0, (s - warm) / (total - warm))))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def ds_forward(batch):
        for k in [k for k in batch if k.startswith("observation.images.") and batch[k].dtype == torch.uint8]: batch[k] = batch[k].to(dtype=torch.float32) / 255.0
        batch = pre(batch)
        with torch.no_grad(): tokens, pad, mod = pol._encode_multimodal_latent(batch)
        return head(tokens.detach(), pad, mod, pol._current_state(batch))

    def pilot_val():
        head.eval(); G, R = {}, {}
        with torch.no_grad():
            for i in range(0, len(cache), 64):
                E = cache[i:i + 64]; o = head(torch.stack([e["tok"] for e in E]).float().to(dev), torch.stack([e["pad"] for e in E]).to(dev), torch.stack([e["mod"] for e in E]).to(dev), torch.stack([e["state"] for e in E]).to(dev))
                gg, rr = (o["grasp_xyz"] * std + mean).cpu().numpy(), (o["ref_xyz"] * std + mean).cpu().numpy()
                for e, x, y in zip(E, gg, rr): G[(e["scene_id"], e["ci"])] = x.astype(np.float64); R[(e["scene_id"], e["ci"])] = y.astype(np.float64)
        head.train(); m = cf_metrics(G, rows, split, "val"); return {t: {k: round(m[t][k], 3) for k in ("sel_acc", "chance", "cos", "realized", "own_err_med_cm")} for t in ("select_poker", "select_mahjong") if t in m}

    def ds_val():
        head.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for b in dl_va:
                eps = b["episode_index"]; o = ds_forward(b); l, _ = stage_loss(o, label_tensors(labels, eps, dev), mean, std); tot += l.item() * len(eps); n += len(eps)
        head.train(); return tot / n

    log, step, t0 = [], 0, time.time(); head.train(); scene_pool = tr_scenes
    while step < a.steps:
        for batch in dl:
            for gp in opt.param_groups: gp["lr"] = lr_at(step)
            tasks = sorted(scene_pool); sids = []
            for _ in range(npairs): t = tasks[rng.integers(len(tasks))]; sids.append(scene_pool[t][rng.integers(len(scene_pool[t]))])
            od = ds_forward(batch); lab = label_tensors(labels, batch["episode_index"], dev); l_ds, parts = stage_loss(od, lab, mean, std)  # = L_grasp(xyz)+L_aux on dataset
            pb = pilot_batch(cache_by, T, sids, rng, dev, mean, std, off); op = head(pb["tok"], pb["pad"], pb["mod"], pb["st"], return_aux=False)
            l_g, l_ref, l_cf, l_heat, pp = pilot_losses(op, pb, mean, std)
            loss = l_ds + l_g + LAM["ref"] * l_ref + LAM["cf"] * l_cf + LAM["heat"] * l_heat
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0); opt.step(); step += 1
            if step % 100 == 0: print(f"step {step} loss {loss.item():.4f} ds {l_ds.item():.4f} pg {l_g.item():.4f} ref {l_ref.item():.4f} cf {l_cf.item():.4f} heat {l_heat.item():.4f} {pp} lr {lr_at(step):.2e} {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % a.val_every == 0 or step == a.steps:
                rec = {"step": step, "train_loss": loss.item(), "ds_val_loss": ds_val(), "pilot_val": pilot_val()}; log.append(rec); print("VAL", rec, flush=True); (out / "train_log.json").write_text(json.dumps(log, indent=1))
            if step >= a.steps: break
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, out / "stageplan_head.safetensors")
    (out / "stageplan_config.json").write_text(json.dumps({"head": head_cfg, "variables": ["grasp_xyz", "ref_xyz", "grasp_rot_sc", "approach_dir", "pre_dist", "wp_rel"], "init_checkpoint": "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/020000/pretrained_model", "steps": a.steps, "batch": a.batch, "seed": 1000,
        "loss_config": {"lambda": LAM, "sigma_ref_m": SIGMA_REF, "dataset_part": "v2 stage_loss"}, "checkpoint_rule": "final step (fixed before training)", "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}, indent=1)); print("saved", out)

if __name__ == "__main__": main()
