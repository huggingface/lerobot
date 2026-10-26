#!/usr/bin/env python
"""G15 (offline, no environment, no training): 2x2 FACTORIAL composition audit on the stored G13 observation bank.
Factor X = target-branch gain gamma (X0 = 1, X1 = 4; scales only target_xyz_encoder's additive contribution, as in G9). Factor Z = wrist gripper appearance (Z0 = original, Z1 = matched OPEN-gripper donor robot-region pasted on the wrist image, G13 swapR_diff_wrist rule: same fill, same donor matching).
A = g1+orig, B = g4+orig, C = g1+open, D = g4+open.  Subset: physical gripper CLOSED (grip_raw > 0.5) AND open donor available AND EE-GT 3D > 6cm.
Primary decoder conditioning: GT anchor (diagnostic only). Secondary: G4d predicted anchor (computed once from the ORIGINAL batch, held fixed over the 4 cells). Features: R = grounded feature recomputed from the (possibly patched) latents (deployed pipeline, primary); F = V2's feature of the original image.
Phase flag = the frame's own phase. Every decode is seeded identically so cells differ only by the factors."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
from g13_interventions import BANK, OUT as O13, KEYS, FixedHead, FixedFeat  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g15"); FAR = 0.06; CELLS = {"A": (1.0, 0), "B": (4.0, 0), "C": (1.0, 1), "D": (4.0, 1)}


class GammaWrap(nn.Module):
    def __init__(self, orig): super().__init__(); self.orig = orig; self.gamma = 1.0
    def forward(self, x): return self.gamma * self.orig(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True); M = np.load(O13 / "mean_images.npy"); B, IM, MK, SRC = [], [], [], []
    for src in ("g4d", "oracle"):
        b = pickle.load(open(BANK / src / "bank.pkl", "rb")); im = np.load(BANK / src / "imgs.npy", mmap_mode="r"); mk = np.load(BANK / src / "masks_packed.npy")
        for i, e in enumerate(b): B.append(e); SRC.append(src); IM.append(np.asarray(im[i])); MK.append(np.unpackbits(mk[i])[:3 * 2 * 480 * 480].reshape(3, 2, 480, 480).astype(bool))
    N = len(B); ee = np.array([e["ee_robot"] for e in B]); anc = np.array([e["gt_anchor_robot"] for e in B]); evec = anc - ee; grip = np.array([e["grip_raw"] > 0.5 for e in B]); d3 = np.linalg.norm(evec, axis=1)
    def donor(i):  # G13 'Dg' rule: different gripper state, EE <= 5cm, target-relative geometry <= 10cm
        d_ee = np.linalg.norm(ee - ee[i], axis=1); d_ev = np.linalg.norm(evec - evec[i], axis=1); ok = (grip != grip[i]) & (d_ee <= 0.05) & (d_ev <= 0.10); ok[i] = False
        return int(np.argmin(np.where(ok, d_ee + 0.5 * d_ev, 1e9))) if ok.any() else -1
    sel = [(i, donor(i)) for i in range(N) if grip[i] and d3[i] > FAR]; n_cf = len(sel); sel = [(i, j) for i, j in sel if j >= 0]
    print(f"bank {N} | closed & EE-GT>6cm: {n_cf} | with open donor: {len(sel)}", flush=True)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); mean, std = policy.action_pos_mean, policy.action_pos_std; dev = "cuda"
    policy.decoder.target_xyz_encoder = GammaWrap(policy.decoder.target_xyz_encoder); gw = policy.decoder.target_xyz_encoder
    enc = SafeDiffVLAPolicy._encode_multimodal_latent; head0, feat0 = policy.target_point_head, policy.visual_grounding_cross_attn; grounder = G5.Grounder("g4d", policy)
    def mk_batch(imgs, e):
        b = {k: torch.from_numpy(imgs[c]).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dev) for c, k in enumerate(KEYS)}; b["observation.state"] = torch.from_numpy(e["state_norm"]).float().unsqueeze(0).to(dev); b["observation.language.tokens"] = torch.from_numpy(e["lang_tokens"]).long().unsqueeze(0).to(dev); b["observation.language.attention_mask"] = torch.from_numpy(e["lang_mask"]).bool().unsqueeze(0).to(dev); return b
    @torch.no_grad()
    def decode(b, a, lat, feat, gamma, phase):
        policy.target_point_head = FixedHead(a); policy.visual_grounding_cross_attn = FixedFeat(feat); policy._encode_multimodal_latent = lambda bb: lat; policy._v2_grasp_phase = phase; gw.gamma = gamma; torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        act = SafeDiffVLAPolicy._plan_temporal_decoder(policy, b)[0][0]; return {"xyz": (act[:, :3] * std + mean).cpu().numpy(), "grip": (act[:, 6] * policy.action_grip_std + policy.action_grip_mean).cpu().numpy()}
    R = []
    for n, (i, j) in enumerate(sel):
        e = B[i]; im = IM[i]; x = im.copy(); left = MK[i][2][0] & ~MK[j][2][0]; x[2][left] = M[2][left]; x[2][MK[j][2][0]] = IM[j][2][MK[j][2][0]]; b0 = mk_batch(im, e); b1 = mk_batch(x, e); rec = {"i": i, "j": j, "src": SRC[i], "ep": e["ep"], "t": e["t"], "ee": e["ee_robot"], "gt": e["gt_anchor_robot"], "phase": e["phase"], "ee_gt_xy": float(np.linalg.norm(evec[i][:2])), "ee_gt_z": float(evec[i][2]), "grip_raw": e["grip_raw"], "n_displaced": e["n_displaced"], "out": {}}
        with torch.no_grad():
            lat = {z: enc(policy, bb) for z, bb in ((0, b0), (1, b1))}; own = {z: feat0(policy.target_query_extractor(*lat[z]), *lat[z]) for z in (0, 1)}
            g4d = grounder.compute(*lat[0], b0)["xyz_n"].reshape(1, -1)[:, :3]; rec["g4d_anchor"] = (g4d[0] * std + mean).cpu().numpy(); gt_n = ((torch.as_tensor(e["gt_anchor_robot"], dtype=std.dtype, device=dev) - mean) / std).unsqueeze(0)
            for an, a in (("GT", gt_n), ("G4D", g4d.to(std.dtype))):
                for fe in ("R", "F"):
                    for c, (g, z) in CELLS.items(): rec["out"][(an, fe, c)] = decode(b1 if z else b0, a, lat[z], own[z] if fe == "R" else own[0], g, e["phase"])
        R.append(rec)
        if (n + 1) % 20 == 0: print(f"{n + 1}/{len(sel)}", flush=True)
    gw.gamma = 1.0; pickle.dump({"R": R, "n_cf": n_cf}, open(OUT / "g15.pkl", "wb")); print("saved", len(R))


if __name__ == "__main__":
    main()
