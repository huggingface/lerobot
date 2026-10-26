#!/usr/bin/env python
"""G16-F fixed-observation capture (offline on the G13 bank: the same 797 observations as G8/G9/G11/G13, no environment). Models: v2 (frozen baseline), g11 (residual conditioner), g16 (geometry conditioner), g4 (V2 with target-branch gamma=4: diagnostic upper bound only).
Per frame, decoder conditioning anchors: GT keypoint ('GT'), nearest wrong card ('WR' = the nearest same-type card's position shifted by the GT keypoint-minus-card offset), G4d prediction computed from the original batch ('G4D'); wrist-image variant Z0 = original. For the G15 subset (closed & EE-GT>6cm & open donor) also Z1 = open-gripper wrist patch (G15 rule), GT anchor.
Phase flag = the frame's own phase; each decode is identically seeded."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
import g11_residual as R11  # noqa: E402
import g16_geometry as G16  # noqa: E402
from g13_interventions import BANK, OUT as O13, KEYS, FixedHead, FixedFeat  # noqa: E402
from g15_composition import GammaWrap  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g16"); FAR = 0.06


def main():
    OUT.mkdir(parents=True, exist_ok=True); M = np.load(O13 / "mean_images.npy"); B, IM, MK, SRC = [], [], [], []
    for src in ("g4d", "oracle"):
        b = pickle.load(open(BANK / src / "bank.pkl", "rb")); im = np.load(BANK / src / "imgs.npy", mmap_mode="r"); mk = np.load(BANK / src / "masks_packed.npy")
        for i, e in enumerate(b): B.append(e); SRC.append(src); IM.append(np.asarray(im[i])); MK.append(np.unpackbits(mk[i])[:3 * 2 * 480 * 480].reshape(3, 2, 480, 480).astype(bool))
    N = len(B); ee = np.array([e["ee_robot"] for e in B]); anc = np.array([e["gt_anchor_robot"] for e in B]); evec = anc - ee; grip = np.array([e["grip_raw"] > 0.5 for e in B]); d3 = np.linalg.norm(evec, axis=1)
    def donor(i):
        d_ee = np.linalg.norm(ee - ee[i], axis=1); d_ev = np.linalg.norm(evec - evec[i], axis=1); ok = (grip != grip[i]) & (d_ee <= 0.05) & (d_ev <= 0.10); ok[i] = False
        return int(np.argmin(np.where(ok, d_ee + 0.5 * d_ev, 1e9))) if ok.any() else -1
    DON = [donor(i) if (grip[i] and d3[i] > FAR) else -1 for i in range(N)]; wrong = []
    for e in B:
        cs = e["cards_xyz_robot"]; tn = e["target_name"]; ot = [n for n in cs if n != tn]; w = min(ot, key=lambda n: np.linalg.norm(np.asarray(cs[n])[:2] - np.asarray(cs[tn])[:2])) if ot else None; wrong.append(np.asarray(e["gt_anchor_robot"]) + (np.asarray(cs[w]) - np.asarray(cs[tn])) if w else None)
    print("frames", N, "subset", sum(d >= 0 for d in DON), "wrong-card available", sum(w is not None for w in wrong), flush=True); res = {}
    for model in sys.argv[1:] or ["v2", "g11", "g16", "g4"]:
        policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); mean, std = policy.action_pos_mean, policy.action_pos_std; dev = "cuda"; gw = None
        if model == "g11": r = R11.install(policy); r.load_state_dict(torch.load("outputs/train/safediff_vla_g11/residual_conditioner.pt")); r.eval()
        if model == "g16": ck = torch.load("outputs/train/safediff_vla_g16/geometry_conditioner.pt", weights_only=False); g = G16.install(policy, ck["alpha_max"]); g.load_state_dict(ck["state"]); g.eval()
        if model == "g4": policy.decoder.target_xyz_encoder = GammaWrap(policy.decoder.target_xyz_encoder); gw = policy.decoder.target_xyz_encoder; gw.gamma = 4.0
        enc = SafeDiffVLAPolicy._encode_multimodal_latent; feat0 = policy.visual_grounding_cross_attn; grounder = G5.Grounder("g4d", policy)
        def mk_batch(imgs, e):
            b = {k: torch.from_numpy(imgs[c]).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dev) for c, k in enumerate(KEYS)}; b["observation.state"] = torch.from_numpy(e["state_norm"]).float().unsqueeze(0).to(dev); b["observation.language.tokens"] = torch.from_numpy(e["lang_tokens"]).long().unsqueeze(0).to(dev); b["observation.language.attention_mask"] = torch.from_numpy(e["lang_mask"]).bool().unsqueeze(0).to(dev); return b
        @torch.no_grad()
        def decode(b, a, lat, feat, phase):
            policy.target_point_head = FixedHead(a); policy.visual_grounding_cross_attn = FixedFeat(feat); policy._encode_multimodal_latent = lambda bb: lat; policy._v2_grasp_phase = phase; torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            act = SafeDiffVLAPolicy._plan_temporal_decoder(policy, b)[0][0]; return {"xyz": (act[:, :3] * std + mean).cpu().numpy(), "grip": (act[:, 6] * policy.action_grip_std + policy.action_grip_mean).cpu().numpy()}
        nrm = lambda a: ((torch.as_tensor(a, dtype=std.dtype, device=dev) - mean) / std).unsqueeze(0); out = []
        for i in range(N):
            e = B[i]; b0 = mk_batch(IM[i], e)
            with torch.no_grad():
                lat0 = enc(policy, b0); own0 = feat0(policy.target_query_extractor(*lat0), *lat0); g4 = grounder.compute(*lat0, b0)["xyz_n"].reshape(1, -1)[:, :3].to(std.dtype)
                rec = {"i": i, "src": SRC[i], "ep": e["ep"], "t": e["t"], "phase": e["phase"], "ee": e["ee_robot"], "gt": e["gt_anchor_robot"], "wrong": wrong[i], "grip_raw": e["grip_raw"], "n_displaced": e["n_displaced"], "any_disp_cm": e["any_card_displaced_cm"], "ee_gt_xy": e["ee_gt_xy"], "ee_gt_z": e["ee_gt_z"], "sub": DON[i] >= 0, "out": {}}
                rec["out"][("GT", 0)] = decode(b0, nrm(e["gt_anchor_robot"]), lat0, own0, e["phase"]); rec["out"][("G4D", 0)] = decode(b0, g4, lat0, own0, e["phase"])
                if wrong[i] is not None: rec["out"][("WR", 0)] = decode(b0, nrm(wrong[i]), lat0, own0, e["phase"])
                if DON[i] >= 0:
                    j = DON[i]; x = IM[i].copy(); left = MK[i][2][0] & ~MK[j][2][0]; x[2][left] = M[2][left]; x[2][MK[j][2][0]] = IM[j][2][MK[j][2][0]]; b1 = mk_batch(x, e); lat1 = enc(policy, b1); own1 = feat0(policy.target_query_extractor(*lat1), *lat1); rec["out"][("GT", 1)] = decode(b1, nrm(e["gt_anchor_robot"]), lat1, own1, e["phase"])
            out.append(rec)
            if (i + 1) % 200 == 0: print(model, i + 1, flush=True)
        res[model] = out; del policy; torch.cuda.empty_cache()
    pickle.dump(res, open(OUT / ("capture_" + "_".join(sys.argv[1:] or ["all"]) + ".pkl"), "wb")); print("saved")


if __name__ == "__main__":
    main()
