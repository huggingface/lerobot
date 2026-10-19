#!/usr/bin/env python
"""G17 capture (offline, no env, no training): the 797 G8-G16 fixed observations. For every frame, the layer-wise decoder runner (g17_decoder) is used on the frozen V2 policy.
Anchors: GT, WR (nearest wrong card), G4D (G4d prediction), GT +-5cm along x / y / z (xp xm yp ym zp zm; z is OOD for poker -> secondary).
Conditions: V2 (== I0 move) | G4 (input gain 4, diagnostic upper bound) | NT (target branch removed) | move@I1..I4 (input branch removed, h_target = target_xyz_encoder(T) added at the location; same norm as the normal branch) | extra@I0..I4 (V2 + ONE extra copy of h_target at the location) | geo@I0..I4 (V2 + the trained G16 conditioner output h_geo(T), weights unchanged, added at the location).
Also records (T0 = GT vs T1 = WR, normal V2 pipeline) the hidden difference at every sub-layer point."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
import g17_decoder as D  # noqa: E402
from g16_geometry import GeometryConditioner  # noqa: E402
from g13_interventions import BANK, KEYS, FixedHead, FixedFeat  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g17"); D5 = 0.05; PTS = ["in"] + [f"L{l}.{p}" for l in range(4) for p in ("sa", "x1", "ca", "x2", "ff", "x3")] + ["out"]


def load_bank():
    B, IM, SRC = [], [], []
    for src in ("g4d", "oracle"):
        b = pickle.load(open(BANK / src / "bank.pkl", "rb")); im = np.load(BANK / src / "imgs.npy", mmap_mode="r")
        for i, e in enumerate(b): B.append(e); SRC.append(src); IM.append(np.asarray(im[i]))
    return B, IM, SRC


def main():
    OUT.mkdir(parents=True, exist_ok=True); B, IM, SRC = load_bank(); N = len(B); dev = "cuda"
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to(dev).eval(); mean, std = policy.action_pos_mean, policy.action_pos_std; dec = policy.decoder; D.capture_install(policy); ck = torch.load("outputs/train/safediff_vla_g16/geometry_conditioner.pt", weights_only=False)
    geo = GeometryConditioner(float(ck["alpha_max"]), out_dim=dec.action_head.in_features).to(dev); geo.load_state_dict(ck["state"]); geo.eval(); s_mean, s_std = policy.state_pos_mean, policy.state_pos_std
    enc = SafeDiffVLAPolicy._encode_multimodal_latent; feat0 = policy.visual_grounding_cross_attn; grounder = G5.Grounder("g4d", policy)
    print("decoder:", len(dec.transformer.layers), "layers, norm_first", dec.transformer.layers[0].norm_first, "final norm", dec.transformer.norm, "hidden", dec.action_head.in_features, flush=True)
    def mk_batch(imgs, e):
        b = {k: torch.from_numpy(imgs[c]).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dev) for c, k in enumerate(KEYS)}; b["observation.state"] = torch.from_numpy(e["state_norm"]).float().unsqueeze(0).to(dev); b["observation.language.tokens"] = torch.from_numpy(e["lang_tokens"]).long().unsqueeze(0).to(dev); b["observation.language.attention_mask"] = torch.from_numpy(e["lang_mask"]).bool().unsqueeze(0).to(dev); return b
    nrm = lambda a: ((torch.as_tensor(a, dtype=std.dtype, device=dev) - mean) / std).reshape(1, 3)
    def dec_out(out): a = policy._decode_action_prediction(out); return {"xyz": (a[0, :, :3] * std + mean).cpu().numpy().astype(np.float32), "grip": (a[0, :, 6] * policy.action_grip_std + policy.action_grip_mean).cpu().numpy().astype(np.float32)}
    res = []; maxdiff = 0.0
    for i in range(N):
        e = B[i]; b0 = mk_batch(IM[i], e); cs = e["cards_xyz_robot"]; tn = e["target_name"]; ot = [n for n in cs if n != tn]; w = min(ot, key=lambda n: np.linalg.norm(np.asarray(cs[n])[:2] - np.asarray(cs[tn])[:2])); gt = np.asarray(e["gt_anchor_robot"], float); wr = gt + np.asarray(cs[w]) - np.asarray(cs[tn])
        with torch.no_grad():
            lat0 = enc(policy, b0); own0 = feat0(policy.target_query_extractor(*lat0), *lat0); g4 = grounder.compute(*lat0, b0)["xyz_n"].reshape(1, -1)[:, :3].to(std.dtype); policy.target_point_head = FixedHead(nrm(gt)); policy.visual_grounding_cross_attn = FixedFeat(own0); policy._encode_multimodal_latent = lambda bb, L=lat0: L; policy._v2_grasp_phase = e["phase"]
            ref = SafeDiffVLAPolicy._plan_temporal_decoder(policy, b0); ctx = dec._ctx; ref_out = dec(ctx["tok"], ctx["pad"], ctx["state"], None, ctx["target"], None, None, ctx["feat"], ctx["phase"]); maxdiff = max(maxdiff, D.check(dec, ctx, ref_out))
            anchors = {"GT": nrm(gt), "WR": nrm(wr), "G4D": g4}
            for ax, nm in enumerate("xyz"):
                for sg, sn in ((1, "p"), (-1, "m")): d = np.zeros(3); d[ax] = sg * D5; anchors[nm + sn] = nrm(gt + d)
            ee_m = ctx["state"][:, :3] * s_std + s_mean; cond = {}
            def hgeo(T): return geo(T * std + mean - ee_m, std)
            for an, T in anchors.items():
                cond[("V2", an)] = dec_out(D.run(dec, ctx, T)[0]); cond[("G4", an)] = dec_out(D.run(dec, ctx, T, in_scale=4.0)[0]); cond[("NT", an)] = dec_out(D.run(dec, ctx, T, in_scale=0.0)[0])
                for loc in D.LOCS[1:]: cond[("move_" + loc, an)] = dec_out(D.place(dec, ctx, T, loc, "move"))
                for loc in D.LOCS: cond[("extra_" + loc, an)] = dec_out(D.place(dec, ctx, T, loc, "extra")); cond[("geo_" + loc, an)] = dec_out(D.place(dec, ctx, T, loc, "extra", vec=hgeo(T)))
            r0 = D.run(dec, ctx, anchors["GT"], record=True)[1]; r1 = D.run(dec, ctx, anchors["WR"], record=True)[1]; prop = {}
            din = (r1["in"] - r0["in"]); prop = {p: {"d": (r1[p] - r0[p]).norm(dim=-1).mean().item(), "h": r0[p].norm(dim=-1).mean().item(), "cos": (torch.nn.functional.cosine_similarity((r1[p] - r0[p]).reshape(-1, 512), din.reshape(-1, 512), dim=-1).mean().item() if p != "out" else float("nan"))} for p in PTS}
            prop["nt_in_norm"] = dec.target_xyz_encoder(anchors["GT"]).norm().item(); prop["hgeo_norm"] = hgeo(anchors["GT"]).norm().item()
            res.append({"i": i, "src": SRC[i], "ep": e["ep"], "t": e["t"], "phase": e["phase"], "ee": e["ee_robot"], "gt": gt, "wr": wr, "grip_raw": e["grip_raw"], "any_disp_cm": e["any_card_displaced_cm"], "ee_gt_xy": e["ee_gt_xy"], "ee_gt_z": e["ee_gt_z"], "cond": cond, "prop": prop})
        if (i + 1) % 100 == 0: print(i + 1, "max |runner - dec.forward| so far", maxdiff, flush=True)
    pickle.dump({"R": res, "maxdiff": maxdiff}, open(OUT / "g17_bank.pkl", "wb")); print("saved; runner-vs-forward max diff", maxdiff)


if __name__ == "__main__":
    main()
