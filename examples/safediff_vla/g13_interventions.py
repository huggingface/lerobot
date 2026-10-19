#!/usr/bin/env python
"""G13 pass 2 (offline, no environment): visual-source causal interventions on the stored observation bank (797 frames). Decoder conditioning fixed to the GT anchor (A1); phase flag = the frame's own phase; NO retraining.
For every intervened image set the VLM latents are recomputed; the decoder is run in two feature variants: F = V2's grounded feature of the ORIGINAL image (isolates the decoder's visual memory path, as in G10) and R = grounded feature recomputed from the intervened latents (the deployed pipeline).
ONE fill rule everywhere: pixels removed from an image are replaced by the same-camera dataset-mean image (mean over 300 poker demonstration frames, resized to 480).
Interventions
  Step 1  cam_front / cam_right / cam_wrist / cam_all : whole camera image replaced by the mean image (other cameras untouched)
  Step 2  robot_all / scene_all / cards_all / robot_wrist / robot_front : sim segmentation (DIAGNOSTIC ONLY): robot pixels (franka/*), non-robot pixels, card+holder pixels replaced by the mean image
  Step 3/4 swapR_diff / swapR_same / swapR_diff_wrist : paste the DONOR's robot pixels over the receiver (receiver robot pixels not covered by the donor -> mean image); donor = similar EE (<=5cm) and similar target-relative geometry (<=10cm) with a DIFFERENT gripper state (diff) or the SAME state (same = procedure-artefact control)
          swapS_diff / swapS_same : receiver robot kept, donor SCENE (non-robot pixels) pasted; donor = similar EE and same gripper state with a different contact state (diff) or same contact state in another episode (same = control)"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

BANK = Path("outputs/eval/safediff_vla_g13_bank"); OUT = Path("outputs/eval/safediff_vla_g13_audit"); KEYS = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3"); CAMN = ("front", "right", "wrist")


class FixedHead(nn.Module):
    def __init__(self, v): super().__init__(); self.v = v
    def forward(self, x): return self.v.to(device=x.device, dtype=x.dtype)


class FixedFeat(nn.Module):
    def __init__(self, v): super().__init__(); self.v = v
    def forward(self, *a, **k): return self.v


def mean_images():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    f = OUT / "mean_images.npy"
    if f.exists(): return np.load(f)
    d = LeRobotDataset("lerobot/vlabench_unified"); fi = d.meta.episodes["dataset_from_index"]; ti = d.meta.episodes["dataset_to_index"]; E = pd.read_json("outputs/eval/safediff_vla_phase_d0_training_data_audit/episodes.json"); P = E[E.task == "select_poker"]; rng = np.random.default_rng(0); acc = np.zeros((3, 224, 224, 3), np.float64); n = 0
    for _, r in P.sample(300, random_state=0).iterrows():
        ep = int(r.episode); idx = int(fi[ep]) + int(rng.integers(0, int(ti[ep]) - int(fi[ep]))); it = d[idx]
        for c, k in enumerate(("observation.images.image", "observation.images.second_image", "observation.images.wrist_image")): im = it[k].permute(1, 2, 0).numpy(); acc[c] += im * (255 if im.max() <= 1.0 else 1)
        n += 1
    m = np.stack([cv2.resize((acc[c] / n).astype(np.float32), (480, 480), interpolation=cv2.INTER_LINEAR) for c in range(3)]).round().clip(0, 255).astype(np.uint8); np.save(f, m); return m


def main():
    OUT.mkdir(parents=True, exist_ok=True); M = mean_images(); print("mean images", M.shape, flush=True)
    B, IM, MK, SRC = [], [], [], []
    for src in ("g4d", "oracle"):
        b = pickle.load(open(BANK / src / "bank.pkl", "rb")); im = np.load(BANK / src / "imgs.npy", mmap_mode="r"); mk = np.load(BANK / src / "masks_packed.npy")
        for i, e in enumerate(b): B.append(e); SRC.append(src); IM.append(np.asarray(im[i])); MK.append(np.unpackbits(mk[i])[:3 * 2 * 480 * 480].reshape(3, 2, 480, 480).astype(bool))
    N = len(B); print("bank frames", N, flush=True); ee = np.array([e["ee_robot"] for e in B]); anc = np.array([e["gt_anchor_robot"] for e in B]); evec = anc - ee; grip = np.array([e["grip_raw"] > 0.5 for e in B]); contact = np.array([e["n_displaced"] >= 1 for e in B]); epk = np.array([f"{s}{e['ep']}" for s, e in zip(SRC, B)])
    def nearest(i, ok, w_e=0.5, max_ee=0.05, max_ev=0.10):
        d_ee = np.linalg.norm(ee - ee[i], axis=1); d_ev = np.linalg.norm(evec - evec[i], axis=1); cand = ok & (d_ee <= max_ee) & (d_ev <= max_ev); cand[i] = False
        return int(np.argmin(np.where(cand, d_ee + w_e * d_ev, 1e9))) if cand.any() else -1
    donors = {k: np.full(N, -1) for k in ("Dg", "Ds", "Sd", "Ss")}
    for i in range(N):
        donors["Dg"][i] = nearest(i, grip != grip[i]); donors["Ds"][i] = nearest(i, (grip == grip[i]) & (epk != epk[i]))
        donors["Sd"][i] = nearest(i, (grip == grip[i]) & (contact != contact[i]), 0.0, 0.05, 1e9); donors["Ss"][i] = nearest(i, (grip == grip[i]) & (contact == contact[i]) & (epk != epk[i]), 0.0, 0.05, 1e9)
    print("donor availability (of %d frames):" % N, {k: int((v >= 0).sum()) for k, v in donors.items()}, flush=True)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); mean, std = policy.action_pos_mean, policy.action_pos_std; dev = "cuda"
    def mk_batch(imgs, e):
        b = {k: torch.from_numpy(imgs[c]).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dev) for c, k in enumerate(KEYS)}; b["observation.state"] = torch.from_numpy(e["state_norm"]).float().unsqueeze(0).to(dev); b["observation.language.tokens"] = torch.from_numpy(e["lang_tokens"]).long().unsqueeze(0).to(dev); b["observation.language.attention_mask"] = torch.from_numpy(e["lang_mask"]).bool().unsqueeze(0).to(dev); return b
    @torch.no_grad()
    def decode(b, a_gt, lat, feat):
        tok, pad, mod = lat; policy.target_point_head = FixedHead(a_gt); policy.visual_grounding_cross_attn = FixedFeat(feat); policy._encode_multimodal_latent = lambda bb: (tok, pad, mod); phase = B_phase[0]
        policy._v2_grasp_phase = phase; act = SafeDiffVLAPolicy._plan_temporal_decoder(policy, b)[0][0]
        return {"xyz": (act[:, :3] * std + mean).cpu().numpy(), "grip": (act[:, 6] * policy.action_grip_std + policy.action_grip_mean).cpu().numpy()}
    enc_orig = SafeDiffVLAPolicy._encode_multimodal_latent; head0, feat0 = policy.target_point_head, policy.visual_grounding_cross_attn; B_phase = [0]; R = {}; C0 = {}
    def conds_for(i):
        im, mk_ = IM[i], MK[i]; out = {}
        for c, n in enumerate(CAMN): x = im.copy(); x[c] = M[c]; out[f"cam_{n}"] = x
        out["cam_all"] = M.copy()
        def maskfill(sel, cams=(0, 1, 2)):
            x = im.copy()
            for c in cams: m_ = sel(mk_[c]); x[c][m_] = M[c][m_]
            return x
        out["robot_all"] = maskfill(lambda m: m[0]); out["scene_all"] = maskfill(lambda m: ~m[0]); out["cards_all"] = maskfill(lambda m: m[1]); out["robot_wrist"] = maskfill(lambda m: m[0], (2,)); out["robot_front"] = maskfill(lambda m: m[0], (0,))
        def swapR(j, cams=(0, 1, 2)):
            x = im.copy()
            for c in cams: left = mk_[c][0] & ~MK[j][c][0]; x[c][left] = M[c][left]; x[c][MK[j][c][0]] = IM[j][c][MK[j][c][0]]
            return x
        def swapS(j):
            x = IM[j].copy()
            for c in range(3): leftover = MK[j][c][0] & ~mk_[c][0]; x[c][leftover] = M[c][leftover]; x[c][mk_[c][0]] = im[c][mk_[c][0]]
            return x
        if donors["Dg"][i] >= 0: out["swapR_diff"] = swapR(donors["Dg"][i]); out["swapR_diff_wrist"] = swapR(donors["Dg"][i], (2,))
        if donors["Ds"][i] >= 0: out["swapR_same"] = swapR(donors["Ds"][i])
        if donors["Sd"][i] >= 0: out["swapS_diff"] = swapS(donors["Sd"][i])
        if donors["Ss"][i] >= 0: out["swapS_same"] = swapS(donors["Ss"][i])
        return out
    import time; t0 = time.time()
    for i in range(N):
        e = B[i]; B_phase[0] = e["phase"]; a_gt = ((torch.as_tensor(e["gt_anchor_robot"], dtype=std.dtype, device=dev) - mean) / std).unsqueeze(0); b0 = mk_batch(IM[i], e)
        with torch.no_grad():
            lat0 = enc_orig(policy, b0); tq = policy.target_query_extractor(*[lat0[0], lat0[1], lat0[2]]); own0 = feat0(tq, lat0[0], lat0[1], lat0[2]); C0[i] = decode(b0, a_gt, lat0, own0)
            for name, x in conds_for(i).items():
                bx = mk_batch(x, e); lat = enc_orig(policy, bx); tq2 = policy.target_query_extractor(lat[0], lat[1], lat[2]); own2 = feat0(tq2, lat[0], lat[1], lat[2]); R[(i, name)] = {"F": decode(bx, a_gt, lat, own0), "R": decode(bx, a_gt, lat, own2)}
        if (i + 1) % 50 == 0: print(f"frame {i + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    pickle.dump({"C0": C0, "R": R, "donors": donors, "bank": B, "src": SRC}, open(OUT / "interventions.pkl", "wb")); print("saved")


if __name__ == "__main__":
    main()
