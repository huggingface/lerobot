#!/usr/bin/env python
"""G4-A representation-only feasibility test (DIAGNOSTIC; the GT crops used here are NOT an inference method).
Question: does the SAME FROZEN VLM recover card identity when a card is shown LARGE?
Features per (scene, card), all probed with the SAME scene-level-split L2 logistic probe as the G3 diagnostic (rank 13 / suit 4 / identity 52):
  A   current full-frame POST-transformer latent, neutral text, GT-mask pooled (G3 diagnostic number)
  A2  same, front and right camera pooled separately (1920-d, same width as the crop features)
  A'  full-frame embed_image tokens (vision tower + connector, PRE-transformer), GT-mask pooled per camera  <- resolution-matched control for C1
  B   raw 40x40 pixel crop (G3 diagnostic number)
  C1  high-res crop -> frozen vision tower + connector (embed_image), mean of the 64 tokens, front|right
  C2  high-res crop -> the FULL frozen VLM prefix pipeline (neutral text; crop in camera1/camera2, black camera3), mean of that camera's 64 image tokens
Crop: square, centred on the GT bbox centre (diagnostic only), side = CROP_MULT x median card bbox max-side (fixed once from training-scene statistics),
shifted to stay inside the image, resized by the VLM's own resize_with_pad to the 512x512 visual input."""
from __future__ import annotations

import copy, json, math, os, sys
import cv2
from pathlib import Path

import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from g1_train import D  # noqa: E402
from g3_identity_diagnostic import RANKS, SUITS, probe, rs  # noqa: E402

from lerobot.envs import make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.common.vla_utils import resize_with_pad  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SNAP = os.environ.get("G4_SNAP") == "1"; DOWN = os.environ.get("G4_DOWN") == "1"; PD = D / "select_poker"; OUT = Path("outputs/eval/safediff_vla_g4_offline"); CROP_MULT = 3.0; NEUTRAL = "primitive: Please pick the poker"


def load_boxes():
    B = {}
    for p in sorted(PD.glob("bboxes_*.json")): B.update(json.load(open(p)))
    return B


def crop_box(b, side, size=480):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    if SNAP: cx, cy = (int(cx // 60) + 0.5) * 60, (int(cy // 60) + 0.5) * 60  # inference-feasible: crop centred on a 60-px coarse CELL centre, not on the GT bbox
    x0 = int(round(min(max(cx - side / 2, 0), size - side))); y0 = int(round(min(max(cy - side / 2, 0), size - side))); return x0, y0, int(side)


def main():
    OUT.mkdir(parents=True, exist_ok=True); B = load_boxes(); splits = json.load(open(PD / "splits.json")); sids = sorted(splits)
    bad = [s for s, v in B.items() if max(v["rgb_mean_abs_diff_vs_stored"][:2]) > 0.5]; print("scenes with bboxes:", len(B), "| scene reproduction mismatches (front/right RGB diff>0.5):", len(bad), flush=True)
    # ---- fixed crop size from TRAINING-scene bbox statistics ----
    ms = {cam: [max(b[cam][2] - b[cam][0], b[cam][3] - b[cam][1]) for s in sids if splits[s] == "train" and s in B for b in B[s]["boxes_xyxy_per_camera"].values() if b[cam]] for cam in (0, 1)}
    side = {cam: int(2 * math.ceil(CROP_MULT * float(np.median(ms[cam])) / 2)) for cam in (0, 1)}
    stats = {"card_bbox_max_side_px": {c: {"median": float(np.median(ms[c])), "p10": float(np.percentile(ms[c], 10)), "p90": float(np.percentile(ms[c], 90))} for c in (0, 1)}, "crop_mult": CROP_MULT, "crop_side_px": side}
    print("bbox stats / crop side:", json.dumps(stats), flush=True)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); cfg = VLABenchEnv(task="select_poker")
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    env_pre, _ = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config); vlm = policy.backbone.model.vlm_with_expert; scale = math.sqrt(960)

    @torch.no_grad()
    def embed(img_uint8):  # [H,W,3] uint8 -> [64,960] frozen vision tower + connector, same scaling as embed_prefix
        t = torch.from_numpy(img_uint8).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
        t = resize_with_pad(t, 512, 512, pad_value=0) * 2.0 - 1.0; return (vlm.embed_image(t)[0].float() * scale).cpu().numpy()

    data = {}; names_all = []
    cf = torch.load(D / "select_poker" / "latent_cache.pt", weights_only=False)["cellfrac"].numpy(); neutral = torch.load(Path("outputs/eval/safediff_vla_g3_offline/g3_neutral_image_latents.pt"), weights_only=False)["tokens"].float().numpy()
    rows = []; F = {k: [] for k in ("A", "A2", "Aprime", "C1", "C1_front", "C2")}
    for si, sid in enumerate(sids):
        z = np.load(PD / f"{sid}.npz"); rec = json.load(open(PD / f"{sid}.json")); rgb = z["rgb"]
        full = [embed(rgb[0]), embed(rgb[1])]
        for k, c in enumerate(rec["candidates"]):
            bx = B[sid]["boxes_xyxy_per_camera"][c["name"]]; w = cf[si, k]; rows.append((si, k, RANKS.index(rs(c["name"])[0]), SUITS.index(rs(c["name"])[1]), splits[sid]))
            ww = lambda sl: (w[sl] / max(w[sl].sum(), 1e-9))
            F["A"].append((w / max(w.sum(), 1e-9))[:, None].T @ neutral[si]); F["A"][-1] = F["A"][-1][0]
            F["A2"].append(np.concatenate([ww(slice(0, 64)) @ neutral[si][:64], ww(slice(64, 128)) @ neutral[si][64:128]]))
            F["Aprime"].append(np.concatenate([ww(slice(0, 64)) @ full[0], ww(slice(64, 128)) @ full[1]]))
            crops = []
            for cam in (0, 1):
                x0, y0, s_ = crop_box(bx[cam], side[cam])
                if DOWN:
                    f = 224 / 480; sm = cv2.resize(rgb[cam], (224, 224), interpolation=cv2.INTER_AREA); a0, b0 = int(round(x0 * f)), int(round(y0 * f)); ss = max(2, int(round(s_ * f))); crops.append(sm[b0:b0 + ss, a0:a0 + ss])
                else: crops.append(rgb[cam][y0:y0 + s_, x0:x0 + s_])
            e = [embed(cr).mean(0) for cr in crops]; F["C1"].append(np.concatenate(e)); F["C1_front"].append(e[0])
            if SNAP: F["C2"].append(np.zeros(1)); continue
            o = preprocess_observation(copy.deepcopy({"pixels": {"image": crops[0][None], "second_image": crops[1][None], "wrist_image": np.zeros_like(crops[0])[None]}, "agent_pos": z["state"][None].astype(np.float64)})); o["task"] = [NEUTRAL]
            with torch.no_grad(): tok, pad, mod = policy._encode_multimodal_latent(pre(env_pre(o)))
            t = tok[0].float().cpu().numpy(); F["C2"].append(np.concatenate([t[:64].mean(0), t[64:128].mean(0)]))
        if (si + 1) % 50 == 0: print(si + 1, "/", len(sids), flush=True)
    rows_a = np.array([(r[0], r[1], r[2], r[3]) for r in rows]); spl = np.array([r[4] for r in rows]); rk, su = rows_a[:, 2], rows_a[:, 3]; ident = rk * 4 + su
    Fs = {k: np.stack(v).astype(np.float32) for k, v in F.items()}; np.savez_compressed(OUT / ("g4_card_features_snapped_224.npz" if DOWN else "g4_card_features_snapped.npz") if SNAP else "g4_card_features.npz", rows=rows_a, split=spl, **Fs)
    tr, va, te, held = spl == "train", spl == "val", spl == "test", spl == "test_heldout_identity"; res = {"crop": stats, "n_cards": int(len(rows)), "probe": {}}
    for name, X in Fs.items():
        if SNAP and name not in ("C1", "C1_front"): continue
        res["probe"][name] = {}
        for lab, y, ch in (("rank", rk, 1 / 13), ("suit", su, 0.25), ("identity", ident, 1 / 48)):
            r = probe(X[tr], y[tr], X[va], y[va], X[te], y[te]); r["chance"] = ch; res["probe"][name][lab] = r
        sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(C=res["probe"][name]["suit"]["C"], max_iter=3000).fit(sc.transform(X[tr]), su[tr]); res["probe"][name]["suit"]["acc_on_heldout_rank9_scenes"] = float((clf.predict(sc.transform(X[held])) == su[held]).mean())
        print(f"{name:9s} test acc rank {res['probe'][name]['rank']['test_acc']:.3f} | suit {res['probe'][name]['suit']['test_acc']:.3f} | identity {res['probe'][name]['identity']['test_acc']:.3f}", flush=True)
    (OUT / ("g4a_crop_probe_cellsnapped_224px.json" if DOWN else "g4a_crop_probe_cellsnapped.json") if SNAP else "g4a_crop_probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
