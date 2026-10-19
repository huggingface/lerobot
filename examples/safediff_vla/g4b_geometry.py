#!/usr/bin/env python
"""G4b STEP 0 -- right->front geometry sanity (no training). Right coarse peak cell centre -> camera ray (fixed calibration) -> intersection with the table plane
(z = single global constant = median card-surface height over TRAIN scenes; NOT per-scene GT) -> front-camera pixel -> 96px front crop box.
GT bboxes/segmentation are used ONLY to score recall / projection error (diagnostic)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from g1_train import D, TaskData  # noqa: E402
from g4_build_cache import crop_geom, nms_peaks  # noqa: E402

PD = D / "select_poker"; OUT = Path("outputs/eval/safediff_vla_g4b_offline"); CAL = json.load(open(D / "camera_calibration.json"))["select_poker"]["cams_policy_order"]


def ray(cam, u, v):  # 480px pixel -> world origin, unit dir
    R = np.array(cam["xmat"]).reshape(3, 3); f = 240 / np.tan(np.deg2rad(cam["fovy"]) / 2); d = R @ np.array([(u - 240) / f, -(v - 240) / f, -1.0]); return np.array(cam["xpos"]), d / np.linalg.norm(d)


def proj(cam, xyz):
    R = np.array(cam["xmat"]).reshape(3, 3); p = R.T @ (np.asarray(xyz) - np.array(cam["xpos"])); f = 240 / np.tan(np.deg2rad(cam["fovy"]) / 2); return f * p[0] / -p[2] + 240, 240 - f * p[1] / -p[2]


def right_to_front(cell, zplane):
    """right-camera 8x8 cell -> front pixel (u,v) via table plane. cell centre in 480px."""
    u, v = (cell % 8 + 0.5) * 60, (cell // 8 + 0.5) * 60; o, d = ray(CAL[1], u, v)
    if abs(d[2]) < 1e-6: return None
    t = (zplane - o[2]) / d[2]
    if t <= 0: return None
    return proj(CAL[0], o + t * d)


def front_box_at(u, v, side=96):
    x0 = min(max(u - side / 2, 0), 480 - side); y0 = min(max(v - side / 2, 0), 480 - side); return x0, y0, x0 + side, y0 + side


def zplane_const(data):
    zs = [json.load(open(PD / f"{data.sids[s]}.json"))["candidates"][0]["object_pose_world"]["xpos"][2] for s in data.idx("train")[:200]]; return float(np.median(zs))


def inside(b, bb): cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2; return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def main():
    data = TaskData("select_poker", "cpu"); B = {}
    for p in sorted(PD.glob("bboxes_*.json")): B.update(json.load(open(p)))
    z = zplane_const(data); print("table-plane z constant (train median card height):", round(z, 4)); Z = torch.load("outputs/train/safediff_vla_g3/M1_s0_preds.pt", weights_only=False); res = {"zplane": z}
    for label, sp, held_sel in (("seen_test", "test", False), ("heldout_rank9", "test_heldout_identity", True)):
        P, W = Z["preds"][sp]; ids = data.idx(sp); R = {K: {"front": 0, "right_proj": 0, "union": 0, "right_orig": 0, "orig_union": 0} for K in (1, 2)}; n = 0; err = []; err_soft = []
        for j, s in enumerate(ids):
            names = data.names[s]; sid = data.sids[s]; ks = [k for k in range(len(names)) if bool(data.valid[s, k])]
            for k in ks:
                if bool(data.held[s, k]) != held_sel: continue
                n += 1; tb = B[sid]["boxes_xyxy_per_camera"][names[k]]; tf, tr = tb[0], tb[1]
                pf = nms_peaks(W[j, k, :64].numpy().reshape(8, 8), 2); pr = nms_peaks(W[j, k, 64:128].numpy().reshape(8, 8), 2)
                fb = [crop_geom(c, 0, 480) for c in pf]; fb = [(x, y, x + s_, y + s_) for x, y, s_ in fb]; rb_orig = [crop_geom(c, 1, 480) for c in pr]; rb_orig = [(x, y, x + s_, y + s_) for x, y, s_ in rb_orig]
                pb = []
                for c in pr:
                    q = right_to_front(c, z); pb.append(front_box_at(*q) if q is not None else None)
                for K in (1, 2):
                    hf = any(inside(b, tf) for b in fb[:K]); hp = any(b is not None and inside(b, tf) for b in pb[:K]); hr = any(inside(b, tr) for b in rb_orig[:K])
                    R[K]["front"] += hf; R[K]["right_proj"] += hp; R[K]["union"] += hf or hp; R[K]["right_orig"] += hr; R[K]["orig_union"] += hf or hr
                q = right_to_front(pr[0], z)
                if q is not None: err.append(float(np.hypot(q[0] - (tf[0] + tf[2]) / 2, q[1] - (tf[1] + tf[3]) / 2)))
        res[label] = {"n": n, "recall": {str(K): {m: v / n for m, v in R[K].items()} for K in R}, "right_top1_proj_err_px_front(diag)": {"median": float(np.median(err)), "mean": float(np.mean(err)), "p90": float(np.percentile(err, 90))}}
        for K in (1, 2): print(f"{label:14s} n={n} K={K}: front-only {R[K]['front'] / n:.3f} | right->front projected {R[K]['right_proj'] / n:.3f} | UNION(front+projected) {R[K]['union'] / n:.3f} | [ref] original front+right(own crops) union {R[K]['orig_union'] / n:.3f}")
        print("   right top-1 -> front projection error vs GT card centre (px, front 480): median %.1f mean %.1f p90 %.1f  (crop half-side 48)" % (np.median(err), np.mean(err), np.percentile(err, 90)))
    (OUT / "g4b_step0_geometry.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
