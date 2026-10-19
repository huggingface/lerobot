"""G4b proposal geometry (official, fixed): coarse 8x8 maps -> world table points -> FRONT-view 144px crops.
front peak (top-2, 3x3 NMS)  : cell centre -> front ray -> table plane (world point)          [front crop is centred on the cell centre]
right peak (top-2, 3x3 NMS)  : 3x3 local mass centroid (in right px) -> right ray -> table plane -> projected into the front camera
Candidate order F1,R1,F2,R2; later candidates whose front-crop centre is < DEDUP*side from an earlier kept one are dropped (<=4 kept).
Every candidate carries a [192] spatial neighbourhood mask: 3x3 cells around its front pixel (front cams) U 3x3 cells around its right pixel (right cam).
Only fixed calibration + the coarse map are used; GT is never used here. Plane z = one global constant (train-scene median card height)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g4_build_cache import nms_peaks  # noqa: E402
from g4b_geometry import CAL, ray, proj  # noqa: E402

SIDE = 144; DEDUP = 0.5; ZPLANE = 0.831; J = 4


def centroid(m, c):
    r, cc = divmod(c, 8); rs = range(max(0, r - 1), min(8, r + 2)); cs = range(max(0, cc - 1), min(8, cc + 2)); w = np.array([[m[a, b] for b in cs] for a in rs]); ys = np.array([(a + .5) * 60 for a in rs]); xs = np.array([(b + .5) * 60 for b in cs])
    if w.sum() <= 0: return (cc + .5) * 60, (r + .5) * 60  # fp16-underflowed neighbourhood: fall back to the peak-cell centre
    return float((w.sum(0) @ xs) / w.sum()), float((w.sum(1) @ ys) / w.sum())


def to_plane(cam_i, u, v):
    o, d = ray(CAL[cam_i], u, v); t = (ZPLANE - o[2]) / d[2]; return o + t * d


def nbr_mask(fu, fv, ru, rv):
    m = np.zeros(192, bool)
    for cam, (u, v) in ((0, (fu, fv)), (1, (ru, rv))):
        c, r = int(np.clip(u // 60, 0, 7)), int(np.clip(v // 60, 0, 7))
        for rr in range(max(0, r - 1), min(8, r + 2)):
            for cc in range(max(0, c - 1), min(8, c + 2)): m[cam * 64 + rr * 8 + cc] = True
    return m


def front_box(u, v, side=SIDE):
    x0 = min(max(u - side / 2, 0), 480 - side); y0 = min(max(v - side / 2, 0), 480 - side); return int(round(x0)), int(round(y0)), side


def make(w, side=SIDE, mode="new", dedup=True):
    """w: [192] coarse map. returns list of dicts (<=J): box (x0,y0,side) in 480px, nbr [192] bool, src, rank. mode 'old96' = cell-centre projection, side 96 (Step-0 baseline)."""
    mf, mr = w[:64].reshape(8, 8), w[64:128].reshape(8, 8); pf, pr = nms_peaks(mf, 2), nms_peaks(mr, 2); cands = []
    for rank in range(2):
        c = pf[rank]; u, v = (c % 8 + .5) * 60, (c // 8 + .5) * 60; p = to_plane(0, u, v); ru, rv = proj(CAL[1], p); cands.append({"src": "front", "rank": rank, "fpx": (u, v), "rpx": (ru, rv)})
        c = pr[rank]; u, v = centroid(mr, c) if mode == "new" else ((c % 8 + .5) * 60, (c // 8 + .5) * 60); p = to_plane(1, u, v); fu, fv = proj(CAL[0], p); cands.append({"src": "right", "rank": rank, "fpx": (fu, fv), "rpx": (u, v)})
    out = []
    for c in cands:
        bx = front_box(*c["fpx"], side)
        if dedup and any(np.hypot((bx[0] + side / 2) - (o["box"][0] + side / 2), (bx[1] + side / 2) - (o["box"][1] + side / 2)) < DEDUP * side for o in out): continue
        c["box"] = bx; c["nbr"] = nbr_mask(*c["fpx"], *c["rpx"]); out.append(c)
        if len(out) == J: break
    return out


def inside(box, bb): cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2; return box[0] <= cx <= box[0] + box[2] and box[1] <= cy <= box[1] + box[2]


def purity(box, boxes_front, k):
    """boxes_front: list of GT front bboxes (None for invalid) for all valid cards; k: target index. GT used for diagnostics / L_fine label only."""
    x0, y0, s = box; cx, cy = x0 + s / 2, y0 + s / 2; ins, cd, inter = [], {}, {}
    for c, bb in enumerate(boxes_front):
        if bb is None: continue
        bcx, bcy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2; cd[c] = np.hypot(bcx - cx, bcy - cy); inter[c] = max(0, min(x0 + s, bb[2]) - max(x0, bb[0])) * max(0, min(y0 + s, bb[3]) - max(y0, bb[1]))
        if x0 <= bcx <= x0 + s and y0 <= bcy <= y0 + s: ins.append(c)
    return {"contains_target": float(k in ins), "n_cards": len(ins), "target_nearest": float(k in ins and min(ins, key=lambda c: cd[c]) == k), "tgt_frac": inter[k] / (s * s), "dis_frac": sum(v for c, v in inter.items() if c != k) / (s * s)}
