#!/usr/bin/env python
"""G4-B/C: INFERENCE-FEASIBLE candidate generation + COARSE RECALL AUDIT (no new network).
Candidate generation uses ONLY the G3-M1 coarse referent score map (3 cameras x 8x8): per static camera (front, right), greedy top-K cells with 3x3 non-maximum
suppression; each peak -> original-RGB coordinate (cell centre) -> a fixed-size square crop box (side from training-scene card statistics, same as G4-A).
GT bboxes are used ONLY to score recall: the instructed card is RECALLED by a crop set iff its GT bbox centre lies inside one of the K crop boxes (stricter: bbox fully
inside). Failure to include the target is recorded as a candidate-generation failure (nothing is moved or removed using GT).
Reference chance: K cells drawn uniformly (same NMS) among the cells that contain ANY candidate card, i.e. the best a map-agnostic picker could do once the card cluster
is found. Reported per camera and as the union (recalled in either static camera)."""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from g1_train import D, TaskData  # noqa: E402
from g3_identity_diagnostic import rs  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g4_offline"); OUTT = Path("outputs/train/safediff_vla_g3"); PD = D / "select_poker"; KS = (1, 2, 3, 4)


def nms_peaks(m, K):
    m = m.copy(); peaks = []
    for _ in range(K):
        if m.max() <= -1e9: break
        r, c = np.unravel_index(int(m.argmax()), m.shape); peaks.append((r, c)); m[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = -1e9
    return peaks


def box_of(rc, side):
    cx, cy = (rc[1] + 0.5) * 60, (rc[0] + 0.5) * 60; x0 = min(max(cx - side / 2, 0), 480 - side); y0 = min(max(cy - side / 2, 0), 480 - side); return x0, y0, x0 + side, y0 + side


def inside_center(b, bb): cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2; return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
def inside_full(b, bb): return b[0] <= bb[0] and b[1] <= bb[1] and b[2] >= bb[2] and b[3] >= bb[3]


def main():
    data = TaskData("select_poker", "cpu"); B = {}
    for p in sorted(PD.glob("bboxes_*.json")): B.update(json.load(open(p)))
    side = {int(k): v for k, v in json.load(open(OUT / "g4a_crop_probe.json"))["crop"]["crop_side_px"].items()}; rng = np.random.default_rng(0); cf = data.cf.numpy(); res = {"crop_side_px": side}
    for seed in range(3):
        Z = torch.load(OUTT / f"M1_s{seed}_preds.pt", weights_only=False); res[f"seed{seed}"] = {}
        for sp, label in (("test", "seen_identity"), ("test_heldout_identity", "heldout_rank9"), ("test_heldout_identity", "seen_rank_in_heldout_scenes")):
            if sp not in Z["preds"]: continue
            P, W = Z["preds"][sp]; ids = data.idx(sp); acc = {cam: {K: [0, 0, 0] for K in KS} for cam in (0, 1, "union")}; chance = {cam: {K: 0.0 for K in KS} for cam in (0, 1, "union")}; n = 0; distract = []
            for j, s in enumerate(ids):
                names = data.names[s]; sid = data.sids[s]; ks = [k for k in range(len(names)) if bool(data.valid[s, k])]
                for k in ks:
                    held = bool(data.held[s, k])
                    if label == "heldout_rank9" and not held: continue
                    if label == "seen_rank_in_heldout_scenes" and held: continue
                    if label == "seen_identity" and held: continue
                    n += 1; bx = B[sid]["boxes_xyxy_per_camera"]; got = {}
                    for cam in (0, 1):
                        m = W[j, k, cam * 64:(cam + 1) * 64].numpy().reshape(8, 8); peaks = nms_peaks(m, max(KS)); boxes = [box_of(p, side[cam]) for p in peaks]; tb = bx[names[k]][cam]
                        # chance: K random cells among card-containing cells
                        cells = np.where(sum(cf[s, c, cam * 64:(cam + 1) * 64] for c in ks) > 0)[0]
                        for K in KS:
                            hit_c = tb is not None and any(inside_center(b, tb) for b in boxes[:K]); hit_f = tb is not None and any(inside_full(b, tb) for b in boxes[:K])
                            got.setdefault(cam, {})[K] = (hit_c, hit_f)
                            acc[cam][K][0] += hit_c; acc[cam][K][1] += hit_f; acc[cam][K][2] += 1
                            if len(cells):
                                dr = 0
                                for _ in range(20):
                                    pk = nms_peaks(np.where(np.isin(np.arange(64), rng.choice(cells, size=min(len(cells), K), replace=False)).reshape(8, 8), 1.0, -1e9), K); dr += tb is not None and any(inside_center(box_of(q, side[cam]), tb) for q in pk)
                                chance[cam][K] += dr / 20
                    for K in KS:
                        acc["union"][K][0] += got[0][K][0] or got[1][K][0]; acc["union"][K][1] += got[0][K][1] or got[1][K][1]; acc["union"][K][2] += 1
                        chance["union"][K] += 0.0
                    # distractors inside the top-1 front crop
                    m = W[j, k, :64].numpy().reshape(8, 8); b1 = box_of(nms_peaks(m, 1)[0], side[0]); distract.append(sum(inside_center(b1, bx[names[c]][0]) for c in ks if c != k and bx[names[c]][0]))
            res[f"seed{seed}"][label] = {"n_queries": n, "recall_center_inside": {str(cam): {str(K): acc[cam][K][0] / max(acc[cam][K][2], 1) for K in KS} for cam in acc},
                                         "recall_bbox_fully_inside": {str(cam): {str(K): acc[cam][K][1] / max(acc[cam][K][2], 1) for K in KS} for cam in acc},
                                         "chance_recall_front_right(K random card-containing cells)": {str(cam): {str(K): chance[cam][K] / max(n, 1) for K in KS} for cam in (0, 1)},
                                         "mean_distractor_cards_inside_top1_front_crop": float(np.mean(distract)) if distract else None}
    (OUT / "g4c_coarse_recall.json").write_text(json.dumps(res, indent=1))
    for label in ("seen_identity", "heldout_rank9", "seen_rank_in_heldout_scenes"):
        for cam in ("0", "1", "union"):
            v = {K: [res[f"seed{s}"][label]["recall_center_inside"][cam][str(K)] for s in range(3) if label in res[f"seed{s}"]] for K in KS}
            print(f"{label:28s} cam={cam:5s} n={res['seed0'][label]['n_queries']:4d} recall@K(center): " + " ".join(f"K{K}={np.mean(v[K]):.3f}[{min(v[K]):.2f},{max(v[K]):.2f}]" for K in KS))
        ch = {K: np.mean([res[f"seed{s}"][label]["chance_recall_front_right(K random card-containing cells)"]["0"][str(K)] for s in range(3)]) for K in KS}
        print(f"{'':28s} chance front: " + " ".join(f"K{K}={ch[K]:.3f}" for K in KS), "| mean distractors in top-1 front crop:", round(float(np.mean([res[f"seed{s}"][label]["mean_distractor_cards_inside_top1_front_crop"] for s in range(3)])), 2))
    # ---- sanity visualisation: crop boxes from the coarse map (seed 0) ----
    Z = torch.load(OUTT / "M1_s0_preds.pt", weights_only=False); picks = []
    def find(sp, cond):
        P, W = Z["preds"][sp]
        for j, s in enumerate(data.idx(sp)):
            ks = [k for k in range(4) if bool(data.valid[s, k])]
            for k in ks:
                if cond(s, k, ks): return sp, j, s, k
    names = data.names
    picks.append(("easy 3-card", find("test", lambda s, k, ks: len(ks) == 3 and not data.held[s, k])))
    picks.append(("difficult 4-card", find("test", lambda s, k, ks: len(ks) == 4 and not data.held[s, k])))
    picks.append(("same-rank/suit distractor", find("test", lambda s, k, ks: any((rs(names[s][c])[0] == rs(names[s][k])[0] or rs(names[s][c])[1] == rs(names[s][k])[1]) for c in ks if c != k) and len(ks) == 4 and not data.held[s, k])))
    picks.append(("held-out rank 9", find("test_heldout_identity", lambda s, k, ks: bool(data.held[s, k]) and len(ks) == 4)))
    fig, ax = plt.subplots(len(picks), 5, figsize=(20, 4.1 * len(picks)))
    for r, (title, (sp, j, s, k)) in enumerate(picks):
        sid = data.sids[s]; rgb = np.load(PD / f"{sid}.npz")["rgb"][0]; P, W = Z["preds"][sp]; m = W[j, k, :64].numpy().reshape(8, 8); peaks = nms_peaks(m, 3); bx = B[sid]["boxes_xyxy_per_camera"]
        a = ax[r, 0]; a.imshow(rgb); a.imshow(m, extent=(0, 480, 480, 0), cmap="jet", alpha=0.4, interpolation="nearest")
        for c in range(len(names[s])):
            if bx[names[s][c]][0]: b = bx[names[s][c]][0]; a.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime" if c == k else "white", lw=2 if c == k else 1))
        for q, p in enumerate(peaks):
            b = box_of(p, side[0]); a.add_patch(plt.Rectangle((b[0], b[1]), side[0], side[0], fill=False, ec="red", lw=1.5, ls="--")); a.text(b[0], b[1] - 3, f"#{q + 1}", color="red", fontsize=9)
        a.set_xlim(120, 360); a.set_ylim(330, 150); a.set_title(f"{title}: {sid}\nquery {names[s][k]}  (green=GT target, white=other cards, red=top-3 crops)", fontsize=8); a.set_xticks([]); a.set_yticks([])
        for q in range(3):
            a = ax[r, 1 + q]
            if q < len(peaks):
                b = [int(v) for v in box_of(peaks[q], side[0])]; a.imshow(rgb[b[1]:b[3], b[0]:b[2]]); hit = bx[names[s][k]][0] and inside_center(b, bx[names[s][k]][0]); a.set_title(f"crop #{q + 1} ({side[0]}px) | contains target: {bool(hit)} | map {m[peaks[q]]:.2f}", fontsize=8)
            a.axis("off")
        ax[r, 4].imshow(np.load(PD / f"{sid}.npz")["rgb"][1]); mm = W[j, k, 64:128].numpy().reshape(8, 8); ax[r, 4].imshow(mm, extent=(0, 480, 480, 0), cmap="jet", alpha=0.4, interpolation="nearest")
        for q, p in enumerate(nms_peaks(mm, 3)): b = box_of(p, side[1]); ax[r, 4].add_patch(plt.Rectangle((b[0], b[1]), side[1], side[1], fill=False, ec="red", lw=1.5, ls="--"))
        bb = bx[names[s][k]][1]; ax[r, 4].add_patch(plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="lime", lw=2)); ax[r, 4].set_title("right camera (top-3 crop boxes)", fontsize=8); ax[r, 4].axis("off")
    fig.tight_layout(); (OUT / "figures").mkdir(exist_ok=True); fig.savefig(OUT / "figures" / "g4_candidate_generation_sanity.png", dpi=70); print("saved figure")


if __name__ == "__main__":
    main()
