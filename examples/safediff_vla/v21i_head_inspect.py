"""[v2.1 integration gate 5] What does the SHARED anchor head actually use to produce the poker (+2.0 cm z) vs mahjong (+1.0 cm z, +0.4 cm y) offsets? Frozen head, NO retraining.
On DEV scenes (GT-matched proposals): error of C under ablations: (n) normal; (g) triangulated-3D input replaced by the TRAIN mean ref; (t) both crop-token inputs replaced by TRAIN mean tokens;
(gt) both; (x) cross-task swap: poker crops with mahjong refs and vice-versa (checks where the task cue lives)."""
import json, pickle, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import v21g_pipeline as GP, v21q_prop_iter as Q, v21hr_candidate as H, v21p_proposals as P0, v21g_right_prop as R
from v21_common import OUT as V21, load_policy, offsets
X = V21 / "x"; pol, pre = load_policy(); off = offsets(); xr = pickle.load(open(X / "referent_table_x.pkl", "rb")); split = json.load(open(X / "split_x.json")); A_, B_ = Q.build_rows("A"), Q.build_rows("B")
tr = [r for r in xr if split[r["scene_id"]] == "train" and r["scene_ok"]] + [r for r in A_ if r["scene_ok"]]; dev = [r for r in B_ if r["scene_ok"]]
fm = Q.Head(64).cuda(); fm.load_state_dict(torch.load(X / "proposal_head_v21q_g64.pt")); fm.eval(); rm = Q.Head(64).cuda(); rm.load_state_dict(torch.load(X / "proposal_head_right_g64.pt")); rm.eval()
ctx = GP.Ctx(pol, pre, None, fm, rm); allr = tr + dev; P = GP.props(fm, rm, P0.patch_feats(pol, allr), R.feats_cam(pol, allr, 1), allr)
Dtr = GP.train_dataset(ctx, tr[::4], P, off); Dd = GP.train_dataset(ctx, dev, P, off)
head = GP.AnchorHead().cuda(); head.load_state_dict(torch.load(X / "anchor_head.pt")); head.eval()
mtf = torch.stack([d["tf"] for d in Dtr]).float().mean(0).cuda(); mtr = torch.stack([d["tr"] for d in Dtr]).float().mean(0).cuda(); mref = torch.tensor(np.mean([d["ref"] for d in Dtr], 0), dtype=torch.float32).cuda()
def run(D, mode, other=None):
    out = []
    with torch.no_grad():
        for i, d in enumerate(D):
            tf, trr, ref = d["tf"].float().cuda()[None], d["tr"].float().cuda()[None], torch.tensor(d["ref"], dtype=torch.float32).cuda()[None]
            if mode in ("g", "gt"): ref = mref[None]
            if mode in ("t", "gt"): tf, trr = mtf[None], mtr[None]
            if mode == "x_ref": ref = torch.tensor(other[i]["ref"], dtype=torch.float32).cuda()[None]
            out.append(head(tf, trr, ref)[0].cpu().numpy())
    return np.array(out)
res = {}
for t in H.TASKS:
    D = [d for d in Dd if d["task"] == t]; Y = np.array([d["res"] for d in D]); o = {}
    for mode in ("n", "g", "t", "gt"):
        p = run(D, mode); e = np.linalg.norm(p - Y, axis=1); o[mode] = {"median_err_cm": float(np.median(e)), "mean_pred_offset_cm": p.mean(0).round(3).tolist()}
    oth = [d for d in Dd if d["task"] != t]; oth = [oth[i % len(oth)] for i in range(len(D))]; p = run(D, "x_ref", oth); o["x_ref(other-task ref geometry, own crops)"] = {"median_err_cm": float(np.median(np.linalg.norm(p - Y, axis=1))), "mean_pred_offset_cm": p.mean(0).round(3).tolist()}
    o["gt_mean_offset_cm"] = Y.mean(0).round(3).tolist(); res[t] = o
json.dump(res, open(X / "head_inspection.json", "w"), indent=1); print(json.dumps(res, indent=1))
