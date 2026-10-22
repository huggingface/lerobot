"""[v2.1 integration gate (b)] DATASET check: on 60 real TRAIN episodes/task (split_constants train list), C = stored StagePlan-v2 label grasp_xyz vs B = observation.state[:3] (measured EE, robot frame) at the close transition tc and at t1."""
import json, pickle, numpy as np, pandas as pd
df = pd.read_pickle("outputs/stageplan/df6.pkl"); L = pickle.load(open("outputs/stageplan/labels_v2.pkl", "rb")); train = set(json.load(open("outputs/stageplan/split_constants.json"))["train"]); out = {}
def st(d):
    d = np.asarray(d) * 100; n = np.linalg.norm(d, axis=1); return {"n": len(d), "median_xyz_cm": np.median(d, 0).round(3).tolist(), "mean_xyz_cm": d.mean(0).round(3).tolist(), "p90_norm_cm": round(float(np.percentile(n, 90)), 3), "max_norm_cm": round(float(n.max()), 3), "median_norm_cm": round(float(np.median(n)), 3)}
for fam in ("select_poker", "select_mahjong"):
    eps = [e for e in sorted(L) if L[e]["fam"] == fam and e in train][:60]; Ctc, Ct1 = [], []
    for e in eps:
        g = df[df.episode_index == e].sort_values("frame_index"); S = np.stack(g["observation.state"].values)[:, :3].astype(float); l = L[e]; Ctc.append(l["grasp_xyz"] - S[l["tc"]]); Ct1.append(l["grasp_xyz"] - S[l["t1"]])
    out[fam] = {"C_minus_B_at_tc": st(Ctc), "C_minus_B_at_t1": st(Ct1)}
json.dump(out, open("outputs/stageplan_v21/x/label_frame_dataset.json", "w"), indent=1); print(json.dumps(out, indent=1))
