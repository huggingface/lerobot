#!/usr/bin/env python
"""G13 visuals: for representative frames, the images before / after the intervention (and the DONOR image for swaps) next to the chunk z curves (receiver original = black, intervened = red, donor's own chunk shifted by the EE-height difference = blue dashed; gray = EE z, green = anchor z)."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g13_interventions import BANK, OUT  # noqa: E402


def main():
    D = pickle.load(open(OUT / "interventions.pkl", "rb")); C0, R, donors, B, SRC = D["C0"], D["R"], D["donors"], D["bank"], D["src"]; M = np.load(OUT / "mean_images.npy"); IM, MK = [], []
    for src in ("g4d", "oracle"):
        im = np.load(BANK / src / "imgs.npy", mmap_mode="r"); mk = np.load(BANK / src / "masks_packed.npy")
        for i in range(len(im)): IM.append(np.asarray(im[i])); MK.append(np.unpackbits(mk[i])[:3 * 2 * 480 * 480].reshape(3, 2, 480, 480).astype(bool))
    N = len(B); dz30 = lambda i, a: (a["xyz"][29, 2] - B[i]["ee_robot"][2]) * 100
    def img_for(i, name, j=None):
        x = IM[i].copy(); mk_ = MK[i]
        if name.startswith("cam_"): c = {"front": 0, "right": 1, "wrist": 2}.get(name[4:]); x = M.copy() if name == "cam_all" else x; (x.__setitem__(c, M[c]) if c is not None else None)
        elif name in ("robot_all", "scene_all", "cards_all"):
            for c in range(3): m_ = mk_[c][0] if name == "robot_all" else (~mk_[c][0] if name == "scene_all" else mk_[c][1]); x[c][m_] = M[c][m_]
        elif name.startswith("swapR"):
            for c in ((2,) if name.endswith("wrist") else (0, 1, 2)): left = mk_[c][0] & ~MK[j][c][0]; x[c][left] = M[c][left]; x[c][MK[j][c][0]] = IM[j][c][MK[j][c][0]]
        elif name.startswith("swapS"):
            x = IM[j].copy()
            for c in range(3): left = MK[j][c][0] & ~mk_[c][0]; x[c][left] = M[c][left]; x[c][mk_[c][0]] = IM[i][c][mk_[c][0]]
        return x
    def pick(name, cond, q=0.5):
        idx = sorted([i for i in range(N) if (i, name) in R and cond(i)], key=lambda i: abs(dz30(i, R[(i, name)]["F"]) - dz30(i, C0[i])))
        return idx[int(q * (len(idx) - 1))]
    prim = lambda i: B[i]["ee_gt_z"] > 0.06
    cases = [("cam_wrist replaced by mean image", "cam_wrist", None, pick("cam_wrist", prim, 0.8), 2), ("robot pixels masked (all cams)", "robot_all", None, pick("robot_all", lambda i: prim(i) and B[i]["grip_raw"] > .5, 0.9), 2), ("scene pixels masked (all cams)", "scene_all", None, pick("scene_all", lambda i: prim(i) and B[i]["grip_raw"] > .5, 0.8), 2),
             ("ROBOT swap, donor gripper OPEN (receiver closed)", "swapR_diff", "Dg", pick("swapR_diff", lambda i: prim(i) and B[i]["grip_raw"] > .5, 0.6), 2), ("SCENE swap (donor other contact state)", "swapS_diff", "Sd", pick("swapS_diff", prim, 0.6), 0)]
    fig, ax = plt.subplots(len(cases), 4, figsize=(18, 4.2 * len(cases)))
    for r, (title, name, dk, i, cam) in enumerate(cases):
        j = donors[dk][i] if dk else None; x = img_for(i, name, j); e = B[i]
        ax[r, 0].imshow(IM[i][cam]); ax[r, 0].set_title(f"original {['front', 'right', 'wrist'][cam]} | {SRC[i]} ep{e['ep']} t={e['t']} grip={'closed' if e['grip_raw'] > .5 else 'open'}", fontsize=8); ax[r, 1].imshow(x[cam]); ax[r, 1].set_title(f"{title}\nintervened", fontsize=8)
        if j is not None and j >= 0: ax[r, 2].imshow(IM[j][cam]); ax[r, 2].set_title(f"donor ({SRC[j]} ep{B[j]['ep']} t={B[j]['t']} grip={'closed' if B[j]['grip_raw'] > .5 else 'open'})", fontsize=8)
        else: ax[r, 2].set_title("(no donor)", fontsize=8)
        for a in ax[r, :3]: a.axis("off")
        a = ax[r, 3]; a.plot(C0[i]["xyz"][:, 2], "k", lw=2, label="original"); a.plot(R[(i, name)]["F"]["xyz"][:, 2], "r", lw=2, label="intervened")
        if j is not None and j >= 0: a.plot(C0[j]["xyz"][:, 2] - (B[j]["ee_robot"][2] - e["ee_robot"][2]), "b--", lw=1.6, label="donor chunk (EE-shifted)")
        a.axhline(e["ee_robot"][2], color="gray", ls=":"); a.axhline(e["gt_anchor_robot"][2], color="green", lw=.8); g0 = np.argmax(C0[i]["grip"] <= .5) if (C0[i]["grip"] <= .5).any() else 50; g1 = np.argmax(R[(i, name)]["F"]["grip"] <= .5) if (R[(i, name)]["F"]["grip"] <= .5).any() else 50
        a.set_title(f"chunk z (m); first close idx: original {g0} -> intervened {g1}", fontsize=8); a.set_xlabel("chunk index", fontsize=7); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "g13_interventions.png", dpi=48); print("saved")


if __name__ == "__main__":
    main()
