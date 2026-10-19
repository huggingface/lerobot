#!/usr/bin/env python
"""G21 analysis: per task x arm (ORACLE+decoder vs ORACLE+geometry, seeds 3000-3004). Oracle = live target-entity grasp keypoint (diagnostic). Stages: REACH/GRASP (reach <3cm, close, close-EE-oracle 3D, contact, lift, retention) vs POST-GRASP/NATIVE SUCCESS. Failure taxonomy for every non-success: E1 never reached <3cm | E2 reached/closed but no lift or grasp | E3 grasp (lift>3cm or is_grasped) but no native success | E4 flag: target or another object displaced >1.5cm BEFORE the close (collision/approach) | E5 flag: native success without any close/lift (metric false-positive candidate)."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np

OUT = Path("outputs/eval/safediff_vla_g21"); TASKS = ["select_poker", "select_book", "select_mahjong", "select_drink", "select_fruit", "select_toy"]; ARMS = ["decoder", "geo"]


def episode(ep, L):
    S = ep["steps"]; tgt = ep["meta"]["target_entity"]; ph0 = [r for r in L if r["phase"] == 0]; n = len(S)
    close_i = next((i for i, s in enumerate(S) if s.get("reactive_close_event")), None); c0 = S[0]["candidates_world"]; tz0 = np.array(S[0]["target_pos"]); d = {"seed": ep["seed"], "success": bool(ep["success"]), "len": n - 1, "target": tgt}
    d["min_xy"] = min(r["dxy"] for r in ph0) * 100; d["min_z"] = min(abs(r["dz"]) for r in ph0) * 100; d["min_3d"] = min(r["d3"] for r in ph0) * 100; d["reach"] = d["min_3d"] < 3; d["t_anchor"] = next((r["t"] for r in ph0 if r["d3"] < .03), None)
    d["close"] = close_i is not None; d["close_d3"] = L[close_i - 1]["d3"] * 100 if close_i is not None and close_i - 1 < len(L) else None
    tp = np.array([s["target_pos"] for s in S]); disp_t = np.linalg.norm(tp - tp[0], axis=1); lift = (tp[:, 2] - tp[0, 2]).max(); d["lift_cm"] = lift * 100; d["lifted"] = bool(lift > 0.03); d["grasped_ever"] = any(s["is_grasped"] for s in S)
    pre_end = close_i if close_i is not None else n; others = [k for k in c0 if k != tgt]; disp_o = np.array([[np.linalg.norm(np.array(S[i]["candidates_world"][k]) - np.array(c0[k])) for k in others] for i in range(n)]) if others else np.zeros((n, 0))
    d["target_contact"] = bool(disp_t.max() > 0.015 or d["grasped_ever"]); d["wrong_contact"] = bool(disp_o.size and disp_o.max() > 0.015); d["E4"] = bool(disp_t[:pre_end].max() > 0.015 or (disp_o.size and disp_o[:pre_end].max() > 0.015))
    post = [s["is_grasped"] for s in S[close_i:]] if close_i is not None else []; d["retention"] = float(np.mean(post)) if post else None; d["E5"] = bool(d["success"] and not (d["close"] and (d["lifted"] or d["grasped_ever"])))
    if d["success"]: d["cat"] = "OK" + ("/E5?" if d["E5"] else "")
    elif not d["reach"]: d["cat"] = "E1"
    elif not (d["lifted"] or d["grasped_ever"]): d["cat"] = "E2"
    else: d["cat"] = "E3"
    if not d["success"] and d["E4"]: d["cat"] += "+E4"
    return d


def sheets(R):
    import cv2, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for t in TASKS:
        picks = []
        for arm in ARMS:
            for d in R[(t, arm)]:
                picks.append((arm, d))
        sel = []
        g = [x for x in picks if x[0] == "geo"]; dd = [x for x in picks if x[0] == "decoder"]
        for name, pool in (("geo SUCCESS", [x for x in g if x[1]["success"]]), ("geo FAILURE", [x for x in g if not x[1]["success"]]), ("decoder FAILURE", [x for x in dd if not x[1]["success"]]), ("decoder SUCCESS", [x for x in dd if x[1]["success"]])):
            if pool: sel.append((name, *pool[0]))
        fig, ax = plt.subplots(len(sel), 6, figsize=(24, 4 * len(sel)), squeeze=False)
        for r, (name, arm, d) in enumerate(sel):
            vids = sorted((OUT / t / arm / "videos").glob("*.mp4")); ix = [d["ep"]]; vp = vids[d["ep"]] if d["ep"] < len(vids) else None
            if vp is None: continue
            cap = cv2.VideoCapture(str(vp)); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fr = np.linspace(0, nf - 1, 6).astype(int)
            for c, f in enumerate(fr): cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read(); ax[r, c].imshow(im[..., ::-1]) if ok else None; ax[r, c].axis("off"); ax[r, c].set_title(f"{name} seed {d['seed']} [{d['cat']}] frame {f}/{nf}" if c == 0 else f"frame {f}", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / f"g21_video_{t}.png", dpi=40); plt.close(fig)


def main():
    R = {}
    for t in TASKS:
        for arm in ARMS:
            p = OUT / t / arm
            if not (p / "episodes_raw.json").exists(): continue
            E = json.load(open(p / "episodes_raw.json")); L = pickle.load(open(p / "g21_log.pkl", "rb")); by = {i: [r for r in L["log"] if r["ep"] == i] for i in range(len(E))}; R[(t, arm)] = [dict(episode(e, by[i]), ep=i) for i, e in enumerate(E)]; R[(t, arm, "chunks")] = L["chunks"]
    print(f"{'task':16s}{'arm':8s}{'n':>3s}{'reach<3cm':>10s}{'close':>6s}{'closeD3cm':>10s}{'tgtContact':>11s}{'wrongCont':>10s}{'lift>3cm':>9s}{'retention':>10s}{'SUCCESS':>8s}{'minXY':>7s}{'minZ':>6s}{'min3D':>7s}{'tAnchor':>8s}")
    for t in TASKS:
        for arm in ARMS:
            if (t, arm) not in R: continue
            r = R[(t, arm)]; n = len(r); f = lambda k: sum(bool(x[k]) for x in r); cl = [x["close_d3"] for x in r if x["close_d3"] is not None]; ta = [x["t_anchor"] for x in r if x["t_anchor"] is not None]; rt = [x["retention"] for x in r if x["retention"] is not None]
            print(f"{t:16s}{arm:8s}{n:3d}{f('reach'):>7d}/{n}{f('close'):>4d}/{n}{(np.mean(cl) if cl else float('nan')):10.1f}{f('target_contact'):>8d}/{n}{f('wrong_contact'):>7d}/{n}{f('lifted'):>6d}/{n}{(np.mean(rt) if rt else float('nan')):10.2f}{f('success'):>6d}/{n}{np.mean([x['min_xy'] for x in r]):7.1f}{np.mean([x['min_z'] for x in r]):6.1f}{np.mean([x['min_3d'] for x in r]):7.1f}{(np.mean(ta) if ta else float('nan')):8.0f}")
    print("\n== per-episode (min3D cm | close d3 cm | lift cm | retention | success | category) ==")
    for t in TASKS:
        for arm in ARMS:
            if (t, arm) in R: print(f" {t[7:]:8s}{arm:8s}" + " ; ".join(f"{x['seed']}: {x['min_3d']:.1f}|{('%.1f' % x['close_d3']) if x['close_d3'] is not None else '-'}|{x['lift_cm']:.1f}|{('%.2f' % x['retention']) if x['retention'] is not None else '-'}|{int(x['success'])}|{x['cat']}" for x in R[(t, arm)]))
    print("\n== FAILURE TAXONOMY of ORACLE+GEOMETRY non-successes (and decoder, for reference) ==")
    for arm in ARMS:
        cats = {}
        for t in TASKS:
            for x in R.get((t, arm), []): cats.setdefault(x["cat"], []).append(t[7:])
        print(f"  {arm}: " + " | ".join(f"{k}: {len(v)} ({','.join(sorted(set(v)))})" for k, v in sorted(cats.items())))
    for t in TASKS:
        c = R.get((t, "geo", "chunks"), [])
        if c: print(f"  geo first-action offset {t[7:]}: mean {np.mean([x['first_off'] for x in c]) * 100:.2f} cm over {len(c)} replans")
    pickle.dump(R, open(OUT / "g21_summary.pkl", "wb")); sheets(R); print("saved sheets")


if __name__ == "__main__":
    main()
