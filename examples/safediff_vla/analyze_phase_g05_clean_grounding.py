#!/usr/bin/env python
"""PHASE G0.5 analysis (offline; reads `phase_g05_clean_grounding_collect.py` outputs only).

Unit of analysis = EPISODE. Primary sample = step-0 observation (what the first planner chunk is
computed from; arm at home pose, scene static). Steps 0..N-1 are near-duplicates of step 0, so they
are used only as a per-episode-mean robustness check, never pooled as independent samples.

Classes (identity margin m = d_distractor_min - d_target, 3D, predicted vs GT object origin):
    D ambiguous            |m| <= EPS_AMBIG
    A identity-wrong       m < -EPS_AMBIG
    B identity-ok, loc-bad m >  EPS_AMBIG and xy error > LOC_THR
    C identity-ok, loc-ok  m >  EPS_AMBIG and xy error <= LOC_THR
Thresholds are fixed a priori (EPS 2 cm, LOC 5 cm) and sensitivity-checked, not tuned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval/safediff_vla_grounded_grasp_v2_phase_g05_clean_grounding")
EPS_AMBIG, LOC_THR = 0.02, 0.05
GRID = 8


def load_all():
    eps = []
    for meta_p in sorted(ROOT.glob("*/seed*/meta.json")):
        meta = json.loads(meta_p.read_text())
        arr = np.load(meta_p.parent / "arrays.npz")
        eps.append({"dir": meta_p.parent, "task": meta["task"], "seed": meta["seed"], "meta": meta, "arr": {k: arr[k] for k in arr.files}})
    return eps


def step_metrics(ep, t, eps_ambig=EPS_AMBIG, loc_thr=LOC_THR):
    s = ep["meta"]["steps"][t]
    a = ep["arr"]
    names, tname = s["cand_names"], s["target_name"]
    pred = np.array(s["pred_world"])
    pos = {n: np.array(s["cand_pos_world"][n]) for n in names}
    others = [n for n in names if n != tname]
    d_t = np.linalg.norm(pred - pos[tname])
    d_o = {n: np.linalg.norm(pred - pos[n]) for n in others}
    near = min(d_o, key=d_o.get)
    m = d_o[near] - d_t
    err_xy = np.linalg.norm((pred - pos[tname])[:2])
    spacing = min(np.linalg.norm(pos[tname] - pos[n]) for n in others)
    cls = "D" if abs(m) <= eps_ambig else ("A" if m < 0 else ("B" if err_xy > loc_thr else "C"))
    # attention mass by region (head-mean attention over the 192 image tokens)
    att = a["attn_img_heads"][t].mean(0)  # [192]
    att = att / max(att.sum(), 1e-12)
    cf = lambda n: a[f"cellfrac::{n}"][t].reshape(-1)  # [192] cam-major, row-major
    f_t, f_near, f_rob = cf(tname), cf(near), cf("__robot__")
    f_dist_all = sum(cf(n) for n in others)
    mass = {"target": float(att @ f_t), "near_distractor": float(att @ f_near), "all_distractors": float(att @ f_dist_all),
            "robot": float(att @ f_rob)}
    mass["background"] = 1.0 - mass["target"] - mass["all_distractors"] - mass["robot"]
    area = {"target": float(f_t.sum() / 192), "near_distractor": float(f_near.sum() / 192),
            "all_distractors": float(f_dist_all.sum() / 192), "robot": float(f_rob.sum() / 192)}
    area["background"] = 1.0 - area["target"] - area["all_distractors"] - area["robot"]
    peak = int(att.argmax())
    peak_region = max(("target", f_t[peak]), ("all_distractors", f_dist_all[peak]), ("robot", f_rob[peak]),
                      ("background", 1 - f_t[peak] - f_dist_all[peak] - f_rob[peak]), key=lambda x: x[1])[0]
    return {
        "task": ep["task"], "seed": ep["seed"], "t": t, "target": tname, "near": near, "cls": cls,
        "d_target": float(d_t), "d_dist": float(d_o[near]), "margin": float(m), "err_xy": float(err_xy),
        "err_3d": float(d_t), "err_signed": (pred - pos[tname]).tolist(), "spacing": float(spacing),
        "margin_norm": float(m / spacing), "mass": mass, "area": area, "peak_region": peak_region,
        "ent": float(s["img_attn_entropy"]), "vis_px": s["target_visible_px"],
        "proj_vs_seg_px": [np.hypot(*(np.array(s["proj_px"][str(c)][tname]) - np.array(s["segcentroid_px"][str(c)])))
                           if s["segcentroid_px"][str(c)] is not None else None for c in range(3)],
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def proportions(rows, tasks):
    out = {}
    for label, sel in [("ALL", rows)] + [(t, [r for r in rows if r["task"] == t]) for t in tasks]:
        n = len(sel)
        out[label] = {"n": n, **{c: {"k": sum(r["cls"] == c for r in sel), "frac": (sum(r["cls"] == c for r in sel) / n if n else None),
                                      "ci95": wilson(sum(r["cls"] == c for r in sel), n)} for c in "ABCD"}}
    return out


def mass_by_class(rows):
    out = {}
    for c in "ABCD":
        sel = [r for r in rows if r["cls"] == c and sum(r["vis_px"]) > 0]
        if not sel:
            continue
        o = {"n": len(sel)}
        for reg in ["target", "near_distractor", "all_distractors", "robot", "background"]:
            m = np.array([r["mass"][reg] for r in sel])
            ar = np.array([r["area"][reg] for r in sel])
            o[reg] = {"mass_mean": float(m.mean()), "area_mean": float(ar.mean()),
                      "lift_of_means": float(m.mean() / max(ar.mean(), 1e-9)),
                      "frac_eps_lift_gt1": float(np.mean(m > ar))}
        o["peak_region_counts"] = {k: int(sum(r["peak_region"] == k for r in sel)) for k in ["target", "all_distractors", "robot", "background"]}
        out[c] = o
    return out


def loeo_probe(X, y, groups, tasks_arr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pred, base = [], []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        pred.append(clf.predict(sc.transform(X[te]))[0])
        # task-only baseline: majority label of the same task in the training fold
        same = tr & (tasks_arr == tasks_arr[te][0])
        base.append(int(y[same].mean() >= 0.5) if same.any() else int(y[tr].mean() >= 0.5))
    order = np.concatenate([np.where(groups == g)[0] for g in np.unique(groups)])
    yt = y[order]
    return {"loeo_acc": float(np.mean(np.array(pred) == yt)), "task_majority_baseline_acc": float(np.mean(np.array(base) == yt)),
            "global_majority_acc": float(max(yt.mean(), 1 - yt.mean()))}


def make_figures(eps, rows0, mass):
    # ---- representative: per class, sample with err_xy closest to the class median (fixed rule) ----
    reps = []
    for c in "ABCD":
        sel = [r for r in rows0 if r["cls"] == c]
        if sel:
            med = np.median([r["err_xy"] for r in sel])
            reps.append(min(sel, key=lambda r: abs(r["err_xy"] - med)))
    if reps:
        fig, axes = plt.subplots(len(reps), 3, figsize=(13, 4.2 * len(reps)))
        axes = np.atleast_2d(axes)
        for i, r in enumerate(reps):
            ep = next(e for e in eps if e["task"] == r["task"] and e["seed"] == r["seed"])
            s = ep["meta"]["steps"][0]
            att = ep["arr"]["attn_img_heads"][0].mean(0)
            att = att / att.sum()
            labels = np.load(ep["dir"] / "step00_labels.npy")
            for cam in range(3):
                ax = axes[i, cam]
                img = plt.imread(ep["dir"] / f"step00_camera{cam + 1}.png")
                ax.imshow(img)
                hm = att[cam * 64:(cam + 1) * 64].reshape(GRID, GRID)
                ax.imshow(hm, cmap="jet", alpha=0.45, extent=(0, 480, 480, 0), interpolation="nearest", vmin=0, vmax=max(att.max(), 1e-6))
                for lv, col in [(1, "lime"), (2, "red"), (3, "cyan")]:
                    if (labels[cam] == lv).any():
                        ax.contour(labels[cam] == lv, levels=[0.5], colors=col, linewidths=1.2)
                tp = s["proj_px"][str(cam)][r["target"]]
                ax.plot(*tp, "+", color="white", ms=14, mew=2)
                ax.set_title(f"[{r['cls']}] {r['task']} s{r['seed']} cam{cam + 1}", fontsize=9)
                ax.axis("off")
            axes[i, 0].text(0.0, -0.04, f"instructed={r['target']} | nearest-to-pred={r['near']} | margin={r['margin'] * 100:.1f}cm | xy err={r['err_xy'] * 100:.1f}cm | "
                            f"mass tgt/distr/robot/bg={r['mass']['target']:.2f}/{r['mass']['all_distractors']:.2f}/{r['mass']['robot']:.2f}/{r['mass']['background']:.2f}",
                            transform=axes[i, 0].transAxes, fontsize=8, va="top")
        fig.suptitle("step-0 attention (8x8 token grid, row-major) | green=target mask, red=distractor, cyan=robot, white +=GT origin projection", fontsize=10)
        fig.tight_layout()
        fig.savefig(ROOT / "representative_cases.png", dpi=100)
        plt.close(fig)

    # ---- aggregate ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    cols = {t: c for t, c in zip(sorted({r["task"] for r in rows0}), plt.cm.tab10.colors)}
    for r in rows0:
        ax[0].scatter(r["margin"] * 100, r["err_xy"] * 100, color=cols[r["task"]], s=25)
    for t, c in cols.items():
        ax[0].scatter([], [], color=c, label=t)
    ax[0].axvline(0, color="k", lw=0.8); ax[0].axvspan(-EPS_AMBIG * 100, EPS_AMBIG * 100, color="gray", alpha=0.15)
    ax[0].axhline(LOC_THR * 100, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel("identity margin m (cm)  [m<0: distractor closer]"); ax[0].set_ylabel("xy error to GT object origin (cm)")
    ax[0].legend(fontsize=7); ax[0].set_title("step-0 samples (1 per episode)")
    regs = ["target", "all_distractors", "robot", "background"]
    w = 0.2
    for j, c in enumerate("ABC"):
        if c in mass:
            ax[1].bar(np.arange(4) + j * w, [mass[c][g]["mass_mean"] for g in regs], w, label=f"{c} (n={mass[c]['n']})")
    ax[1].plot(np.arange(4) + w, [np.mean([r["area"][g] for r in rows0]) for g in regs], "k_", ms=18, label="area share (uniform attn)")
    ax[1].set_xticks(np.arange(4) + w); ax[1].set_xticklabels(regs, fontsize=8); ax[1].legend(fontsize=7); ax[1].set_title("attention mass by region")
    for c, col in zip("ABCD", ["tab:red", "tab:orange", "tab:green", "tab:gray"]):
        v = [r["ent"] for r in rows0 if r["cls"] == c]
        if v:
            ax[2].scatter([c] * len(v), v, color=col, alpha=0.6)
    ax[2].set_title("image-attention entropy (nats) by class -- NOT confidence"); ax[2].set_ylabel("entropy")
    fig.tight_layout(); fig.savefig(ROOT / "aggregate.png", dpi=100); plt.close(fig)


def main():
    eps = load_all()
    tasks = sorted({e["task"] for e in eps})
    rows0 = [step_metrics(e, 0) for e in eps]
    summary = {"n_episodes": len(eps), "tasks": tasks, "seeds_per_task": {t: sorted(e["seed"] for e in eps if e["task"] == t) for t in tasks}}

    # --- [1] validation ---
    val = {}
    for p in sorted(ROOT.glob("token_pixel_validation_*.json")):
        v = json.loads(p.read_text())
        cs = v["cases"]
        within1 = [(abs(c["latent_argmax"] // 64 - c["camera"]) == 0 and abs((c["latent_argmax"] % 64) // 8 - c["cell_rc"][0]) <= 1
                    and abs((c["latent_argmax"] % 64) % 8 - c["cell_rc"][1]) <= 1) for c in cs]
        within1e = [(abs(c["embed_argmax"] // 64 - c["camera"]) == 0 and abs((c["embed_argmax"] % 64) // 8 - c["cell_rc"][0]) <= 1
                     and abs((c["embed_argmax"] % 64) % 8 - c["cell_rc"][1]) <= 1) for c in cs]
        val[p.stem.replace("token_pixel_validation_", "")] = {
            "n_cases": len(cs), "meta": v["meta"],
            "embed_argmax_exact": float(np.mean([c["embed_argmax"] == c["expected_token"] for c in cs])),
            "embed_argmax_within_1cell": float(np.mean(within1e)),
            "latent_argmax_exact": float(np.mean([c["latent_argmax"] == c["expected_token"] for c in cs])),
            "latent_argmax_within_1cell": float(np.mean(within1)),
            "latent_argmax_equals_transposed": float(np.mean([c["latent_argmax"] == c["transposed_token"] for c in cs if c["expected_token"] != c["transposed_token"]])),
            "latent_mass_in_3x3_mean(chance=9/192=0.047)": float(np.mean([c["latent_mass_in_3x3"] for c in cs])),
        }
    summary["[1]_token_pixel_validation"] = val
    pe = [x for r in rows0 for x in r["proj_vs_seg_px"] if x is not None]
    summary["[1]_projection_vs_segmask_centroid_px"] = {"n": len(pe), "median": float(np.median(pe)), "p90": float(np.percentile(pe, 90))}

    # --- [3] margin stats ---
    ms = {}
    for t in ["ALL"] + tasks:
        sel = rows0 if t == "ALL" else [r for r in rows0 if r["task"] == t]
        m = np.array([r["margin"] for r in sel]) * 100
        ms[t] = {"n": len(sel), "margin_cm_quantiles_5_25_50_75_95": np.percentile(m, [5, 25, 50, 75, 95]).round(1).tolist(),
                 "frac_margin_neg": float(np.mean(m < 0)), "median_margin_norm_by_spacing": float(np.median([r["margin_norm"] for r in sel])),
                 "median_candidate_spacing_cm": float(np.median([r["spacing"] for r in sel]) * 100)}
    summary["[3]_margin"] = ms
    ext = {t: float(np.median([max(v["extent_xyz_m"][:2]) for e in eps if e["task"] == t for v in e["meta"]["extents"].values()]) * 100) for t in tasks}
    summary["[3]_object_xy_extent_cm_median_(bounding-sphere_upper_bound)"] = ext

    # --- [4] A/B/C/D ---
    summary["[4]_class_proportions_step0"] = proportions(rows0, tasks)
    sens = {}
    for eps_a in (0.01, 0.02, 0.03):
        for thr in (0.03, 0.05, 0.08):
            rr = [step_metrics(e, 0, eps_a, thr) for e in eps]
            sens[f"eps{eps_a}_thr{thr}"] = {c: sum(r["cls"] == c for r in rr) for c in "ABCD"}
    summary["[4]_threshold_sensitivity_counts"] = sens
    T = max(len(e["meta"]["steps"]) for e in eps)
    rows_all = [[step_metrics(e, t) for t in range(len(e["meta"]["steps"]))] for e in eps]
    summary["[4]_robustness_first30_per_episode_majority_class_counts"] = {
        c: sum(max("ABCD", key=lambda k: sum(r["cls"] == k for r in er)) == c for er in rows_all) for c in "ABCD"}
    summary["[4]_robustness_frac_steps_by_class_pooled(NOT independent)"] = {
        c: float(np.mean([r["cls"] == c for er in rows_all for r in er])) for c in "ABCD"}
    cond = {}
    for lab, f in [("identity_ok(m>eps)", lambda r: r["margin"] > EPS_AMBIG), ("identity_wrong(m<-eps)", lambda r: r["margin"] < -EPS_AMBIG), ("ambiguous", lambda r: abs(r["margin"]) <= EPS_AMBIG)]:
        sel = [r for r in rows0 if f(r)]
        if sel:
            cond[lab] = {"n": len(sel), "err_xy_cm_median": float(np.median([r["err_xy"] for r in sel]) * 100),
                         "err_xy_cm_mean": float(np.mean([r["err_xy"] for r in sel]) * 100), "err3d_cm_median": float(np.median([r["err_3d"] for r in sel]) * 100)}
    summary["[4]_error_conditioned_on_identity"] = cond
    se = {t: np.mean([r["err_signed"] for r in rows0 if r["task"] == t], 0).round(3).tolist() for t in tasks}
    summary["[4]_mean_signed_error_xyz_m_(pred-GT_origin)_per_task"] = se

    # --- [2] attention mass ---
    summary["[2]_attention_mass_by_class_step0"] = mass_by_class(rows0)
    pooled = mass_by_class([r for er in rows_all for r in er])
    summary["[2]_attention_mass_by_class_pooled_first30(not independent; descriptive)"] = pooled
    # episode-mean of steps 0..N for robustness
    allr = {"ALL": rows0}
    summary["[2]_vis_px_target_zero_in_all_cams_n"] = int(sum(sum(r["vis_px"]) == 0 for r in rows0))

    # --- [7] entropy ---
    ent = {}
    for c in "ABCD":
        v = [r["ent"] for r in rows0 if r["cls"] == c]
        if v:
            ent[c] = {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))}
    for lab, f in [("identity_ok", lambda r: r["margin"] > EPS_AMBIG), ("identity_wrong", lambda r: r["margin"] < -EPS_AMBIG),
                   ("loc_bad(xy>5cm)", lambda r: r["err_xy"] > LOC_THR), ("loc_good", lambda r: r["err_xy"] <= LOC_THR)]:
        v = [r["ent"] for r in rows0 if f(r)]
        if v:
            ent[lab] = {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))}
    allent = np.array([r["ent"] for r in rows0])
    wl = [r for r in rows0 if r["cls"] == "A" and r["ent"] <= np.percentile(allent, 25)]
    ent["wrong_but_low_entropy(A & entropy<=pooled 25th pct)"] = {"n": len(wl), "of_A": sum(r["cls"] == "A" for r in rows0)}
    summary["[7]_entropy_step0"] = ent

    # --- Q4 & [5] ---
    lq = np.stack([e["arr"]["lang_query"][0] for e in eps]); ft = np.stack([e["arr"]["feat"][0] for e in eps])
    nrm = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    tk = np.array([e["task"] for e in eps])
    def cos_stats(X):
        S = nrm(X) @ nrm(X).T
        same = [S[i, j] for i in range(len(eps)) for j in range(i) if tk[i] == tk[j]]
        diff = [S[i, j] for i in range(len(eps)) for j in range(i) if tk[i] != tk[j]]
        return {"within_task_mean_cos": float(np.mean(same)), "across_task_mean_cos": float(np.mean(diff))}
    summary["Q4_language_query_cosine_step0"] = cos_stats(lq)
    summary["Q4_grounded_feature_cosine_step0"] = cos_stats(ft)
    y = np.array([1 if r["margin"] > EPS_AMBIG else 0 for r in rows0 if abs(r["margin"]) > EPS_AMBIG])
    keep = np.array([abs(r["margin"]) > EPS_AMBIG for r in rows0])
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    probe = {"n_identity_ok": n_pos, "n_identity_wrong": n_neg, "unit": "episode (step-0 grounded_target_feature), leave-one-episode-out"}
    if min(n_pos, n_neg) >= 8:
        probe.update(loeo_probe(ft[keep], y, np.arange(len(y)), tk[keep]))
        probe["verdict"] = "valid only if loeo_acc clearly exceeds task_majority_baseline_acc"
    else:
        probe["verdict"] = "INSUFFICIENT DATA (<8 episodes in minority class) -- no probe run"
    summary["[5]_feature_separability_leakage_free"] = probe
    # within-task cosine: same-label vs diff-label (descriptive, no classifier)
    wt = {}
    for t in tasks:
        idx = [i for i in range(len(eps)) if tk[i] == t and abs(rows0[i]["margin"]) > EPS_AMBIG]
        lab = np.array([rows0[i]["margin"] > 0 for i in idx])
        if len(idx) > 3 and 0 < lab.sum() < len(lab):
            S = nrm(ft[idx]) @ nrm(ft[idx]).T
            same = [S[a, b] for a in range(len(idx)) for b in range(a) if lab[a] == lab[b]]
            diff = [S[a, b] for a in range(len(idx)) for b in range(a) if lab[a] != lab[b]]
            wt[t] = {"n_ok": int(lab.sum()), "n_wrong": int((~lab).sum()), "cos_same_label": float(np.mean(same)), "cos_diff_label": float(np.mean(diff))}
    summary["[5]_within_task_feature_cosine_by_identity_label"] = wt

    make_figures(eps, rows0, summary["[2]_attention_mass_by_class_step0"])
    (ROOT / "g05_summary.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "g05_step0_rows.json").write_text(json.dumps(rows0, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
