#!/usr/bin/env python
"""PHASE G0.6 analysis. Offline; reads `phase_g06_counterfactual_instruction.py` outputs.

A-priori rules (fixed BEFORE looking at results; not tuned):
  * pair excluded from per-pair ratios if ||dGT_xy|| < 1 cm (division guard only; pooled follow
    uses ratio-of-sums and needs no exclusion)
  * stage "invariant" if S_stage < 0.10, where S_stage = (mean relative change of that stage under an
    instruction swap, scene fixed) / (mean relative change of that stage between DIFFERENT scenes of
    the same task with their own correct instruction)  -- the denominator is the stage's natural
    variation scale; repeat-control gives the noise floor
  * "instruction conditioning present" (L4) if pooled R_follow_xy >= 0.5
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval/safediff_vla_grounded_grasp_v2_phase_g06_counterfactual_instruction")
G05 = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_g05_clean_grounding")
INV_S, FOLLOW_OK, GUARD = 0.10, 0.5, 0.01
TASK_ORDER = ["select_book", "select_poker", "select_mahjong", "select_drink", "select_fruit", "select_toy"]
STAGES = ["text_latent_mean", "lang_q", "feat", "pred_xy"]
rng = np.random.default_rng(0)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def rel(a, b):
    return float(np.linalg.norm(a - b) / (0.5 * (np.linalg.norm(a) + np.linalg.norm(b)) + 1e-12))


def load():
    sc = json.loads((ROOT / "g06_raw.json").read_text())
    lat = np.load(ROOT / "g06_image_latents.npz")
    for s in sc:
        base = np.array(s["base"])
        s["gt"] = {n: np.array(v) for n, v in s["gt_world"].items()}
        for k, c in s["conds"].items():
            for f in ("pred_robot", "lang_q", "feat", "text_latent_mean", "image_latent_mean", "img_attn", "text_attn"):
                c[f] = np.array(c[f])
            c["pred"] = c["pred_robot"] + base
        s["lat"] = {k: lat[f"{s['task']}|{s['seed']}|{k}|image_latent"].astype(np.float32) for k in s["conds"]}
    return sc


def pairs_metrics(scenes):
    rows = []
    for s in scenes:
        for i, j in combinations(s["cands"], 2):
            dg = s["gt"][j] - s["gt"][i]
            dp = s["conds"][f"cand::{j}"]["pred"] - s["conds"][f"cand::{i}"]["pred"]
            r = {"task": s["task"], "seed": s["seed"]}
            for tag, sl in (("xy", slice(0, 2)), ("xyz", slice(0, 3))):
                g, p = dg[sl], dp[sl]
                r[f"dgt_{tag}"] = float(np.linalg.norm(g))
                r[f"num_{tag}"] = float(p @ g)
                r[f"den_{tag}"] = float(g @ g)
                r[f"mag_{tag}"] = float(np.linalg.norm(p) / max(np.linalg.norm(g), 1e-9))
                r[f"dir_{tag}"] = cos(p, g)
                r[f"follow_{tag}"] = float(p @ g / max(g @ g, 1e-9))
            qi, qj = s["conds"][f"cand::{i}"], s["conds"][f"cand::{j}"]
            for f in ("text_latent_mean", "lang_q", "feat"):
                r[f"rel_{f}"] = rel(qi[f], qj[f])
                r[f"cos_{f}"] = cos(qi[f], qj[f])
            r["rel_pred_xy"] = float(np.linalg.norm(qi["pred"][:2] - qj["pred"][:2]))
            r["rel_image_latent"] = float(np.mean(np.linalg.norm(s["lat"][f"cand::{i}"] - s["lat"][f"cand::{j}"], axis=1) /
                                                  (np.linalg.norm(s["lat"][f"cand::{i}"], axis=1) + 1e-9)))
            r["cos_img_attn"] = cos(qi["img_attn"], qj["img_attn"])
            r["valid"] = r["dgt_xy"] >= GUARD
            rows.append(r)
    return rows


def boot_follow(rows, tag="xy", n=2000):
    scenes = sorted({r["seed"] for r in rows})
    by = {sd: [r for r in rows if r["seed"] == sd] for sd in scenes}
    vals = []
    for _ in range(n):
        pick = rng.choice(scenes, len(scenes))
        num = sum(r[f"num_{tag}"] for sd in pick for r in by[sd])
        den = sum(r[f"den_{tag}"] for sd in pick for r in by[sd])
        vals.append(num / den)
    return np.percentile(vals, [2.5, 97.5]).round(3).tolist()


def scene_scale(scenes, task):
    """Mean relative change of each stage between DIFFERENT scenes (own correct instruction)."""
    ss = [s for s in scenes if s["task"] == task]
    out = {f: [] for f in STAGES}
    for a, b in combinations(ss, 2):
        ca, cb = a["conds"][f"cand::{a['target']}"], b["conds"][f"cand::{b['target']}"]
        for f in ("text_latent_mean", "lang_q", "feat"):
            out[f].append(rel(ca[f], cb[f]))
        out["pred_xy"].append(float(np.linalg.norm(ca["pred"][:2] - cb["pred"][:2])))
    return {f: float(np.mean(v)) for f, v in out.items()}


def main():
    scenes = load()
    tasks = [t for t in TASK_ORDER if any(s["task"] == t for s in scenes)]
    summary = {"n_scenes": len(scenes)}

    # ---- validation of reuse + controls ----
    rep = []
    ctrl = []
    for s in scenes:
        m0 = json.loads((G05 / s["task"] / f"seed{s['seed']}" / "meta.json").read_text())["steps"][0]
        rep.append(float(np.abs(s["conds"]["task_name"]["pred"] - np.array(m0["pred_world"])).max()))
        ctrl.append(float(np.abs(s["conds"]["repeat"]["pred"] - s["conds"][f"cand::{s['cands'][0]}"]["pred"]).max()))
    summary["validation"] = {"max_abs_diff_task_name_pred_vs_G05_pred_m": max(rep),
                             "control_A_repeat_max_abs_diff_m": max(ctrl),
                             "scenes_where_fresh_reset_differs_from_G05_image(px_diff>0.5)": int(sum(max(s["px_diff_vs_g05"]) > 0.5 for s in scenes)),
                             "language_tokens_task_name_vs_instruction_median": {
                                 t: [int(np.median([s["conds"]["task_name"]["n_text_tokens"] for s in scenes if s["task"] == t])),
                                     int(np.median([s["conds"][f"cand::{s['target']}"]["n_text_tokens"] for s in scenes if s["task"] == t]))] for t in tasks}}

    rows = pairs_metrics(scenes)
    per_task = {}
    scales = {t: scene_scale(scenes, t) for t in tasks}
    for t in tasks:
        R = [r for r in rows if r["task"] == t]
        V = [r for r in R if r["valid"]]
        sc = [s for s in scenes if s["task"] == t]
        out = {"n_scenes": len(sc), "n_pairs": len(R), "n_pairs_valid(>=1cm)": len(V)}
        for tag in ("xy", "xyz"):
            out[f"R_follow_{tag}_pooled"] = float(sum(r[f"num_{tag}"] for r in R) / sum(r[f"den_{tag}"] for r in R))
            out[f"R_follow_{tag}_pooled_ci95_scene_bootstrap"] = boot_follow(R, tag)
            for k in ("follow", "mag", "dir"):
                out[f"R_{k}_{tag}_median"] = float(np.median([r[f"{k}_{tag}"] for r in V]))
        # variance across instructions vs across candidates (xy, scene-wise, then pooled)
        vp = vg = 0.0
        for s in sc:
            P = np.stack([s["conds"][f"cand::{c}"]["pred"][:2] for c in s["cands"]])
            G = np.stack([s["gt"][c][:2] for c in s["cands"]])
            vp += ((P - P.mean(0)) ** 2).sum(1).mean()
            vg += ((G - G.mean(0)) ** 2).sum(1).mean()
        out["pred_std_across_instructions_cm"] = float(np.sqrt(vp / len(sc)) * 100)
        out["GT_std_across_candidates_cm"] = float(np.sqrt(vg / len(sc)) * 100)
        out["variance_ratio_pred_over_GT"] = float(vp / vg)
        # absolute-referent test
        cnt = {"instructed_cand": 0, "other_cand": 0, "candidate_centroid": 0, "scene_pred_mean(fixed mode)": 0}
        n_inst = 0
        id_task_name = id_instr = id_noun = 0
        for s in sc:
            cents = np.mean([s["gt"][c][:2] for c in s["cands"]], 0)
            pmean = np.mean([s["conds"][f"cand::{c}"]["pred"][:2] for c in s["cands"]], 0)
            for c in s["cands"]:
                p = s["conds"][f"cand::{c}"]["pred"][:2]
                anchors = {**{("instructed_cand" if k == c else "other_cand") + f"::{k}": s["gt"][k][:2] for k in s["cands"]},
                           "candidate_centroid": cents, "scene_pred_mean(fixed mode)": pmean}
                best = min(anchors, key=lambda a: np.linalg.norm(p - anchors[a]))
                cnt[best.split("::")[0]] += 1
                n_inst += 1
            near = lambda p, s=s: min(s["cands"], key=lambda k: np.linalg.norm(p[:2] - s["gt"][k][:2]))
            id_task_name += near(s["conds"]["task_name"]["pred"]) == s["target"]
            id_instr += near(s["conds"][f"cand::{s['target']}"]["pred"]) == s["target"]
            id_noun += near(s["conds"]["noun_only"]["pred"]) == s["target"]
        out["nearest_anchor_counts_over_instructions"] = {**cnt, "n": n_inst}
        out["chance_identity_rate"] = float(np.mean([1 / len(s["cands"]) for s in sc]))
        out["target_identity_rate(nearest cand==target)"] = {"task_name(G0.5 input)": id_task_name / len(sc), "correct_instruction": id_instr / len(sc),
                                                             "noun_only(control B)": id_noun / len(sc)}
        e = lambda k: float(np.median([np.linalg.norm(s["conds"][k(s)]["pred"][:2] - s["gt"][s["target"]][:2]) for s in sc]) * 100)
        out["median_xy_err_to_target_cm"] = {"task_name": e(lambda s: "task_name"), "correct_instruction": e(lambda s: f"cand::{s['target']}"),
                                             "noun_only": e(lambda s: "noun_only")}
        # control B: noun-only pred vs candidate centroid / vs mean of instructed preds
        d_c = np.mean([np.linalg.norm(s["conds"]["noun_only"]["pred"][:2] - np.mean([s["gt"][c][:2] for c in s["cands"]], 0)) for s in sc]) * 100
        d_m = np.mean([np.linalg.norm(s["conds"]["noun_only"]["pred"][:2] - np.mean([s["conds"][f"cand::{c}"]["pred"][:2] for c in s["cands"]], 0)) for s in sc]) * 100
        out["controlB_noun_only_dist_cm"] = {"to_candidate_centroid": float(d_c), "to_mean_of_instructed_preds": float(d_m)}
        # stage sensitivities
        st = {}
        for f, key in (("text_latent_mean", "rel_text_latent_mean"), ("lang_q", "rel_lang_q"), ("feat", "rel_feat"), ("pred_xy", "rel_pred_xy")):
            swap = float(np.mean([r[key] for r in R]))
            st[f] = {"instr_swap_delta": swap, "scene_scale": scales[t][f], "S": swap / scales[t][f]}
        st["image_latent_rel_delta_swap"] = float(np.mean([r["rel_image_latent"] for r in R]))
        st["cos_lang_q_mean"] = float(np.mean([r["cos_lang_q"] for r in R]))
        st["cos_feat_mean"] = float(np.mean([r["cos_feat"] for r in R]))
        st["cos_img_attn_mean"] = float(np.mean([r["cos_img_attn"] for r in R]))
        out["stage_sensitivity"] = st
        # task_name -> correct instruction change (what the missing instruction costs, per stage)
        out["task_name_vs_instruction_rel_change"] = {
            "lang_q": float(np.mean([rel(s["conds"]["task_name"]["lang_q"], s["conds"][f"cand::{s['target']}"]["lang_q"]) for s in sc])),
            "feat": float(np.mean([rel(s["conds"]["task_name"]["feat"], s["conds"][f"cand::{s['target']}"]["feat"]) for s in sc])),
            "pred_xy_cm": float(np.mean([np.linalg.norm(s["conds"]["task_name"]["pred"][:2] - s["conds"][f"cand::{s['target']}"]["pred"][:2]) for s in sc]) * 100)}
        # case label (a-priori rule)
        S = {f: st[f]["S"] for f in STAGES}
        if out["R_follow_xy_pooled"] >= FOLLOW_OK:
            case = "L4 (xyz follows referent)"
        elif S["lang_q"] < INV_S:
            case = "L1 (query invariant)"
        elif S["feat"] < INV_S:
            case = "L2 (feature invariant)"
        else:
            case = "L3 (feature moves, xyz does not follow)"
        out["case_a_priori_rule"] = case
        per_task[t] = out
    summary["per_task"] = per_task

    # ---- G0.5 consistency of ordering ----
    g05 = {"select_book": -0.01, "select_poker": 0.27, "select_mahjong": 0.40, "select_drink": 0.46, "select_fruit": 0.67, "select_toy": 0.75}
    from scipy.stats import spearmanr  # noqa: PLC0415

    xs = [g05[t] for t in tasks]; ys = [per_task[t]["R_follow_xy_pooled"] for t in tasks]
    summary["spearman_G05_follow_vs_G06_R_follow"] = float(spearmanr(xs, ys).correlation)

    # ---- class counts under task-name vs correct instruction (A/B/C/D, 2cm/5cm) ----
    cls = {"task_name": {c: 0 for c in "ABCD"}, "correct_instruction": {c: 0 for c in "ABCD"}}
    for s in scenes:
        for key, ck in (("task_name", "task_name"), ("correct_instruction", f"cand::{s['target']}")):
            p = s["conds"][ck]["pred"]
            d_t = np.linalg.norm(p - s["gt"][s["target"]])
            d_o = min(np.linalg.norm(p - s["gt"][c]) for c in s["cands"] if c != s["target"])
            m = d_o - d_t
            exy = np.linalg.norm((p - s["gt"][s["target"]])[:2])
            c = "D" if abs(m) <= 0.02 else ("A" if m < 0 else ("B" if exy > 0.05 else "C"))
            cls[key][c] += 1
    summary["ABCD_task_name_vs_correct_instruction"] = cls

    (ROOT / "g06_summary.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "g06_pair_rows.json").write_text(json.dumps(rows))

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    cols = dict(zip(tasks, plt.cm.tab10.colors))
    for t in tasks:
        R = [r for r in rows if r["task"] == t]
        ax[0].scatter([r["dgt_xy"] * 100 for r in R], [r["num_xy"] / max(r["dgt_xy"], 1e-9) * 100 for r in R], color=cols[t], s=14, alpha=0.7, label=t)
    lim = max(r["dgt_xy"] for r in rows) * 100
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.8); ax[0].axhline(0, color="gray", lw=0.6)
    ax[0].set_xlabel("|dGT_xy| (cm)"); ax[0].set_ylabel("dPred_xy projected on dGT direction (cm)"); ax[0].legend(fontsize=7); ax[0].set_title("per-pair instruction-swap response (y=x: follows)")
    xs_ = np.arange(len(tasks))
    ax[1].bar(xs_, [per_task[t]["R_follow_xy_pooled"] for t in tasks],
              yerr=[[per_task[t]["R_follow_xy_pooled"] - per_task[t]["R_follow_xy_pooled_ci95_scene_bootstrap"][0] for t in tasks],
                    [per_task[t]["R_follow_xy_pooled_ci95_scene_bootstrap"][1] - per_task[t]["R_follow_xy_pooled"] for t in tasks]], color=[cols[t] for t in tasks], capsize=3)
    ax[1].axhline(1, color="k", ls="--", lw=0.8); ax[1].axhline(0, color="gray", lw=0.6)
    ax[1].set_xticks(xs_); ax[1].set_xticklabels([t.replace("select_", "") for t in tasks]); ax[1].set_title("R_follow (xy, pooled; 95% CI over scenes)")
    for t in tasks:
        ax[2].plot(["text latent", "lang query", "grounded feat", "xyz"], [per_task[t]["stage_sensitivity"][f]["S"] for f in STAGES], "o-", color=cols[t], label=t)
    ax[2].axhline(INV_S, color="k", ls=":", lw=0.8); ax[2].set_yscale("log"); ax[2].set_ylabel("S = instr-swap change / natural scene change")
    ax[2].set_title("stage-wise instruction sensitivity (dotted = a-priori invariance cutoff)"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT / "g06_summary.png", dpi=100); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
