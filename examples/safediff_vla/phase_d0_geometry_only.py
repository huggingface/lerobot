#!/usr/bin/env python
"""PHASE D0 step E analysis: language-free CANDIDATE SELECTION from scene geometry only (sim-sampled scenes, seeds 5000+).
Candidate set = same-type objects; the instructed target is the label. Features use ONLY positions in the robot
frame (candidate xy relative to robot base / to the candidate centroid) + the dataset's task-mean anchor (a fixed spatial
mode). No names, no instruction, no image. Fixed rule set + one tiny conditional-logit; scene-level train/test.
Selection protocol (no leakage): the best fixed rule is chosen on the first half of scenes and scored on the second half;
the conditional-logit is 5-fold cross-validated by scene. Chance = mean(1/K).
Also: sim target-position distribution vs the dataset's anchor distribution (is the sim scene generator the dataset's?)
and referent coverage of the sim candidates in the training instruction set."""
from __future__ import annotations

import glob, json
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

OUT = Path("outputs/eval/safediff_vla_phase_d0_training_data_audit")
rng = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def feats(P, base, mode):
    """P [K,3] world -> per-candidate geometry features (robot-frame xy)."""
    xy = P[:, :2] - base[:2]; c = xy.mean(0)
    r = np.linalg.norm(xy, axis=1); rc = np.linalg.norm(xy - c, axis=1)
    rk = lambda v: (np.argsort(np.argsort(v)) / max(len(v) - 1, 1))
    return np.stack([xy[:, 0], xy[:, 1], r, xy[:, 0] - c[0], xy[:, 1] - c[1], rc, rk(xy[:, 0]), rk(xy[:, 1]), rk(r), np.linalg.norm(xy - mode[:2], axis=1)], 1), xy, r, rc


RULES = {"nearest_to_base": lambda xy, r, rc, md: np.argmin(r), "farthest_from_base": lambda xy, r, rc, md: np.argmax(r),
         "min_x": lambda xy, r, rc, md: np.argmin(xy[:, 0]), "max_x": lambda xy, r, rc, md: np.argmax(xy[:, 0]),
         "min_y": lambda xy, r, rc, md: np.argmin(xy[:, 1]), "max_y": lambda xy, r, rc, md: np.argmax(xy[:, 1]),
         "nearest_centroid": lambda xy, r, rc, md: np.argmin(rc), "farthest_centroid": lambda xy, r, rc, md: np.argmax(rc),
         "nearest_dataset_task_mean_anchor": lambda xy, r, rc, md: np.argmin(md)}


def main():
    eps = pd.read_json(OUT / "episodes.json"); eps = eps[eps.has_anchor]
    train_instr = set(eps.instruction); train_ref = {t: set(g.referent) for t, g in eps.groupby("task")}
    res = {}
    for task in ["select_book", "select_poker", "select_mahjong", "select_drink", "select_fruit", "select_toy"]:
        sc = json.load(open(OUT / f"sim_scenes_{task}.json")); sc = [s for s in sc if len(s["cands"]) >= 2 and s["target"] in s["cands"]]
        mode = np.stack(eps[eps.task == task].anchor.to_numpy()).mean(0)
        data = []
        for s in sc:
            names = list(s["cands"]); P = np.array([s["cands"][n] for n in names]); base = np.array(s["base"])
            F, xy, r, rc = feats(P, base, mode); md = np.linalg.norm(xy - mode[:2], axis=1)
            data.append((F, xy, r, rc, md, names.index(s["target"]), len(names), s))
        n = len(data); half = n // 2; chance = float(np.mean([1 / d[6] for d in data]))
        acc_rule = {nm: float(np.mean([f(d[1], d[2], d[3], d[4]) == d[5] for d in data])) for nm, f in RULES.items()}
        best = max(RULES, key=lambda nm: np.mean([RULES[nm](d[1], d[2], d[3], d[4]) == d[5] for d in data[:half]]))
        held = float(np.mean([RULES[best](d[1], d[2], d[3], d[4]) == d[5] for d in data[half:]]))
        # conditional logit (binary per-candidate logistic, argmax), 5-fold CV by scene
        idx = rng.permutation(n); hits = 0
        for te in [idx[i::5] for i in range(5)]:
            tr = np.setdiff1d(idx, te)
            X = np.concatenate([data[i][0] for i in tr]); y = np.concatenate([(np.arange(data[i][6]) == data[i][5]).astype(int) for i in tr])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit((X - X.mean(0)) / (X.std(0) + 1e-9), y)
            mu, sd = X.mean(0), X.std(0) + 1e-9
            hits += sum(int(np.argmax(clf.decision_function((data[i][0] - mu) / sd)) == data[i][5]) for i in te)
        tgt_robot = np.array([np.array(d[7]["cands"][d[7]["target"]]) - np.array(d[7]["base"]) for d in data])
        anc = np.stack(eps[eps.task == task].anchor.to_numpy())
        ref_seen = [s["instruction"].replace("primitive: ", "") for s in sc]
        res[task] = {"n_scenes": n, "mean_K": float(np.mean([d[6] for d in data])), "chance": chance,
                     "rule_accuracy_all_scenes": {k: round(v, 3) for k, v in acc_rule.items()},
                     "best_rule_selected_on_first_half": best, "best_rule_heldout_acc": held, "best_rule_heldout_ci95": wilson(int(round(held * (n - half))), n - half),
                     "conditional_logit_cv_acc": hits / n, "conditional_logit_ci95": wilson(hits, n),
                     "sim_target_xy_mean_std_cm": [(tgt_robot[:, :2].mean(0) * 100).round(1).tolist(), (tgt_robot[:, :2].std(0) * 100).round(1).tolist()],
                     "dataset_anchor_xy_mean_std_cm": [(anc[:, :2].mean(0) * 100).round(1).tolist(), (anc[:, :2].std(0) * 100).round(1).tolist()],
                     "frac_sim_instructions_exactly_in_training_set": float(np.mean([s["instruction"] in train_instr for s in sc])),
                     "frac_sim_targets_with_seen_referent": float(np.mean([any(s["target"].replace("_", " ") in r or s["target"] in r for r in train_ref[task]) for s in sc]))}
        print(f"{task:14s} n={n} K={res[task]['mean_K']:.1f} chance={chance:.2f} | best rule {best} held-out {held:.2f} {res[task]['best_rule_heldout_ci95']} | logit-CV {hits / n:.2f} {res[task]['conditional_logit_ci95']}")
        print("     all-scene rule acc:", {k: round(v, 2) for k, v in acc_rule.items()})
        print("     sim target xy mean/std cm:", res[task]["sim_target_xy_mean_std_cm"], "| dataset anchor:", res[task]["dataset_anchor_xy_mean_std_cm"], "| sim instr in train:", round(res[task]["frac_sim_instructions_exactly_in_training_set"], 2))
    (OUT / "geometry_only.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
