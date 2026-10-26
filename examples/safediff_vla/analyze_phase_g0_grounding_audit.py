#!/usr/bin/env python
"""Phase G0 analysis: nearest-object identity rates (section 3), attention-entropy comparison
(section 5), feature-separability probe (section 4, diagnostic-only logistic regression + PCA),
and the attention-heatmap-overlay figure (section 2) for the 4 representative episodes captured by
`phase_g0_grounding_audit_rollout.py`.

Usage:
    uv run python examples/safediff_vla/analyze_phase_g0_grounding_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_g0_grounding_audit")
TASKS = ["select_poker", "select_book", "select_fruit", "select_toy"]
GRID_SIDE = 8


def load(task: str) -> dict:
    return json.loads((ROOT / task / "steps.json").read_text())


def section3_identity_rates() -> dict:
    out = {}
    for task in TASKS:
        d = load(task)
        steps = d["steps"]
        nc = [s.get("nearest_candidate") for s in steps if s.get("nearest_candidate") is not None]
        is_instr = [n["is_instructed"] for n in nc if n.get("is_instructed") is not None]
        dist_correct = [n["nearest_dist_m"] for n in nc if n.get("is_instructed")]
        dist_wrong = [n["nearest_dist_m"] for n in nc if n.get("is_instructed") is False]
        out[task] = {
            "success": d["success"], "n_steps": len(steps),
            "correct_identity_rate": float(np.mean(is_instr)) if is_instr else None,
            "mean_dist_when_identity_correct_m": float(np.mean(dist_correct)) if dist_correct else None,
            "mean_dist_when_identity_wrong_m": float(np.mean(dist_wrong)) if dist_wrong else None,
            "n_correct": len(dist_correct), "n_wrong": len(dist_wrong),
        }
    return out


def section5_entropy() -> dict:
    out = {}
    for task in TASKS:
        d = load(task)
        steps = d["steps"]
        ent = [s["image_attn_entropy"] for s in steps if s.get("image_attn_entropy") is not None]
        mx = [s["image_attn_max"] for s in steps if s.get("image_attn_max") is not None]
        text_ent = [s["text_attn_entropy"] for s in steps if s.get("text_attn_entropy") is not None]
        # split by identity-correctness to check whether entropy discriminates
        nc = [s.get("nearest_candidate") for s in steps]
        ent_correct = [s["image_attn_entropy"] for s, n in zip(steps, nc) if n and n.get("is_instructed") is True]
        ent_wrong = [s["image_attn_entropy"] for s, n in zip(steps, nc) if n and n.get("is_instructed") is False]
        out[task] = {
            "image_attn_entropy_mean": float(np.mean(ent)), "image_attn_max_mean": float(np.mean(mx)),
            "text_attn_entropy_mean": float(np.mean(text_ent)),
            "image_attn_entropy_when_identity_correct": float(np.mean(ent_correct)) if ent_correct else None,
            "image_attn_entropy_when_identity_wrong": float(np.mean(ent_wrong)) if ent_wrong else None,
        }
    return out


def section4_feature_separability() -> dict:
    feats, labels, task_labels = [], [], []
    for task in TASKS:
        d = load(task)
        for s in d["steps"]:
            nc = s.get("nearest_candidate")
            if nc is None or nc.get("is_instructed") is None:
                continue
            feats.append(s["grounded_target_feature"])
            labels.append(1 if nc["is_instructed"] else 0)
            task_labels.append(task)
    X = np.array(feats)
    y = np.array(labels)
    result = {"n_samples": len(y), "n_correct": int(y.sum()), "n_wrong": int((1 - y).sum())}

    if len(set(y)) > 1 and len(y) > 20:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        scores = cross_val_score(clf, X, y, cv=5)
        result["probe_5fold_cv_accuracy_mean"] = float(scores.mean())
        result["probe_5fold_cv_accuracy_std"] = float(scores.std())
        result["probe_baseline_majority_class_rate"] = float(max(y.mean(), 1 - y.mean()))

    pca = PCA(n_components=2)
    xy = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"select_poker": "tab:green", "select_book": "tab:red", "select_fruit": "tab:blue", "select_toy": "tab:orange"}
    markers = {1: "o", 0: "x"}
    for task in TASKS:
        for lab in (1, 0):
            mask = [(t == task and l == lab) for t, l in zip(task_labels, labels)]
            if not any(mask):
                continue
            pts = xy[np.array(mask)]
            ax.scatter(pts[:, 0], pts[:, 1], c=colors[task], marker=markers[lab], s=18, alpha=0.6,
                       label=f"{task} ({'correct' if lab else 'wrong'} identity)")
    ax.set_title("grounded_target_feature PCA (o=identity correct, x=identity wrong)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(ROOT / "feature_pca.png", dpi=110)
    plt.close(fig)
    result["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    return result


def plot_attention_heatmap_figure() -> None:
    """One figure, 4 rows (episodes) x (3 cameras + info panel). Picks, per episode, the saved
    image snapshot step closest to the episode's midpoint (a representative, non-cherry-picked
    choice rule, not hand-tuned per episode)."""
    fig, axes = plt.subplots(len(TASKS), 4, figsize=(16, 4 * len(TASKS)))
    for row, task in enumerate(TASKS):
        d = load(task)
        steps = d["steps"]
        images_dir = ROOT / task / "images"
        # Image filenames are indexed by `len(capture["log"])-1` at save time, which (since the
        # env's reset-time record occupies `steps[0]` with no policy data yet) corresponds to
        # `steps[log_index + 1]`, NOT `steps[log_index]` -- and `steps` itself was truncated to
        # `MAX_STEPS` while image capture was not, so only snapshot steps within that truncated
        # range have a matching `steps[]` entry at all.
        all_snapshot_steps = sorted({int(p.name.split("_")[0][4:]) for p in images_dir.glob("step*_camera1.png")})
        snapshot_steps = [t for t in all_snapshot_steps if t + 1 < len(steps)]
        target_step = snapshot_steps[len(snapshot_steps) // 2] if snapshot_steps else None
        if target_step is None:
            continue
        attn = np.load(images_dir / f"step{target_step:04d}_image_attn.npy")  # [192]
        s = steps[target_step + 1]
        instr_name = s.get("target_entity")
        nc = s.get("nearest_candidate") or {}

        for col, cam in enumerate(["camera1", "camera2", "camera3"]):
            ax = axes[row, col]
            img_path = images_dir / f"step{target_step:04d}_{cam}.png"
            if not img_path.exists():
                ax.axis("off")
                continue
            img = plt.imread(img_path)
            ax.imshow(img)
            cam_idx = {"camera1": 0, "camera2": 1, "camera3": 2}[cam]
            cam_attn = attn[cam_idx * 64 : (cam_idx + 1) * 64].reshape(GRID_SIDE, GRID_SIDE)
            ax.imshow(cam_attn, cmap="jet", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0), interpolation="bilinear")
            ax.set_title(f"{task} step{target_step} {cam}", fontsize=8)
            ax.axis("off")

        ax = axes[row, 3]
        ax.axis("off")
        info = (
            f"task={task}\nsuccess={d['success']}\nstep={target_step}/{len(steps)}\n"
            f"instructed={instr_name}\nnearest_pred={nc.get('nearest_name')}\n"
            f"is_instructed={nc.get('is_instructed')}\nnearest_dist={nc.get('nearest_dist_m')}\n"
            f"predicted_dist_to_GT={s.get('predicted_target_dist_to_gt')}\n"
            f"img_attn_entropy={s.get('image_attn_entropy'):.3f}" if s.get("image_attn_entropy") is not None else ""
        )
        ax.text(0.02, 0.98, info, va="top", ha="left", fontsize=9, transform=ax.transAxes, family="monospace")

    fig.tight_layout()
    fig.savefig(ROOT / "attention_heatmap_figure.png", dpi=110)
    plt.close(fig)


def main() -> None:
    s3 = section3_identity_rates()
    s5 = section5_entropy()
    s4 = section4_feature_separability()
    plot_attention_heatmap_figure()

    print("=== Section 3: nearest-object identity rates ===")
    print(json.dumps(s3, indent=2))
    print("\n=== Section 5: attention entropy comparison ===")
    print(json.dumps(s5, indent=2))
    print("\n=== Section 4: feature-separability probe ===")
    print(json.dumps(s4, indent=2))

    (ROOT / "g0_analysis_summary.json").write_text(
        json.dumps({"identity_rates": s3, "attention_entropy": s5, "feature_separability": s4}, indent=2)
    )
    print(f"\nwrote {ROOT / 'g0_analysis_summary.json'}, {ROOT / 'feature_pca.png'}, {ROOT / 'attention_heatmap_figure.png'}")


if __name__ == "__main__":
    main()
