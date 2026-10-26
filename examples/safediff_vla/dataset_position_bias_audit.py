#!/usr/bin/env python
"""Dataset-level audit: does select_poker's target card identity correlate with a fixed physical
slot, and is any one slot (e.g. a "second from the right"-type position) over-represented as the
demonstrated target across episodes? Pure parquet/JSON audit, no simulator, no policy.

Reuses first_close_action_xyz per episode (already computed and saved by
grounded_grasp_supervision_audit.py in grasp_target_label_stats.json) as the per-episode target
position proxy, joined against each episode's own task-string-encoded card identity.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans

REPO_ID = "lerobot/vlabench_unified"
OUT_DIR = Path("/home/outputs/eval/safediff_vla_grounded_grasp_position_bias_audit")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    label_stats = json.loads(
        Path("/home/outputs/eval/safediff_vla_grounded_grasp_supervision_audit/grasp_target_label_stats.json").read_text()
    )
    episodes = {e["episode_index"]: e for e in label_stats["episodes"] if e["first_close_action_xyz"] is not None}

    tasks_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="meta/tasks.parquet")
    tasks_df = pd.read_parquet(tasks_path).reset_index()
    task_to_identity = {}
    for _, row in tasks_df.iterrows():
        t = row["task"]
        if "poker" not in t.lower():
            continue
        # "primitive: Please pick the poker 7 of spades" -> "7_of_spades"
        tail = t.split("poker")[-1].strip()
        identity = tail.replace(" of ", "_of_").replace(" ", "_")
        task_to_identity[row["task_index"]] = identity

    episodes_meta_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename="meta/episodes/chunk-000/file-000.parquet"
    )
    episodes_meta = pd.read_parquet(episodes_meta_path, columns=["episode_index", "tasks"])

    rows = []
    for _, row in episodes_meta.iterrows():
        ep_idx = int(row["episode_index"])
        if ep_idx not in episodes:
            continue
        poker_tasks = [t for t in row["tasks"] if "poker" in t.lower()]
        if not poker_tasks:
            continue
        t = poker_tasks[0]
        tail = t.split("poker")[-1].strip()
        identity = tail.replace(" of ", "_of_").replace(" ", "_")
        xyz = episodes[ep_idx]["first_close_action_xyz"]
        rows.append({"episode_index": ep_idx, "identity": identity, "x": xyz[0], "y": xyz[1], "z": xyz[2]})

    df = pd.DataFrame(rows)
    print(f"joined {len(df)} episodes with identity + xyz")

    # --- 1. episode count per card identity ---
    identity_counts = df["identity"].value_counts().to_dict()

    # --- 2. x/y distribution ---
    xy = df[["x", "y"]].to_numpy()
    xy_stats = {
        "x": {"mean": float(df["x"].mean()), "std": float(df["x"].std()), "min": float(df["x"].min()), "max": float(df["x"].max())},
        "y": {"mean": float(df["y"].mean()), "std": float(df["y"].std()), "min": float(df["y"].min()), "max": float(df["y"].max())},
    }

    # --- 3. slot clustering: how many distinct physical "slots" does first_close_xyz cluster into? ---
    cluster_results = {}
    for k in (3, 4, 5, 6, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(xy)
        counts = Counter(km.labels_.tolist())
        cluster_results[f"k={k}"] = {
            "cluster_sizes": sorted(counts.values(), reverse=True),
            "cluster_centers_xy": km.cluster_centers_.tolist(),
            "inertia": float(km.inertia_),
            "largest_cluster_frac": max(counts.values()) / len(df),
        }

    # --- 4. is a specific identity biased toward a specific slot? ---
    # For identities with >=3 episodes, how tight is their own xy spread vs the dataset-wide spread?
    identity_spread = {}
    for identity, group in df.groupby("identity"):
        if len(group) < 3:
            continue
        identity_spread[identity] = {
            "n_episodes": len(group),
            "x_std": float(group["x"].std()),
            "y_std": float(group["y"].std()),
            "x_mean": float(group["x"].mean()),
            "y_mean": float(group["y"].mean()),
        }
    dataset_wide_x_std = float(df["x"].std())
    dataset_wide_y_std = float(df["y"].std())

    # --- 5. "second from the right" check: bin x into rank-order slots per some fixed grid, see
    # if one x-bin dominates. Since we don't have per-episode DISTRACTOR positions (dataset stores
    # no object pose at all, target only, via first_close_xyz), we cannot compute a true per-scene
    # "rank from the right among visible cards" -- only the marginal distribution of the target's
    # own x position across episodes. Report that distribution's concentration directly.
    n_bins = 6
    x_bin_edges = np.linspace(df["x"].min(), df["x"].max(), n_bins + 1)
    x_bin_idx = np.digitize(df["x"], x_bin_edges[1:-1])
    x_bin_counts = Counter(x_bin_idx.tolist())
    x_bin_report = {
        f"bin_{i}_[{x_bin_edges[i]:.3f},{x_bin_edges[i+1]:.3f})": x_bin_counts.get(i, 0) for i in range(n_bins)
    }

    result = {
        "n_episodes": len(df),
        "n_distinct_target_identities": df["identity"].nunique(),
        "episode_count_per_identity": identity_counts,
        "identity_count_stats": {
            "min": min(identity_counts.values()),
            "max": max(identity_counts.values()),
            "mean": float(np.mean(list(identity_counts.values()))),
            "std": float(np.std(list(identity_counts.values()))),
        },
        "first_close_xy_distribution": xy_stats,
        "first_close_x_marginal_histogram": x_bin_report,
        "slot_clustering_kmeans": cluster_results,
        "identity_specific_spatial_spread": {
            "dataset_wide_x_std": dataset_wide_x_std,
            "dataset_wide_y_std": dataset_wide_y_std,
            "per_identity_with_ge3_episodes": identity_spread,
            "note": (
                "If a given card identity's own x_std/y_std across its episodes is much SMALLER "
                "than the dataset-wide std, that identity is disproportionately placed in the same "
                "physical slot across episodes -- a genuine identity<->slot confound in the data "
                "itself, separate from anything about the policy."
            ),
        },
        "caveat": (
            "Distractor card positions are not recoverable from this dataset (no object-pose "
            "field at all -- see grasp_target_label_stats.json's target_card_xyz_availability "
            "note), so a true per-scene 'rank from the right among all visible cards' cannot be "
            "computed here; only the TARGET's own marginal x/y distribution and clustering across "
            "episodes is available, which is what's reported above."
        ),
    }
    (OUT_DIR / "position_bias_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("episode_count_per_identity",)}, indent=2))
    print("wrote", OUT_DIR / "position_bias_audit.json")


if __name__ == "__main__":
    main()
