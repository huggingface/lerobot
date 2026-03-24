"""
Visualize end-effector workspace density and trajectory clusters for OpenArm datasets.
Downloads joint position data (no videos) from HuggingFace, computes forward
kinematics per episode, clusters trajectories with K-means, and renders
2D projections comparing dataset coverage and multimodality.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.cluster import KMeans

DATASETS = [
    {"repo_id": "lerobot-data-collection/level2_final_quality3", "label": "HQ curated"},
    {"repo_id": "lerobot-data-collection/level12_rac_2_2026-02-08_1", "label": "Full collection"},
]
OUTPUT_DIR = Path("/Users/pepijnkooijmans/Documents/GitHub_local/progress_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

N_CLUSTERS = 10
WAYPOINTS = 50
SEED = 42
DPI = 180

CLUSTER_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#dcbeff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
]

# FK chains extracted from OpenArm bimanual URDF.
# Each entry: (rpy, xyz, revolute_axis_or_None).
LEFT_CHAIN = [
    ((-np.pi / 2, 0, 0), (0, 0.031, 0.698), None),
    ((0, 0, 0), (0, 0, 0.0625), (0, 0, 1)),
    ((-np.pi / 2, 0, 0), (-0.0301, 0, 0.06), (-1, 0, 0)),
    ((0, 0, 0), (0.0301, 0, 0.06625), (0, 0, 1)),
    ((0, 0, 0), (0, 0.0315, 0.15375), (0, 1, 0)),
    ((0, 0, 0), (0, -0.0315, 0.0955), (0, 0, 1)),
    ((0, 0, 0), (0.0375, 0, 0.1205), (1, 0, 0)),
    ((0, 0, 0), (-0.0375, 0, 0), (0, -1, 0)),
    ((0, 0, 0), (0, 0, 0.1001), None),
    ((0, 0, 0), (0, 0, 0.08), None),
]
RIGHT_CHAIN = [
    ((np.pi / 2, 0, 0), (0, -0.031, 0.698), None),
    ((0, 0, 0), (0, 0, 0.0625), (0, 0, 1)),
    ((np.pi / 2, 0, 0), (-0.0301, 0, 0.06), (-1, 0, 0)),
    ((0, 0, 0), (0.0301, 0, 0.06625), (0, 0, 1)),
    ((0, 0, 0), (0, 0.0315, 0.15375), (0, 1, 0)),
    ((0, 0, 0), (0, -0.0315, 0.0955), (0, 0, 1)),
    ((0, 0, 0), (0.0375, 0, 0.1205), (1, 0, 0)),
    ((0, 0, 0), (-0.0375, 0, 0), (0, 1, 0)),
    ((0, 0, 0), (0, 0, 0.1001), None),
    ((0, 0, 0), (0, 0, 0.08), None),
]


# ── FK math ─────────────────────────────────────────────


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _tf(rpy: tuple, xyz: tuple) -> np.ndarray:
    """Build a 4x4 homogeneous transform from URDF rpy + xyz."""
    r, p, y = rpy
    mat = np.eye(4)
    mat[:3, :3] = _rot_z(y) @ _rot_y(p) @ _rot_x(r)
    mat[:3, 3] = xyz
    return mat


def _batch_axis_rot(axis: tuple, angles: np.ndarray) -> np.ndarray:
    """Batched Rodrigues rotation: (n,) angles around a fixed axis → (n, 4, 4)."""
    n = len(angles)
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    x, y, z = ax
    c = np.cos(angles)
    s = np.sin(angles)
    t = 1 - c
    rot = np.zeros((n, 4, 4))
    rot[:, 0, 0] = t * x * x + c
    rot[:, 0, 1] = t * x * y - s * z
    rot[:, 0, 2] = t * x * z + s * y
    rot[:, 1, 0] = t * x * y + s * z
    rot[:, 1, 1] = t * y * y + c
    rot[:, 1, 2] = t * y * z - s * x
    rot[:, 2, 0] = t * x * z - s * y
    rot[:, 2, 1] = t * y * z + s * x
    rot[:, 2, 2] = t * z * z + c
    rot[:, 3, 3] = 1.0
    return rot


def batch_fk(chain: list, joint_angles: np.ndarray) -> np.ndarray:
    """Vectorized FK: (n, 7) radians → (n, 3) TCP positions in world frame."""
    n = joint_angles.shape[0]
    tf_batch = np.tile(np.eye(4), (n, 1, 1))
    qi = 0
    for rpy, xyz, axis in chain:
        tf_batch = tf_batch @ _tf(rpy, xyz)
        if axis is not None:
            rot = _batch_axis_rot(axis, joint_angles[:, qi])
            tf_batch = np.einsum("nij,njk->nik", tf_batch, rot)
            qi += 1
    return tf_batch[:, :3, 3]


# ── Data loading ────────────────────────────────────────


def _flatten_names(obj: object) -> list[str]:
    """Recursively flatten a names structure (list, dict, or nested) into a flat string list."""
    if isinstance(obj, dict):
        out: list[str] = []
        for v in obj.values():
            out.extend(_flatten_names(v))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            if isinstance(item, (list, tuple, dict)):
                out.extend(_flatten_names(item))
            else:
                out.append(str(item))
        return out
    return [str(obj)]


def _detect_and_convert(vals: np.ndarray) -> np.ndarray:
    """Auto-detect servo ticks / degrees / radians and convert to radians."""
    mx = np.max(np.abs(vals))
    if mx > 360:
        print(f"    Unit detection: servo ticks (max={mx:.0f})")
        return (vals - 2048) / 2048 * np.pi
    if mx > 6.3:
        print(f"    Unit detection: degrees (max={mx:.1f})")
        return np.deg2rad(vals)
    print(f"    Unit detection: radians (max={mx:.3f})")
    return vals.astype(np.float64)


def _find_joint_indices(features: dict, state_col: str, n_dim: int) -> tuple[list[int], list[int]]:
    """Try to find left/right joint indices from info.json feature names."""
    feat = features.get("observation.state", features.get(state_col, {}))
    names = _flatten_names(feat.get("names", []))

    left_idx: list[int] = []
    right_idx: list[int] = []
    if names and len(names) == n_dim:
        names_l = [n.lower() for n in names]
        print(f"  Feature names: {names[:4]}…{names[-4:]}")
        for j in range(1, 8):
            for i, nm in enumerate(names_l):
                if f"left_joint_{j}" in nm and i not in left_idx:
                    left_idx.append(i)
                    break
            for i, nm in enumerate(names_l):
                if f"right_joint_{j}" in nm and i not in right_idx:
                    right_idx.append(i)
                    break

    if len(left_idx) == 7 and len(right_idx) == 7:
        print(f"  Matched by name: left={left_idx} right={right_idx}")
        return left_idx, right_idx
    if n_dim >= 16:
        print("  Falling back to positional: [0:7]=left, [8:15]=right")
        return list(range(7)), list(range(8, 15))
    if n_dim >= 14:
        print("  Falling back to positional: [0:7]=left, [7:14]=right")
        return list(range(7)), list(range(7, 14))
    raise RuntimeError(f"State dim {n_dim} too small for bimanual 7-DOF robot")


def download_data(repo_id: str) -> Path:
    print(f"  Downloading {repo_id} (parquet only) …")
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["meta/**", "data/**"],
            ignore_patterns=["*.mp4", "videos/**"],
        )
    )


def resample_trajectory(traj: np.ndarray, n_waypoints: int) -> np.ndarray:
    """Resample a (F, 3) trajectory to exactly n_waypoints via linear interpolation."""
    f = traj.shape[0]
    if f == n_waypoints:
        return traj
    old_t = np.linspace(0, 1, f)
    new_t = np.linspace(0, 1, n_waypoints)
    return np.column_stack([np.interp(new_t, old_t, traj[:, d]) for d in range(3)])


def load_episode_trajectories(local: Path) -> list[dict]:
    """
    Load per-episode joint data, compute FK, return list of trajectory dicts.
    Each dict: {"left_tcp": (F,3), "right_tcp": (F,3), "episode_index": int}.
    Uses all episodes in the dataset for a fair comparison.
    """
    info = json.loads((local / "meta" / "info.json").read_text())
    features = info.get("features", {})

    dfs = [pd.read_parquet(pq) for pq in sorted((local / "data").glob("**/*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total frames: {len(df):,}")

    state_col = next((c for c in df.columns if "observation.state" in c), None)
    if state_col is None:
        raise RuntimeError(f"No observation.state column. Available: {list(df.columns)}")

    first = df[state_col].iloc[0]
    if not hasattr(first, "__len__"):
        raise RuntimeError(f"observation.state is scalar ({type(first)}), expected array")

    state = np.stack(df[state_col].values).astype(np.float64)
    n_dim = state.shape[1]
    print(f"  State dim: {n_dim}  max|val|: {np.max(np.abs(state)):.1f}")

    left_idx, right_idx = _find_joint_indices(features, state_col, n_dim)

    ep_col = next((c for c in df.columns if c == "episode_index"), None)
    if ep_col is None:
        raise RuntimeError(f"No episode_index column. Available: {list(df.columns)}")

    episode_ids = df[ep_col].values
    unique_eps = np.unique(episode_ids)
    print(f"  Episodes: {len(unique_eps):,}")

    left_raw = state[:, left_idx]
    right_raw = state[:, right_idx]
    left_all = _detect_and_convert(left_raw)
    right_all = _detect_and_convert(right_raw)

    print("  Computing FK per episode …")
    trajectories = []
    for ep_id in unique_eps:
        mask = episode_ids == ep_id
        left_tcp = batch_fk(LEFT_CHAIN, left_all[mask])
        right_tcp = batch_fk(RIGHT_CHAIN, right_all[mask])
        if len(left_tcp) < 3:
            continue
        trajectories.append({"left_tcp": left_tcp, "right_tcp": right_tcp, "episode_index": int(ep_id)})

    print(f"  Valid trajectories: {len(trajectories):,}")
    return trajectories


# ── Clustering ──────────────────────────────────────────


def cluster_trajectories(
    trajectories: list[dict], n_clusters: int, n_waypoints: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-means on resampled trajectory features.
    Combines left+right TCP into a single feature vector per episode.
    Returns (labels, centroid_trajs (k, waypoints, 6), spread_per_cluster (k,) in metres).
    Spread = mean per-waypoint Euclidean distance from each trajectory to its centroid.
    """
    feat_vecs = []
    for t in trajectories:
        left_rs = resample_trajectory(t["left_tcp"], n_waypoints)
        right_rs = resample_trajectory(t["right_tcp"], n_waypoints)
        feat_vecs.append(np.concatenate([left_rs.ravel(), right_rs.ravel()]))
    feat_matrix = np.array(feat_vecs)

    k = min(n_clusters, len(feat_vecs))
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    labels = km.fit_predict(feat_matrix)

    centroids_flat = km.cluster_centers_
    centroid_trajs = np.zeros((k, n_waypoints, 6))
    for ci in range(k):
        left_flat = centroids_flat[ci, : n_waypoints * 3]
        right_flat = centroids_flat[ci, n_waypoints * 3 :]
        centroid_trajs[ci, :, :3] = left_flat.reshape(n_waypoints, 3)
        centroid_trajs[ci, :, 3:] = right_flat.reshape(n_waypoints, 3)

    # Mean per-waypoint distance to centroid (in metres) for each cluster
    spread = np.zeros(k)
    for ci in range(k):
        members = np.where(labels == ci)[0]
        if len(members) == 0:
            continue
        centroid_left = centroid_trajs[ci, :, :3]
        centroid_right = centroid_trajs[ci, :, 3:]
        dists = []
        for mi in members:
            t = trajectories[mi]
            left_rs = resample_trajectory(t["left_tcp"], n_waypoints)
            right_rs = resample_trajectory(t["right_tcp"], n_waypoints)
            d_left = np.linalg.norm(left_rs - centroid_left, axis=1).mean()
            d_right = np.linalg.norm(right_rs - centroid_right, axis=1).mean()
            dists.append((d_left + d_right) / 2)
        spread[ci] = np.mean(dists)

    return labels, centroid_trajs, spread


# ── Visualization ───────────────────────────────────────

PROJ_VIEWS = [
    ("XZ (side)", 0, 2, "X (m)", "Z (m)"),
    ("XY (top)", 0, 1, "X (m)", "Y (m)"),
    ("YZ (front)", 1, 2, "Y (m)", "Z (m)"),
]


def render(results: list[dict], out_path: Path) -> None:
    """
    2-row × 3-col grid per dataset (3 projections × 2 datasets).
    Trajectory lines colored by cluster, centroid trajectories drawn thick.
    """
    n_ds = len(results)
    n_proj = len(PROJ_VIEWS)
    fig, axes = plt.subplots(n_ds, n_proj, figsize=(7 * n_proj, 7 * n_ds), facecolor="#0d1117")
    if n_ds == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(results):
        trajectories = r["trajectories"]
        labels = r["labels"]
        centroids = r["centroids"]
        k = centroids.shape[0]

        cluster_sizes = np.bincount(labels, minlength=k)
        size_order = np.argsort(-cluster_sizes)
        pcts = cluster_sizes / len(labels) * 100
        spread = r["spread"]

        for col, (view_name, dim_a, dim_b, xlabel, ylabel) in enumerate(PROJ_VIEWS):
            ax = axes[row, col]
            ax.set_facecolor("#0d1117")

            for ti, traj in enumerate(trajectories):
                color = CLUSTER_COLORS[labels[ti] % len(CLUSTER_COLORS)]
                for tcp_key in ("left_tcp", "right_tcp"):
                    pts = traj[tcp_key]
                    ax.plot(pts[:, dim_a], pts[:, dim_b], color=color, alpha=0.12, linewidth=0.4)

            for ci in range(k):
                color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
                left_c = centroids[ci, :, :3]
                right_c = centroids[ci, :, 3:]
                lw = 1.5 + 2.0 * cluster_sizes[ci] / cluster_sizes.max()
                for c_pts in (left_c, right_c):
                    ax.plot(
                        c_pts[:, dim_a],
                        c_pts[:, dim_b],
                        color=color,
                        linewidth=lw,
                        alpha=0.95,
                        zorder=10,
                    )
                    ax.plot(
                        c_pts[0, dim_a],
                        c_pts[0, dim_b],
                        "o",
                        color=color,
                        markersize=4,
                        zorder=11,
                    )
                    ax.plot(
                        c_pts[-1, dim_a],
                        c_pts[-1, dim_b],
                        "s",
                        color=color,
                        markersize=4,
                        zorder=11,
                    )

            ax.set_xlabel(xlabel, color="#888", fontsize=9)
            ax.set_ylabel(ylabel, color="#888", fontsize=9)
            ax.tick_params(colors="#555", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#333")
            ax.set_aspect("equal")

            mean_spread_cm = np.average(spread, weights=cluster_sizes) * 100
            if col == 0:
                ax.set_title(
                    f"{r['label']}  ({r['n_episodes']:,} episodes, {k} clusters, "
                    f"avg spread {mean_spread_cm:.1f}cm)",
                    color="white",
                    fontsize=11,
                    pad=10,
                )
            else:
                ax.set_title(view_name, color="#aaa", fontsize=10, pad=8)

        # Cluster size + spread legend on the rightmost panel
        legend_ax = axes[row, -1]
        for ci in size_order:
            color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            spread_cm = spread[ci] * 100
            label = f"C{ci}: {cluster_sizes[ci]} eps ({pcts[ci]:.0f}%) ±{spread_cm:.1f}cm"
            legend_ax.plot([], [], color=color, linewidth=3, label=label)
        legend_ax.legend(
            loc="upper right",
            fontsize=7,
            frameon=True,
            facecolor="#1a1a2e",
            edgecolor="#333",
            labelcolor="white",
            handlelength=1.5,
        )

    fig.suptitle(
        "End-Effector Trajectory Clusters (FK · K-means)",
        color="white",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n✓ Saved: {out_path}")


# ── Main ────────────────────────────────────────────────


def main() -> None:
    results = []

    for ds in DATASETS:
        repo_id, label = ds["repo_id"], ds["label"]
        print(f"\n{'=' * 60}")
        print(f"  {label}: {repo_id}")
        print(f"{'=' * 60}")

        local = download_data(repo_id)
        trajectories = load_episode_trajectories(local)
        labels, centroids, spread = cluster_trajectories(trajectories, N_CLUSTERS, WAYPOINTS)

        cluster_sizes = np.bincount(labels, minlength=centroids.shape[0])
        print(f"  Cluster sizes: {sorted(cluster_sizes, reverse=True)}")
        for ci in np.argsort(-cluster_sizes):
            print(
                f"    C{ci}: {cluster_sizes[ci]} eps ({cluster_sizes[ci] / len(labels) * 100:.0f}%) "
                f"spread ±{spread[ci] * 100:.1f}cm"
            )

        results.append(
            {
                "label": label,
                "trajectories": trajectories,
                "labels": labels,
                "centroids": centroids,
                "spread": spread,
                "n_episodes": len(trajectories),
            }
        )

    out = OUTPUT_DIR / "workspace_trajectory_clusters.jpg"
    render(results, out)


if __name__ == "__main__":
    main()
