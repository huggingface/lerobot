"""
Action-state consistency analysis for imitation learning datasets.
For each frame, finds K nearest neighbors in state space (from other episodes)
and measures the variance of corresponding actions. High variance at similar
states = contradictory supervision for the policy.

Outputs a comparison figure with histograms, per-episode curves, and spatial
heatmaps showing where demonstrations conflict.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree

DATASETS = [
    {"repo_id": "lerobot-data-collection/level2_final_quality3", "label": "HQ curated"},
    {"repo_id": "lerobot-data-collection/level12_rac_2_2026-02-08_1", "label": "Full collection"},
]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FRAMES = 10_000
K_NEIGHBORS = 50
ACTION_CHUNK_SIZE = 30
SEED = 42
DPI = 150

CONSISTENCY_CMAP = LinearSegmentedColormap.from_list(
    "consistency", ["#0a2e0a", "#1a8e1a", "#88cc22", "#ffaa22", "#ff2222"]
)

# FK chains from OpenArm bimanual URDF (same as workspace_density.py).
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
    r, p, y = rpy
    mat = np.eye(4)
    mat[:3, :3] = _rot_z(y) @ _rot_y(p) @ _rot_x(r)
    mat[:3, 3] = xyz
    return mat


def _batch_axis_rot(axis: tuple, angles: np.ndarray) -> np.ndarray:
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


# ── Data helpers ────────────────────────────────────────


def _flatten_names(obj: object) -> list[str]:
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


# ── Data loading ────────────────────────────────────────


def _build_action_chunks(
    actions: np.ndarray, episode_ids: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build action chunks: for each frame, concatenate the next chunk_size actions
    from the same episode. Returns (action_chunks, valid_mask).
    Frames too close to episode end to form a full chunk are marked invalid.
    """
    n = len(actions)
    act_dim = actions.shape[1]
    chunks = np.zeros((n, chunk_size * act_dim), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        end = i + chunk_size
        if end > n:
            continue
        # All frames in the chunk must belong to the same episode
        if episode_ids[i] != episode_ids[end - 1]:
            continue
        chunks[i] = actions[i:end].ravel()
        valid[i] = True

    return chunks, valid


def load_state_action_data(local: Path, max_frames: int, chunk_size: int, rng: np.random.Generator) -> dict:
    """
    Load observation.state and action columns, build action chunks of size
    chunk_size (matching what the policy learns), subsample, normalize.
    """
    info = json.loads((local / "meta" / "info.json").read_text())
    features = info.get("features", {})

    dfs = [pd.read_parquet(pq) for pq in sorted((local / "data").glob("**/*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    n_total = len(df)
    print(f"  Total frames: {n_total:,}")

    state_col = next((c for c in df.columns if "observation.state" in c), None)
    action_col = next((c for c in df.columns if c == "action"), None)
    if state_col is None:
        raise RuntimeError(f"No observation.state column. Available: {list(df.columns)}")
    if action_col is None:
        raise RuntimeError(f"No action column. Available: {list(df.columns)}")

    ep_col = next((c for c in df.columns if c == "episode_index"), None)
    if ep_col is None:
        raise RuntimeError(f"No episode_index column. Available: {list(df.columns)}")

    state_all = np.stack(df[state_col].values).astype(np.float64)
    action_all = np.stack(df[action_col].values).astype(np.float64)
    episode_all = df[ep_col].values.astype(np.int64)

    n_dim = state_all.shape[1]
    act_dim = action_all.shape[1]
    print(f"  State dim: {n_dim}  Action dim: {act_dim}  Chunk size: {chunk_size}")
    print(f"  Action chunk dim: {chunk_size * act_dim}")

    left_idx, right_idx = _find_joint_indices(features, state_col, n_dim)

    # Build action chunks within episode boundaries
    print("  Building action chunks …")
    action_chunks, valid = _build_action_chunks(action_all, episode_all, chunk_size)
    valid_idx = np.where(valid)[0]
    print(f"  Valid frames (with full action chunk): {len(valid_idx):,} / {n_total:,}")

    # Subsample from valid frames only
    if len(valid_idx) > max_frames:
        chosen = np.sort(rng.choice(valid_idx, max_frames, replace=False))
    else:
        chosen = valid_idx
    print(f"  Using {len(chosen):,} frames")

    state_raw = state_all[chosen]
    action_raw = action_chunks[chosen]
    episode_ids = episode_all[chosen]

    # Z-score normalize for fair KNN distance
    state_mean = state_raw.mean(axis=0)
    state_std = state_raw.std(axis=0)
    state_std[state_std < 1e-8] = 1.0
    state_norm = (state_raw - state_mean) / state_std

    action_mean = action_raw.mean(axis=0)
    action_std = action_raw.std(axis=0)
    action_std[action_std < 1e-8] = 1.0
    action_norm = (action_raw - action_mean) / action_std

    return {
        "state_raw": state_raw,
        "state_norm": state_norm,
        "action_raw": action_raw,
        "action_norm": action_norm,
        "episode_ids": episode_ids,
        "left_joint_idx": left_idx,
        "right_joint_idx": right_idx,
        "n_total": n_total,
    }


# ── KNN consistency ─────────────────────────────────────


def compute_consistency(
    state_norm: np.ndarray,
    action_norm: np.ndarray,
    episode_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    For each frame, find K nearest neighbors in state space from *other* episodes.
    Return per-frame action variance (mean across action dims).
    """
    n = len(state_norm)
    print(f"  Building KD-tree on {n:,} state vectors …")
    tree = cKDTree(state_norm)

    # Query extra neighbors to have room after filtering same-episode
    k_query = min(k * 3, n - 1)
    print(f"  Querying {k_query} neighbors per frame …")
    dists, indices = tree.query(state_norm, k=k_query + 1)

    # indices[:, 0] is the point itself — skip it
    indices = indices[:, 1:]

    print("  Computing cross-episode action variance …")
    variance = np.zeros(n)
    for i in range(n):
        ep_i = episode_ids[i]
        neighbors = indices[i]
        cross_ep = neighbors[episode_ids[neighbors] != ep_i][:k]
        if len(cross_ep) < 2:
            variance[i] = 0.0
            continue
        neighbor_actions = action_norm[cross_ep]
        variance[i] = np.mean(np.var(neighbor_actions, axis=0))

    return variance


# ── Visualization ───────────────────────────────────────


def render(results: list[dict], out_path: Path) -> None:
    n_ds = len(results)
    fig, axes = plt.subplots(3, n_ds, figsize=(9 * n_ds, 18), facecolor="#0d1117")
    if n_ds == 1:
        axes = axes[:, np.newaxis]

    headline_parts = []

    for col, r in enumerate(results):
        variance = r["variance"]
        episode_ids = r["episode_ids"]
        tcp_xz = r["tcp_xz"]
        label = r["label"]

        median_var = np.median(variance)
        mean_var = np.mean(variance)
        headline_parts.append(f"{label}: median={median_var:.3f}, mean={mean_var:.3f}")

        # Row 0: Histogram of per-frame action variance
        ax = axes[0, col]
        ax.set_facecolor("#0d1117")
        nonzero = variance[variance > 0]
        if len(nonzero) > 0:
            bins = np.logspace(np.log10(nonzero.min().clip(1e-6)), np.log10(nonzero.max()), 60)
            ax.hist(nonzero, bins=bins, color="#4363d8", alpha=0.8, edgecolor="#222")
        ax.set_xscale("log")
        ax.axvline(median_var, color="#ff6600", linewidth=2, label=f"median={median_var:.3f}")
        ax.axvline(mean_var, color="#ff2222", linewidth=2, linestyle="--", label=f"mean={mean_var:.3f}")
        ax.set_xlabel("Action variance (log scale)", color="#888", fontsize=10)
        ax.set_ylabel("Frame count", color="#888", fontsize=10)
        ax.set_title(f"{label}\nPer-frame action variance distribution", color="white", fontsize=12, pad=10)
        ax.tick_params(colors="#555", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

        # Row 1: Per-episode mean inconsistency curve (sorted)
        ax = axes[1, col]
        ax.set_facecolor("#0d1117")
        unique_eps = np.unique(episode_ids)
        ep_means = np.array([variance[episode_ids == ep].mean() for ep in unique_eps])
        sorted_means = np.sort(ep_means)[::-1]
        ep_x = np.arange(len(sorted_means))

        p90 = np.percentile(ep_means, 90)
        above_p90 = np.sum(ep_means > p90)

        ax.fill_between(ep_x, sorted_means, alpha=0.3, color="#4363d8")
        ax.plot(ep_x, sorted_means, color="#4363d8", linewidth=1.2)
        ax.axhline(
            np.median(ep_means), color="#ff6600", linewidth=1.5, label=f"median={np.median(ep_means):.3f}"
        )
        ax.axhline(
            p90, color="#ff2222", linewidth=1, linestyle=":", label=f"p90={p90:.3f} ({above_p90} eps above)"
        )
        ax.set_xlabel("Episode rank (worst → best)", color="#888", fontsize=10)
        ax.set_ylabel("Mean action variance", color="#888", fontsize=10)
        ax.set_title(
            f"{label}\nPer-episode inconsistency ({len(unique_eps):,} episodes)",
            color="white",
            fontsize=12,
            pad=10,
        )
        ax.tick_params(colors="#555", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

        # Row 2: Spatial heatmap (XZ side view) colored by local action variance
        ax = axes[2, col]
        ax.set_facecolor("#0d1117")
        order = np.argsort(variance)
        pts = tcp_xz[order]
        var_sorted = variance[order]

        vmin = np.percentile(variance[variance > 0], 5) if np.any(variance > 0) else 0
        vmax = np.percentile(variance[variance > 0], 95) if np.any(variance > 0) else 1

        sc = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=var_sorted,
            cmap=CONSISTENCY_CMAP,
            s=0.5,
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_xlabel("X (m)", color="#888", fontsize=10)
        ax.set_ylabel("Z (m)", color="#888", fontsize=10)
        ax.set_title(
            f"{label}\nAction variance by TCP position (XZ side)",
            color="white",
            fontsize=12,
            pad=10,
        )
        ax.tick_params(colors="#555", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.set_aspect("equal")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Action variance", color="white", fontsize=9)
        cbar.ax.tick_params(colors="#aaa", labelsize=7)

    fig.suptitle(
        f"Action-State Consistency Analysis  (action chunk = {ACTION_CHUNK_SIZE})\n"
        + "  |  ".join(headline_parts),
        color="white",
        fontsize=15,
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n✓ Saved: {out_path}")


# ── Main ────────────────────────────────────────────────


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = []

    for ds in DATASETS:
        repo_id, label = ds["repo_id"], ds["label"]
        print(f"\n{'=' * 60}")
        print(f"  {label}: {repo_id}")
        print(f"{'=' * 60}")

        local = download_data(repo_id)
        data = load_state_action_data(local, MAX_FRAMES, ACTION_CHUNK_SIZE, rng)

        variance = compute_consistency(
            data["state_norm"], data["action_norm"], data["episode_ids"], K_NEIGHBORS
        )
        print(
            f"  Variance stats: median={np.median(variance):.4f}  mean={np.mean(variance):.4f}  "
            f"p90={np.percentile(variance, 90):.4f}"
        )

        # Compute FK for spatial heatmap (left arm TCP, XZ projection)
        print("  Computing FK for spatial heatmap …")
        left_raw = data["state_raw"][:, data["left_joint_idx"]]
        left_rad = _detect_and_convert(left_raw)
        left_tcp = batch_fk(LEFT_CHAIN, left_rad)
        tcp_xz = left_tcp[:, [0, 2]]

        results.append(
            {
                "label": label,
                "variance": variance,
                "episode_ids": data["episode_ids"],
                "tcp_xz": tcp_xz,
                "n_total": data["n_total"],
            }
        )

    out = OUTPUT_DIR / "action_consistency_comparison.jpg"
    render(results, out)


if __name__ == "__main__":
    main()
