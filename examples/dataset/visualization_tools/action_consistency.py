"""
Action consistency analysis for imitation learning datasets.

Two parallel analyses per dataset:
  1. State-based: KNN in joint-state space → action chunk variance
  2. Image-based: KNN in SigLIP embedding space → action chunk variance

Comparing them reveals whether visual similarity and proprioceptive similarity
agree on where the data is inconsistent — and images are what the policy
primarily sees.
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.spatial import cKDTree
from transformers import AutoModel, AutoProcessor

DATASETS = [
    {"repo_id": "lerobot-data-collection/level2_final_quality3", "label": "HQ curated"},
    {"repo_id": "lerobot-data-collection/level12_rac_2_2026-02-08_1", "label": "Full collection"},
]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FRAMES = 100_000
K_NEIGHBORS = 50
ACTION_CHUNK_SIZE = 30
CAMERA_KEY = "observation.images.base"
ENCODER_MODEL = "google/siglip-base-patch16-224"
ENCODE_BATCH_SIZE = 512
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


def download_data(repo_id: str, camera_key: str) -> Path:
    print(f"  Downloading {repo_id} (parquet + {camera_key} videos) …")
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[
                "meta/**",
                "data/**",
                f"videos/{camera_key}/**",
            ],
        )
    )


# ── Data loading ────────────────────────────────────────


def _build_action_chunks(
    actions: np.ndarray, episode_ids: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each frame, concatenate the next chunk_size actions from the same episode.
    Returns (action_chunks, valid_mask).
    """
    n = len(actions)
    act_dim = actions.shape[1]
    chunks = np.zeros((n, chunk_size * act_dim), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        end = i + chunk_size
        if end > n:
            continue
        if episode_ids[i] != episode_ids[end - 1]:
            continue
        chunks[i] = actions[i:end].ravel()
        valid[i] = True

    return chunks, valid


def load_state_action_data(local: Path, max_frames: int, chunk_size: int, rng: np.random.Generator) -> dict:
    """
    Load observation.state and action, build action chunks, subsample, normalize.
    Also returns the original row indices (`chosen_idx`) for video frame mapping.
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

    print("  Building action chunks …")
    action_chunks, valid = _build_action_chunks(action_all, episode_all, chunk_size)
    valid_idx = np.where(valid)[0]
    print(f"  Valid frames (with full action chunk): {len(valid_idx):,} / {n_total:,}")

    if len(valid_idx) > max_frames:
        chosen = np.sort(rng.choice(valid_idx, max_frames, replace=False))
    else:
        chosen = valid_idx
    print(f"  Using {len(chosen):,} frames")

    state_raw = state_all[chosen]
    action_raw = action_chunks[chosen]
    episode_ids = episode_all[chosen]

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
        "episode_all": episode_all,
        "left_joint_idx": left_idx,
        "right_joint_idx": right_idx,
        "n_total": n_total,
        "chosen_idx": chosen,
        "df": df,
    }


# ── Video → frame extraction ──────────────────────────────


def build_video_lookup(local: Path, camera_key: str) -> dict:
    """
    Build a mapping from episode_index → {video_path, fps, from_ts}.
    """
    info = json.loads((local / "meta" / "info.json").read_text())
    fps = info["fps"]
    video_template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )

    ep_rows = []
    for pq in sorted((local / "meta" / "episodes").glob("**/*.parquet")):
        ep_rows.append(pd.read_parquet(pq))
    ep_df = pd.concat(ep_rows, ignore_index=True)

    chunk_col = f"videos/{camera_key}/chunk_index"
    file_col = f"videos/{camera_key}/file_index"
    ts_from = f"videos/{camera_key}/from_timestamp"
    if chunk_col not in ep_df.columns:
        chunk_col = f"{camera_key}/chunk_index"
        file_col = f"{camera_key}/file_index"
        ts_from = f"{camera_key}/from_timestamp"

    lookup: dict[int, dict] = {}
    for _, row in ep_df.iterrows():
        ci = int(row[chunk_col])
        fi = int(row[file_col])
        video_rel = video_template.format(video_key=camera_key, chunk_index=ci, file_index=fi)
        lookup[int(row["episode_index"])] = {
            "video_path": local / video_rel,
            "from_ts": float(row[ts_from]),
            "fps": fps,
        }
    return lookup


def extract_frames(
    chosen_idx: np.ndarray,
    episode_all: np.ndarray,
    video_lookup: dict,
) -> list[np.ndarray | None]:
    """
    Extract BGR frames for each chosen global index.
    Uses episode boundaries + fps to compute the seek timestamp.
    Returns list of (H, W, 3) BGR arrays (or None on failure).
    """
    # Build per-episode local frame index: for each row in the dataset,
    # its position within its episode
    unique_eps = np.unique(episode_all)
    ep_start: dict[int, int] = {}
    for ep in unique_eps:
        ep_start[int(ep)] = int(np.where(episode_all == ep)[0][0])

    frames: list[np.ndarray | None] = []
    # Group by video file for efficient sequential access
    jobs: list[tuple[int, int, str, float]] = []
    for out_i, global_i in enumerate(chosen_idx):
        ep = int(episode_all[global_i])
        info = video_lookup.get(ep)
        if info is None:
            jobs.append((out_i, -1, "", 0.0))
            continue
        local_frame = global_i - ep_start[ep]
        seek_ts = info["from_ts"] + local_frame / info["fps"]
        jobs.append((out_i, global_i, str(info["video_path"]), seek_ts))

    jobs.sort(key=lambda x: (x[2], x[3]))

    frames = [None] * len(chosen_idx)
    current_cap = None
    current_path = ""
    extracted = 0
    for out_i, _global_i, vpath, seek_ts in jobs:
        if not vpath:
            continue
        if vpath != current_path:
            if current_cap is not None:
                current_cap.release()
            current_cap = cv2.VideoCapture(vpath)
            current_path = vpath
        if current_cap is None or not current_cap.isOpened():
            continue
        current_cap.set(cv2.CAP_PROP_POS_MSEC, seek_ts * 1000.0)
        ret, frame = current_cap.read()
        if ret:
            frames[out_i] = frame
            extracted += 1
    if current_cap is not None:
        current_cap.release()

    print(f"  Extracted {extracted:,} / {len(chosen_idx):,} frames from video")
    return frames


# ── SigLIP encoding ─────────────────────────────────────


def encode_frames_siglip(
    frames: list[np.ndarray | None],
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Encode BGR frames through SigLIP vision encoder.
    Returns (N, embed_dim) float32 array. Frames that are None get a zero vector.
    """
    print(f"  Loading SigLIP model: {model_name} …")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embed_dim = model.config.vision_config.hidden_size

    n = len(frames)
    embeddings = np.zeros((n, embed_dim), dtype=np.float32)

    # Collect valid frame indices
    valid_indices = [i for i, f in enumerate(frames) if f is not None]
    print(f"  Encoding {len(valid_indices):,} valid frames in batches of {batch_size} …")

    for batch_start in range(0, len(valid_indices), batch_size):
        batch_idx = valid_indices[batch_start : batch_start + batch_size]
        pil_images = []
        for i in batch_idx:
            bgr = frames[i]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))

        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # L2-normalize embeddings for cosine-like KNN
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        embeddings[batch_idx] = image_features.cpu().numpy()

        done = min(batch_start + batch_size, len(valid_indices))
        if done % (batch_size * 10) == 0 or done == len(valid_indices):
            print(f"    {done:,} / {len(valid_indices):,} encoded")

    del model, processor
    torch.cuda.empty_cache()
    return embeddings


# ── KNN consistency ─────────────────────────────────────


def compute_consistency(
    features: np.ndarray,
    action_norm: np.ndarray,
    episode_ids: np.ndarray,
    k: int,
    label: str = "",
) -> np.ndarray:
    """
    For each frame, find K nearest neighbors in feature space from other episodes.
    Return per-frame action variance (mean across action dims).
    """
    n = len(features)
    print(f"  Building KD-tree on {n:,} vectors ({label}) …")
    tree = cKDTree(features)

    k_query = min(k * 3, n - 1)
    print(f"  Querying {k_query} neighbors per frame …")
    _dists, indices = tree.query(features, k=k_query + 1)
    indices = indices[:, 1:]

    print(f"  Computing cross-episode action variance ({label}) …")
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


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#555", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")


def _plot_histogram(ax: plt.Axes, variance: np.ndarray, title: str, color: str) -> None:
    _style_ax(ax)
    median_var = np.median(variance)
    mean_var = np.mean(variance)
    nonzero = variance[variance > 0]
    if len(nonzero) > 0:
        bins = np.logspace(np.log10(nonzero.min().clip(1e-6)), np.log10(nonzero.max()), 60)
        ax.hist(nonzero, bins=bins, color=color, alpha=0.8, edgecolor="#222")
    ax.set_xscale("log")
    ax.axvline(median_var, color="#ff6600", linewidth=2, label=f"median={median_var:.3f}")
    ax.axvline(mean_var, color="#ff2222", linewidth=2, linestyle="--", label=f"mean={mean_var:.3f}")
    ax.set_xlabel("Action variance (log scale)", color="#888", fontsize=10)
    ax.set_ylabel("Frame count", color="#888", fontsize=10)
    ax.set_title(title, color="white", fontsize=11, pad=10)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")


def _plot_episode_curves(
    ax: plt.Axes,
    var_state: np.ndarray,
    var_image: np.ndarray,
    episode_ids: np.ndarray,
    title: str,
) -> None:
    _style_ax(ax)
    unique_eps = np.unique(episode_ids)

    ep_means_s = np.array([var_state[episode_ids == ep].mean() for ep in unique_eps])
    ep_means_i = np.array([var_image[episode_ids == ep].mean() for ep in unique_eps])

    sorted_s = np.sort(ep_means_s)[::-1]
    sorted_i = np.sort(ep_means_i)[::-1]
    ep_x = np.arange(len(unique_eps))

    ax.fill_between(ep_x, sorted_s, alpha=0.2, color="#4363d8")
    ax.plot(ep_x, sorted_s, color="#4363d8", linewidth=1.2, label=f"State (med={np.median(ep_means_s):.3f})")
    ax.fill_between(ep_x, sorted_i, alpha=0.2, color="#e6194b")
    ax.plot(ep_x, sorted_i, color="#e6194b", linewidth=1.2, label=f"Image (med={np.median(ep_means_i):.3f})")

    ax.set_xlabel("Episode rank (worst → best)", color="#888", fontsize=10)
    ax.set_ylabel("Mean action variance", color="#888", fontsize=10)
    ax.set_title(title, color="white", fontsize=11, pad=10)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")


def _plot_heatmap(
    ax: plt.Axes, fig: plt.Figure, tcp_xz: np.ndarray, variance: np.ndarray, title: str
) -> None:
    _style_ax(ax)
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
    ax.set_title(title, color="white", fontsize=11, pad=10)
    ax.set_aspect("equal")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Action variance", color="white", fontsize=9)
    cbar.ax.tick_params(colors="#aaa", labelsize=7)


def render(results: list[dict], out_path: Path) -> None:
    """
    4-row x N-column figure:
      Row 0: State-based variance histogram
      Row 1: Image-based variance histogram
      Row 2: Per-episode curves (both overlaid)
      Row 3: Spatial heatmap (image-based variance)
    """
    n_ds = len(results)
    fig, axes = plt.subplots(4, n_ds, figsize=(9 * n_ds, 24), facecolor="#0d1117")
    if n_ds == 1:
        axes = axes[:, np.newaxis]

    headline_parts = []
    for col, r in enumerate(results):
        label = r["label"]
        var_s = r["var_state"]
        var_i = r["var_image"]
        tcp_xz = r["tcp_xz"]
        episode_ids = r["episode_ids"]

        med_s = np.median(var_s)
        med_i = np.median(var_i)
        headline_parts.append(f"{label}: state={med_s:.3f}, image={med_i:.3f}")

        _plot_histogram(axes[0, col], var_s, f"{label}\nState-based variance (K={K_NEIGHBORS})", "#4363d8")
        _plot_histogram(
            axes[1, col], var_i, f"{label}\nImage-based variance (SigLIP, K={K_NEIGHBORS})", "#e6194b"
        )
        _plot_episode_curves(
            axes[2, col],
            var_s,
            var_i,
            episode_ids,
            f"{label}\nPer-episode inconsistency ({len(np.unique(episode_ids)):,} episodes)",
        )
        _plot_heatmap(
            axes[3, col],
            fig,
            tcp_xz,
            var_i,
            f"{label}\nImage-based variance by TCP position (XZ)",
        )

    fig.suptitle(
        f"Action Consistency: State vs Image  (chunk={ACTION_CHUNK_SIZE}, K={K_NEIGHBORS})\n"
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rng = np.random.default_rng(SEED)
    results = []

    for ds in DATASETS:
        repo_id, label = ds["repo_id"], ds["label"]
        print(f"\n{'=' * 60}")
        print(f"  {label}: {repo_id}")
        print(f"{'=' * 60}")

        local = download_data(repo_id, CAMERA_KEY)
        data = load_state_action_data(local, MAX_FRAMES, ACTION_CHUNK_SIZE, rng)

        # --- State-based KNN ---
        var_state = compute_consistency(
            data["state_norm"], data["action_norm"], data["episode_ids"], K_NEIGHBORS, "state"
        )
        print(
            f"  State variance: median={np.median(var_state):.4f}  "
            f"mean={np.mean(var_state):.4f}  p90={np.percentile(var_state, 90):.4f}"
        )

        # --- Image-based KNN ---
        print("\n  Preparing image embeddings …")
        video_lookup = build_video_lookup(local, CAMERA_KEY)
        frames = extract_frames(data["chosen_idx"], data["episode_all"], video_lookup)
        embeddings = encode_frames_siglip(frames, ENCODER_MODEL, ENCODE_BATCH_SIZE, device)
        del frames  # free memory

        var_image = compute_consistency(
            embeddings, data["action_norm"], data["episode_ids"], K_NEIGHBORS, "image"
        )
        print(
            f"  Image variance: median={np.median(var_image):.4f}  "
            f"mean={np.mean(var_image):.4f}  p90={np.percentile(var_image, 90):.4f}"
        )

        # FK for spatial heatmap
        print("  Computing FK for spatial heatmap …")
        left_raw = data["state_raw"][:, data["left_joint_idx"]]
        left_rad = _detect_and_convert(left_raw)
        left_tcp = batch_fk(LEFT_CHAIN, left_rad)
        tcp_xz = left_tcp[:, [0, 2]]

        results.append(
            {
                "label": label,
                "var_state": var_state,
                "var_image": var_image,
                "episode_ids": data["episode_ids"],
                "tcp_xz": tcp_xz,
                "n_total": data["n_total"],
            }
        )

    out = OUTPUT_DIR / "action_consistency_comparison.jpg"
    render(results, out)

    # Save worst-episodes summary (image-based, since that's the stronger signal)
    worst_summary = {}
    for r in results:
        unique_eps = np.unique(r["episode_ids"])
        ep_means = {int(ep): float(r["var_image"][r["episode_ids"] == ep].mean()) for ep in unique_eps}
        ranked = sorted(ep_means.items(), key=lambda x: x[1], reverse=True)[:50]
        worst_summary[r["label"]] = [{"episode": ep, "mean_variance": v} for ep, v in ranked]
    worst_path = OUTPUT_DIR / "action_consistency_worst_episodes.json"
    worst_path.write_text(json.dumps(worst_summary, indent=2))
    print(f"✓ Saved worst episodes: {worst_path}")


if __name__ == "__main__":
    main()
