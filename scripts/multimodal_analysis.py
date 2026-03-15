#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Chunk-level multi-modality analysis for comparing full/mixed vs curated datasets.

Treats each action chunk (sliding window of CHUNK_SIZE consecutive frames) as the
atomic unit, tagged by the SARM progress score at its start frame. For each
progress band, compares the full vs HQ dataset on:

  1. Intra-band action variance
  2. Progress delta per chunk
  3. GMM + BIC optimal K (number of distinct strategies)
  4. PCA embedding (visual cluster inspection)

Usage:
    python chunk_multimodality_analysis.py \\
        --full-dataset lerobot-data-collection/level12_rac_2_2026-02-08_1 \\
        --hq-dataset   lerobot-data-collection/level2_final_quality3 \\
        --output-dir   ./chunk_analysis
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Visual style ──────────────────────────────────────────────────────────

BG = "#0e1117"
CARD = "#1a1d27"
BORDER = "#2a2d3a"
SUB = "#8b8fa8"
TEXT = "#e8eaf0"
C_FULL = "#f7934f"
C_HQ = "#4dc98a"


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(CARD)
    ax.tick_params(colors=SUB, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Step 0: Load episodes ────────────────────────────────────────────────

def _load_sarm_progress(repo_id: str) -> pd.DataFrame | None:
    """Try to download sarm_progress.parquet from the Hub."""
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename="sarm_progress.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        col = "progress_sparse" if "progress_sparse" in df.columns else "progress_dense"
        if col not in df.columns:
            logger.warning("sarm_progress.parquet has no progress columns — ignoring")
            return None
        logger.info("Loaded SARM progress (%s) for %s (%d rows)", col, repo_id, len(df))
        return df.rename(columns={col: "progress"})[["episode_index", "frame_index", "progress"]]
    except Exception as exc:
        logger.warning("Could not load sarm_progress.parquet for %s: %s", repo_id, exc)
        return None


def load_episodes(
    repo_id: str,
    n_joints: int = 16,
    max_episodes: int | None = None,
) -> list[dict]:
    dataset = LeRobotDataset(repo_id, download_videos=False)
    raw = dataset.hf_dataset

    sarm_df = _load_sarm_progress(repo_id)
    # Build per-episode progress arrays from SARM parquet (indexed by frame_index)
    sarm_by_ep: dict[int, dict[int, float]] = {}
    if sarm_df is not None:
        if max_episodes is not None:
            sarm_df = sarm_df[sarm_df["episode_index"] < max_episodes]
        for ep_id, grp in sarm_df.groupby("episode_index"):
            sarm_by_ep[int(ep_id)] = dict(
                zip(grp["frame_index"].astype(int), grp["progress"].astype(float))
            )

    episodes: dict[int, dict] = defaultdict(lambda: {"actions": [], "progress": []})
    for row in raw:
        ep = int(row["episode_index"])
        if max_episodes is not None and ep >= max_episodes:
            continue
        action = np.array(row["action"], dtype=np.float32)[:n_joints]
        episodes[ep]["actions"].append(action)
        fi = int(row["frame_index"])
        ep_prog = sarm_by_ep.get(ep, {})
        episodes[ep]["progress"].append(ep_prog.get(fi, float("nan")))

    has_sarm = len(sarm_lookup) > 0
    result = []
    for ep_id, d in sorted(episodes.items()):
        actions = np.stack(d["actions"])
        T = len(actions)
        if has_sarm:
            prog = np.array(d["progress"], dtype=np.float32)
            prog = np.clip(np.nan_to_num(prog, nan=0.0), 0.0, 1.0)
            prog = np.maximum.accumulate(prog)
        else:
            prog = np.linspace(0.0, 1.0, T, dtype=np.float32)
        result.append({"episode": ep_id, "actions": actions, "progress": prog})

    src = "SARM" if has_sarm else "time-based"
    logger.info("Progress source: %s", src)
    return result


# ── Step 1: Filter short episodes ────────────────────────────────────────

def auto_length_threshold(
    episodes_full: list[dict], episodes_hq: list[dict]
) -> int:
    all_lengths = np.array(
        [e["actions"].shape[0] for e in episodes_full + episodes_hq]
    )
    kde = gaussian_kde(all_lengths, bw_method=0.25)
    xs = np.linspace(all_lengths.min(), np.percentile(all_lengths, 40), 300)
    return int(xs[np.argmin(kde(xs))])


def plot_length_distribution(
    episodes_full: list[dict],
    episodes_hq: list[dict],
    threshold: int,
    out_path: Path,
) -> None:
    lens_full = np.array([e["actions"].shape[0] for e in episodes_full])
    lens_hq = np.array([e["actions"].shape[0] for e in episodes_hq])
    all_lens = np.concatenate([lens_full, lens_hq])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    bins = np.linspace(all_lens.min(), all_lens.max(), 50)
    ax.hist(lens_full, bins=bins, alpha=0.5, color=C_FULL, label="Full/Mixed")
    ax.hist(lens_hq, bins=bins, alpha=0.5, color=C_HQ, label="HQ")

    xs = np.linspace(all_lens.min(), all_lens.max(), 300)
    kde = gaussian_kde(all_lens, bw_method=0.25)
    ax.plot(xs, kde(xs) * len(all_lens) * (bins[1] - bins[0]), color=TEXT, lw=1.5, label="KDE (combined)")

    ax.axvline(threshold, color="#ff4b4b", ls="--", lw=1.5, label=f"Threshold = {threshold}")
    ax.set_xlabel("Episode length (frames)", color=SUB)
    ax.set_ylabel("Count", color=SUB)
    ax.set_title("Episode Length Distribution", color=TEXT, fontsize=13)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    _save(fig, out_path)


def filter_episodes(episodes: list[dict], min_length: int) -> list[dict]:
    kept = [e for e in episodes if e["actions"].shape[0] >= min_length]
    logger.info("Kept %d / %d episodes (min_length=%d)", len(kept), len(episodes), min_length)
    return kept


# ── Step 2: Extract chunks ───────────────────────────────────────────────

def extract_chunks(
    episodes: list[dict],
    chunk_size: int = 30,
    chunk_stride: int = 15,
) -> list[dict]:
    chunks = []
    for ep in episodes:
        actions = ep["actions"]
        T = len(actions)
        prog = ep["progress"]

        for t in range(0, T - chunk_size, chunk_stride):
            chunk = actions[t : t + chunk_size]
            p_start = float(prog[t])
            p_end = float(prog[min(t + chunk_size, T - 1)])

            chunks.append({
                "action_mean": chunk.mean(axis=0).astype(np.float32),
                "action_flat": chunk.flatten().astype(np.float32),
                "progress_start": p_start,
                "progress_delta": p_end - p_start,
                "episode": ep["episode"],
            })
    return chunks


# ── Step 3: Adaptive progress bands ─────────────────────────────────────

def make_bands(n_bands: int = 5) -> list[tuple[float, float]]:
    edges = np.linspace(0.0, 1.0, n_bands + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_bands)]


def assign_bands(
    chunks: list[dict], band_edges: list[tuple[float, float]]
) -> list[dict]:
    n = len(band_edges)
    for c in chunks:
        p = c["progress_start"]
        c["band"] = next(
            (bi for bi, (lo, hi) in enumerate(band_edges) if p < hi),
            n - 1,
        )
    return chunks


def split_by_band(chunks: list[dict], n_bands: int) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {b: [] for b in range(n_bands)}
    for c in chunks:
        out[c["band"]].append(c)
    return out


# ── Step 4: Intra-band action variance ──────────────────────────────────

def band_variance_matrix(
    bands: dict[int, list[dict]], n_bands: int, n_joints: int
) -> np.ndarray:
    var_mat = np.full((n_bands, n_joints), np.nan)
    for b, clist in bands.items():
        if len(clist) < 3:
            continue
        means = np.stack([c["action_mean"] for c in clist])
        var_mat[b] = np.var(means, axis=0)
    return var_mat


def plot_variance_heatmap(
    var_full: np.ndarray,
    var_hq: np.ndarray,
    band_edges: list[tuple[float, float]],
    out_path: Path,
) -> None:
    n_bands = var_full.shape[0]
    vmin = 0.0
    vmax = max(np.nanmax(var_full), np.nanmax(var_hq))

    band_labels = [f"{lo:.0%}–{hi:.0%}" for lo, hi in band_edges]
    joint_labels = [f"J{j}" for j in range(var_full.shape[1])]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 3, 2]})
    fig.patch.set_facecolor(BG)
    fig.suptitle("Intra-Band Action Variance", color=TEXT, fontsize=14, y=0.98)

    for ax_idx, (mat, label) in enumerate([(var_full, "Full/Mixed"), (var_hq, "HQ")]):
        ax = axes[ax_idx]
        _style_ax(ax)
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
        ax.set_yticks(range(n_bands))
        ax.set_yticklabels(band_labels, fontsize=7, color=SUB)
        ax.set_xticks(range(var_full.shape[1]))
        ax.set_xticklabels(joint_labels, fontsize=7, color=SUB)
        ax.set_title(f"Panel {'A' if ax_idx == 0 else 'B'}: {label}", color=TEXT, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    with np.errstate(invalid="ignore"):
        mean_full = np.nanmean(var_full, axis=1)
        mean_hq = np.nanmean(var_hq, axis=1)
    ratio = np.where(np.isnan(mean_full) | np.isnan(mean_hq), np.nan,
                     mean_full / (mean_hq + 1e-8))
    ax_bar = axes[2]
    _style_ax(ax_bar)
    colors = [
        "#ff4b4b" if r > 2.0 else "#ffaa33" if r > 1.2 else C_HQ
        for r in ratio
    ]
    ax_bar.bar(range(n_bands), ratio, color=colors, edgecolor=BORDER)
    ax_bar.axhline(1.0, color=SUB, ls="--", lw=0.8)
    ax_bar.set_xticks(range(n_bands))
    ax_bar.set_xticklabels(band_labels, fontsize=7, color=SUB)
    ax_bar.set_ylabel("Variance ratio\n(Full / HQ)", color=SUB, fontsize=9)
    ax_bar.set_title("Panel C: Variance Ratio per Band", color=TEXT, fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_path)


# ── Step 5: Progress delta per band ──────────────────────────────────────

def plot_progress_delta(
    bands_full: dict[int, list[dict]],
    bands_hq: dict[int, list[dict]],
    band_edges: list[tuple[float, float]],
    out_path: Path,
) -> None:
    n_bands = len(band_edges)
    band_labels = [f"{lo:.0%}–{hi:.0%}" for lo, hi in band_edges]
    x = np.arange(n_bands)
    w = 0.35

    means_full, stds_full = [], []
    means_hq, stds_hq = [], []
    all_deltas_full, all_deltas_hq = [], []

    for b in range(n_bands):
        df = np.array([c["progress_delta"] for c in bands_full.get(b, [])])
        dh = np.array([c["progress_delta"] for c in bands_hq.get(b, [])])
        means_full.append(np.mean(df) if len(df) > 0 else 0)
        stds_full.append(np.std(df) if len(df) > 0 else 0)
        means_hq.append(np.mean(dh) if len(dh) > 0 else 0)
        stds_hq.append(np.std(dh) if len(dh) > 0 else 0)
        all_deltas_full.extend(df.tolist())
        all_deltas_hq.extend(dh.tolist())

    fig, (ax_bar, ax_viol) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor(BG)
    fig.suptitle("Progress Delta per Chunk", color=TEXT, fontsize=14)

    _style_ax(ax_bar)
    ax_bar.bar(x - w / 2, means_full, w, yerr=stds_full, color=C_FULL, edgecolor=BORDER,
               capsize=3, label="Full/Mixed", error_kw={"ecolor": SUB})
    ax_bar.bar(x + w / 2, means_hq, w, yerr=stds_hq, color=C_HQ, edgecolor=BORDER,
               capsize=3, label="HQ", error_kw={"ecolor": SUB})
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(band_labels, fontsize=7, color=SUB, rotation=30)
    ax_bar.set_ylabel("Mean progress Δ", color=SUB)
    ax_bar.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

    _style_ax(ax_viol)
    data_viol = [np.array(all_deltas_full), np.array(all_deltas_hq)]
    if all(len(d) > 0 for d in data_viol):
        parts = ax_viol.violinplot(data_viol, positions=[0, 1], showmeans=True, showmedians=True)
        for pc, c in zip(parts["bodies"], [C_FULL, C_HQ]):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)
        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color(SUB)
    ax_viol.set_xticks([0, 1])
    ax_viol.set_xticklabels(["Full", "HQ"], color=SUB)
    ax_viol.set_ylabel("Progress Δ", color=SUB)
    ax_viol.set_title("Overall Distribution", color=TEXT, fontsize=10)

    fig.tight_layout()
    _save(fig, out_path)


# ── Step 6: GMM + BIC per band ──────────────────────────────────────────

def gmm_optimal_k(
    band_chunks: list[dict],
    pca_components: int = 15,
    max_k: int = 12,
    seed: int = 42,
) -> int | None:
    if len(band_chunks) < 20:
        return None
    X = np.stack([c["action_flat"] for c in band_chunks])
    X = StandardScaler().fit_transform(X)
    n = min(pca_components, X.shape[1], X.shape[0] - 1)
    X_r = PCA(n_components=n, random_state=seed).fit_transform(X)
    bics = []
    for k in range(1, min(max_k + 1, len(X_r) // 6)):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            n_init=5, max_iter=300, random_state=seed,
        )
        gmm.fit(X_r)
        bics.append((k, gmm.bic(X_r)))
    if not bics:
        return None
    return min(bics, key=lambda x: x[1])[0]


def plot_gmm_bic(
    bands_full: dict[int, list[dict]],
    bands_hq: dict[int, list[dict]],
    band_edges: list[tuple[float, float]],
    seed: int,
    out_path: Path,
) -> tuple[list[int | None], list[int | None]]:
    n_bands = len(band_edges)
    ks_full = [gmm_optimal_k(bands_full.get(b, []), seed=seed) for b in range(n_bands)]
    ks_hq = [gmm_optimal_k(bands_hq.get(b, []), seed=seed) for b in range(n_bands)]

    band_labels = [f"{lo:.0%}–{hi:.0%}" for lo, hi in band_edges]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    xs = np.arange(n_bands)
    valid_full = [(i, k) for i, k in enumerate(ks_full) if k is not None]
    valid_hq = [(i, k) for i, k in enumerate(ks_hq) if k is not None]

    if valid_full:
        xi, yi = zip(*valid_full)
        ax.plot(xi, yi, "o-", color=C_FULL, label="Full/Mixed", lw=2, markersize=7)
    if valid_hq:
        xi, yi = zip(*valid_hq)
        ax.plot(xi, yi, "o-", color=C_HQ, label="HQ", lw=2, markersize=7)

    if valid_full and valid_hq:
        all_x = sorted(set([i for i, _ in valid_full]) & set([i for i, _ in valid_hq]))
        if len(all_x) >= 2:
            kf_interp = {i: k for i, k in valid_full}
            kh_interp = {i: k for i, k in valid_hq}
            shared_x = [i for i in all_x if i in kf_interp and i in kh_interp]
            yf = [kf_interp[i] for i in shared_x]
            yh = [kh_interp[i] for i in shared_x]
            ax.fill_between(shared_x, yf, yh, alpha=0.15, color=TEXT)

    ax.set_xticks(xs)
    ax.set_xticklabels(band_labels, fontsize=7, color=SUB, rotation=30)
    ax.set_ylabel("Optimal K (GMM-BIC)", color=SUB)
    ax.set_title("Number of Distinct Strategies per Band", color=TEXT, fontsize=13)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    _save(fig, out_path)
    return ks_full, ks_hq


# ── Step 7: PCA scatter per band ────────────────────────────────────────

def plot_pca_scatter(
    bands_full: dict[int, list[dict]],
    bands_hq: dict[int, list[dict]],
    band_edges: list[tuple[float, float]],
    out_path: Path,
) -> None:
    n_plot = min(4, len(band_edges))
    fig, axes = plt.subplots(2, n_plot, figsize=(4 * n_plot, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle("PCA of Action Chunks per Band", color=TEXT, fontsize=14)

    if n_plot == 1:
        axes = axes.reshape(2, 1)

    for col, b in enumerate(range(n_plot)):
        cf = bands_full.get(b, [])
        ch = bands_hq.get(b, [])
        lo, hi = band_edges[b]

        for row, (clist, color, label) in enumerate([
            (cf, C_FULL, "Full/Mixed"), (ch, C_HQ, "HQ")
        ]):
            ax = axes[row, col]
            _style_ax(ax)
            if row == 0:
                ax.set_title(f"{lo:.0%}–{hi:.0%}", color=TEXT, fontsize=10)
            if col == 0:
                ax.set_ylabel(label, color=SUB, fontsize=9)

            if len(cf) < 3 or len(ch) < 3:
                ax.text(0.5, 0.5, "Too few\nchunks", transform=ax.transAxes,
                        ha="center", va="center", color=SUB, fontsize=9)
                continue

            X_full_b = np.stack([c["action_flat"] for c in cf])
            X_hq_b = np.stack([c["action_flat"] for c in ch])
            X_all = np.vstack([X_full_b, X_hq_b])
            X_all = StandardScaler().fit_transform(X_all)
            X_2d = PCA(n_components=2, random_state=42).fit_transform(X_all)

            X_2d_full = X_2d[: len(cf)]
            X_2d_hq = X_2d[len(cf) :]

            pts = X_2d_full if row == 0 else X_2d_hq
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5, color=color, edgecolors="none")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_path)


# ── Plot 1: Chunk counts per band ───────────────────────────────────────

def plot_chunk_counts(
    bands_full: dict[int, list[dict]],
    bands_hq: dict[int, list[dict]],
    band_edges: list[tuple[float, float]],
    out_path: Path,
) -> None:
    n_bands = len(band_edges)
    band_labels = [f"{lo:.0%}–{hi:.0%}" for lo, hi in band_edges]
    x = np.arange(n_bands)
    w = 0.35

    counts_full = [len(bands_full.get(b, [])) for b in range(n_bands)]
    counts_hq = [len(bands_hq.get(b, [])) for b in range(n_bands)]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    ax.bar(x - w / 2, counts_full, w, color=C_FULL, edgecolor=BORDER, label="Full/Mixed")
    ax.bar(x + w / 2, counts_hq, w, color=C_HQ, edgecolor=BORDER, label="HQ")
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=7, color=SUB, rotation=30)
    ax.set_ylabel("Chunk count", color=SUB)
    ax.set_title("Chunk Counts per Progress Band", color=TEXT, fontsize=13)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


# ── Summary figure ───────────────────────────────────────────────────────

def plot_summary(
    var_full: np.ndarray,
    var_hq: np.ndarray,
    band_edges: list[tuple[float, float]],
    ks_full: list[int | None],
    ks_hq: list[int | None],
    bands_full: dict[int, list[dict]],
    bands_hq: dict[int, list[dict]],
    out_path: Path,
) -> None:
    with np.errstate(invalid="ignore"):
        mean_full = np.nanmean(var_full, axis=1)
        mean_hq = np.nanmean(var_hq, axis=1)
    ratio = np.where(np.isnan(mean_full) | np.isnan(mean_hq), np.nan,
                     mean_full / (mean_hq + 1e-8))
    valid_ratio = ratio[~np.isnan(ratio)]
    mean_ratio = float(np.mean(valid_ratio)) if len(valid_ratio) > 0 else float("nan")
    peak_idx = int(np.argmax(valid_ratio)) if len(valid_ratio) > 0 else 0
    peak_ratio = float(valid_ratio[peak_idx]) if len(valid_ratio) > 0 else float("nan")
    lo, hi = band_edges[peak_idx]
    peak_band = f"{lo:.0%}–{hi:.0%}"

    valid_kf = [k for k in ks_full if k is not None]
    valid_kh = [k for k in ks_hq if k is not None]
    mean_k_full = np.mean(valid_kf) if valid_kf else float("nan")
    mean_k_hq = np.mean(valid_kh) if valid_kh else float("nan")

    n_bands = len(band_edges)
    deltas_full = [c["progress_delta"] for b in range(n_bands) for c in bands_full.get(b, [])]
    deltas_hq = [c["progress_delta"] for b in range(n_bands) for c in bands_hq.get(b, [])]
    mean_delta_full = float(np.mean(deltas_full)) if deltas_full else float("nan")
    mean_delta_hq = float(np.mean(deltas_hq)) if deltas_hq else float("nan")

    rows = [
        ("Mean variance ratio (Full / HQ)", f"{mean_ratio:.2f}x"),
        ("Peak variance ratio", f"{peak_ratio:.2f}x at {peak_band}"),
        ("Mean GMM K — Full", f"{mean_k_full:.1f}"),
        ("Mean GMM K — HQ", f"{mean_k_hq:.1f}"),
        ("Mean progress Δ — Full", f"{mean_delta_full:.4f}"),
        ("Mean progress Δ — HQ", f"{mean_delta_hq:.4f}"),
    ]

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.axis("off")

    table = ax.table(
        cellText=[[m, v] for m, v in rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_facecolor(CARD)
        cell.set_text_props(color=TEXT)
        if key[0] == 0:
            cell.set_text_props(color=TEXT, fontweight="bold")
    table.scale(1, 1.6)
    ax.set_title("Summary Statistics", color=TEXT, fontsize=13, pad=15)
    fig.tight_layout()
    _save(fig, out_path)

    for metric, value in rows:
        logger.info("  %s: %s", metric, value)


# ── Main ─────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FULL dataset: %s", args.full_dataset)
    episodes_full = load_episodes(args.full_dataset, args.n_joints, args.max_episodes)
    logger.info("Loading HQ dataset: %s", args.hq_dataset)
    episodes_hq = load_episodes(args.hq_dataset, args.n_joints, args.max_episodes)
    logger.info("Loaded %d full episodes, %d HQ episodes", len(episodes_full), len(episodes_hq))

    # Step 1: length threshold + filter
    if args.min_episode_length is not None:
        threshold = args.min_episode_length
    else:
        threshold = auto_length_threshold(episodes_full, episodes_hq)
    logger.info("Episode length threshold: %d", threshold)

    plot_length_distribution(episodes_full, episodes_hq, threshold, out / "0_length_distribution.png")
    episodes_full = filter_episodes(episodes_full, threshold)
    episodes_hq = filter_episodes(episodes_hq, threshold)

    # Step 2: extract chunks
    chunks_full = extract_chunks(episodes_full, args.chunk_size, args.chunk_stride)
    chunks_hq = extract_chunks(episodes_hq, args.chunk_size, args.chunk_stride)
    logger.info("Extracted %d full chunks, %d HQ chunks", len(chunks_full), len(chunks_hq))

    # Step 3: fixed equal-width bands over episode-relative progress
    band_edges = make_bands(args.n_bands)
    n_bands = len(band_edges)
    logger.info("Progress bands (%d): %s", n_bands,
                [f"{lo:.0%}–{hi:.0%}" for lo, hi in band_edges])

    chunks_full = assign_bands(chunks_full, band_edges)
    chunks_hq = assign_bands(chunks_hq, band_edges)
    bands_full = split_by_band(chunks_full, n_bands)
    bands_hq = split_by_band(chunks_hq, n_bands)

    # Plot 1: chunk counts
    plot_chunk_counts(bands_full, bands_hq, band_edges, out / "1_chunk_counts_per_band.png")

    # Step 4: variance heatmap
    var_full = band_variance_matrix(bands_full, n_bands, args.n_joints)
    var_hq = band_variance_matrix(bands_hq, n_bands, args.n_joints)
    plot_variance_heatmap(var_full, var_hq, band_edges, out / "2_variance_heatmap.png")

    # Step 5: progress delta
    plot_progress_delta(bands_full, bands_hq, band_edges, out / "3_progress_delta_per_band.png")

    # Step 6: GMM BIC
    ks_full, ks_hq = plot_gmm_bic(bands_full, bands_hq, band_edges, args.seed, out / "4_gmm_bic_per_band.png")

    # Step 7: PCA scatter
    plot_pca_scatter(bands_full, bands_hq, band_edges, out / "5_pca_per_band.png")

    # Summary
    plot_summary(var_full, var_hq, band_edges, ks_full, ks_hq,
                 bands_full, bands_hq, out / "6_summary.png")

    logger.info("All figures saved to %s", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Chunk-level multi-modality analysis: Full/Mixed vs HQ dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--full-dataset", default="lerobot-data-collection/level12_rac_2_2026-02-08_1")
    p.add_argument("--hq-dataset", default="lerobot-data-collection/level2_final_quality3_trim_0_hil_data")
    p.add_argument("--output-dir", default="./chunk_analysis")
    p.add_argument("--chunk-size", type=int, default=30)
    p.add_argument("--chunk-stride", type=int, default=15)
    p.add_argument("--n-bands", type=int, default=5, help="Number of equal-width progress bands")
    p.add_argument("--max-episodes", type=int, default=500)
    p.add_argument("--n-joints", type=int, default=16)
    p.add_argument("--min-episode-length", type=int, default=None,
                   help="Override auto-detected length filter threshold")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
