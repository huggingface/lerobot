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
Analyze the quality of a LeRobot dataset and produce a per-episode quality report.

Scores episodes on multiple dimensions (dead frames, smoothness, consistency,
coverage, temporal) and gives actionable recommendations for which episodes
to consider removing before training.

Usage Examples:

Analyze a dataset from the Hub:
    lerobot-analyze-dataset --repo_id lerobot/pusht

Analyze specific episodes:
    lerobot-analyze-dataset --repo_id lerobot/pusht --episodes "[0, 1, 2]"

Analyze a local dataset and write JSON report:
    lerobot-analyze-dataset --repo_id lerobot/pusht --root /path/to/data \
        --output_json report.json

Disable specific analyzers:
    lerobot-analyze-dataset --repo_id lerobot/pusht --run_consistency false
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AnalyzeDatasetConfig:
    repo_id: str = ""
    root: str | None = None
    episodes: list[int] | None = None

    # Which analyzers to run
    run_dead_frames: bool = True
    run_smoothness: bool = True
    run_consistency: bool = True
    run_coverage: bool = True
    run_temporal: bool = True

    # Thresholds
    dead_frame_threshold: float = 1e-4  # base threshold at reference_fps (50 Hz)
    dead_frame_reference_fps: int = 50  # threshold scales by reference_fps/actual_fps
    consistency_outlier_std: float = 2.0
    min_episode_length_ratio: float = 0.3

    # Output
    output_json: str | None = None
    top_k_worst: int = 5


# ---------------------------------------------------------------------------
# Analyzer: Dead Frames
# ---------------------------------------------------------------------------


def analyze_dead_frames(actions: np.ndarray, threshold: float) -> dict:
    """Detect frames where action change is negligible.

    Args:
        actions: (T, action_dim) array of actions.
        threshold: Maximum L2 norm of action delta to count as "dead".

    Returns:
        Dict with dead_frame_count, dead_frame_ratio, dead_frame_indices,
        longest_dead_streak, and a 0-1 score (higher = fewer dead frames).
    """
    if len(actions) < 2:
        return {
            "dead_frame_count": 0,
            "dead_frame_ratio": 0.0,
            "dead_frame_indices": [],
            "longest_dead_streak": 0,
            "score": 1.0,
        }

    deltas = np.diff(actions, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    dead_mask = norms < threshold

    dead_indices = np.where(dead_mask)[0].tolist()
    dead_count = len(dead_indices)
    dead_ratio = dead_count / len(norms)

    # Longest consecutive streak
    longest_streak = 0
    current_streak = 0
    for is_dead in dead_mask:
        if is_dead:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0

    score = 1.0 - dead_ratio

    return {
        "dead_frame_count": dead_count,
        "dead_frame_ratio": float(dead_ratio),
        "dead_frame_indices": dead_indices,
        "longest_dead_streak": longest_streak,
        "score": float(score),
    }


# ---------------------------------------------------------------------------
# Analyzer: Smoothness (Delta CV)
# ---------------------------------------------------------------------------


def analyze_smoothness(actions: np.ndarray, fps: int) -> dict:
    """Measure motion smoothness via delta consistency and direction coherence.

    Combines two complementary signals:
    1. **Delta CV** — coefficient of variation of frame-to-frame step
       sizes. High CV = erratic speed changes.
    2. **Reversal ratio** — fraction of consecutive frame pairs where
       the motion direction reverses (cosine similarity < 0). High
       ratio = oscillatory / sawtooth motion.

    The final score is ``exp(-delta_cv / 2) * (1 - reversal_ratio)``.

    A secondary mean-jerk value is also reported for the JSON output.

    Args:
        actions: (T, action_dim) array.
        fps: Dataset frames per second.

    Returns:
        Dict with delta statistics and a 0-1 smoothness score.
    """
    if len(actions) < 2:
        return {
            "delta_mean": 0.0,
            "delta_std": 0.0,
            "delta_cv": 0.0,
            "reversal_ratio": 0.0,
            "mean_jerk": 0.0,
            "smoothness_score": 1.0,
        }

    # Delta magnitudes
    diffs = np.diff(actions, axis=0)  # (T-1, D)
    deltas = np.linalg.norm(diffs, axis=1)  # (T-1,)
    delta_mean = float(np.mean(deltas))
    delta_std = float(np.std(deltas))
    delta_cv = delta_std / (delta_mean + 1e-10)

    # Direction reversals: fraction of consecutive delta pairs with negative dot product
    reversal_ratio = 0.0
    if len(diffs) >= 2:
        dot_products = np.sum(diffs[:-1] * diffs[1:], axis=1)  # (T-2,)
        reversal_ratio = float(np.mean(dot_products < 0))

    # Combined score — exp(-cv/2) so that a typical CV of ~1.0 yields ~0.61
    cv_score = float(np.exp(-delta_cv / 2))
    direction_score = 1.0 - reversal_ratio
    smoothness_score = float(np.clip(cv_score * direction_score, 0.0, 1.0))

    # Secondary: mean jerk magnitude (for the JSON report)
    mean_jerk = 0.0
    if len(actions) >= 4:
        dt = 1.0 / fps if fps > 0 else 1.0
        velocity = np.diff(actions, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt
        jerk = np.diff(acceleration, axis=0) / dt
        jerk_mag = np.linalg.norm(jerk, axis=1)
        mean_jerk = float(np.mean(jerk_mag))

    return {
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_cv": delta_cv,
        "reversal_ratio": reversal_ratio,
        "mean_jerk": mean_jerk,
        "smoothness_score": smoothness_score,
    }


# ---------------------------------------------------------------------------
# Analyzer: Consistency (Inter-Episode Distance)
# ---------------------------------------------------------------------------


def _resample(trajectory: np.ndarray, target_length: int) -> np.ndarray:
    """Resample trajectory to target_length via linear interpolation."""
    t_orig = np.linspace(0, 1, len(trajectory))
    t_new = np.linspace(0, 1, target_length)
    resampled = np.stack(
        [np.interp(t_new, t_orig, trajectory[:, d]) for d in range(trajectory.shape[1])],
        axis=1,
    )
    return resampled


def analyze_consistency(all_episode_actions: list[np.ndarray], outlier_std: float) -> dict:
    """Measure how similar episodes are to each other.

    Compares each episode to the mean trajectory after resampling to the
    median length. O(N) complexity.

    Args:
        all_episode_actions: List of (T_i, action_dim) arrays.
        outlier_std: Episodes whose distance from the mean exceeds
            mean_dist + outlier_std * std_dist are flagged.

    Returns:
        Dict with per-episode distances, outlier indices, and a 0-1 score.
    """
    n = len(all_episode_actions)
    if n < 2:
        return {
            "mean_pairwise_distance": 0.0,
            "per_episode_mean_distance": [0.0] * n,
            "outlier_episodes": [],
            "consistency_score": 1.0,
        }

    lengths = [len(a) for a in all_episode_actions]
    target_len = int(np.median(lengths))
    target_len = max(target_len, 1)

    resampled = np.stack([_resample(a, target_len) for a in all_episode_actions])  # (N, T, D)
    mean_traj = np.mean(resampled, axis=0)  # (T, D)

    distances = [float(np.mean(np.linalg.norm(resampled[i] - mean_traj, axis=1))) for i in range(n)]

    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))

    threshold = mean_dist + outlier_std * std_dist
    outliers = [i for i, d in enumerate(distances) if d > threshold]

    # Score: fraction of non-outlier episodes, adjusted by distance spread
    if std_dist < 1e-8:
        consistency_score = 1.0
    else:
        outlier_ratio = len(outliers) / n
        consistency_score = float(
            np.clip(1.0 - outlier_ratio - 0.1 * (std_dist / (mean_dist + 1e-8)), 0.0, 1.0)
        )

    return {
        "mean_pairwise_distance": mean_dist,
        "per_episode_mean_distance": distances,
        "outlier_episodes": outliers,
        "consistency_score": consistency_score,
    }


# ---------------------------------------------------------------------------
# Analyzer: Coverage
# ---------------------------------------------------------------------------


def analyze_coverage(all_episode_states: list[np.ndarray]) -> dict:
    """Analyze state-space coverage across episodes.

    Args:
        all_episode_states: List of (T_i, state_dim) arrays.

    Returns:
        Dict with per-dimension range, std, and a 0-1 utilization score.
    """
    all_states = np.concatenate(all_episode_states, axis=0)

    mins = all_states.min(axis=0)
    maxs = all_states.max(axis=0)
    ranges = maxs - mins
    stds = all_states.std(axis=0)

    # Utilization: how much of the range each dimension's std covers.
    # A uniform distribution has std ≈ range / sqrt(12) ≈ 0.289 * range.
    # Score per dim = min(1, std / (0.289 * range + eps)).
    ideal_std = 0.289 * ranges
    per_dim_utilization = np.clip(stds / (ideal_std + 1e-8), 0.0, 1.0)
    coverage_score = float(np.mean(per_dim_utilization))

    return {
        "state_dim_ranges": [(float(lo), float(hi)) for lo, hi in zip(mins, maxs, strict=True)],
        "per_dim_std": stds.tolist(),
        "per_dim_utilization": per_dim_utilization.tolist(),
        "coverage_score": float(coverage_score),
    }


# ---------------------------------------------------------------------------
# Analyzer: Temporal
# ---------------------------------------------------------------------------


def analyze_temporal(
    episode_lengths: list[int],
    fps: int,
    min_episode_length_ratio: float,
) -> dict:
    """Analyze episode length distribution.

    Args:
        episode_lengths: Number of frames per episode.
        fps: Dataset fps.
        min_episode_length_ratio: Episodes shorter than this fraction of the
            median (or longer than 1/ratio * median) are flagged.

    Returns:
        Dict with length stats, flagged episodes, and a 0-1 score.
    """
    lengths = np.array(episode_lengths)
    median_len = float(np.median(lengths))
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))

    short_thresh = min_episode_length_ratio * median_len
    long_thresh = median_len / min_episode_length_ratio if min_episode_length_ratio > 0 else float("inf")

    short_episodes = [int(i) for i, length in enumerate(lengths) if length < short_thresh]
    long_episodes = [int(i) for i, length in enumerate(lengths) if length > long_thresh]

    flagged_ratio = (len(short_episodes) + len(long_episodes)) / max(len(lengths), 1)
    # Also penalize high coefficient of variation
    cv = std_len / (mean_len + 1e-8)
    temporal_score = float(np.clip(1.0 - flagged_ratio - 0.2 * cv, 0.0, 1.0))

    durations = (lengths / fps).tolist() if fps > 0 else lengths.tolist()

    return {
        "lengths": lengths.tolist(),
        "durations_s": durations,
        "mean_length": mean_len,
        "median_length": median_len,
        "std_length": std_len,
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "short_episodes": short_episodes,
        "long_episodes": long_episodes,
        "temporal_score": temporal_score,
    }


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS = {
    "dead_frames": 0.25,
    "smoothness": 0.20,
    "consistency": 0.25,
    "coverage": 0.10,
    "temporal": 0.20,
}


def compute_composite_score(dimension_scores: dict[str, float]) -> float:
    """Weighted average of per-dimension 0-1 scores."""
    total = sum(
        DIMENSION_WEIGHTS[k] * dimension_scores[k] for k in DIMENSION_WEIGHTS if k in dimension_scores
    )
    weight_sum = sum(DIMENSION_WEIGHTS[k] for k in DIMENSION_WEIGHTS if k in dimension_scores)
    if weight_sum == 0:
        return 0.0
    return total / weight_sum


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def _score_status(score: float, detail: str = "") -> str:
    if score >= 0.8:
        return "OK"
    elif score >= 0.5:
        msg = "WARNING"
        if detail:
            msg += f" - {detail}"
        return msg
    else:
        msg = "POOR"
        if detail:
            msg += f" - {detail}"
        return msg


def print_report(
    repo_id: str,
    total_episodes: int,
    total_frames: int,
    fps: int,
    overall_score: float,
    dimension_scores: dict[str, float],
    dimension_details: dict[str, str],
    worst_episodes: list[dict],
    recommendations: list[str],
) -> None:
    """Print a formatted quality report to the terminal."""
    w = 66

    def line(text: str = "") -> str:
        return f"  {text}"

    lines = []
    lines.append("=" * w)
    lines.append(line("Dataset Quality Report"))
    lines.append(line(f"repo_id: {repo_id}"))
    lines.append(line(f"episodes: {total_episodes}  |  frames: {total_frames}  |  fps: {fps}"))
    lines.append("=" * w)
    lines.append("")
    lines.append(line(f"OVERALL SCORE: {overall_score:.2f} / 1.00"))
    lines.append("")

    # Dimension table
    hdr = f"  {'Dimension':<20s} {'Score':>5s}  {'Status'}"
    lines.append(hdr)
    lines.append("  " + "-" * (w - 4))
    for dim_key, label in [
        ("dead_frames", "Dead Frames"),
        ("smoothness", "Smoothness"),
        ("consistency", "Consistency"),
        ("coverage", "Coverage"),
        ("temporal", "Temporal"),
    ]:
        if dim_key in dimension_scores:
            s = dimension_scores[dim_key]
            detail = dimension_details.get(dim_key, "")
            status = _score_status(s, detail)
            lines.append(f"  {label:<20s} {s:>5.2f}  {status}")
    lines.append("")

    if worst_episodes:
        lines.append(line("WORST EPISODES:"))
        for ep in worst_episodes:
            flags = ", ".join(ep.get("flags", []))
            flag_str = f"  [{flags}]" if flags else ""
            lines.append(line(f"  #{ep['episode_index']:<4d} score={ep['composite_score']:.2f}{flag_str}"))
        lines.append("")

    if recommendations:
        lines.append(line("RECOMMENDATIONS:"))
        for rec in recommendations:
            lines.append(line(f"  - {rec}"))
        lines.append("")

    lines.append("=" * w)
    print("\n".join(lines))


def build_json_report(
    repo_id: str,
    total_episodes: int,
    total_frames: int,
    fps: int,
    overall_score: float,
    dimension_scores: dict[str, float],
    episode_results: list[dict],
    recommendations: list[str],
) -> dict:
    """Build a JSON-serializable report dict."""
    return {
        "repo_id": repo_id,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": fps,
        "overall_score": overall_score,
        "dimension_scores": dimension_scores,
        "episodes": episode_results,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(data) -> np.ndarray:
    """Convert various data types returned by HF datasets to a 2-D numpy array."""
    if isinstance(data, torch.Tensor):
        return data.numpy()
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, list):
        # List of tensors, list of lists, etc.
        return np.array([t.numpy() if isinstance(t, torch.Tensor) else t for t in data])
    return np.array(data)


# ---------------------------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------------------------


@parser.wrap()
def analyze_dataset(cfg: AnalyzeDatasetConfig) -> dict:
    if not cfg.repo_id:
        raise ValueError("repo_id must be specified")

    logger.info(f"Loading dataset {cfg.repo_id}...")
    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root)
    fps = dataset.meta.fps
    total_episodes = dataset.meta.total_episodes
    total_frames = dataset.meta.total_frames

    # Determine episodes to analyze
    ep_indices = cfg.episodes if cfg.episodes is not None else list(range(total_episodes))
    analyzing_subset = cfg.episodes is not None

    logger.info(f"Analyzing {len(ep_indices)} episodes (fps={fps})...")

    # ------------------------------------------------------------------
    # Extract per-episode data from parquet (no video decode)
    # ------------------------------------------------------------------
    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    episode_lengths: list[int] = []
    has_state = "observation.state" in dataset.meta.features

    for ep_idx in ep_indices:
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
        length = to_idx - from_idx
        episode_lengths.append(length)

        rows = dataset.hf_dataset[from_idx:to_idx]
        actions = _to_numpy(rows["action"])
        all_actions.append(actions)

        if has_state:
            states = _to_numpy(rows["observation.state"])
            all_states.append(states)

    # ------------------------------------------------------------------
    # Run per-episode analyzers
    # ------------------------------------------------------------------
    # Scale dead-frame threshold by reference_fps / actual_fps.  At low fps
    # each frame spans more wall-clock time, so a truly idle robot is more
    # suspicious — we use a *higher* threshold to be stricter about motion.
    dead_threshold = cfg.dead_frame_threshold
    if cfg.run_dead_frames:
        fps_scale = cfg.dead_frame_reference_fps / max(fps, 1)
        dead_threshold = cfg.dead_frame_threshold * fps_scale

    episode_results: list[dict] = []
    for i, ep_idx in enumerate(ep_indices):
        result: dict = {"episode_index": int(ep_idx), "length": episode_lengths[i], "flags": []}

        if cfg.run_dead_frames:
            df_result = analyze_dead_frames(all_actions[i], dead_threshold)
            result["dead_frames"] = df_result
            if df_result["dead_frame_ratio"] > 0.3:
                result["flags"].append(f"dead_frames: {df_result['dead_frame_ratio']:.0%}")

        if cfg.run_smoothness:
            sm_result = analyze_smoothness(all_actions[i], fps)
            result["smoothness"] = sm_result

        episode_results.append(result)

    # ------------------------------------------------------------------
    # Run dataset-level analyzers
    # ------------------------------------------------------------------
    consistency_result = None
    coverage_result = None
    temporal_result = None

    if cfg.run_consistency and len(all_actions) >= 2:
        logger.info("Running consistency analysis...")
        consistency_result = analyze_consistency(all_actions, cfg.consistency_outlier_std)
        # Annotate per-episode results
        for i, ep_res in enumerate(episode_results):
            ep_res["consistency_distance"] = consistency_result["per_episode_mean_distance"][i]
            if i in consistency_result["outlier_episodes"]:
                ep_res["flags"].append("consistency: outlier")
    elif cfg.run_consistency and len(all_actions) < 2:
        logger.info("Skipping consistency analysis (need >= 2 episodes).")

    if cfg.run_coverage and all_states:
        logger.info("Running coverage analysis...")
        coverage_result = analyze_coverage(all_states)
    elif cfg.run_coverage and not all_states:
        logger.info("Skipping coverage analysis (no observation.state found).")

    if cfg.run_temporal:
        logger.info("Running temporal analysis...")
        temporal_result = analyze_temporal(episode_lengths, fps, cfg.min_episode_length_ratio)
        # Annotate per-episode results
        for i in temporal_result.get("short_episodes", []):
            if i < len(episode_results):
                episode_results[i]["flags"].append(
                    f"temporal: short ({episode_lengths[i]} frames vs median {temporal_result['median_length']:.0f})"
                )
        for i in temporal_result.get("long_episodes", []):
            if i < len(episode_results):
                episode_results[i]["flags"].append(
                    f"temporal: long ({episode_lengths[i]} frames vs median {temporal_result['median_length']:.0f})"
                )

    # ------------------------------------------------------------------
    # Compute per-episode composite scores
    # ------------------------------------------------------------------
    for ep_res in episode_results:
        ep_dim_scores: dict[str, float] = {}

        if "dead_frames" in ep_res:
            ep_dim_scores["dead_frames"] = ep_res["dead_frames"]["score"]
        if "smoothness" in ep_res:
            ep_dim_scores["smoothness"] = ep_res["smoothness"]["smoothness_score"]
        if consistency_result is not None:
            # Per-episode consistency score: inverse of normalized distance
            dist = ep_res.get("consistency_distance", 0.0)
            max_dist = (
                max(consistency_result["per_episode_mean_distance"])
                if consistency_result["per_episode_mean_distance"]
                else 1.0
            )
            ep_dim_scores["consistency"] = float(np.clip(1.0 - dist / (max_dist + 1e-8), 0.0, 1.0))
        if coverage_result is not None:
            ep_dim_scores["coverage"] = coverage_result["coverage_score"]
        if temporal_result is not None:
            # Per-episode temporal score based on how close length is to median
            median_len = temporal_result["median_length"]
            ep_len = ep_res["length"]
            ratio = min(ep_len, median_len) / max(ep_len, median_len) if median_len > 0 else 1.0
            ep_dim_scores["temporal"] = float(ratio)

        ep_res["composite_score"] = compute_composite_score(ep_dim_scores)

    # ------------------------------------------------------------------
    # Dataset-level dimension scores
    # ------------------------------------------------------------------
    dimension_scores: dict[str, float] = {}
    dimension_details: dict[str, str] = {}

    if cfg.run_dead_frames:
        scores = [r["dead_frames"]["score"] for r in episode_results if "dead_frames" in r]
        dimension_scores["dead_frames"] = float(np.mean(scores)) if scores else 1.0
        high_dead = sum(
            1 for r in episode_results if "dead_frames" in r and r["dead_frames"]["dead_frame_ratio"] > 0.3
        )
        if high_dead:
            dimension_details["dead_frames"] = f"{high_dead} episodes >30% dead"

    if cfg.run_smoothness:
        scores = [r["smoothness"]["smoothness_score"] for r in episode_results if "smoothness" in r]
        dimension_scores["smoothness"] = float(np.mean(scores)) if scores else 1.0
        cvs = [r["smoothness"]["delta_cv"] for r in episode_results if "smoothness" in r]
        if cvs:
            median_cv = float(np.median(cvs))
            high_cv = sum(1 for cv in cvs if cv > 2.0 * median_cv)
            if high_cv:
                dimension_details["smoothness"] = f"{high_cv} episodes with erratic motion"

    if consistency_result is not None:
        dimension_scores["consistency"] = consistency_result["consistency_score"]
        n_outliers = len(consistency_result["outlier_episodes"])
        if n_outliers:
            dimension_details["consistency"] = f"{n_outliers} outlier episodes"

    if coverage_result is not None:
        dimension_scores["coverage"] = coverage_result["coverage_score"]

    if temporal_result is not None:
        dimension_scores["temporal"] = temporal_result["temporal_score"]
        n_short = len(temporal_result["short_episodes"])
        n_long = len(temporal_result["long_episodes"])
        parts = []
        if n_short:
            parts.append(f"{n_short} short")
        if n_long:
            parts.append(f"{n_long} long")
        if parts:
            dimension_details["temporal"] = " + ".join(parts) + " episodes"

    overall_score = compute_composite_score(dimension_scores)

    # Use analyzed frames (not total dataset frames) when analyzing a subset
    analyzed_frames = sum(episode_lengths) if analyzing_subset else total_frames

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    recommendations: list[str] = []
    episodes_to_remove: list[int] = []

    # Collect episodes to potentially remove
    for ep_res in episode_results:
        if ep_res["composite_score"] < 0.4:
            episodes_to_remove.append(ep_res["episode_index"])

    if episodes_to_remove:
        recommendations.append(f"Consider removing episodes: {episodes_to_remove}")

    if consistency_result and consistency_result["outlier_episodes"]:
        outlier_ep_indices = [ep_indices[i] for i in consistency_result["outlier_episodes"]]
        recommendations.append(f"{len(outlier_ep_indices)} episodes are outliers in action consistency")

    high_dead_eps = [
        r["episode_index"]
        for r in episode_results
        if "dead_frames" in r and r["dead_frames"]["dead_frame_ratio"] > 0.3
    ]
    if high_dead_eps:
        recommendations.append(f"{len(high_dead_eps)} episodes have >30% dead frames")

    if temporal_result and temporal_result["short_episodes"]:
        short_ep_indices = [ep_indices[i] for i in temporal_result["short_episodes"]]
        recommendations.append(f"{len(short_ep_indices)} episodes are suspiciously short")

    if not recommendations:
        recommendations.append("Dataset looks good! No major quality issues detected.")

    # ------------------------------------------------------------------
    # Worst episodes
    # ------------------------------------------------------------------
    sorted_eps = sorted(episode_results, key=lambda r: r["composite_score"])
    worst_episodes = sorted_eps[: cfg.top_k_worst]

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print_report(
        repo_id=cfg.repo_id,
        total_episodes=len(ep_indices),
        total_frames=analyzed_frames,
        fps=fps,
        overall_score=overall_score,
        dimension_scores=dimension_scores,
        dimension_details=dimension_details,
        worst_episodes=worst_episodes,
        recommendations=recommendations,
    )

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    report = build_json_report(
        repo_id=cfg.repo_id,
        total_episodes=len(ep_indices),
        total_frames=analyzed_frames,
        fps=fps,
        overall_score=overall_score,
        dimension_scores=dimension_scores,
        episode_results=episode_results,
        recommendations=recommendations,
    )

    if cfg.output_json:
        output_path = Path(cfg.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types for JSON serialization
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"JSON report written to {output_path}")

    return report


def main() -> None:
    init_logging()
    analyze_dataset()


if __name__ == "__main__":
    main()
