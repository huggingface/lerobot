#!/usr/bin/env python3
"""
Plot results from latency-adaptive async inference experiments.

Usage:
    # Plot a single CSV file
    uv run python examples/experiments/plot_results.py \
        --input results/custom_custom_20260108_211827.csv

    # Plot all CSVs in a directory
    uv run python examples/experiments/plot_results.py \
        --input results/ \
        --output results/combined_plot.png

    # Compare estimators (filter by pattern)
    uv run python examples/experiments/plot_results.py \
        --input results/ \
        --filter "estimator_" \
        --mode estimator_comparison \
        --output results/estimator_comparison.png

    # Show latency spikes with measured RTT
    uv run python examples/experiments/plot_results.py \
        --input results/spike_experiment.csv \
        --mode detailed \
        --output results/spike_detailed.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Kandinsky-inspired color palette (from trajectory_viz.html)
CHUNK_COLORS = [
    "#c1272d",  # vermillion
    "#1a3a6e",  # ultramarine
    "#f4c430",  # cadmium yellow
    "#e85d04",  # orange
    "#5c3d6e",  # purple
    "#2d6a4f",  # deep green
    "#1a1a1a",  # black
    "#8b7355",  # ochre
    "#0077b6",  # cerulean
    "#9d4edd",  # violet
]


def load_trajectory_data(csv_path: Path) -> dict | None:
    """Load trajectory JSON data corresponding to a CSV file.

    The trajectory file is expected to be at the same path as the CSV
    but with a '.trajectory.json' suffix.
    """
    trajectory_path = csv_path.with_suffix(".trajectory.json")
    if not trajectory_path.exists():
        return None

    with open(trajectory_path) as f:
        return json.load(f)


def plot_trajectory_on_axis(
    ax,
    trajectory_data: dict,
    joint_idx: int = 0,
    time_offset: float = 0.0,
    fps: float = 30.0,
    max_chunks: int = 20,
    label_prefix: str = "",
):
    """Plot chunk trajectories and executed actions on a given axis.

    Args:
        ax: Matplotlib axis to plot on
        trajectory_data: Dict with 'chunks' and 'executed' lists
        joint_idx: Which joint/dimension to plot (default 0)
        time_offset: Time offset to align x-axis with other plots (seconds from start)
        fps: Frames per second for step-to-time conversion
        max_chunks: Maximum number of recent chunks to display
        label_prefix: Prefix for legend labels
    """
    if trajectory_data is None:
        ax.text(0.5, 0.5, "No trajectory data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        return

    chunks = trajectory_data.get("chunks", [])
    executed = trajectory_data.get("executed", [])

    if not chunks and not executed:
        ax.text(0.5, 0.5, "No trajectory data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        return

    # Calculate time offset from first executed action timestamp
    if not executed:
        ax.text(0.5, 0.5, "No executed actions", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        return

    t0 = min(e["t"] for e in executed)

    # Plot executed actions as scatter points using actual execution timestamps
    # Gaps in points show stalls (when no actions were executed)
    exec_times = [(e["t"] - t0) for e in executed if joint_idx < len(e["action"])]
    exec_values = [e["action"][joint_idx] for e in executed if joint_idx < len(e["action"])]
    if exec_times:
        ax.scatter(exec_times, exec_values, s=3, color="#1a1a1a",
                   label=f"{label_prefix}Executed" if label_prefix else "Executed", zorder=10)

    ax.set_ylabel(f"Joint {joint_idx}")
    ax.grid(True, alpha=0.3)


def load_experiment_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess experiment CSV data."""
    df = pd.read_csv(csv_path)

    # Normalize timestamps to start at 0
    df["t_relative"] = df["t"] - df["t"].iloc[0]

    # Calculate rolling stall rate (30-sample window ~= 1 second at 30fps)
    df["stall_rolling"] = df["stall"].rolling(window=30, min_periods=1).mean()

    # Convert measured_latency_ms to numeric (may have empty strings)
    if "measured_latency_ms" in df.columns:
        df["measured_latency_ms"] = pd.to_numeric(df["measured_latency_ms"], errors="coerce")

    # Convert L2 columns to numeric
    for col in ["chunk_mean_l2", "chunk_max_l2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_single_experiment(df: pd.DataFrame, title: str, ax_stall, ax_cooldown, ax_latency, ax_events):
    """Plot a single experiment's data across 4 subplots."""
    t = df["t_relative"]

    # Calculate summary stats
    total_ticks = len(df)
    stall_count = df["stall"].sum()
    stall_fraction = stall_count / total_ticks if total_ticks > 0 else 0
    obs_sent_count = df["obs_sent"].sum()
    action_received_count = df["action_received"].sum()

    # 1. Stall Rate (rolling)
    ax_stall.plot(t, df["stall_rolling"], linewidth=1, label=title)
    ax_stall.set_ylabel("Stall Rate (rolling)")
    ax_stall.set_ylim(-0.05, 1.05)
    ax_stall.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, linewidth=0.5)
    ax_stall.grid(True, alpha=0.3)

    # 2. Cooldown counter
    ax_cooldown.plot(t, df["cooldown"], linewidth=0.5, alpha=0.7, label=title)
    ax_cooldown.set_ylabel("Cooldown Counter")
    ax_cooldown.grid(True, alpha=0.3)

    # 3. Latency estimate
    ax_latency.plot(t, df["latency_estimate_steps"], linewidth=1, label=title)
    ax_latency.set_ylabel("Latency Est. (steps)")
    ax_latency.grid(True, alpha=0.3)

    # 4. Events timeline
    obs_times = t[df["obs_sent"] == 1]
    action_times = t[df["action_received"] == 1]

    ax_events.scatter(obs_times, [1] * len(obs_times), marker="|", s=20, alpha=0.7, label=f"obs sent ({obs_sent_count})")
    ax_events.scatter(action_times, [0] * len(action_times), marker="|", s=20, alpha=0.7, label=f"action recv ({action_received_count})")
    ax_events.set_ylabel("Events")
    ax_events.set_ylim(-0.5, 1.5)
    ax_events.set_yticks([0, 1])
    ax_events.set_yticklabels(["Action", "Obs"])
    ax_events.grid(True, alpha=0.3)

    return {
        "total_ticks": total_ticks,
        "stall_count": stall_count,
        "stall_fraction": stall_fraction,
        "obs_sent_count": obs_sent_count,
        "action_received_count": action_received_count,
    }


def plot_estimator_comparison(
    dfs: dict[str, pd.DataFrame],
    output_path: Path,
    csv_paths: dict[str, Path] | None = None,
):
    """Plot latency estimator comparison with detailed metrics.

    Expects dfs to be a dict mapping experiment name to DataFrame,
    where names contain 'jk' or 'max_last_10' to identify the estimator.

    Args:
        dfs: Dict mapping experiment name to DataFrame
        output_path: Path to save the plot image
        csv_paths: Optional dict mapping experiment name to CSV path (for loading trajectory data)
    """
    # Check if we have trajectory data
    trajectories: dict[str, dict | None] = {}
    if csv_paths:
        for name, csv_path in csv_paths.items():
            trajectories[name] = load_trajectory_data(csv_path)

    has_trajectory_data = any(t is not None for t in trajectories.values())

    # Create figure with extra rows for trajectory if available
    if has_trajectory_data:
        # 3 main plots + 2 trajectory plots (one for JK, one for Max10)
        fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1, 2, 2, 2]})
        ax_latency, ax_measured, ax_stall, ax_traj_jk, ax_traj_max = axes
    else:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        ax_latency, ax_measured, ax_stall = axes
        ax_traj_jk = None
        ax_traj_max = None

    colors = {"jk": "#2ecc71", "max_last_10": "#e74c3c"}

    for name, df in dfs.items():
        t = df["t_relative"]

        # Determine estimator type from name
        if "jk" in name.lower():
            estimator = "jk"
            label = f"JK: {name}"
        elif "max" in name.lower():
            estimator = "max_last_10"
            label = f"Max10: {name}"
        else:
            estimator = "jk"
            label = name

        color = colors.get(estimator, "#3498db")

        # Latency estimate
        ax_latency.plot(t, df["latency_estimate_steps"], linewidth=1.5, color=color, label=label)

        # Measured RTT as scatter on same axis (converted to steps assuming 30fps)
        if "measured_latency_ms" in df.columns:
            measured = df[df["measured_latency_ms"].notna()]
            if len(measured) > 0:
                # Convert ms to steps (assuming ~33ms per step at 30fps)
                measured_steps = measured["measured_latency_ms"] / 33.3
                ax_measured.scatter(
                    measured["t_relative"],
                    measured_steps,
                    s=10,
                    alpha=0.6,
                    color=color,
                    label=f"RTT ({estimator})",
                )

        # Stall rate
        ax_stall.plot(t, df["stall_rolling"], linewidth=1, color=color, alpha=0.8, label=label)

        # Plot trajectory data if available
        if has_trajectory_data and name in trajectories and trajectories[name] is not None:
            traj_ax = ax_traj_jk if estimator == "jk" else ax_traj_max
            if traj_ax is not None:
                plot_trajectory_on_axis(
                    traj_ax,
                    trajectories[name],
                    joint_idx=0,  # Plot joint 0 as representative
                    label_prefix=f"{estimator.upper()}: ",
                )
                traj_ax.set_title(f"Executed Actions: {estimator.upper()} (Joint 0)")

    ax_latency.set_ylabel("Latency Estimate (steps)")
    ax_latency.legend(loc="upper right")
    ax_latency.grid(True, alpha=0.3)
    ax_latency.set_title("Latency Estimation: JK vs Max-of-Last-10")

    ax_measured.set_ylabel("Measured RTT (steps)")
    ax_measured.legend(loc="upper right")
    ax_measured.grid(True, alpha=0.3)

    ax_stall.set_ylabel("Stall Rate (rolling)")
    ax_stall.set_ylim(-0.05, 1.05)
    ax_stall.legend(loc="upper right")
    ax_stall.grid(True, alpha=0.3)

    # Set x-axis label on the bottom-most plot
    if has_trajectory_data and ax_traj_max is not None:
        ax_traj_max.set_xlabel("Time (seconds)")
    else:
        ax_stall.set_xlabel("Time (seconds)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Estimator comparison plot saved to: {output_path}")


def plot_detailed(df: pd.DataFrame, title: str, output_path: Path):
    """Detailed plot for a single experiment with schedule size and L2 metrics."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    ax_schedule, ax_latency, ax_cooldown, ax_stall, ax_l2 = axes
    t = df["t_relative"]

    # 1. Schedule size
    if "schedule_size" in df.columns:
        ax_schedule.plot(t, df["schedule_size"], linewidth=1, color="#3498db")
        ax_schedule.set_ylabel("Schedule Size")
        ax_schedule.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=0.5)
        ax_schedule.grid(True, alpha=0.3)
        ax_schedule.set_title(f"Detailed Analysis: {title}")

    # 2. Latency estimate with measured RTT overlay
    ax_latency.plot(t, df["latency_estimate_steps"], linewidth=1.5, color="#2ecc71", label="Estimate")
    if "measured_latency_ms" in df.columns:
        measured = df[df["measured_latency_ms"].notna()]
        if len(measured) > 0:
            measured_steps = measured["measured_latency_ms"] / 33.3
            ax_latency.scatter(
                measured["t_relative"],
                measured_steps,
                s=15,
                alpha=0.7,
                color="#e74c3c",
                label="Measured RTT",
                zorder=5,
            )
    ax_latency.set_ylabel("Latency (steps)")
    ax_latency.legend(loc="upper right")
    ax_latency.grid(True, alpha=0.3)

    # 3. Cooldown counter
    ax_cooldown.plot(t, df["cooldown"], linewidth=0.5, color="#9b59b6", alpha=0.8)
    ax_cooldown.set_ylabel("Cooldown")
    ax_cooldown.grid(True, alpha=0.3)

    # 4. Stall indicator
    ax_stall.fill_between(t, 0, df["stall"], alpha=0.5, color="#e74c3c", step="mid")
    ax_stall.set_ylabel("Stall")
    ax_stall.set_ylim(-0.1, 1.1)
    ax_stall.grid(True, alpha=0.3)

    # 5. L2 discrepancy (if available)
    if "chunk_mean_l2" in df.columns:
        l2_data = df[df["chunk_mean_l2"].notna()]
        if len(l2_data) > 0:
            ax_l2.scatter(l2_data["t_relative"], l2_data["chunk_mean_l2"], s=10, alpha=0.7, color="#f39c12")
            ax_l2.set_ylabel("Chunk L2")
            ax_l2.grid(True, alpha=0.3)

    ax_l2.set_xlabel("Time (seconds)")

    # Summary stats
    total_ticks = len(df)
    stall_count = df["stall"].sum()
    stall_frac = stall_count / total_ticks if total_ticks > 0 else 0
    fig.suptitle(f"{title} | Stalls: {stall_count} ({stall_frac:.1%})", fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Detailed plot saved to: {output_path}")


def plot_results(input_path: Path, output_path: Path, mode: str = "basic", filter_pattern: str | None = None):
    """Load CSV(s) and generate plots.

    Args:
        input_path: Path to CSV file or directory
        output_path: Path to save the plot image
        mode: Plot mode - 'basic', 'detailed', or 'estimator_comparison'
        filter_pattern: Optional pattern to filter CSV files by name
    """
    # Collect CSV files
    if input_path.is_file():
        csv_files = [input_path]
    elif input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if filter_pattern:
            csv_files = [f for f in csv_files if filter_pattern in f.name]
        if not csv_files:
            print(f"No CSV files found in {input_path}" + (f" matching '{filter_pattern}'" if filter_pattern else ""))
            return
    else:
        print(f"Input path does not exist: {input_path}")
        return

    print(f"Found {len(csv_files)} CSV file(s)")

    # Load all data
    dfs = {}
    csv_paths = {}
    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}")
        dfs[csv_file.stem] = load_experiment_data(csv_file)
        csv_paths[csv_file.stem] = csv_file

    # Route to appropriate plotting function based on mode
    if mode == "estimator_comparison":
        plot_estimator_comparison(dfs, output_path, csv_paths=csv_paths)
        return
    elif mode == "detailed" and len(csv_files) == 1:
        plot_detailed(list(dfs.values())[0], list(dfs.keys())[0], output_path)
        return

    # Default: basic multi-experiment comparison
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax_stall, ax_cooldown, ax_latency, ax_events = axes

    all_stats = []

    for name, df in dfs.items():
        stats = plot_single_experiment(
            df,
            title=name,
            ax_stall=ax_stall,
            ax_cooldown=ax_cooldown,
            ax_latency=ax_latency,
            ax_events=ax_events,
        )
        stats["file"] = name
        all_stats.append(stats)

    # Set common x-axis label
    ax_events.set_xlabel("Time (seconds)")

    # Add legends
    if len(csv_files) > 1:
        ax_stall.legend(loc="upper right", fontsize=8)

    ax_events.legend(loc="upper right", fontsize=8)

    # Create title with summary stats
    if len(csv_files) == 1:
        s = all_stats[0]
        title = (
            f"{list(dfs.keys())[0]}\n"
            f"Ticks: {s['total_ticks']} | Stalls: {s['stall_count']} ({s['stall_fraction']:.1%}) | "
            f"Obs Sent: {s['obs_sent_count']} | Actions Recv: {s['action_received_count']}"
        )
    else:
        title = f"Experiment Comparison ({len(csv_files)} files)"

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Print summary table
    print("\nSummary:")
    print("-" * 80)
    print(f"{'File':<40} {'Ticks':>8} {'Stalls':>8} {'Stall%':>8} {'ObsSent':>8} {'ActRecv':>8}")
    print("-" * 80)
    for s in all_stats:
        print(
            f"{s['file']:<40} {s['total_ticks']:>8} {s['stall_count']:>8} "
            f"{s['stall_fraction']:>7.1%} {s['obs_sent_count']:>8} {s['action_received_count']:>8}"
        )

    # Show plot interactively if not in headless mode
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot latency-adaptive experiment results")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to CSV file or directory containing CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plot.png"),
        help="Output path for the plot image (default: results/plot.png)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "detailed", "estimator_comparison"],
        default="basic",
        help="Plot mode: basic (default), detailed (single file), or estimator_comparison",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter CSV files by pattern (e.g., 'estimator_' to only plot estimator experiments)",
    )

    args = parser.parse_args()
    plot_results(args.input, args.output, mode=args.mode, filter_pattern=args.filter)


if __name__ == "__main__":
    main()
