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
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_experiment_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess experiment CSV data."""
    df = pd.read_csv(csv_path)

    # Normalize timestamps to start at 0
    df["t_relative"] = df["t"] - df["t"].iloc[0]

    # Calculate rolling stall rate (30-sample window ~= 1 second at 30fps)
    df["stall_rolling"] = df["stall"].rolling(window=30, min_periods=1).mean()

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


def plot_results(input_path: Path, output_path: Path):
    """Load CSV(s) and generate plots."""
    # Collect CSV files
    if input_path.is_file():
        csv_files = [input_path]
    elif input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {input_path}")
            return
    else:
        print(f"Input path does not exist: {input_path}")
        return

    print(f"Found {len(csv_files)} CSV file(s)")

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax_stall, ax_cooldown, ax_latency, ax_events = axes

    all_stats = []

    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}")
        df = load_experiment_data(csv_file)
        stats = plot_single_experiment(
            df,
            title=csv_file.stem,
            ax_stall=ax_stall,
            ax_cooldown=ax_cooldown,
            ax_latency=ax_latency,
            ax_events=ax_events,
        )
        stats["file"] = csv_file.name
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
            f"{csv_files[0].stem}\n"
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

    args = parser.parse_args()
    plot_results(args.input, args.output)


if __name__ == "__main__":
    main()
