#!/usr/bin/env python3
"""
Plot RTC Parameter Sweep Results

Visualizes the results from an RTC parameter sweep, showing:
- Heatmap of mean L2 distance vs (sigma_d, full_trajectory_alignment)
- Time series of L2 discrepancy for each configuration
- Summary statistics

Usage:
    uv run python examples/experiments/plot_rtc_sweep.py \
        --input results/rtc_sweep/ \
        --output results/rtc_sweep/analysis.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def load_experiment_data(csv_path: Path) -> pd.DataFrame | None:
    """Load and preprocess experiment CSV data.

    Returns:
        DataFrame with preprocessed data, or None if the file is empty.
    """
    df = pd.read_csv(csv_path)

    # Handle empty CSV files
    if len(df) == 0:
        return None

    # Normalize timestamps to start at 0
    df["t_relative"] = df["t"] - df["t"].iloc[0]

    # Convert L2 columns to numeric (they may have empty strings)
    for col in ["chunk_mean_l2", "chunk_max_l2", "chunk_overlap_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def extract_batch_and_config(filename: str) -> tuple[str | None, int | None]:
    """Extract batch number and config index from filename.

    Handles formats:
    - batch1_config0.csv -> ("1", 0)
    - rtc_sigma0.4_fulltrajFalse_20260110_180310.csv -> (None, None)
    """
    import re
    match = re.match(r"batch(\d+)_config(\d+)", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def load_all_results(
    input_dir: Path,
    batch_filter: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load all experiment results from a sweep directory.

    Args:
        input_dir: Directory containing experiment CSVs and config JSONs.
        batch_filter: If provided, only load results from this batch.

    Returns:
        summary_df: DataFrame with aggregate stats per configuration
        timeseries: Dict mapping config name to DataFrame with timeseries data
    """
    results = []
    timeseries = {}

    # Try both patterns: config JSONs and batch CSV files
    config_files = sorted(input_dir.glob("*_config.json"))
    batch_csvs = sorted(input_dir.glob("batch*_config*.csv"))

    # Process config JSON files (old rtc_sweep.py format)
    for config_file in config_files:
        with open(config_file) as f:
            config = json.load(f)

        csv_file = config_file.with_name(config_file.stem.replace("_config", "") + ".csv")
        if not csv_file.exists():
            print(f"  Skipping {config_file.name}: CSV not found")
            continue

        # Extract batch from config if present
        batch = config.get("batch", None)
        if batch_filter is not None and str(batch) != str(batch_filter):
            continue

        df = load_experiment_data(csv_file)
        if df is None:
            print(f"  Skipping {csv_file.name}: empty CSV")
            continue

        key = f"{config['name']}_b{batch}" if batch else config["name"]
        timeseries[key] = df

        df_chunks = df[df["chunk_mean_l2"].notna()]
        if len(df_chunks) == 0:
            print(f"  Skipping {csv_file.name}: no L2 data")
            continue

        results.append({
            "name": config["name"],
            "batch": batch,
            "sigma_d": config["rtc_sigma_d"],
            "full_traj_alignment": config["rtc_full_trajectory_alignment"],
            "mean_l2_avg": df_chunks["chunk_mean_l2"].mean(),
            "mean_l2_std": df_chunks["chunk_mean_l2"].std(),
            "mean_l2_max": df_chunks["chunk_mean_l2"].max(),
            "max_l2_max": df_chunks["chunk_max_l2"].max(),
            "chunk_count": len(df_chunks),
            "stall_count": df["stall"].sum() if "stall" in df.columns else 0,
            "stall_fraction": df["stall"].mean() if "stall" in df.columns else 0,
            "csv_file": csv_file.name,
        })

    # RTC sweep config mapping (same as in robot_client_improved.py)
    RTC_SWEEP_CONFIGS = [
        (0.1, False), (0.1, True),   # configs 0, 1
        (0.2, False), (0.2, True),   # configs 2, 3
        (0.4, False), (0.4, True),   # configs 4, 5
        (0.6, False), (0.6, True),   # configs 6, 7
        (0.8, False), (0.8, True),   # configs 8, 9
        (1.0, False), (1.0, True),   # configs 10, 11
    ]

    # Process batch CSV files (new env var format: batch1_config0.csv)
    for csv_file in batch_csvs:
        batch, config_idx = extract_batch_and_config(csv_file.stem)
        if batch is None:
            continue

        if batch_filter is not None and batch != batch_filter:
            continue

        # Get sigma_d and full_traj from config index
        if config_idx >= len(RTC_SWEEP_CONFIGS):
            print(f"  Skipping {csv_file.name}: invalid config index {config_idx}")
            continue

        sigma_d, full_traj = RTC_SWEEP_CONFIGS[config_idx]

        df = load_experiment_data(csv_file)
        if df is None:
            print(f"  Skipping {csv_file.name}: empty CSV")
            continue

        key = f"b{batch}_c{config_idx}"
        timeseries[key] = df

        df_chunks = df[df["chunk_mean_l2"].notna()]
        if len(df_chunks) == 0:
            print(f"  Skipping {csv_file.name}: no L2 data")
            continue

        results.append({
            "name": f"sigma{sigma_d}_fta{full_traj}",
            "batch": batch,
            "config_idx": config_idx,
            "sigma_d": sigma_d,
            "full_traj_alignment": full_traj,
            "mean_l2_avg": df_chunks["chunk_mean_l2"].mean(),
            "mean_l2_std": df_chunks["chunk_mean_l2"].std(),
            "mean_l2_max": df_chunks["chunk_mean_l2"].max(),
            "max_l2_max": df_chunks["chunk_max_l2"].max(),
            "chunk_count": len(df_chunks),
            "stall_count": df["stall"].sum() if "stall" in df.columns else 0,
            "stall_fraction": df["stall"].mean() if "stall" in df.columns else 0,
            "csv_file": csv_file.name,
        })

    if not results:
        raise ValueError("No valid results found in directory")

    return pd.DataFrame(results), timeseries


def plot_heatmap(ax, summary_df: pd.DataFrame):
    """Plot heatmap of mean L2 vs (sigma_d, full_trajectory_alignment)."""
    # Use pivot_table with aggfunc='mean' to handle duplicates (multiple batches)
    pivot = summary_df.pivot_table(
        index="full_traj_alignment",
        columns="sigma_d",
        values="mean_l2_avg",
        aggfunc="mean",
    )

    # Create custom colormap (green = good/low, red = bad/high)
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
    cmap = LinearSegmentedColormap.from_list("l2_cmap", colors)

    # Plot heatmap
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, label="Mean L2 Distance")

    # Set labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(["False", "True"] if len(pivot.index) == 2 else pivot.index)

    ax.set_xlabel("rtc_sigma_d")
    ax.set_ylabel("rtc_full_trajectory_alignment")
    ax.set_title("Mean L2 Distance by RTC Parameters")

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > pivot.values.mean() else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontsize=9)

    return im


def plot_timeseries(ax, timeseries: dict[str, pd.DataFrame], summary_df: pd.DataFrame):
    """Plot L2 discrepancy timeseries for each configuration."""
    # Sort by mean_l2_avg to show best configs first
    sorted_df = summary_df.sort_values("mean_l2_avg")

    colors = plt.cm.viridis(np.linspace(0, 1, len(timeseries)))

    for idx, (_, row) in enumerate(sorted_df.iterrows()):
        name = row["name"]
        if name not in timeseries:
            continue

        df = timeseries[name]
        df_chunks = df[df["chunk_mean_l2"].notna()]
        if len(df_chunks) == 0:
            continue

        label = f"σ={row['sigma_d']:.1f}, FTA={row['full_traj_alignment']}"
        ax.plot(
            df_chunks["t_relative"],
            df_chunks["chunk_mean_l2"],
            label=label,
            color=colors[idx],
            alpha=0.7,
            linewidth=1,
        )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean L2 Distance")
    ax.set_title("L2 Discrepancy Over Time by Configuration")
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_bar_comparison(ax, summary_df: pd.DataFrame):
    """Plot bar chart comparing configurations."""
    # Sort by mean_l2_avg
    sorted_df = summary_df.sort_values("mean_l2_avg")

    x = range(len(sorted_df))
    bars = ax.bar(x, sorted_df["mean_l2_avg"], yerr=sorted_df["mean_l2_std"], capsize=3)

    # Color best bar green
    bars[0].set_color("#2ecc71")

    # Labels
    labels = [f"σ={r['sigma_d']:.1f}\nFTA={r['full_traj_alignment']}"
              for _, r in sorted_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Mean L2 Distance")
    ax.set_title("Configurations Ranked by Mean L2 (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)


def plot_stall_vs_l2(ax, summary_df: pd.DataFrame):
    """Plot stall fraction vs L2 distance trade-off."""
    # Create scatter with different markers for full_traj_alignment
    for fta in [False, True]:
        subset = summary_df[summary_df["full_traj_alignment"] == fta]
        marker = "o" if not fta else "s"
        label = f"full_traj_alignment={fta}"

        scatter = ax.scatter(
            subset["stall_fraction"] * 100,
            subset["mean_l2_avg"],
            c=subset["sigma_d"],
            cmap="viridis",
            marker=marker,
            s=100,
            label=label,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Stall Fraction (%)")
    ax.set_ylabel("Mean L2 Distance")
    ax.set_title("Stall Rate vs Smoothness Trade-off")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add colorbar for sigma_d
    cbar = plt.colorbar(scatter, ax=ax, label="sigma_d")


def plot_results(input_path: Path, output_path: Path, batch_filter: str | None = None):
    """Generate comprehensive visualization of RTC sweep results."""
    print(f"Loading results from: {input_path}")
    if batch_filter:
        print(f"Filtering to batch: {batch_filter}")

    summary_df, timeseries = load_all_results(input_path, batch_filter=batch_filter)

    print(f"Found {len(summary_df)} configurations")

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 12))

    # 2x2 layout
    ax1 = fig.add_subplot(2, 2, 1)  # Heatmap
    ax2 = fig.add_subplot(2, 2, 2)  # Bar comparison
    ax3 = fig.add_subplot(2, 2, 3)  # Timeseries
    ax4 = fig.add_subplot(2, 2, 4)  # Stall vs L2

    # Generate plots
    plot_heatmap(ax1, summary_df)
    plot_bar_comparison(ax2, summary_df)
    plot_timeseries(ax3, timeseries, summary_df)
    plot_stall_vs_l2(ax4, summary_df)

    # Add title
    best = summary_df.loc[summary_df["mean_l2_avg"].idxmin()]
    batch_info = f" (Batch {batch_filter})" if batch_filter else ""
    fig.suptitle(
        f"RTC Parameter Sweep Analysis{batch_info}\n"
        f"Best: σ_d={best['sigma_d']:.1f}, full_traj_alignment={best['full_traj_alignment']} "
        f"(mean_L2={best['mean_l2_avg']:.4f})",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary (sorted by mean L2):")
    print("=" * 80)
    sorted_df = summary_df.sort_values("mean_l2_avg")
    print(sorted_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Plot RTC parameter sweep results")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/rtc_sweep"),
        help="Path to directory containing sweep results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the plot image (default: auto-generated in input dir)",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Filter to only show results from this batch number",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input path does not exist: {args.input}")
        return 1

    # Auto-generate output path if not specified
    if args.output is None:
        if args.batch:
            output_path = args.input / f"batch{args.batch}_analysis.png"
        else:
            output_path = args.input / "analysis.png"
    else:
        output_path = args.output

    plot_results(args.input, output_path, batch_filter=args.batch)
    return 0


if __name__ == "__main__":
    exit(main())
