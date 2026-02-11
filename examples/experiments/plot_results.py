#!/usr/bin/env python3
"""
Plot results from latency-adaptive async inference experiments.

Usage:
    # Plot an experiment directory (finds CSV + trajectory JSON automatically)
    uv run python examples/experiments/plot_results.py \
        --input results/experiments/drop_obs_00

    # Plot a single CSV file
    uv run python examples/experiments/plot_results.py \
        --input results/experiments/drop_obs_00/drop_obs_00.csv

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


# Colors for simulation event types
_SIM_EVENT_COLORS = {
    "obs_dropped": "#c1272d",       # vermillion
    "obs_reorder_held": "#f4c430",  # cadmium yellow
    "obs_reorder_swapped": "#e85d04",  # orange
    "obs_duplicated": "#5c3d6e",    # purple
    "action_dropped": "#1a3a6e",    # ultramarine
    "action_reorder_held": "#0077b6",  # cerulean
    "action_reorder_swapped": "#2d6a4f",  # deep green
    "action_duplicated": "#9d4edd", # violet
}

# Y-position for each event type so they don't overlap
_SIM_EVENT_YPOS = {
    "obs_dropped": 7,
    "obs_reorder_held": 6,
    "obs_reorder_swapped": 5,
    "obs_duplicated": 4,
    "action_dropped": 3,
    "action_reorder_held": 2,
    "action_reorder_swapped": 1,
    "action_duplicated": 0,
}

# Shading colors for configured simulation windows
_SIM_CONFIG_COLORS = {
    "drop_obs": ("#c1272d", 0.12),
    "drop_action": ("#1a3a6e", 0.12),
    "dup_obs": ("#5c3d6e", 0.10),
    "dup_action": ("#9d4edd", 0.10),
    "reorder_obs": ("#f4c430", 0.10),
    "reorder_action": ("#2d6a4f", 0.10),
}


def plot_sim_events_on_axis(
    ax, trajectory_data: dict, sim_config_offset: float = 0.0,
) -> None:
    """Plot simulation events timeline on a given axis.

    Shows actual recorded sim events as scatter markers and overlays
    configured simulation windows (from ``simulation_config``) as shaded
    regions.  Spike events are shown as vertical dashed lines.

    Args:
        ax: Matplotlib axis.
        trajectory_data: Trajectory JSON dict.
        sim_config_offset: Seconds between experiment start (CSV t0) and
            the first executed action (trajectory t0).  The ``start_s``
            values in the simulation config are relative to experiment
            start, so we subtract this offset to align them with the
            trajectory t0 used by all other subplots.
    """
    sim_events = trajectory_data.get("sim_events", [])
    sim_config = trajectory_data.get("simulation_config", {})
    executed = trajectory_data.get("executed", [])

    if not sim_events and not sim_config:
        ax.text(
            0.5, 0.5, "No simulation events",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        return

    # Derive t0 from executed actions (same baseline as trajectory plot)
    t0 = min(e["t"] for e in executed) if executed else 0.0

    # --- Overlay configured windows as shaded regions ---
    # start_s is relative to experiment start; subtract the warmup offset
    # so the shaded regions align with the trajectory t0 baseline.
    for config_key, (color, alpha) in _SIM_CONFIG_COLORS.items():
        windows = sim_config.get(config_key, [])
        for i, w in enumerate(windows):
            start = w.get("start_s", 0) - sim_config_offset
            dur = w.get("duration_s", 0)
            label = config_key.replace("_", " ") if i == 0 else None
            ax.axvspan(start, start + dur, alpha=alpha, color=color, label=label)

    # Overlay spike config as vertical dashed lines
    spikes = sim_config.get("spikes", [])
    for i, spike in enumerate(spikes):
        start = spike.get("start_s", 0) - sim_config_offset
        delay_ms = spike.get("delay_ms", 0)
        label = f"spike ({delay_ms}ms)" if i == 0 else None
        ax.axvline(start, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.7, label=label)

    # --- Plot actual recorded events ---
    # Group events by type
    events_by_type: dict[str, list[float]] = {}
    for ev in sim_events:
        etype = ev["event_type"]
        t_rel = ev["t"] - t0
        events_by_type.setdefault(etype, []).append(t_rel)

    active_types = [k for k in _SIM_EVENT_YPOS if k in events_by_type]

    for etype, times in events_by_type.items():
        y = _SIM_EVENT_YPOS.get(etype, 0)
        color = _SIM_EVENT_COLORS.get(etype, "#888888")
        ax.scatter(
            times, [y] * len(times),
            marker="|", s=30, color=color, alpha=0.8,
            label=f"{etype} ({len(times)})",
        )

    # Configure axis
    if active_types:
        ax.set_yticks([_SIM_EVENT_YPOS[k] for k in active_types])
        ax.set_yticklabels([k.replace("_", " ") for k in active_types], fontsize=7)
    ax.set_ylabel("Sim Events")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_register_state(
    ax,
    trajectory_data: dict,
) -> None:
    """Plot LWW register writes over time, showing control_step for each register.

    Accepted writes are shown as filled markers; rejected writes as hollow
    markers with an ``x``.  For the ``client_action`` register the
    ``chunk_start_step`` is plotted as a secondary series.

    Falls back to the legacy ``executed`` provenance data when
    ``register_events`` is not present in the trajectory JSON, so older
    experiment files still render.

    Args:
        ax: Matplotlib axis.
        trajectory_data: Trajectory JSON dict (must contain ``register_events``).
    """
    events = trajectory_data.get("register_events", [])

    # ---- Fallback: legacy trajectory without register_events ----
    if not events:
        executed = trajectory_data.get("executed", [])
        if not executed:
            ax.text(
                0.5, 0.5, "No register events",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray",
            )
            return

        # Synthesise from executed provenance data (all accepted by definition)
        t0 = min(e["t"] for e in executed)
        fields = {
            "src_control_step": {"color": "#1a3a6e", "label": "src control step"},
            "chunk_start_step": {"color": "#c1272d", "label": "chunk start step"},
        }
        for field, style in fields.items():
            times = [e["t"] - t0 for e in executed if field in e]
            values = [e[field] for e in executed if field in e]
            if times:
                ax.plot(times, values, linewidth=1, color=style["color"],
                        alpha=0.8, label=style["label"])

        ax.set_ylabel("Control Step")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        return

    # ---- Normal path: register_events present ----
    t0 = min(ev["t"] for ev in events)

    # Group events by register name
    registers: dict[str, list[dict]] = {}
    for ev in events:
        registers.setdefault(ev["register_name"], []).append(ev)

    # Style per register
    _STYLES = {
        "client_obs_request": {"color": "#2ecc71", "label": "obs request"},
        "client_action": {"color": "#3498db", "label": "action (ctrl step)"},
    }

    for reg_name, reg_events in sorted(registers.items()):
        style = _STYLES.get(reg_name, {"color": "#7f8c8d", "label": reg_name})

        # Separate accepted vs rejected
        acc_t = [ev["t"] - t0 for ev in reg_events if ev["accepted"]]
        acc_v = [ev["control_step"] for ev in reg_events if ev["accepted"]]
        rej_t = [ev["t"] - t0 for ev in reg_events if not ev["accepted"]]
        rej_v = [ev["control_step"] for ev in reg_events if not ev["accepted"]]

        # Accepted: filled markers + connecting line
        if acc_t:
            ax.plot(acc_t, acc_v, linewidth=1, color=style["color"],
                    alpha=0.6, zorder=3)
            ax.scatter(acc_t, acc_v, s=12, color=style["color"],
                       alpha=0.8, label=style["label"], zorder=4)

        # Rejected: hollow x markers
        if rej_t:
            ax.scatter(rej_t, rej_v, s=18, marker="x", linewidths=0.8,
                       color=style["color"], alpha=0.4,
                       label=f"{style['label']} (rejected)", zorder=4)

        # For action registers, also plot chunk_start_step
        if reg_name == "client_action":
            css_t = [ev["t"] - t0 for ev in reg_events
                     if ev["accepted"] and ev.get("chunk_start_step") is not None]
            css_v = [ev["chunk_start_step"] for ev in reg_events
                     if ev["accepted"] and ev.get("chunk_start_step") is not None]
            if css_t:
                ax.plot(css_t, css_v, linewidth=1, color="#e74c3c",
                        alpha=0.6, zorder=3)
                ax.scatter(css_t, css_v, s=12, color="#e74c3c",
                           alpha=0.8, label="action (chunk start)", zorder=4)

    ax.set_ylabel("Control Step")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)


def load_experiment_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess experiment CSV data.

    Handles corrupted CSVs that contain concatenated runs (caused by a
    race between ``signal_stop`` and ``stop`` both calling ``flush()``).
    Detection: when ``step`` resets back to -1 after having been positive,
    the earlier rows are discarded and only the last complete run is kept.
    Rows with clearly anomalous timestamps (> 3 IQR from the median) are
    also dropped.
    """
    df = pd.read_csv(csv_path)

    # --- Detect concatenated runs ---
    # When the CSV contains data from two flushes of the same experiment,
    # `step` jumps from a positive value back to -1.  Keep only the last
    # contiguous run (the complete one).
    if "step" in df.columns:
        step = df["step"].values
        # Find indices where step resets from >= 0 back to -1
        reset_indices = []
        for i in range(1, len(step)):
            if step[i] == -1 and step[i - 1] >= 0:
                reset_indices.append(i)
        if reset_indices:
            last_reset = reset_indices[-1]
            n_dropped = last_reset
            df = df.iloc[last_reset:].reset_index(drop=True)
            print(f"    WARNING: CSV contains concatenated runs; "
                  f"dropped first {n_dropped} rows, keeping last run ({len(df)} rows)")

    # --- Drop rows with anomalous timestamps ---
    # A flush race can also truncate a timestamp (e.g. '773263.12' instead
    # of '1770773263.12').  Detect these as outliers relative to the median.
    t = df["t"]
    t_median = t.median()
    t_iqr = t.quantile(0.75) - t.quantile(0.25)
    if t_iqr > 0:
        lower = t_median - 10 * t_iqr
        upper = t_median + 10 * t_iqr
        bad_mask = (t < lower) | (t > upper)
        n_bad = bad_mask.sum()
        if n_bad > 0:
            df = df[~bad_mask].reset_index(drop=True)
            print(f"    WARNING: Dropped {n_bad} rows with anomalous timestamps")

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


def plot_single_experiment(df: pd.DataFrame, title: str, ax_cooldown, ax_latency, ax_events):
    """Plot a single experiment's data across 3 subplots."""
    t = df["t_relative"]

    # Calculate summary stats
    total_ticks = len(df)
    stall_count = df["stall"].sum()
    stall_fraction = stall_count / total_ticks if total_ticks > 0 else 0
    obs_sent_count = df["obs_sent"].sum()
    action_received_count = df["action_received"].sum()

    # 1. Cooldown counter
    ax_cooldown.plot(t, df["cooldown"], linewidth=0.5, alpha=0.7, label=title)
    ax_cooldown.set_ylabel("Cooldown Counter")
    ax_cooldown.grid(True, alpha=0.3)

    # 2. Latency estimate + measured RTT overlay
    ax_latency.plot(t, df["latency_estimate_steps"], linewidth=1, color="#3498db", label="Estimate")
    # Overlay measured RTT converted to steps (red scatter)
    if "measured_latency_ms" in df.columns:
        measured = df[df["measured_latency_ms"].notna()]
        if len(measured) > 0:
            # Infer fps from the data to convert ms -> steps
            t_span = df["t_relative"].iloc[-1]
            fps = (len(df) - 1) / t_span if t_span > 0 else 60.0
            measured_steps = measured["measured_latency_ms"] / 1000.0 * fps
            ax_latency.scatter(
                measured["t_relative"], measured_steps,
                s=15, alpha=0.8, color="#e74c3c", label="Measured RTT",
                zorder=5,
            )
    ax_latency.set_ylabel("Latency (steps)")
    ax_latency.legend(loc="upper right", fontsize=7)
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
        # 2 main plots + 2 trajectory plots (one for JK, one for Max10)
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 2, 2, 2]})
        ax_measured, ax_latency, ax_traj_jk, ax_traj_max = axes
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax_measured, ax_latency = axes
        ax_traj_jk = None
        ax_traj_max = None

    colors = {"jk": "#2ecc71", "max_last_10": "#e74c3c"}
    linestyles = {"jk": "-", "max_last_10": "--"}

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
        linestyle = linestyles.get(estimator, "-")

        # Measured RTT in milliseconds (line plot with distinct linestyle)
        if "measured_latency_ms" in df.columns:
            measured = df[df["measured_latency_ms"].notna()]
            if len(measured) > 0:
                ax_measured.plot(
                    measured["t_relative"],
                    measured["measured_latency_ms"],
                    linewidth=1.5,
                    linestyle=linestyle,
                    color=color,
                    label=f"RTT ({estimator})",
                )

        # Latency estimate (line plot with distinct linestyle)
        ax_latency.plot(t, df["latency_estimate_steps"], linewidth=1.5, linestyle=linestyle, color=color, label=label)

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

    ax_measured.set_ylabel("Measured RTT (ms)")
    ax_measured.legend(loc="upper right")
    ax_measured.grid(True, alpha=0.3)
    ax_measured.set_title("Latency Estimation: JK vs Max-of-Last-10")

    ax_latency.set_ylabel("Latency Estimate (steps)")
    ax_latency.legend(loc="upper right")
    ax_latency.grid(True, alpha=0.3)

    # Set x-axis label on the bottom-most plot
    if has_trajectory_data and ax_traj_max is not None:
        ax_traj_max.set_xlabel("Time (seconds)")
    else:
        ax_latency.set_xlabel("Time (seconds)")

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

    When ``input_path`` is a directory the function looks for CSV files inside
    it.  If exactly one CSV is found it also loads the matching
    ``.trajectory.json`` (if present) and includes a trajectory subplot.

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

    # For a single experiment, try to load trajectory data for extra subplots
    trajectory_data: dict | None = None
    if len(csv_files) == 1:
        trajectory_data = load_trajectory_data(csv_files[0])
        if trajectory_data is not None:
            traj_path = csv_files[0].with_suffix(".trajectory.json")
            print(f"  Loading: {traj_path.name}")

    # Decide subplot layout:
    #   Without trajectory: 3 base rows (cooldown, latency, events)
    #   With trajectory:    3 base + trajectory + sim_events + provenance = 6
    has_trajectory = trajectory_data is not None
    if has_trajectory:
        n_rows = 6
        # base rows are height 1, trajectory gets 2, the rest 1
        height_ratios = [1, 1, 1, 2, 1, 1]
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(14, 16), sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        (ax_cooldown, ax_latency, ax_events,
         ax_traj, ax_sim_events, ax_provenance) = axes
    else:
        n_rows = 3
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 8), sharex=True)
        ax_cooldown, ax_latency, ax_events = axes
        ax_traj = None
        ax_sim_events = None
        ax_provenance = None

    all_stats = []

    for name, df in dfs.items():
        stats = plot_single_experiment(
            df,
            title=name,
            ax_cooldown=ax_cooldown,
            ax_latency=ax_latency,
            ax_events=ax_events,
        )
        stats["file"] = name
        all_stats.append(stats)

    # Plot trajectory-derived subplots if available
    if trajectory_data is not None:
        # Compute the offset between CSV t0 (experiment start) and trajectory
        # t0 (first executed action) so that sim config windows align with
        # the event scatter markers and other trajectory subplots.
        df0 = list(dfs.values())[0]
        csv_t0 = df0["t"].iloc[0]
        traj_executed = trajectory_data.get("executed", [])
        traj_t0 = min(e["t"] for e in traj_executed) if traj_executed else csv_t0
        sim_config_offset = traj_t0 - csv_t0

        # 4. Executed actions trajectory (joint 0)
        if ax_traj is not None:
            plot_trajectory_on_axis(ax_traj, trajectory_data, joint_idx=0)
            ax_traj.set_title("Executed Actions (Joint 0)")
            ax_traj.legend(loc="upper right", fontsize=8)

        # 5. Simulation events timeline
        if ax_sim_events is not None:
            plot_sim_events_on_axis(
                ax_sim_events, trajectory_data,
                sim_config_offset=sim_config_offset,
            )
            ax_sim_events.set_title("Simulation Events")

        # 6. LWW register state (control_step per register over time)
        if ax_provenance is not None:
            plot_register_state(ax_provenance, trajectory_data)
            ax_provenance.set_title("LWW Register State")

    # Set common x-axis label on the bottom-most plot
    bottom_ax = ax_provenance or ax_traj or ax_events
    bottom_ax.set_xlabel("Time (seconds)")

    # Add legends
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
        default=None,
        help="Output path for the plot image (default: saved beside the input CSV)",
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

    # Default output path: save plot beside the input file/directory
    output = args.output
    if output is None:
        if args.input.is_file():
            output = args.input.parent / "plot.png"
        else:
            output = args.input / "plot.png"

    plot_results(args.input, output, mode=args.mode, filter_pattern=args.filter)


if __name__ == "__main__":
    main()
