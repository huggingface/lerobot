#!/usr/bin/env python3
"""
Plot results from DRTC experiments.

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
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def setup_paper_style():
    """Configure matplotlib rcParams for clean, academic paper-ready plots."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 100,
    })


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in *text*."""
    # Order matters: ampersand first so we don't double-escape later subs.
    for char, replacement in [
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        text = text.replace(char, replacement)
    return text


# Human-readable labels for each config key, in display order.
_CONFIG_DISPLAY = [
    # Robot / hardware
    ("robot_type", "Robot type"),
    ("gpu", "GPU"),
    ("client_host", "Client host"),
    ("server_host", "Server host"),
    ("num_cameras", "Number of cameras"),
    ("cameras", "Cameras"),
    # Policy
    ("policy_type", "Policy type"),
    ("pretrained_name_or_path", "Model path"),
    ("chunk_size", "Chunk size"),
    ("fps", "FPS"),
    ("s_min", r"$s_{\min}$"),
    ("epsilon", r"$\epsilon$"),
    ("latency_estimator_type", "Latency estimator"),
    ("latency_alpha", r"$\alpha$"),
    ("latency_beta", r"$\beta$"),
    ("latency_k", r"$K$"),
    # Flow matching / RTC
    ("num_flow_matching_steps", "Flow matching steps"),
    ("rtc_enabled", "RTC enabled"),
    ("rtc_max_guidance_weight", r"RTC max guidance weight ($\beta$)"),
    ("rtc_prefix_attention_schedule", "RTC attention schedule"),
    ("rtc_sigma_d", r"RTC $\sigma_d$"),
    ("rtc_full_trajectory_alignment", "RTC full trajectory alignment"),
    # Action filter
    ("filter_type", "Filter type"),
    ("filter_cutoff", "Filter cutoff (Hz)"),
    ("gain", "Gain"),
]

# Human-readable names for latency estimator types.
_ESTIMATOR_DISPLAY_NAMES = {
    "jk": "Jacobson--Karels",
    "max_last_10": "Max of last 10",
    "fixed": "Fixed",
}

# Human-readable labels for simulation config fault-injection keys.
_SIM_CONFIG_DISPLAY = [
    ("drop_obs", "Drop obs"),
    ("drop_action", "Drop action"),
    ("dup_obs", "Duplicate obs"),
    ("dup_action", "Duplicate action"),
    ("reorder_obs", "Reorder obs"),
    ("reorder_action", "Reorder action"),
    ("disconnect", "Disconnect"),
    ("spikes", "Spikes"),
]


def _format_fault_windows(windows: list[dict]) -> str:
    """Format a list of fault-injection window dicts into a compact string.

    Each window is rendered as ``start_s`` -- ``+duration_s`` (or for spikes,
    ``start_s`` @ ``delay_ms`` ms).  Multiple windows are comma-separated.
    Returns ``---`` when the list is empty.
    """
    if not windows:
        return "---"
    parts = []
    for w in windows:
        if "delay_ms" in w:
            # Spike event
            parts.append(f"{w.get('start_s', 0):.1f}s @ {w['delay_ms']}ms")
        else:
            start = w.get("start_s", 0)
            dur = w.get("duration_s", 0)
            parts.append(f"{start:.1f}s +{dur:.1f}s")
    return ", ".join(parts)


def generate_config_table(
    experiment_config: dict,
    simulation_config: dict | None = None,
) -> str:
    """Return a LaTeX ``table`` environment summarising the experiment config.

    The table has two columns (Parameter / Value) and uses the ``booktabs``
    package for clean horizontal rules.  Long model paths are wrapped in
    ``\\texttt``.

    When *simulation_config* is provided, a second section is appended
    showing fault-injection windows (drops, duplicates, reorders,
    disconnects, spikes).

    Args:
        experiment_config: Dict of config key/value pairs (as written by
            ``ExperimentMetricsWriter``).
        simulation_config: Optional dict of fault-injection window lists
            (as written to the trajectory JSON under ``simulation_config``).

    Returns:
        A string of LaTeX source for the config table (no surrounding
        ``\\begin{document}``).
    """
    rows: list[str] = []
    for key, label in _CONFIG_DISPLAY:
        value = experiment_config.get(key)
        if value is None:
            value_str = "N/A"
        elif key == "pretrained_name_or_path":
            # Render model path in monospace; allow line-break at slashes.
            escaped = _latex_escape(str(value))
            value_str = r"\texttt{" + escaped + "}"
        elif key == "latency_estimator_type":
            display = _ESTIMATOR_DISPLAY_NAMES.get(str(value), str(value))
            value_str = _latex_escape(display)
        elif isinstance(value, float):
            # Use a sensible number of decimals.
            value_str = f"{value:g}"
        else:
            value_str = _latex_escape(str(value))
        rows.append(f"    {label} & {value_str} \\\\")

    # Fault-injection section (only if any faults are configured)
    if simulation_config:
        has_any_faults = any(
            simulation_config.get(key) for key, _ in _SIM_CONFIG_DISPLAY
        )
        if has_any_faults:
            rows.append("    \\midrule")
            rows.append(
                "    \\multicolumn{2}{l}{\\textit{Fault Injection}} \\\\"
            )
            for key, label in _SIM_CONFIG_DISPLAY:
                windows = simulation_config.get(key, [])
                if windows:
                    value_str = _latex_escape(_format_fault_windows(windows))
                    rows.append(f"    {_latex_escape(label)} & {value_str} \\\\")

    rows_str = "\n".join(rows)
    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Experiment Configuration}\n"
        "  \\begin{tabular}{l p{0.65\\textwidth}}\n"
        "    \\toprule\n"
        "    Parameter & Value \\\\\n"
        "    \\midrule\n"
        f"{rows_str}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


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


# Default joint names for the SO101 follower robot
_SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def plot_trajectory_on_axis(
    ax,
    trajectory_data: dict,
    joint_names: list[str] | None = None,
):
    """Plot all joint trajectories on a given axis.

    Plots each joint dimension as a separate coloured line.  Gaps in the
    lines correspond to stalls (timesteps where no action was executed).

    Args:
        ax: Matplotlib axis to plot on.
        trajectory_data: Dict with ``executed`` list of action records.
        joint_names: Human-readable names for each joint dimension.
            Defaults to the SO101 follower joint names.
    """
    if trajectory_data is None:
        ax.text(0.5, 0.5, "No trajectory data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        return

    executed = trajectory_data.get("executed", [])

    if not executed:
        ax.text(0.5, 0.5, "No executed actions", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        return

    if joint_names is None:
        joint_names = _SO101_JOINT_NAMES

    t0 = min(e["t"] for e in executed)
    n_joints = len(executed[0]["action"])

    for j in range(n_joints):
        times = [(e["t"] - t0) for e in executed if j < len(e["action"])]
        values = [e["action"][j] for e in executed if j < len(e["action"])]
        label = joint_names[j] if j < len(joint_names) else f"joint {j}"
        color = CHUNK_COLORS[j % len(CHUNK_COLORS)]
        ax.scatter(times, values, s=3, color=color, alpha=0.8, label=label, linewidths=0)

    ax.set_ylabel("Position")


# Colors for event types (sim events + obs/action from CSV)
_SIM_EVENT_COLORS = {
    "obs_triggered": "#3498db",      # blue
    "action_received": "#e67e22",   # orange
    "obs_dropped": "#c1272d",       # vermillion
    "obs_reorder_held": "#f4c430",  # cadmium yellow
    "obs_reorder_swapped": "#e85d04",  # orange
    "obs_duplicated": "#5c3d6e",    # purple
    "action_dropped": "#1a3a6e",    # ultramarine
    "action_reorder_held": "#0077b6",  # cerulean
    "action_reorder_swapped": "#2d6a4f",  # deep green
    "action_duplicated": "#9d4edd", # violet
    "disconnect": "#333333",        # dark gray
    "spike": "#e74c3c",             # red
}

# Y-position for each event type so they don't overlap (contiguous)
_SIM_EVENT_YPOS = {
    "obs_triggered": 11,
    "action_received": 10,
    "disconnect": 9,
    "spike": 8,
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
    "disconnect": ("#333333", 0.18),
    "spikes": ("#e74c3c", 0.12),
}


def plot_gantt_on_axis(
    ax,
    trajectory_data: dict,
    sim_config_offset: float = 0.0,
) -> None:
    """Plot a Gantt chart of configured fault-injection windows.

    Each fault type gets its own horizontal lane with coloured bars showing
    when the fault is active.  Spikes are rendered as bars whose width
    corresponds to the injected delay duration.

    Args:
        ax: Matplotlib axis.
        trajectory_data: Trajectory JSON dict (must contain ``simulation_config``).
        sim_config_offset: Seconds between experiment start (CSV t0) and the
            first executed action (trajectory t0).  Subtracted from ``start_s``
            values so the bars align with other subplots.
    """
    sim_config = trajectory_data.get("simulation_config", {})
    if not sim_config:
        ax.text(
            0.5, 0.5, "No simulation config",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        return

    # Collect lanes: only include fault types that have at least one window.
    # Order follows _SIM_CONFIG_COLORS so related faults are grouped.
    lane_keys: list[str] = []
    for key in _SIM_CONFIG_COLORS:
        windows = sim_config.get(key, [])
        if windows:
            lane_keys.append(key)

    if not lane_keys:
        ax.text(
            0.5, 0.5, "No fault windows configured",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        return

    n_lanes = len(lane_keys)
    bar_height = 0.6

    for i, lane in enumerate(lane_keys):
        y = i  # y-position for this lane
        windows = sim_config.get(lane, [])
        color, _alpha = _SIM_CONFIG_COLORS.get(lane, ("#888888", 0.5))

        is_spike = lane == "spikes"

        # Build (start, width) tuples and labels for each window.
        # Spikes use delay_ms converted to seconds; other faults use duration_s.
        bars: list[tuple[float, float]] = []
        labels: list[str] = []
        for w in windows:
            start = w.get("start_s", 0) - sim_config_offset
            if is_spike:
                dur = w.get("delay_ms", 0) / 1000.0
                labels.append(f"{w.get('delay_ms', 0):.0f}ms")
            else:
                dur = w.get("duration_s", 0)
                labels.append(f"{dur:.1f}s")
            bars.append((start, dur))

        ax.broken_barh(
            bars, (y - bar_height / 2, bar_height),
            facecolors=color, alpha=0.7, edgecolors="white", linewidth=0.5,
        )
        for (start, dur), label in zip(bars, labels):
            if is_spike:
                ax.text(
                    start + dur / 2, y,
                    label,
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold",
                    rotation=-90, rotation_mode="anchor",
                )
            else:
                ax.text(
                    start + dur / 2, y,
                    label,
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold",
                )

    # Configure axis
    ax.set_yticks(range(n_lanes))
    ax.set_yticklabels(
        [lane.replace("_", " ") for lane in lane_keys], fontsize=9,
    )
    # ax.set_ylim(-0.5, n_lanes - 0.5)
    # ax.set_ylabel("Fault Schedule")


# Lane configuration for the latency breakdown Gantt chart.
_LATENCY_GANTT_LANES = [
    ("total", "Total", "#e74c3c"),              # red
    ("client_to_server", "Client \u2192 Server", "#0077b6"),   # cerulean
    ("model_inference", "Model Inference", "#2d6a4f"),         # deep green
    ("server_to_client", "Server \u2192 Client", "#e85d04"),   # orange
]


def plot_latency_gantt_on_axis(
    ax,
    df: pd.DataFrame,
    time_offset: float = 0.0,
) -> None:
    """Plot a two-lane horizontal stacked bar chart of inference latency breakdown.

    Each inference round-trip (rows where ``action_received == 1`` and all
    timestamp columns are present) is rendered across two lanes: a Total
    bar (top) and a Breakdown bar (bottom) with three end-to-end colored
    segments (Client -> Server, Model Inference, Server -> Client).

    Supports both new timestamp columns (``obs_sent_ts``, etc.) and legacy
    duration columns (``client_to_server_ms``, etc.) for backward
    compatibility with older CSVs.

    Args:
        ax: Matplotlib axis.
        df: Experiment DataFrame.
        time_offset: Seconds to subtract from ``t_relative`` so bar
            positions align with other subplots.
    """
    # Detect whether we have new timestamp columns or legacy duration columns
    ts_cols = ["obs_sent_ts", "server_obs_received_ts", "server_action_sent_ts", "action_received_ts"]
    legacy_cols = ["measured_latency_ms", "client_to_server_ms", "model_inference_ms", "server_to_client_ms"]
    use_timestamps = all(c in df.columns for c in ts_cols)
    use_legacy = all(c in df.columns for c in legacy_cols)

    if not use_timestamps and not use_legacy:
        ax.text(
            0.5, 0.5, "No latency breakdown data",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        return

    # Filter to rows where action was received and all required cols exist
    check_cols = ts_cols if use_timestamps else legacy_cols
    mask = (df["action_received"] == 1)
    for col in check_cols:
        mask = mask & df[col].notna()
    rtt_rows = df[mask]

    if len(rtt_rows) == 0:
        ax.text(
            0.5, 0.5, "No latency breakdown data",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        return

    bar_height = 0.6

    # Compute a minimum visual bar width so that very short phases
    # (e.g. 1ms client->server on a 25s axis) are still visible.
    t_all = df["t_relative"] - time_offset
    t_span = t_all.max() - t_all.min() if len(t_all) > 1 else 1.0
    min_bar_width = t_span * 0.004  # ~0.4% of axis width

    # Reference t0 for converting absolute timestamps to relative x-axis values
    csv_t0 = df["t"].iloc[0]

    # Colors for the two-lane layout
    total_color = "#e74c3c"       # red
    c2s_color = "#0077b6"         # cerulean
    model_color = "#2d6a4f"       # deep green
    s2c_color = "#e85d04"         # orange

    for _, row in rtt_rows.iterrows():
        if use_timestamps:
            # Derive bar positions directly from wall-clock timestamps
            t_send = row["obs_sent_ts"] - csv_t0 - time_offset
            t_server_recv = row["server_obs_received_ts"] - csv_t0 - time_offset
            t_server_send = row["server_action_sent_ts"] - csv_t0 - time_offset
            t_recv = row["action_received_ts"] - csv_t0 - time_offset

            rtt_s = t_recv - t_send
            c2s_s = t_server_recv - t_send
            model_s = t_server_send - t_server_recv
            s2c_s = t_recv - t_server_send
        else:
            # Legacy: derive from duration columns
            t_recv = row["t_relative"] - time_offset
            rtt_s = row["measured_latency_ms"] / 1000.0
            c2s_s = row["client_to_server_ms"] / 1000.0
            model_s = row["model_inference_ms"] / 1000.0
            s2c_s = row["server_to_client_ms"] / 1000.0
            t_send = t_recv - rtt_s
            t_server_recv = t_send + c2s_s
            t_server_send = t_server_recv + model_s

        # Lane 1 (top, y=1): Total round-trip bar
        ax.broken_barh(
            [(t_send, max(rtt_s, min_bar_width))],
            (1 - bar_height / 2, bar_height),
            facecolors=total_color, alpha=0.7,
            edgecolors="white", linewidth=0.3,
        )

        # Lane 0 (bottom, y=0): Breakdown segments tiled end-to-end.
        # Compute visual widths first, then place each segment right
        # after the previous one so min_bar_width clamping never causes
        # overlap.
        vis_c2s = max(c2s_s, min_bar_width)
        vis_model = max(model_s, min_bar_width)
        vis_s2c = max(s2c_s, min_bar_width)

        seg_start = t_send
        for dur, color in [
            (vis_c2s, c2s_color),
            (vis_model, model_color),
            (vis_s2c, s2c_color),
        ]:
            ax.broken_barh(
                [(seg_start, dur)],
                (0 - bar_height / 2, bar_height),
                facecolors=color, alpha=0.7,
                edgecolors="white", linewidth=0.3,
            )
            seg_start += dur

        # Vertical guide lines at phase boundaries (span both lanes)
        for bx in [t_send, t_send + vis_c2s, t_send + vis_c2s + vis_model, seg_start]:
            ax.axvline(bx, color="#888888", linewidth=0.4, alpha=0.5)

    # Configure axis -- 2 lanes
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Breakdown", "Total"], fontsize=9)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title("Latency Breakdown")

    # Legend with proxy artists for all 4 colors
    legend_handles = [
        Patch(facecolor=total_color, alpha=0.7, label="Total"),
        Patch(facecolor=c2s_color, alpha=0.7, label="Client \u2192 Server"),
        Patch(facecolor=model_color, alpha=0.7, label="Model Inference"),
        Patch(facecolor=s2c_color, alpha=0.7, label="Server \u2192 Client"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)


def plot_sim_events_on_axis(
    ax,
    trajectory_data: dict,
    sim_config_offset: float = 0.0,
    df: pd.DataFrame | None = None,
) -> None:
    """Plot events timeline on a given axis.

    Shows actual recorded sim events as scatter markers.  Spike events
    from the simulation config are shown as scatter markers in a
    dedicated lane.  When *df* is provided, ``obs_triggered`` and
    ``action_received`` events from the CSV are also plotted as
    additional lanes.

    Args:
        ax: Matplotlib axis.
        trajectory_data: Trajectory JSON dict.
        sim_config_offset: Seconds between experiment start (CSV t0) and
            the first executed action (trajectory t0).  The ``start_s``
            values in the simulation config are relative to experiment
            start, so we subtract this offset to align them with the
            trajectory t0 used by all other subplots.
        df: Optional experiment DataFrame (for obs_triggered / action_received).
    """
    sim_events = trajectory_data.get("sim_events", [])
    sim_config = trajectory_data.get("simulation_config", {})
    executed = trajectory_data.get("executed", [])

    if not sim_events and not sim_config and df is None:
        ax.text(
            0.5, 0.5, "No events",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        return

    # Derive t0 from executed actions (same baseline as trajectory plot)
    t0 = min(e["t"] for e in executed) if executed else 0.0

    # --- Collect all events (sim events + obs/action from CSV) ---
    events_by_type: dict[str, list[float]] = {}

    # Sim events from trajectory JSON
    for ev in sim_events:
        etype = ev["event_type"]
        t_rel = ev["t"] - t0
        events_by_type.setdefault(etype, []).append(t_rel)

    # Spike config entries (point events from simulation_config)
    spikes = sim_config.get("spikes", [])
    for spike in spikes:
        t_spike = spike.get("start_s", 0) - sim_config_offset
        events_by_type.setdefault("spike", []).append(t_spike)

    # Obs sent / action received from CSV DataFrame
    if df is not None:
        # CSV times are relative to CSV t0; shift by sim_config_offset
        # so they align with the trajectory t0 baseline.
        t_csv = df["t_relative"] - sim_config_offset
        obs_mask = df["obs_triggered"] == 1
        if obs_mask.any():
            events_by_type["obs_triggered"] = t_csv[obs_mask].tolist()
        act_mask = df["action_received"] == 1
        if act_mask.any():
            events_by_type["action_received"] = t_csv[act_mask].tolist()

    # Determine active lanes (preserve order from _SIM_EVENT_YPOS)
    # Assign contiguous y-positions so there are no gaps between active lanes.
    active_types = [k for k in _SIM_EVENT_YPOS if k in events_by_type]
    active_ypos = {k: i for i, k in enumerate(reversed(active_types))}

    for etype in active_types:
        times = events_by_type[etype]
        y = active_ypos[etype]
        color = _SIM_EVENT_COLORS.get(etype, "#888888")
        ax.scatter(
            times, [y] * len(times),
            marker="|", s=40, color=color, alpha=0.8,
        )

    # Configure axis
    if active_types:
        ax.set_yticks([active_ypos[k] for k in active_types])
        ax.set_yticklabels([k.replace("_", " ") for k in active_types], fontsize=8)
        ax.set_ylim(-0.5, len(active_types) - 0.5)


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

    # Convert measured_latency_ms and timestamp columns to numeric (may have empty strings)
    for col in ["measured_latency_ms", "latency_estimate_ms", "obs_sent_ts", "server_obs_received_ts", "server_action_sent_ts", "action_received_ts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Backward compat: also handle legacy duration columns from older CSVs
    for col in ["client_to_server_ms", "model_inference_ms", "server_to_client_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Backward compat: older CSVs used "obs_sent" instead of "obs_triggered"
    if "obs_triggered" not in df.columns and "obs_sent" in df.columns:
        df.rename(columns={"obs_sent": "obs_triggered"}, inplace=True)

    # Convert L2 columns to numeric
    for col in ["chunk_mean_l2", "chunk_max_l2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_single_experiment(df: pd.DataFrame, title: str, ax_cooldown, ax_latency, ax_schedule=None, ax_events=None, time_offset: float = 0.0):
    """Plot a single experiment's data across 2-4 subplots.

    When *ax_events* is ``None`` (trajectory mode), the obs/action events
    are merged into the combined events subplot elsewhere.

    Args:
        ax_schedule: Optional axis for the action schedule size plot.
        time_offset: Seconds to subtract from ``t_relative`` so that the
            cooldown / latency x-values align with the trajectory t0.
    """
    t = df["t_relative"] - time_offset

    # Calculate summary stats
    total_ticks = len(df)
    stall_count = df["stall"].sum()
    stall_fraction = stall_count / total_ticks if total_ticks > 0 else 0
    obs_triggered_count = df["obs_triggered"].sum()
    action_received_count = df["action_received"].sum()

    # 1. Cooldown counter + quantized latency estimate (both in steps)
    ax_cooldown.plot(t, df["cooldown"], linewidth=1.5, alpha=0.7,
                     color="#9b59b6", label="Cooldown")
    if "latency_estimate_steps" in df.columns:
        ax_cooldown.plot(t, df["latency_estimate_steps"], drawstyle="steps-post",
                         linewidth=1.2, color="#2ecc71", alpha=0.7,
                         label="Latency estimate (steps)")
    ax_cooldown.set_title("Cooldown & Latency Estimate (steps)")
    ax_cooldown.set_ylabel("Steps")
    ax_cooldown.legend(loc="upper right", fontsize=8)

    # 2. Latency estimate (smooth) + measured RTT overlay (all in ms)
    if "latency_estimate_ms" in df.columns:
        estimate_ms = pd.to_numeric(df["latency_estimate_ms"], errors="coerce")
    else:
        # Fallback for older CSVs: convert quantized steps to ms
        t_span = df["t_relative"].iloc[-1]
        fps = (len(df) - 1) / t_span if t_span > 0 else 60.0
        estimate_ms = df["latency_estimate_steps"] / fps * 1000.0
    ax_latency.plot(t, estimate_ms, linewidth=1.5, color="#3498db",
                    label="Estimate")
    # Overlay measured RTT in ms (red scatter)
    if "measured_latency_ms" in df.columns:
        measured = df[df["measured_latency_ms"].notna()]
        if len(measured) > 0:
            ax_latency.scatter(
                measured["t_relative"] - time_offset, measured["measured_latency_ms"],
                s=25, alpha=0.8, color="#e74c3c", label="Measured RTT",
                zorder=5,
            )
    ax_latency.legend(loc="upper right", fontsize=8)
    ax_latency.set_title("Inference Latency")
    ax_latency.set_ylabel("ms")

    # 3. Action schedule size
    if ax_schedule is not None and "schedule_size" in df.columns:
        ax_schedule.plot(t, df["schedule_size"], linewidth=1, color="#3498db")
        ax_schedule.axhline(y=0, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=0.5)
        ax_schedule.set_title("Action Schedule Size")
        ax_schedule.set_ylabel("Actions")

    # 4. Events timeline (only when a separate events axis is provided)
    if ax_events is not None:
        obs_times = t[df["obs_triggered"] == 1]
        action_times = t[df["action_received"] == 1]

        ax_events.scatter(obs_times, [1] * len(obs_times), marker="|", s=30, alpha=0.7, label=f"obs triggered ({obs_triggered_count})")
        ax_events.scatter(action_times, [0] * len(action_times), marker="|", s=30, alpha=0.7, label=f"action recv ({action_received_count})")
        ax_events.set_ylabel("Events")
        ax_events.set_ylim(-0.5, 1.5)
        ax_events.set_yticks([0, 1])
        ax_events.set_yticklabels(["Action", "Obs"])

    return {
        "total_ticks": total_ticks,
        "stall_count": stall_count,
        "stall_fraction": stall_fraction,
        "obs_triggered_count": obs_triggered_count,
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
    setup_paper_style()

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
                )
                traj_ax.set_title(f"Trajectory: {estimator.upper()}")
                traj_ax.legend(loc="upper right", ncol=3)

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
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Estimator comparison plot saved to: {output_path}")


def plot_detailed(df: pd.DataFrame, title: str, output_path: Path):
    """Detailed plot for a single experiment with schedule size and L2 metrics."""
    setup_paper_style()
    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)
    ax_schedule, ax_latency, ax_latency_gantt, ax_cooldown, ax_stall, ax_l2 = axes
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
    ax_latency.legend(loc="upper right", fontsize=8)
    ax_latency.grid(True, alpha=0.3)

    # 3. Latency breakdown Gantt chart
    plot_latency_gantt_on_axis(ax_latency_gantt, df)

    # 4. Cooldown counter
    ax_cooldown.plot(t, df["cooldown"], linewidth=0.5, color="#9b59b6", alpha=0.8)
    ax_cooldown.set_ylabel("Cooldown")
    ax_cooldown.grid(True, alpha=0.3)

    # 5. Stall indicator
    ax_stall.fill_between(t, 0, df["stall"], alpha=0.5, color="#e74c3c", step="mid")
    ax_stall.set_ylabel("Stall")
    ax_stall.set_ylim(-0.1, 1.1)
    ax_stall.grid(True, alpha=0.3)

    # 6. L2 discrepancy (if available)
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
    plt.savefig(output_path, bbox_inches="tight")
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
    setup_paper_style()

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
    #   Without trajectory: 4 base rows (latency, cooldown, schedule, events)
    #   With trajectory:    trajectory + gantt + events + latency + cooldown + schedule = 6
    #     (obs/action events are merged into the sim events plot)
    has_trajectory = trajectory_data is not None
    if has_trajectory:
        n_rows = 6
        # trajectory gets 2, the rest 1
        height_ratios = [2, 1, 1, 1, 1, 1]
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(14, 16), sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        (ax_traj, ax_gantt, ax_sim_events,
         ax_latency, ax_cooldown, ax_schedule) = axes
        ax_events = None  # obs/action events merged into sim_events
    else:
        n_rows = 4
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 10), sharex=True)
        ax_latency, ax_cooldown, ax_schedule, ax_events = axes
        ax_traj = None
        ax_gantt = None
        ax_sim_events = None

    all_stats = []

    # Compute the time offset between CSV t0 and trajectory t0 so that
    # cooldown / latency plots align with trajectory-derived subplots.
    time_offset = 0.0
    if has_trajectory:
        df0 = list(dfs.values())[0]
        csv_t0 = df0["t"].iloc[0]
        traj_executed = trajectory_data.get("executed", [])
        traj_t0 = min(e["t"] for e in traj_executed) if traj_executed else csv_t0
        time_offset = traj_t0 - csv_t0

    for name, df in dfs.items():
        stats = plot_single_experiment(
            df,
            title=name,
            ax_cooldown=ax_cooldown,
            ax_latency=ax_latency,
            ax_schedule=ax_schedule,
            ax_events=ax_events,
            time_offset=time_offset,
        )
        stats["file"] = name
        all_stats.append(stats)

    # Plot trajectory-derived subplots if available
    if trajectory_data is not None:
        sim_config_offset = time_offset  # already computed above
        df0 = list(dfs.values())[0]

        # 4. Trajectory (all joints)
        if ax_traj is not None:
            plot_trajectory_on_axis(ax_traj, trajectory_data)
            ax_traj.set_title("Trajectory")
            ax_traj.legend(loc="upper right", ncol=3)

        # 5. Gantt chart of fault injection windows
        if ax_gantt is not None:
            plot_gantt_on_axis(
                ax_gantt, trajectory_data,
                sim_config_offset=sim_config_offset,
            )
            ax_gantt.set_title("Fault Injection Schedule")

        # 6. Events timeline (sim events + obs/action from CSV)
        if ax_sim_events is not None:
            plot_sim_events_on_axis(
                ax_sim_events, trajectory_data,
                sim_config_offset=sim_config_offset,
                df=df0,
            )
            ax_sim_events.set_title("Events")

    # Label only the bottom subplot with "Time (seconds)"
    axes[-1].set_xlabel("Time (seconds)")

    # Add legends (only for the separate events axis in non-trajectory mode)
    if ax_events is not None:
        ax_events.legend(loc="upper right")

    # Create title
    if len(csv_files) == 1:
        title = list(dfs.keys())[0]
    else:
        title = f"Experiment Comparison ({len(csv_files)} files)"

    plt.tight_layout()

    # Save figure as both PNG and PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("").as_posix().rstrip(".")
    png_path = Path(f"{stem}.png")
    pdf_path = Path(f"{stem}.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"\nPlot saved to: {png_path}")
    print(f"Plot saved to: {pdf_path}")

    # Generate a LaTeX document that imports the PDF figure
    tex_path = Path(f"{stem}.tex")
    pdf_filename = pdf_path.name

    # Build an optional config table from trajectory data
    config_table_tex = ""
    if trajectory_data is not None:
        exp_config = trajectory_data.get("experiment_config")
        sim_config = trajectory_data.get("simulation_config")
        if exp_config:
            config_table_tex = "\n" + generate_config_table(
                exp_config, simulation_config=sim_config,
            ) + "\n"

    tex_content = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage{{booktabs}}
\begin{{document}}
{config_table_tex}
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=\textwidth]{{{pdf_filename}}}
  \caption{{Experiment results.}}
  \label{{fig:{pdf_path.stem}}}
\end{{figure}}

\end{{document}}
"""
    tex_path.write_text(tex_content)
    print(f"LaTeX saved to: {tex_path}")

    # Compile LaTeX to PDF if pdflatex is available
    if shutil.which("pdflatex"):
        tex_pdf_path = tex_path.with_suffix(".pdf")
        # Rename the plot PDF temporarily so pdflatex output doesn't collide
        plot_pdf_tmp = pdf_path.with_suffix(".plot.pdf")
        pdf_path.rename(plot_pdf_tmp)
        # Update the tex to reference the temp name
        tex_content_tmp = tex_content.replace(pdf_filename, plot_pdf_tmp.name)
        tex_path.write_text(tex_content_tmp)
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_path.name],
                cwd=tex_path.parent,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                # pdflatex produced the compiled doc; rename files back
                compiled_pdf = tex_path.with_suffix(".pdf")
                latex_out = tex_path.with_name(f"{tex_path.stem}_doc.pdf")
                compiled_pdf.rename(latex_out)
                plot_pdf_tmp.rename(pdf_path)
                # Restore original tex content
                tex_path.write_text(tex_content)
                print(f"LaTeX PDF saved to: {latex_out}")
            else:
                # Restore on failure
                plot_pdf_tmp.rename(pdf_path)
                tex_path.write_text(tex_content)
                print(f"pdflatex failed (exit {result.returncode}). Check {tex_path}")
        except subprocess.TimeoutExpired:
            plot_pdf_tmp.rename(pdf_path)
            tex_path.write_text(tex_content)
            print("pdflatex timed out")
        finally:
            # Clean up pdflatex auxiliary files
            for ext in (".aux", ".log", ".out"):
                aux = tex_path.with_suffix(ext)
                if aux.exists():
                    aux.unlink()
    else:
        print("pdflatex not found; skipping LaTeX compilation")

    # Print summary table
    print("\nSummary:")
    print("-" * 80)
    print(f"{'File':<40} {'Ticks':>8} {'Stalls':>8} {'Stall%':>8} {'ObsSent':>8} {'ActRecv':>8}")
    print("-" * 80)
    for s in all_stats:
        print(
            f"{s['file']:<40} {s['total_ticks']:>8} {s['stall_count']:>8} "
            f"{s['stall_fraction']:>7.1%} {s['obs_triggered_count']:>8} {s['action_received_count']:>8}"
        )

    # Show plot interactively if not in headless mode
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot DRTC experiment results")
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

    # Default output path (stem only – both .png and .pdf are generated)
    output = args.output
    if output is None:
        if args.input.is_file():
            output = args.input.parent / args.input.stem
        else:
            output = args.input / args.input.name

    plot_results(args.input, output, mode=args.mode, filter_pattern=args.filter)


if __name__ == "__main__":
    main()
