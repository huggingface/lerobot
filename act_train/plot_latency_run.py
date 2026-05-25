"""Plot a latency run: telemetry overview + latency-stats figures.

Picks up `data.csv` and `events.csv` from a run directory written by
`measure_ur10_latency.py`. If no directory is passed, the most recent
`run_*` under `act_train/latency_runs/` is used.

Usage:
    python act_train/plot_latency_run.py
    python act_train/plot_latency_run.py --run-dir act_train/latency_runs/run_YYYYMMDD_HHMMSS
    python act_train/plot_latency_run.py --no-show

Produces two PNGs alongside the CSVs:
    overview.png       — TCP x/y/z (target vs actual), stick, speed, event markers
    latency_stats.png  — distribution, latency-vs-time, status breakdown

Tracking lag (per axis, single number) is computed by cross-correlating the
commanded `target_*` signal against the measured `tcp_*` signal and reading
off the shift at the peak. That gives a robust "how many ms is the actual
trailing the commanded trajectory" number that's valid *during continuous
motion* — complementary to the discrete event latencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUNS_DIR = Path(__file__).parent / "latency_runs"

# Colour palette — muted, print-friendly. Same hues reused for x/y/z across
# panels so a reader sees the consistency at a glance.
C_TARGET   = "#9bb6dd"   # dashed target line
C_ACTUAL   = "#1c4d8c"   # solid actual line
C_SPEED    = "#333333"
C_STICK_X  = "#d97a4b"
C_STICK_Y  = "#4ba16a"
C_STICK_Z  = "#9b59b6"
C_OK       = "#c0392b"
C_TIMEOUT  = "#e67e22"
C_SKIPPED  = "#888888"


def _latest_run() -> Path:
    runs = sorted(p for p in DEFAULT_RUNS_DIR.glob("run_*") if (p / "data.csv").exists())
    if not runs:
        raise SystemExit(f"No runs found under {DEFAULT_RUNS_DIR}")
    return runs[-1]


def _xcorr_lag_ms(target: np.ndarray, actual: np.ndarray, dt: float, max_lag_s: float = 1.0) -> float:
    """Shift (ms) at which the actual signal best matches the target signal.

    Positive → actual trails target (the usual case). Returns NaN if either
    signal has no variance (idle period) — there's no lag to estimate then.
    """
    if len(target) < 3 or len(actual) < 3:
        return float("nan")
    target = target - target.mean()
    actual = actual - actual.mean()
    if target.std() < 1e-9 or actual.std() < 1e-9:
        return float("nan")
    target /= target.std()
    actual /= actual.std()
    n = len(target)
    corr = np.correlate(actual, target, mode="full")
    center = n - 1
    max_lag_samples = max(1, min(int(max_lag_s / dt), n - 1))
    window = corr[center : center + max_lag_samples + 1]
    return float(np.argmax(window) * dt * 1000.0)


def _apply_pro_style() -> None:
    """Tasteful defaults: white background, subtle grid, no top/right spines."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.25,
        "grid.linestyle":   "-",
        "grid.linewidth":   0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlesize":   11,
        "axes.titleweight": "semibold",
        "axes.labelsize":   10,
        "legend.frameon":   True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#dddddd",
        "legend.fontsize":  9,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "font.size":        10,
    })


def plot_overview(data: pd.DataFrame, events: pd.DataFrame, run_dir: Path) -> Path:
    """5-row time-series figure: TCP x, y, z, stick channels, TCP speed."""
    t = data["t_rel_s"].to_numpy()
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02

    lags = {ax: _xcorr_lag_ms(data[f"target_{ax}"].to_numpy(),
                              data[f"tcp_{ax}"].to_numpy(), dt)
            for ax in ("x", "y", "z")}

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, constrained_layout=True)

    # -- TCP x, y, z: target vs actual ------------------------------------------
    for ax, axis_name in zip(axes[:3], ("x", "y", "z")):
        ax.plot(t, data[f"target_{axis_name}"] * 1000, ls="--", lw=1.2,
                color=C_TARGET, label="commanded target")
        ax.plot(t, data[f"tcp_{axis_name}"]    * 1000, lw=1.6,
                color=C_ACTUAL, label="measured TCP")
        ax.set_ylabel(f"{axis_name}  [mm]")
        ax.legend(loc="upper right", ncol=2)
        lag = lags[axis_name]
        lag_str = f"{lag:.1f} ms" if np.isfinite(lag) else "n/a"
        ax.text(
            0.01, 0.96, f"tracking lag (xcorr): {lag_str}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="#555",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc", lw=0.7),
        )

    # -- Stick channels ---------------------------------------------------------
    ax_stick = axes[3]
    ax_stick.plot(t, data["dx"], lw=1.0, color=C_STICK_X, label="dx")
    ax_stick.plot(t, data["dy"], lw=1.0, color=C_STICK_Y, label="dy")
    ax_stick.plot(t, data["dz"], lw=1.0, color=C_STICK_Z, label="dz")
    ax_stick.axhline(0, color="#bbbbbb", lw=0.6)
    ax_stick.set_ylim(-1.08, 1.08)
    ax_stick.set_ylabel("stick")
    ax_stick.legend(loc="upper right", ncol=3)

    # -- TCP linear speed + latency annotations --------------------------------
    ax_speed = axes[4]
    ax_speed.plot(t, data["tcp_speed"] * 1000, lw=1.0, color=C_SPEED, label="|v_TCP|")
    ax_speed.set_ylabel("|v|  [mm/s]")
    ax_speed.set_xlabel("time  [s]")
    ax_speed.legend(loc="upper right")

    # -- Event markers across all rows ------------------------------------------
    ok      = events[events["status"] == "OK"]
    tout    = events[events["status"] == "TIMEOUT"]
    skipped = events[events["status"] == "SKIPPED"]

    def _vline(t_cmd: float, color: str, alpha: float, lw: float) -> None:
        for ax in axes:
            ax.axvline(t_cmd, color=color, ls=":", lw=lw, alpha=alpha)

    for _, ev in ok.iterrows():
        _vline(ev["t_cmd_rel_s"], C_OK, 0.55, 0.9)

    # Annotate latencies on the speed row, staggered if events are clustered.
    y_top = ax_speed.get_ylim()[1] if ax_speed.get_ylim()[1] > 0 else 1.0
    last_x = -np.inf
    stagger = 0
    for _, ev in ok.iterrows():
        x = ev["t_motion_rel_s"]
        if not np.isfinite(x):
            continue
        # If two events fall within < 2% of the timeline, stagger vertically so labels don't overlap.
        if x - last_x < (t[-1] - t[0]) * 0.02 if len(t) > 1 else False:
            stagger = (stagger + 1) % 3
        else:
            stagger = 0
        last_x = x
        y = y_top * (0.92 - 0.10 * stagger)
        ax_speed.annotate(
            f"{ev['latency_ms']:.0f} ms",
            xy=(x, y),
            ha="center", va="center",
            fontsize=8, color=C_OK,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=C_OK, lw=0.7),
        )

    for _, ev in tout.iterrows():
        _vline(ev["t_cmd_rel_s"], C_TIMEOUT, 0.45, 0.8)
    for _, ev in skipped.iterrows():
        _vline(ev["t_cmd_rel_s"], C_SKIPPED, 0.35, 0.7)

    fig.suptitle(
        f"UR10e telemetry — {run_dir.name}   "
        f"({len(ok)} OK, {len(tout)} timeouts, {len(skipped)} skipped)",
        fontsize=13, fontweight="semibold",
    )

    out = run_dir / "overview.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    return out


def plot_latency_stats(events: pd.DataFrame, run_dir: Path) -> Path:
    """Three-panel: distribution, latency-vs-time, status counts."""
    ok = events[events["status"] == "OK"]
    tout = events[events["status"] == "TIMEOUT"]
    skipped = events[events["status"] == "SKIPPED"]
    lat = ok["latency_ms"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # -- Histogram --------------------------------------------------------------
    ax = axes[0]
    if len(lat) >= 1:
        bins = max(8, min(30, int(np.sqrt(len(lat)) * 2)))
        ax.hist(lat, bins=bins, color=C_ACTUAL, edgecolor="white", linewidth=1.0)
        mean, med = float(np.mean(lat)), float(np.median(lat))
        ax.axvline(mean, color=C_OK, ls="--", lw=1.6, label=f"mean {mean:.1f}")
        ax.axvline(med,  color="#27ae60", ls="--", lw=1.6, label=f"median {med:.1f}")
        if len(lat) >= 5:
            p95 = float(np.percentile(lat, 95))
            ax.axvline(p95, color=C_TIMEOUT, ls="--", lw=1.6, label=f"p95 {p95:.1f}")
        ax.legend(loc="upper right")
    ax.set_xlabel("latency  [ms]")
    ax.set_ylabel("count")
    ax.set_title("Latency distribution (OK events)")

    # -- Latency over wall time -------------------------------------------------
    ax = axes[1]
    if len(lat):
        ax.plot(ok["t_cmd_rel_s"], lat, "o-",
                color=C_ACTUAL, ms=5, lw=0.9, mfc="white", mew=1.4)
        ax.set_xlabel("event time  [s]")
        ax.set_ylabel("latency  [ms]")
    ax.set_title("Latency vs run time (any drift?)")

    # -- Status breakdown -------------------------------------------------------
    ax = axes[2]
    counts = {"OK": len(ok), "TIMEOUT": len(tout), "SKIPPED": len(skipped)}
    bars = ax.bar(counts.keys(), counts.values(),
                  color=[C_ACTUAL, C_TIMEOUT, C_SKIPPED], edgecolor="white", linewidth=1.0)
    for bar, v in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(v),
                ha="center", va="bottom", fontsize=10, fontweight="semibold")
    ax.set_title("Event statuses")
    ax.set_ylabel("count")
    ax.margins(y=0.18)

    fig.suptitle(f"Latency stats — {run_dir.name}", fontsize=13, fontweight="semibold")

    out = run_dir / "latency_stats.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    return out


def _summary_lines(events: pd.DataFrame) -> list[str]:
    ok = events[events["status"] == "OK"]
    if not len(ok):
        return ["No OK events to summarise."]
    lat = ok["latency_ms"].to_numpy(dtype=float)
    lines = [
        f"OK events       : {len(ok)}",
        f"  mean latency  : {float(np.mean(lat)):7.1f} ms",
        f"  median        : {float(np.median(lat)):7.1f} ms",
        f"  min / max     : {float(np.min(lat)):7.1f} ms / {float(np.max(lat)):7.1f} ms",
    ]
    if len(lat) >= 5:
        lines.append(f"  p95           : {float(np.percentile(lat, 95)):7.1f} ms")
    if len(lat) >= 2:
        lines.append(f"  stdev         : {float(np.std(lat, ddof=1)):7.1f} ms")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Run directory with data.csv + events.csv. Default: latest under act_train/latency_runs/")
    ap.add_argument("--no-show", action="store_true",
                    help="Save PNGs but don't open the interactive viewer.")
    args = ap.parse_args()

    run_dir = args.run_dir or _latest_run()
    data_path = run_dir / "data.csv"
    events_path = run_dir / "events.csv"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}")
    if not events_path.exists():
        raise SystemExit(f"Missing {events_path}")

    print(f"Reading: {run_dir}")
    data = pd.read_csv(data_path)
    events = pd.read_csv(events_path)
    print(f"  data.csv:   {len(data):,} rows ({data['t_rel_s'].iloc[-1]:.1f} s span)" if len(data) else "  data.csv: empty")
    print(f"  events.csv: {len(events):,} rows")

    _apply_pro_style()
    overview_path = plot_overview(data, events, run_dir)
    print(f"  saved {overview_path.name}")
    stats_path = plot_latency_stats(events, run_dir)
    print(f"  saved {stats_path.name}")

    print()
    for line in _summary_lines(events):
        print("  " + line)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
