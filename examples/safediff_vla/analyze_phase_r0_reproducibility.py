#!/usr/bin/env python
"""Phase R0 analysis: pairwise EE-trajectory divergence across repeats, per episode/mode, plus the
monitor-on vs. monitor-off side-effect check under the deterministic configuration, plus a
DETERMINISTIC / PRACTICALLY DETERMINISTIC / CHAOTIC-SENSITIVE classification per episode/mode.

Usage:
    uv run python examples/safediff_vla/analyze_phase_r0_reproducibility.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_r0_reproducibility")
EPISODES = [("select_poker", 1001), ("select_fruit", 1001), ("select_book", 1002), ("select_mahjong", 1000)]
DIVERGENCE_THRESHOLD_M = 0.01  # presentation threshold for "first divergence step", not a tuned model parameter


def load_repeats(task: str, mode: str, monitor: str) -> list[dict]:
    d = OUT_DIR / mode / f"monitor_{monitor}" / task
    repeats = []
    for p in sorted(d.glob("repeat_*.json")):
        repeats.append(json.loads(p.read_text()))
    return repeats


def ee_xyz(record: dict) -> np.ndarray:
    steps = record.get("steps", [])
    return np.array([s["ee_pos"] for s in steps], dtype=float)


def pairwise_divergence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = min(len(a), len(b))
    return np.linalg.norm(a[:n] - b[:n], axis=1)


def first_divergence_step(div: np.ndarray, threshold: float) -> int | None:
    idxs = np.where(div > threshold)[0]
    return int(idxs[0]) if len(idxs) else None


def classify(max_final_div: float, len_mismatch: bool, success_mismatch: bool, close_step_mismatch: float | None) -> str:
    if len_mismatch or success_mismatch or max_final_div > 0.05 or (close_step_mismatch is not None and close_step_mismatch > 20):
        return "CHAOTIC-SENSITIVE"
    if max_final_div < 1e-4:
        return "DETERMINISTIC"
    return "PRACTICALLY DETERMINISTIC"


def close_step(record: dict) -> int | None:
    steps = record.get("steps", [])
    return next((i for i, s in enumerate(steps) if s.get("reactive_close_event")), None)


def main() -> None:
    summary = {}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Phase R0: pairwise EE-position divergence across repeats (top=current, bottom=deterministic)")

    for col, (task, seed) in enumerate(EPISODES):
        for row, mode in enumerate(["current", "deterministic"]):
            repeats = load_repeats(task, mode, "on")
            ax = axes[row, col]
            lens = [r["episode_len"] for r in repeats]
            successes = [r["success"] for r in repeats]
            closes = [close_step(r) for r in repeats]
            max_final_div = 0.0
            first_divs = []
            for i in range(len(repeats)):
                for j in range(i + 1, len(repeats)):
                    a, b = ee_xyz(repeats[i]), ee_xyz(repeats[j])
                    div = pairwise_divergence(a, b)
                    ax.plot(div, label=f"rep{i} vs rep{j}", lw=1)
                    max_final_div = max(max_final_div, float(div[-1]) if len(div) else 0.0)
                    fd = first_divergence_step(div, DIVERGENCE_THRESHOLD_M)
                    if fd is not None:
                        first_divs.append(fd)
            ax.axhline(DIVERGENCE_THRESHOLD_M, color="black", ls=":", lw=0.7)
            ax.set_title(f"{task} ({mode})\nlens={lens} success={successes}")
            ax.set_xlabel("step")
            ax.set_ylabel("||EE_i - EE_j|| (m)")
            ax.legend(fontsize=6)

            len_mismatch = len(set(lens)) > 1
            success_mismatch = len(set(successes)) > 1
            valid_closes = [c for c in closes if c is not None]
            close_step_mismatch = (max(valid_closes) - min(valid_closes)) if len(valid_closes) >= 2 else (
                None if len(valid_closes) == len(closes) else 999
            )
            verdict = classify(max_final_div, len_mismatch, success_mismatch, close_step_mismatch)

            summary[f"{task}__{mode}"] = {
                "episode_lens": lens, "successes": successes, "close_steps": closes,
                "max_final_pairwise_divergence_m": max_final_div,
                "first_divergence_steps": first_divs,
                "len_mismatch": len_mismatch, "success_mismatch": success_mismatch,
                "close_step_mismatch_steps": close_step_mismatch,
                "verdict": verdict,
            }

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "divergence_curves.png", dpi=110)
    plt.close(fig)

    # monitor-on vs monitor-off side-effect check, deterministic mode only
    print("=== monitor-on vs monitor-off (deterministic mode) side-effect check ===")
    monitor_check = {}
    for task, seed in EPISODES:
        on_repeats = load_repeats(task, "deterministic", "on")
        off_repeats = load_repeats(task, "deterministic", "off")
        if not on_repeats or not off_repeats:
            continue
        off = off_repeats[0]
        divs_vs_on = []
        for on in on_repeats:
            a, b = ee_xyz(on), ee_xyz(off)
            div = pairwise_divergence(a, b)
            divs_vs_on.append(float(div[-1]) if len(div) else None)
        lens_match = len({r["episode_len"] for r in on_repeats} | {off["episode_len"]}) == 1
        print(
            f"  {task}: monitor_off len={off['episode_len']} success={off['success']}  "
            f"vs monitor_on lens={[r['episode_len'] for r in on_repeats]} successes={[r['success'] for r in on_repeats]}  "
            f"final_div_vs_each_on_repeat={divs_vs_on}  all_lens_match={lens_match}"
        )
        monitor_check[task] = {
            "monitor_off_len": off["episode_len"], "monitor_off_success": off["success"],
            "monitor_on_lens": [r["episode_len"] for r in on_repeats],
            "final_div_vs_each_on_repeat": divs_vs_on, "all_lens_match": lens_match,
        }

    # captured nondeterminism warnings (deterministic mode)
    print("\n=== captured torch.use_deterministic_algorithms(warn_only=True) warnings ===")
    all_warnings = set()
    for task, seed in EPISODES:
        for monitor in ("on", "off"):
            for r in load_repeats(task, "deterministic", monitor):
                all_warnings.update(r.get("captured_warnings", []))
    if all_warnings:
        for w in sorted(all_warnings):
            print(f"  {w}")
    else:
        print("  (none captured across any deterministic-mode run)")

    print("\n=== per episode/mode verdict ===")
    for k, v in summary.items():
        print(
            f"  {k:30s} verdict={v['verdict']:24s} max_final_div={v['max_final_pairwise_divergence_m']:.4f}m "
            f"lens={v['episode_lens']} successes={v['successes']} close_steps={v['close_steps']}"
        )

    (OUT_DIR / "divergence_summary.json").write_text(
        json.dumps({"per_episode_mode": summary, "monitor_side_effect_check": monitor_check, "nondeterminism_warnings": sorted(all_warnings)}, indent=2)
    )
    print(f"\nwrote {OUT_DIR / 'divergence_summary.json'} and {OUT_DIR / 'divergence_curves.png'}")


if __name__ == "__main__":
    main()
