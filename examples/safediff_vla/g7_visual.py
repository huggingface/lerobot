#!/usr/bin/env python
"""G7 paired visuals: top row raw G4d (G5 run), bottom row ORACLE-anchor run (same seed). Overlays as in g6_ab_visual (planner-target cross: in the oracle row it is the GT anchor; magenta star = what G4d itself selects, logged only)."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from g6_ab_visual import load, row, FIG  # noqa: E402


def case(name, seed, steps, note, raw, orc):
    fig, ax = plt.subplots(2, len(steps) + 1, figsize=(3.2 * (len(steps) + 1), 7.2)); row(ax[0], "g4d", seed, steps, raw); row(ax[1], "oracle", seed, steps, orc); fig.suptitle(note, fontsize=9); fig.tight_layout(); fig.savefig(FIG / f"g7_case_{name}.png", dpi=52); plt.close(fig)


def main():
    raw, orc = load("g4d"), load("oracle")
    case("1_raw_wrong_oracle_recovers_3003", 3003, [1, 40, 61, 120, 250, 450], "CASE 1 - raw anchor wrong (seed 3003: raw closes on a wrong card at t=61) -> oracle anchor", raw, orc)
    case("1b_raw_wrong_oracle_closest_3007", 3007, [1, 60, 160, 260, 360, 460], "CASE 1b - seed 3007: raw vs oracle (oracle gets within 0.1cm XY but never closes)", raw, orc)
    case("2_raw_correct_control_weak_3004", 3004, [1, 60, 120, 200, 300, 450], "CASE 2 - raw anchor mostly correct, control weak (seed 3004) -> does the oracle change anything?", raw, orc)
    case("3_oracle_wrong_card_interaction_3002", 3002, [1, 30, 60, 100, 200, 400], "CASE 3 - oracle run with wrong-card interaction (seed 3002; raw succeeded at t=67)", raw, orc)
    case("4_oracle_close_earlier_3009", 3009, [1, 30, 52, 62, 100, 160], "CASE 4 - oracle closes earlier near the correct card (seed 3009: close t=52 vs 163 raw)", raw, orc)
    case("5_oracle_fails_3008", 3008, [1, 60, 150, 250, 350, 480], "CASE 5 - oracle anchor but failure (seed 3008; raw succeeded at t=285)", raw, orc)
    print("saved")


if __name__ == "__main__":
    main()
