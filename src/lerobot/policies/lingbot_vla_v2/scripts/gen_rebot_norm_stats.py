#!/usr/bin/env python
"""Generate the lingbot_vla_v2 norm_stats JSON from a rebot LeRobot dataset's meta/stats.json.

Slot slices:
  observation.state.arm.position      <- observation.state[0:6]  (6 arm joints, degrees)
  observation.state.effector.position <- observation.state[6:7]  (gripper)
  action.* slices the action vector the same way.

Two normalization profiles (matching the official real/sim recipes):
  - real:     canonical_norm_type = meanstd — only mean/std needed (default)
  - robotwin: canonical_norm_type = bounds_99_woclip — needs the q01/q99 quantiles
Pass --quantiles to also emit q01/q99 so a single JSON works for both profiles
(only the robotwin conversion profile actually reads q01/q99).

A LeRobot dataset's meta/stats.json already carries real q01/q99 (computed by
RunningQuantileStats), so we just re-slice and re-store them — no need to
recompute from the raw data.

Usage:
  # real (meanstd, default)
  python -m lerobot.policies.lingbot_vla_v2.scripts.gen_rebot_norm_stats \
      --dataset-root /path/to/rebot_dataset --out rebot_norm_stats.json
  # robotwin (with q01/q99)
  python -m lerobot.policies.lingbot_vla_v2.scripts.gen_rebot_norm_stats \
      --dataset-root /path/to/rebot_dataset --quantiles --out rebot_norm_stats.robotwin.json
"""

import argparse
import json
from pathlib import Path

SLOT_SLICES = {
    "arm.position": (0, 6),       # shoulder_pan .. wrist_roll
    "effector.position": (6, 7),  # gripper
}

# The robotwin profile's bounds_99_woclip reads these two quantile keys.
QUANTILE_KEYS = ("q01", "q99")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="rebot LeRobot dataset root (containing meta/stats.json)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "rebot_norm_stats.json"))
    ap.add_argument(
        "--quantiles",
        action="store_true",
        help="also emit q01/q99 quantiles (for the robotwin profile's bounds_99_woclip)",
    )
    args = ap.parse_args()

    stats_path = Path(args.dataset_root) / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())

    out = {"norm_stats": {}}
    for raw_key in ("observation.state", "action"):
        if raw_key not in stats:
            raise KeyError(f"{raw_key} not found in {stats_path}; keys={list(stats.keys())}")
        feat = stats[raw_key]
        mean, std = feat["mean"], feat["std"]
        if len(mean) != 7:
            raise ValueError(f"{raw_key} should be 7-dim, got {len(mean)}")
        if args.quantiles:
            missing_q = [q for q in QUANTILE_KEYS if q not in feat]
            if missing_q:
                raise KeyError(
                    f"{raw_key} is missing quantiles {missing_q} (dataset too old? only newer "
                    f"LeRobot stats carry q01/q99). Available stat keys: {sorted(feat.keys())}"
                )
        for slot, (s, e) in SLOT_SLICES.items():
            # Guard: a near-constant dim (e.g. a mostly-static gripper) with a tiny std
            # would amplify noise, so clamp the std to a small floor.
            slot_std = [max(float(v), 1e-3) for v in std[s:e]]
            entry = {
                "mean": [float(v) for v in mean[s:e]],
                "std": slot_std,
            }
            if args.quantiles:
                for q in QUANTILE_KEYS:
                    entry[q] = [float(v) for v in feat[q][s:e]]
            out["norm_stats"][f"{raw_key}.{slot}"] = entry

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"written: {args.out}  (quantiles={'on' if args.quantiles else 'off'})")
    for k, v in out["norm_stats"].items():
        print(f"  {k}: dim={len(v['mean'])} keys={sorted(v.keys())}")


if __name__ == "__main__":
    main()
