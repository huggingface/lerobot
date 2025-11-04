#!/usr/bin/env python

import argparse
import json
import time
import sys
import select
from pathlib import Path

from lerobot.utils.import_utils import register_third_party_devices
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader


SO101_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture SO101 leader joint degree ranges interactively")
    parser.add_argument("--leader-port", default="/dev/ttyACM0", help="Serial port for SO101 leader")
    parser.add_argument("--outfile", default="leader_ranges_so101.json", help="Where to save ranges JSON")
    parser.add_argument("--fps", type=int, default=30, help="Sampling rate while recording")
    args = parser.parse_args()

    register_third_party_devices()

    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, use_degrees=True))
    leader.connect()

    try:
        print("Connected to SO101 leader. We'll record min/max in DEGREES for each joint.")
        print("Instructions: For each joint, press ENTER to start, move the joint to its extremes, then press ENTER to stop.")

        ranges: dict[str, tuple[float, float]] = {}
        for jname in SO101_JOINTS:
            input(f"\nReady to record '{jname}'. Press ENTER to start...")
            print("Recording... move to both extremes. Press ENTER to stop this joint (or Ctrl+C).")
            min_v = float("inf")
            max_v = float("-inf")
            try:
                while True:
                    # Stop on ENTER without blocking the loop
                    if select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.readline()
                        break

                    act = leader.get_action()
                    v = float(act.get(f"{jname}.pos", 0.0))
                    updated = False
                    if v < min_v:
                        min_v = v
                        updated = True
                    if v > max_v:
                        max_v = v
                        updated = True
                    if updated:
                        print(f"{jname} range so far: ({min_v:.2f}, {max_v:.2f})")
                    time.sleep(1.0 / max(1, args.fps))
            except KeyboardInterrupt:
                # Treat Ctrl+C as stop for this joint (do not abort whole session)
                print("\nStopped this joint with Ctrl+C.")

            # ensure sensible ordering
            if min_v == float("inf") or max_v == float("-inf"):
                min_v, max_v = 0.0, 0.0
            if max_v < min_v:
                min_v, max_v = max_v, min_v

            ranges[jname] = (round(min_v, 2), round(max_v, 2))
            print(f"Captured {jname}: {ranges[jname]}")

        # Save JSON
        out_path = Path(args.outfile)
        out_path.write_text(json.dumps(ranges, indent=2))
        print(f"\nSaved ranges to {out_path.resolve()}")

        # Print Python dict for paste into tele_test.py
        print("\nPaste this into tele_test.py as 'leader_ranges':")
        print("{" + ", ".join([f"\"{k}\": ({v[0]}, {v[1]})" for k, v in ranges.items()]) + "}")

    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()


