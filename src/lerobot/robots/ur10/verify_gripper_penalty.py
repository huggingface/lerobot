#!/usr/bin/env python

"""
Thin wrapper around `verify_dataset` that runs only the gripper-penalty check.

Kept as a separate entry point for muscle memory; for new use prefer

    python -m lerobot.robots.ur10.verify_dataset --checks gripper_penalty ...

which lets you run the full audit battery (schema, finite, action_bounds, state_bounds,
gripper_penalty, gripper_activity, stationary_frames, timestamp_monotonicity,
episode_lengths) in a single dataset pass.

Usage (unchanged from the original script):
    python -m lerobot.robots.ur10.verify_gripper_penalty \
        --repo_id local/ur10_usb_insertion --penalty -0.02
"""

from __future__ import annotations

import sys

from lerobot.robots.ur10 import verify_dataset


def main() -> int:
    # Force --checks gripper_penalty regardless of what the user passes.
    argv = sys.argv[1:]
    if "--checks" not in argv:
        argv += ["--checks", "gripper_penalty"]
    sys.argv = [sys.argv[0]] + argv
    return verify_dataset.main()


if __name__ == "__main__":
    sys.exit(main())
